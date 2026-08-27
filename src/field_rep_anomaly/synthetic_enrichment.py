"""Deterministic representative mapping and business-coherent synthetic enrichment."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from .data_loader import stable_int


GRAIN = ["rep_id", "product_name", "territory_id", "date"]


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip().upper()).strip("_")
    return text[:24] or "UNKNOWN"


def _mode(series: pd.Series) -> str:
    values = series.dropna().astype(str)
    if values.empty:
        return "Unknown"
    modes = values.mode()
    return str(sorted(modes.tolist())[0])


def _safe_divide(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> np.ndarray:
    left = np.asarray(numerator, dtype=float)
    right = np.asarray(denominator, dtype=float)
    return np.divide(left, right, out=np.zeros_like(left, dtype=float), where=np.isfinite(right) & (np.abs(right) > 1e-12))


def create_rep_mapping(
    transactions: pd.DataFrame, reps_per_territory: int = 3, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign every customer deterministically to one synthetic rep per org-territory."""
    if reps_per_territory < 1:
        raise ValueError("reps_per_territory must be positive")
    frame = transactions.copy()
    frame["territory_id"] = [
        f"TERR_{_slug(country)}_{_slug(city)}" for country, city in zip(frame["country"], frame["city"])
    ]
    org_cols = ["sales_team", "sales_manager", "territory_id"]
    orgs = frame[org_cols].drop_duplicates().sort_values(org_cols, kind="stable").reset_index(drop=True)
    rep_rows: list[dict[str, Any]] = []
    counter = 1
    for org in orgs.itertuples(index=False):
        for slot in range(reps_per_territory):
            rep_id = f"REP_{counter:05d}"
            token = f"{org.sales_team}|{org.sales_manager}|{org.territory_id}|{slot}"
            rep_rows.append(
                {
                    "sales_team": org.sales_team,
                    "sales_manager": org.sales_manager,
                    "manager_id": f"MANAGER_{_slug(org.sales_manager)}",
                    "territory_id": org.territory_id,
                    "rep_slot": slot,
                    "rep_id": rep_id,
                    "rep_tenure_months": 6 + stable_int(token + "|tenure", seed) % 115,
                    "rep_capacity": 80 + stable_int(token + "|capacity", seed) % 41,
                }
            )
            counter += 1
    rep_table = pd.DataFrame(rep_rows)

    customers = (
        frame[org_cols + ["customer"]]
        .drop_duplicates()
        .sort_values(org_cols + ["customer"], kind="stable")
        .reset_index(drop=True)
    )
    customers["rep_slot"] = [
        stable_int("|".join(map(str, row)), seed) % reps_per_territory
        for row in customers[org_cols + ["customer"]].itertuples(index=False, name=None)
    ]
    mapping = customers.merge(rep_table, on=org_cols + ["rep_slot"], how="left", validate="many_to_one")
    if mapping["rep_id"].isna().any():
        raise RuntimeError("Rep mapping failed for one or more customers.")
    assigned = frame.merge(mapping, on=org_cols + ["customer"], how="left", validate="many_to_one")
    if assigned["rep_id"].isna().any():
        raise RuntimeError("Some transactions could not be assigned to a representative.")
    return assigned, mapping


def _hhi_by_group(data: pd.DataFrame, group_cols: list[str], category: str, value: str, name: str) -> pd.DataFrame:
    grouped = data.groupby(group_cols + [category], dropna=False, observed=True)[value].sum().rename("category_value").reset_index()
    total = grouped.groupby(group_cols, dropna=False, observed=True)["category_value"].transform("sum")
    grouped["share_sq"] = np.square(_safe_divide(grouped["category_value"], total))
    return grouped.groupby(group_cols, dropna=False, observed=True)["share_sq"].sum().rename(name).reset_index()


def _tiered_incentive_multiplier(attainment: np.ndarray) -> np.ndarray:
    attainment = np.asarray(attainment, dtype=float)
    return np.select(
        [attainment < 0.70, attainment < 0.90, attainment < 1.00, attainment < 1.20],
        [0.20 * attainment / 0.70, 0.60 + 1.0 * (attainment - 0.70), 0.80 + 2.0 * (attainment - 0.90), 1.00 + 1.5 * (attainment - 1.00)],
        default=np.minimum(2.0, 1.30 + 1.2 * (attainment - 1.20)),
    )


def aggregate_and_enrich(assigned: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Aggregate to rep-product-territory-month and add relational activity/target/incentive fields."""
    frame = assigned.dropna(subset=["date"]).copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp()
    for field in ("sales", "quantity", "price", "latitude", "longitude"):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    frame["sales"] = frame["sales"].fillna(0.0)
    frame["quantity"] = frame["quantity"].fillna(0.0)
    group_cols = GRAIN
    agg = (
        frame.groupby(group_cols, dropna=False, observed=True)
        .agg(
            total_sales=("sales", "sum"),
            total_quantity=("quantity", "sum"),
            average_price=("price", "mean"),
            unique_customers=("customer", "nunique"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            country=("country", _mode),
            city=("city", _mode),
            product_class=("product_class", _mode),
            channel=("channel", _mode),
            subchannel=("subchannel", _mode),
            sales_manager=("sales_manager", _mode),
            manager_id=("manager_id", "first"),
            sales_team=("sales_team", _mode),
            distributor=("distributor", _mode),
            rep_tenure_months=("rep_tenure_months", "first"),
            rep_capacity=("rep_capacity", "first"),
            transaction_count=("sales", "size"),
        )
        .reset_index()
    )
    agg["average_price"] = agg["average_price"].fillna(
        pd.Series(_safe_divide(agg["total_sales"], agg["total_quantity"]), index=agg.index)
    )

    customer_hhi = _hhi_by_group(frame, group_cols, "customer", "sales", "customer_concentration")
    channel_hhi = _hhi_by_group(frame, group_cols, "channel", "sales", "channel_mix")
    subchannel_hhi = _hhi_by_group(frame, group_cols, "subchannel", "sales", "subchannel_mix")
    agg = agg.merge(customer_hhi, on=group_cols, how="left").merge(channel_hhi, on=group_cols, how="left").merge(subchannel_hhi, on=group_cols, how="left")

    portfolio = assigned.groupby("rep_id", observed=True)["customer"].nunique().rename("assigned_customer_portfolio")
    agg = agg.merge(portfolio, on="rep_id", how="left")
    territory_stats = (
        assigned.groupby("territory_id", observed=True)
        .agg(
            territory_customer_count=("customer", "nunique"),
            territory_lat_spread=("latitude", "std"),
            territory_lon_spread=("longitude", "std"),
        )
        .reset_index()
    )
    spread_area = (1.0 + territory_stats["territory_lat_spread"].fillna(0).abs() * 111.0) * (
        1.0 + territory_stats["territory_lon_spread"].fillna(0).abs() * 85.0
    )
    territory_stats["territory_customer_density"] = territory_stats["territory_customer_count"] / spread_area.clip(lower=1.0)
    agg = agg.merge(territory_stats, on="territory_id", how="left")

    agg = agg.sort_values(["rep_id", "date", "product_name"], kind="stable").reset_index(drop=True)
    rep_month_cols = ["rep_id", "territory_id", "date"]
    rep_month_sales = agg.groupby(rep_month_cols, observed=True)["total_sales"].transform("sum")
    products_in_month = agg.groupby(rep_month_cols, observed=True)["product_name"].transform("nunique").clip(lower=1)
    product_share = pd.Series(_safe_divide(agg["total_sales"], rep_month_sales), index=agg.index)
    activity_weight = 0.30 / products_in_month + 0.70 * product_share
    weight_sum = activity_weight.groupby([agg[col] for col in rep_month_cols]).transform("sum")
    activity_weight = activity_weight / weight_sum.replace(0, 1)

    rng = np.random.default_rng(seed + 101)
    n = len(agg)
    month_position = (agg["date"].dt.year - agg["date"].dt.year.min()) * 12 + agg["date"].dt.month
    product_territory_median = agg.groupby(["product_name", "territory_id", "date"], observed=True)["total_sales"].transform("median")
    product_global_median = agg.groupby(["product_name", "date"], observed=True)["total_sales"].transform("median")
    expected_sales = product_territory_median.where(product_territory_median > 0, product_global_median).fillna(agg["total_sales"].median())
    target_noise = rng.lognormal(0.0, 0.10, n)
    agg["sales_target"] = np.maximum(1.0, expected_sales.to_numpy() * target_noise * (1.0 + 0.002 * month_position.to_numpy()))
    agg["quantity_target"] = np.maximum(1.0, _safe_divide(agg["sales_target"], agg["average_price"].replace(0, np.nan)))
    agg["target_attainment_pct"] = _safe_divide(agg["total_sales"], agg["sales_target"]) * 100.0

    rep_month_key = agg[rep_month_cols].astype(str).agg("|".join, axis=1)
    contact_fraction = np.clip(0.40 + 0.32 * np.tanh((rep_month_sales / rep_month_sales.median()) - 0.7) + rng.normal(0, 0.07, n), 0.20, 1.0)
    rep_month_contacts = np.maximum(1, np.rint(agg["assigned_customer_portfolio"] * contact_fraction)).astype(int)
    working_days = np.array([18 + stable_int(token + "|days", seed) % 6 for token in rep_month_key])
    rep_calls_total = np.maximum(rep_month_contacts, np.rint(rep_month_contacts * rng.uniform(2.0, 3.8, n))).astype(int)
    agg["unique_customers_contacted"] = np.maximum(1, np.rint(rep_month_contacts * (0.35 + 0.65 * np.sqrt(product_share.clip(0, 1))))).astype(int)
    agg["total_calls"] = np.maximum(1, np.rint(rep_calls_total * activity_weight)).astype(int)
    agg["priority_customer_calls"] = np.minimum(agg["total_calls"], np.rint(agg["total_calls"] * rng.uniform(0.22, 0.46, n))).astype(int)
    agg["digital_engagements"] = np.maximum(0, np.rint(agg["total_calls"] * rng.uniform(0.10, 0.35, n))).astype(int)
    agg["working_days"] = working_days
    base_travel = 4.0 + 0.14 * agg["territory_customer_count"] + 16.0 / np.maximum(agg["territory_customer_density"], 0.05)
    agg["travel_distance_km"] = np.maximum(1.0, base_travel * activity_weight * rng.lognormal(0.0, 0.18, n))
    agg["customer_coverage_pct"] = np.minimum(100.0, _safe_divide(agg["unique_customers_contacted"], agg["assigned_customer_portfolio"]) * 100.0)
    agg["call_plan_adherence_pct"] = np.clip(52.0 + 0.36 * agg["customer_coverage_pct"] + rng.normal(0, 6.0, n), 25.0, 100.0)

    territory_month_sales = agg.groupby(["territory_id", "date"], observed=True)["total_sales"].transform("sum")
    territory_baseline = agg.groupby("territory_id", observed=True)["total_sales"].transform("median")
    agg["territory_sales_potential"] = np.maximum(1.0, 0.65 * territory_month_sales + 0.35 * territory_baseline * products_in_month)
    customer_scale = agg["territory_customer_count"] / max(float(agg["territory_customer_count"].median()), 1.0)
    agg["territory_market_potential"] = agg["territory_sales_potential"] * (0.75 + 0.25 * customer_scale)
    agg["opportunity_index_raw"] = customer_scale * (
        agg["territory_market_potential"] / max(float(agg["territory_market_potential"].median()), 1.0)
    )
    agg["workload_index_raw"] = (
        0.55 * _safe_divide(agg["total_calls"], agg["rep_capacity"])
        + 0.45 * _safe_divide(agg["assigned_customer_portfolio"], agg["rep_capacity"])
    )

    tenure_factor = 0.90 + 0.0015 * np.minimum(agg["rep_tenure_months"], 80)
    agg["target_incentive"] = np.maximum(250.0, (720.0 + 0.018 * agg["sales_target"]) * tenure_factor)
    attainment = agg["target_attainment_pct"].to_numpy() / 100.0
    agg["calculated_incentive"] = agg["target_incentive"] * _tiered_incentive_multiplier(attainment)
    routine_adjustment = rng.normal(0.0, 0.018, n) * agg["calculated_incentive"]
    routine_override_mask = rng.random(n) < 0.018
    routine_adjustment = np.where(routine_override_mask, routine_adjustment * 2.5, routine_adjustment)
    agg["actual_incentive_paid"] = np.maximum(0.0, agg["calculated_incentive"] + routine_adjustment)
    agg["payout_adjustment"] = agg["actual_incentive_paid"] - agg["calculated_incentive"]
    agg["manual_override_amount"] = np.where(routine_override_mask, agg["payout_adjustment"], 0.0)
    agg["incentive_attainment_pct"] = _safe_divide(agg["actual_incentive_paid"], agg["target_incentive"]) * 100.0

    numeric = agg.select_dtypes(include=[np.number]).columns
    agg[numeric] = agg[numeric].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if agg.duplicated(GRAIN).any():
        raise RuntimeError("Analytical grain is not unique after aggregation.")
    return agg


def build_enriched_analytical_dataset(
    transactions: pd.DataFrame, reps_per_territory: int = 3, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the deterministic rep layer, analytical data, and field-lineage table."""
    assigned, mapping = create_rep_mapping(transactions, reps_per_territory=reps_per_territory, seed=seed)
    analytical = aggregate_and_enrich(assigned, seed=seed)
    original_like = {
        "total_sales", "total_quantity", "average_price", "unique_customers", "latitude", "longitude",
        "country", "city", "product_name", "product_class", "channel", "subchannel", "sales_manager",
        "sales_team", "distributor", "date", "transaction_count", "customer_concentration", "channel_mix", "subchannel_mix",
    }
    mapping_fields = {"rep_id", "territory_id", "manager_id", "rep_tenure_months", "rep_capacity", "assigned_customer_portfolio"}
    evaluation_fields = {"injected_anomaly_flag", "anomaly_type", "anomaly_severity"}
    lineage_rows = []
    for field in analytical.columns:
        if field in original_like:
            category = "aggregated_from_commercial_foundation"
        elif field in mapping_fields:
            category = "deterministic_synthetic_mapping"
        elif field in evaluation_fields:
            category = "evaluation_label_only"
        else:
            category = "business_coherent_synthetic_enrichment"
        lineage_rows.append({"field": field, "lineage": category})
    return analytical, mapping, pd.DataFrame(lineage_rows)
