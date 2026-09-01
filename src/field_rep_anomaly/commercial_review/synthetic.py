"""Relational, reproducible synthetic datasets anchored to observed sales rows."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from .foundation import stable_id
from .policy import build_policy_rules, calculate_incentives


def _mode(values: pd.Series, fallback: str = "Unknown") -> str:
    modes = values.dropna().astype(str).mode()
    return modes.sort_values().iloc[0] if len(modes) else fallback


def _safe_ratio(numerator: Any, denominator: Any) -> np.ndarray:
    a = np.asarray(numerator, dtype=float)
    b = np.asarray(denominator, dtype=float)
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=np.abs(b) > 1e-12)


def _quantile_band(values: pd.Series, labels: list[str]) -> pd.Series:
    ranked = values.rank(method="average", pct=True).fillna(0.5)
    bins = np.linspace(0, 1, len(labels) + 1)
    return pd.cut(ranked, bins=bins, labels=labels, include_lowest=True).astype(str)


def _month_end(period: pd.Series) -> pd.Series:
    return pd.to_datetime(period) + pd.offsets.MonthEnd(1)


def _configured_payout_date(period: pd.Timestamp, config: dict[str, Any]) -> pd.Timestamp:
    period = pd.Timestamp(period)
    for version in config["synthetic"]["policy_versions"]:
        if pd.Timestamp(version["effective_start_date"]) <= period <= pd.Timestamp(
            version["effective_end_date"]
        ):
            return period + pd.offsets.MonthEnd(1) + pd.Timedelta(
                days=int(version["payout_delay_days"])
            )
    raise ValueError(f"No configured payout delay for period {period.date()}")


def build_masters(
    source: pd.DataFrame, config: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    """Create stable rep, manager, team, customer, product, and territory masters."""
    seed = int(config["project"]["seed"])
    rng = np.random.default_rng(seed)
    latest = pd.Timestamp(source["period"].max())
    currency = str(config["project"].get("currency_code", "UNK"))
    reference_end = pd.Timestamp(config.get("model", {}).get("train_end", latest))
    reference_mask = pd.to_datetime(source["period"]).le(reference_end)
    # Freeze behavior-derived master attributes before validation/test.  If an
    # entity first appears later, include only its first observed row so the
    # relational lookup remains complete without importing its future history.
    for key in ["rep_id", "manager_id", "team_id", "customer_id", "product_id", "territory_id"]:
        known = set(source.loc[reference_mask, key].dropna())
        missing = set(source[key].dropna()) - known
        if missing:
            first_indices = (
                source.loc[source[key].isin(missing)]
                .sort_values(["period", "source_row_id"], kind="mergesort")
                .groupby(key, observed=True)
                .head(1)
                .index
            )
            reference_mask.loc[first_indices] = True
    reference_source = source.loc[reference_mask].copy()

    territory_group = reference_source.groupby(
        ["territory_id", "territory_name", "team_id", "team_name", "country"], observed=True
    )
    territory = (
        territory_group.agg(
            city=("city", _mode),
            customer_count=("customer_id", "nunique"),
            source_sales=("sales", "sum"),
            source_transaction_count=("transaction_id", "nunique"),
            source_rep_count=("rep_id", "nunique"),
            source_latitude_centroid=("latitude", "mean"),
            source_longitude_centroid=("longitude", "mean"),
            distinct_cities=("city", "nunique"),
            distinct_products=("product_id", "nunique"),
        )
        .reset_index()
        .sort_values("territory_id")
        .reset_index(drop=True)
    )
    territory["state"] = territory["country"]
    territory["region"] = territory["country"]
    territory["urbanicity"] = np.select(
        [territory["distinct_cities"].ge(300), territory["distinct_cities"].ge(120)],
        ["urban", "mixed"],
        default="distributed",
    )
    territory["territory_potential"] = (
        50
        + 25
        * (
            territory["source_sales"].rank(pct=True)
            + territory["customer_count"].rank(pct=True)
            - 1
        )
    ).clip(1, 100)
    territory["customer_density"] = territory["customer_count"] / territory["distinct_cities"].clip(lower=1)
    lat_span = territory_group["latitude"].std().reset_index(name="lat_span")["lat_span"].fillna(0)
    lon_span = territory_group["longitude"].std().reset_index(name="lon_span")["lon_span"].fillna(0)
    territory["travel_complexity_index"] = (
        1
        + np.hypot(lat_span.to_numpy(), lon_span.to_numpy()) / 8
        + territory["distinct_cities"].rank(pct=True) * 2
    ).clip(1, 5)
    territory["average_distance_between_customers"] = (
        5 + 11 * territory["travel_complexity_index"] + rng.normal(0, 2, len(territory))
    ).clip(3, 85)
    territory["product_complexity_index"] = (
        1 + 4 * territory["distinct_products"].rank(pct=True)
    )
    territory["expected_monthly_workload_hours"] = (
        territory["customer_count"] * 0.85 + territory["source_transaction_count"] / 28 * 0.05
    )
    territory["expected_rep_capacity_hours"] = territory["source_rep_count"] * 118.0
    territory["priority_customer_count"] = 0
    territory["currency_code"] = currency
    territory["attribute_reference_end_date"] = reference_end
    territory["data_lineage"] = "synthetic_derived"

    customer = (
        reference_source.groupby(["customer_id", "customer_name"], observed=True)
        .agg(
            territory_id=("territory_id", _mode),
            primary_rep_id=("rep_id", _mode),
            geography=("city", _mode),
            country=("country", _mode),
            channel=("channel", _mode),
            customer_type=("sub_channel", _mode),
            source_sales=("sales", "sum"),
            expected_monthly_volume=("quantity", lambda values: float(values.abs().sum()) / 28),
            source_return_rows=("sales", lambda values: int(values.lt(0).sum())),
            source_rows=("transaction_id", "size"),
            usual_product_classes=("product_class", lambda values: "|".join(sorted(set(values.astype(str))))),
        )
        .reset_index()
        .sort_values("customer_id")
        .reset_index(drop=True)
    )
    customer["customer_segment"] = _quantile_band(
        customer["source_sales"].abs(), ["emerging", "core", "strategic"]
    )
    customer["potential_score"] = (
        10 + 90 * customer["source_sales"].abs().rank(pct=True)
    ).clip(1, 100)
    customer["expected_sales_band"] = _quantile_band(
        customer["source_sales"].abs(), ["low", "medium", "high"]
    )
    customer["required_visit_frequency"] = np.select(
        [customer["potential_score"].ge(75), customer["potential_score"].ge(40)],
        [3, 2],
        default=1,
    ).astype(int)
    customer["customer_priority"] = np.select(
        [customer["potential_score"].ge(75), customer["potential_score"].ge(40)],
        ["high", "medium"],
        default="low",
    )
    customer["historical_return_rate"] = _safe_ratio(
        customer["source_return_rows"], customer["source_rows"]
    )
    centroids = territory.set_index("territory_id")[["source_latitude_centroid", "source_longitude_centroid"]]
    customer = customer.join(centroids, on="territory_id")
    jitter_scale = 0.08 + customer["territory_id"].map(
        territory.set_index("territory_id")["travel_complexity_index"]
    ).fillna(2) * 0.025
    customer["synthetic_latitude"] = customer["source_latitude_centroid"].fillna(50) + rng.normal(
        0, jitter_scale
    )
    customer["synthetic_longitude"] = customer["source_longitude_centroid"].fillna(15) + rng.normal(
        0, jitter_scale
    )
    customer["coordinate_lineage"] = "synthetic around source territory centroid; no geocoding"
    customer["currency_code"] = currency
    customer["attribute_reference_end_date"] = reference_end
    customer["data_lineage"] = "synthetic_derived"
    customer = customer.drop(columns=["source_latitude_centroid", "source_longitude_centroid"])
    priority_counts = customer.loc[customer["customer_priority"].eq("high")].groupby("territory_id").size()
    territory["priority_customer_count"] = territory["territory_id"].map(priority_counts).fillna(0).astype(int)

    product = (
        reference_source.groupby(["product_id", "product_name"], observed=True)
        .agg(
            product_class=("product_class", _mode),
            first_observed_period=("period", "min"),
            list_price=("price", "median"),
            source_sales=("sales", "sum"),
            source_quantity=("quantity", lambda values: float(values.abs().sum())),
        )
        .reset_index()
        .sort_values("product_id")
        .reset_index(drop=True)
    )
    product["product_category"] = product["product_class"]
    product["launch_date"] = product["first_observed_period"] - pd.to_timedelta(
        rng.integers(180, 1800, len(product)), unit="D"
    )
    product["expected_price_band"] = (
        (product["list_price"] * 0.82).round(2).astype(str)
        + "–"
        + (product["list_price"] * 1.18).round(2).astype(str)
    )
    product["margin_pct"] = rng.uniform(0.28, 0.68, len(product))
    product["incentive_eligible_flag"] = rng.random(len(product)) > 0.08
    incentive_weight_draw = rng.random(len(product))
    product["incentive_weight"] = np.select(
        [incentive_weight_draw < 0.20, incentive_weight_draw < 0.50],
        [1.35, 1.15],
        default=1.0,
    )
    product["expected_discount_pct"] = rng.uniform(0.025, 0.105, len(product))
    product["expected_return_rate"] = rng.uniform(0.008, 0.055, len(product))
    product["product_complexity_score"] = rng.uniform(1, 5, len(product))
    product["required_call_intensity"] = np.select(
        [product["product_complexity_score"].ge(4), product["product_complexity_score"].ge(2.5)],
        [3, 2],
        default=1,
    )
    product["currency_code"] = currency
    product["attribute_reference_end_date"] = reference_end
    product["data_lineage"] = "synthetic_derived"

    rep = (
        reference_source.groupby(["rep_id", "rep_name"], observed=True)
        .agg(
            manager_id=("manager_id", _mode),
            team_id=("team_id", _mode),
            territory_id=("territory_id", _mode),
            first_observed_period=("period", "min"),
            last_observed_period=("period", "max"),
            product_specialization=("product_class", lambda values: _mode(values)),
            source_sales=("sales", "sum"),
        )
        .reset_index()
        .sort_values("rep_id")
        .reset_index(drop=True)
    )
    rep["hire_date"] = rep["first_observed_period"] - pd.to_timedelta(
        rng.integers(90, 2200, len(rep)), unit="D"
    )
    rep["tenure_months"] = (
        (latest.year - rep["hire_date"].dt.year) * 12 + latest.month - rep["hire_date"].dt.month
    ).clip(lower=1)
    rep["employment_status"] = "active"
    rep["role_grade"] = np.select(
        [rep["tenure_months"].ge(60), rep["tenure_months"].ge(30)],
        ["senior", "experienced"],
        default="developing",
    )
    rep["historical_performance_band"] = _quantile_band(
        rep["source_sales"], ["developing", "core", "leading"]
    )
    rep["standard_field_hours_per_day"] = 8.0
    rep["standard_working_days_per_month"] = 21
    rep["baseline_visit_capacity"] = np.select(
        [rep["role_grade"].eq("senior"), rep["role_grade"].eq("experienced")],
        [92, 84],
        default=74,
    )
    rep["training_hours"] = rng.integers(4, 13, len(rep))
    rep["administrative_hours"] = rng.integers(16, 29, len(rep))
    rep["leave_days"] = rng.integers(0, 3, len(rep))
    rep["monthly_available_hours"] = (
        rep["standard_field_hours_per_day"]
        * (rep["standard_working_days_per_month"] - rep["leave_days"])
        - rep["training_hours"]
        - rep["administrative_hours"]
    ).clip(lower=40)
    rep["currency_code"] = currency
    rep["attribute_reference_end_date"] = reference_end
    rep["data_lineage"] = "synthetic_derived"

    team = (
        reference_source.groupby(["team_id", "team_name"], observed=True)
        .agg(
            manager_id=("manager_id", _mode),
            region=("country", lambda values: "|".join(sorted(set(values.astype(str))))),
            active_rep_count=("rep_id", "nunique"),
        )
        .reset_index()
    )
    team["management_span"] = team["active_rep_count"]
    team["attribute_reference_end_date"] = reference_end
    team["data_lineage"] = "synthetic_derived"
    manager = (
        reference_source.groupby(["manager_id", "manager_name"], observed=True)
        .agg(
            team_id=("team_id", _mode),
            team_name=("team_name", _mode),
            region=("country", lambda values: "|".join(sorted(set(values.astype(str))))),
            active_rep_count=("rep_id", "nunique"),
        )
        .reset_index()
    )
    manager["management_span"] = manager["active_rep_count"]
    manager["attribute_reference_end_date"] = reference_end
    manager["data_lineage"] = "synthetic_derived"
    return {
        "rep_master": rep,
        "manager_master": manager,
        "team_master": team,
        "customer_master": customer,
        "product_master": product,
        "territory_master": territory,
    }


def build_orders_discounts_returns(
    source: pd.DataFrame,
    rep_master: pd.DataFrame,
    customer_master: pd.DataFrame,
    product_master: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create line-level order, discount, and return facts without altering source sales."""
    seed = int(config["project"]["seed"])
    rng = np.random.default_rng(seed + 11)
    currency = str(config["project"].get("currency_code", "UNK"))
    source_columns = [
        "source_row_id",
        "transaction_id",
        "transaction_date",
        "period",
        "rep_id",
        "customer_id",
        "product_id",
        "territory_id",
        "quantity",
        "price",
        "sales",
        "channel",
    ]
    order = source[source_columns].copy()
    order["order_id"] = order["transaction_id"].map(lambda value: stable_id("ORD", value, length=16))
    order["order_line_id"] = order["transaction_id"].map(lambda value: stable_id("LINE", value, length=16))
    order["order_date"] = pd.to_datetime(order["transaction_date"])
    order["invoice_date"] = order["order_date"] + pd.to_timedelta(rng.integers(0, 3, len(order)), unit="D")
    order["fulfillment_date"] = order["invoice_date"] + pd.to_timedelta(rng.integers(1, 7, len(order)), unit="D")
    order = order.merge(
        product_master[["product_id", "expected_discount_pct"]],
        on="product_id",
        how="left",
        validate="many_to_one",
    ).merge(
        customer_master[["customer_id", "customer_segment"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )
    segment_adjustment = order["customer_segment"].map(
        {"emerging": 0.0, "core": 0.01, "strategic": 0.025}
    ).fillna(0)
    channel_adjustment = order["channel"].astype(str).str.casefold().map(
        {"hospital": 0.012, "pharmacy": 0.0}
    ).fillna(0)
    volume_adjustment = np.log1p(order["quantity"].abs().clip(lower=0)) * 0.003
    order["discount_pct"] = (
        order["expected_discount_pct"].fillna(0.06)
        + segment_adjustment
        + channel_adjustment
        + volume_adjustment
        + rng.normal(0, float(config["synthetic"]["discount_noise_sd"]), len(order))
    ).clip(0, 0.28)
    order["unit_list_price"] = order["price"]
    order["source_observed_sales"] = order["sales"]
    order["gross_sales"] = order["source_observed_sales"]
    order["discount_amount"] = order["gross_sales"].abs() * order["discount_pct"]
    order["net_sales"] = order["gross_sales"] - np.sign(order["gross_sales"]) * order["discount_amount"]
    order["order_status"] = np.where(order["gross_sales"].lt(0), "source_return", "fulfilled")
    order["payment_status"] = np.where(rng.random(len(order)) < 0.97, "paid", "pending")
    order["end_of_period_flag"] = order["order_date"].dt.day.ge(25)
    order["currency_code"] = currency
    order["data_lineage"] = "source_observed"
    order = order.drop(columns=["sales", "price", "expected_discount_pct", "customer_segment"])

    approvals = order["discount_pct"].gt(0.15)
    manager_by_rep = rep_master.set_index("rep_id")["manager_id"]
    discount = pd.DataFrame(
        {
            "discount_id": [f"DISC_{index:09d}" for index in range(1, len(order) + 1)],
            "order_line_id": order["order_line_id"],
            "rep_id": order["rep_id"],
            "discount_pct": order["discount_pct"],
            "discount_amount": order["discount_amount"],
            "discount_reason": np.select(
                [order["discount_pct"].ge(0.15), order["discount_pct"].ge(0.08)],
                ["volume or contract exception", "customer segment pricing"],
                default="standard commercial discount",
            ),
            "approval_required_flag": approvals,
            "approved_flag": ~approvals | (rng.random(len(order)) < 0.96),
            "approver_id": np.where(
                approvals, order["rep_id"].map(manager_by_rep), pd.NA
            ),
            "approval_date": pd.to_datetime(order["order_date"]) - pd.to_timedelta(
                rng.integers(0, 4, len(order)), unit="D"
            ),
            "exception_flag": approvals,
            "currency_code": currency,
            "data_lineage": "synthetic_normal",
        }
    )

    returns: list[dict[str, Any]] = []
    negative = order.loc[order["gross_sales"].lt(0)].copy()
    positive_orders = order.loc[order["gross_sales"].gt(0)].sort_values(
        ["order_date", "order_line_id"], kind="mergesort"
    )
    for index, row in enumerate(negative.itertuples(index=False), start=1):
        return_date = pd.Timestamp(row.order_date)
        prior = positive_orders.loc[
            pd.to_datetime(positive_orders["order_date"]).le(return_date)
        ]
        candidates = prior.loc[
            prior["customer_id"].eq(row.customer_id)
            & prior["product_id"].eq(row.product_id)
        ]
        if candidates.empty:
            candidates = prior.loc[prior["customer_id"].eq(row.customer_id)]
        if candidates.empty:
            candidates = prior.loc[prior["rep_id"].eq(row.rep_id)]
        if candidates.empty:
            linked = pd.Series(row._asdict())
        else:
            linked = candidates.iloc[-1]
        original_date = pd.Timestamp(linked["order_date"])
        days = int((return_date - original_date).days)
        payout_period = pd.Timestamp(linked["period"])
        returns.append(
            {
                "return_id": f"RET_SRC_{index:08d}",
                "order_id": linked["order_id"],
                "order_line_id": linked["order_line_id"],
                "rep_id": linked["rep_id"],
                "customer_id": linked["customer_id"],
                "product_id": linked["product_id"],
                "original_order_date": original_date,
                "return_date": return_date,
                "return_quantity": abs(float(row.quantity)),
                "return_amount": abs(float(row.net_sales)),
                "cancellation_flag": False,
                "return_reason": "source-observed negative commercial line",
                "payout_period": payout_period,
                "after_incentive_payout_flag": return_date
                > _configured_payout_date(payout_period, config),
                "days_after_order": days,
                "currency_code": currency,
                "data_lineage": "source_observed",
            }
        )
    positive = order.loc[order["gross_sales"].gt(0)].copy()
    expected_return = product_master.set_index("product_id")["expected_return_rate"]
    probability = positive["product_id"].map(expected_return).fillna(
        float(config["synthetic"]["base_return_rate"])
    )
    selected = positive.loc[rng.random(len(positive)) < probability.to_numpy()].copy()
    selected = selected.reset_index(drop=True)
    for index, row in enumerate(selected.itertuples(index=False), start=len(returns) + 1):
        days = int(rng.integers(7, 62))
        return_date = pd.Timestamp(row.order_date) + pd.Timedelta(days=days)
        fraction = float(rng.choice([0.25, 0.5, 1.0], p=[0.42, 0.38, 0.20]))
        cancellation = bool(rng.random() < float(config["synthetic"]["base_cancellation_rate"]) * 4)
        returns.append(
            {
                "return_id": f"RET_SYN_{index:08d}",
                "order_id": row.order_id,
                "order_line_id": row.order_line_id,
                "rep_id": row.rep_id,
                "customer_id": row.customer_id,
                "product_id": row.product_id,
                "original_order_date": pd.Timestamp(row.order_date),
                "return_date": return_date,
                "return_quantity": abs(float(row.quantity)) * fraction,
                "return_amount": abs(float(row.net_sales)) * fraction,
                "cancellation_flag": cancellation,
                "return_reason": "customer stock adjustment" if not cancellation else "order cancellation",
                "payout_period": pd.Timestamp(row.period),
                "after_incentive_payout_flag": return_date
                > _configured_payout_date(pd.Timestamp(row.period), config),
                "days_after_order": days,
                "currency_code": currency,
                "data_lineage": "synthetic_normal",
            }
        )
    returns_frame = pd.DataFrame(returns)
    order_columns = [
        "order_id",
        "order_line_id",
        "source_row_id",
        "source_observed_sales",
        "order_date",
        "invoice_date",
        "fulfillment_date",
        "period",
        "rep_id",
        "customer_id",
        "product_id",
        "territory_id",
        "quantity",
        "unit_list_price",
        "gross_sales",
        "discount_pct",
        "discount_amount",
        "net_sales",
        "order_status",
        "payment_status",
        "end_of_period_flag",
        "currency_code",
        "data_lineage",
    ]
    return order[order_columns].reset_index(drop=True), discount.reset_index(drop=True), returns_frame


def build_targets(
    orders: pd.DataFrame,
    rep_master: pd.DataFrame,
    customer_master: pd.DataFrame,
    product_master: pd.DataFrame,
    territory_master: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Generate prior-informed targets that are not fixed multiples of current sales."""
    rng = np.random.default_rng(int(config["project"]["seed"]) + 23)
    product_plan = product_master.set_index("product_id")
    target_orders = orders.copy()
    target_orders["_positive_sales"] = target_orders["net_sales"].clip(lower=0.0)
    target_orders["_priority_product_flag"] = (
        target_orders["product_id"]
        .map(product_plan["incentive_eligible_flag"])
        .fillna(False)
        & target_orders["product_id"]
        .map(product_plan["incentive_weight"])
        .fillna(1.0)
        .ge(1.15)
    )
    target_orders["_priority_product_sales"] = target_orders[
        "_positive_sales"
    ].where(target_orders["_priority_product_flag"], 0.0)
    base = (
        target_orders.groupby(["rep_id", "period"], observed=True)
        .agg(
            actual_sales=("net_sales", "sum"),
            positive_sales=("_positive_sales", "sum"),
            actual_units=("quantity", "sum"),
            actual_priority_product_sales=("_priority_product_sales", "sum"),
        )
        .reset_index()
        .sort_values(["rep_id", "period"])
    )
    base["prior_sales_median"] = base.groupby("rep_id")["actual_sales"].transform(
        lambda values: values.shift().rolling(3, min_periods=1).median()
    )
    base["prior_units_median"] = base.groupby("rep_id")["actual_units"].transform(
        lambda values: values.shift().rolling(3, min_periods=1).median()
    )
    prior_positive_sales = base.groupby("rep_id")["positive_sales"].transform(
        lambda values: values.cumsum().shift(fill_value=0.0)
    )
    prior_priority_sales = base.groupby("rep_id")[
        "actual_priority_product_sales"
    ].transform(lambda values: values.cumsum().shift(fill_value=0.0))
    base["prior_priority_product_mix"] = np.divide(
        prior_priority_sales.to_numpy(float),
        prior_positive_sales.to_numpy(float),
        out=np.full(len(base), np.nan, dtype=float),
        where=prior_positive_sales.to_numpy(float) > 0,
    )
    base["prior_priority_product_mix"] = base[
        "prior_priority_product_mix"
    ].fillna(
        float(config["synthetic"].get("cold_start_priority_product_mix", 0.25))
    ).clip(0.05, 0.75)
    # Territory context for a target must be known when that target becomes
    # effective.  Derive the dominant rep-period territory and its potential
    # from cumulative prior orders; the descriptive all-history master value is
    # deliberately not used here.
    territory_counts = (
        orders.groupby(["rep_id", "period", "territory_id"], observed=True)
        .size()
        .rename("order_count")
        .reset_index()
        .sort_values(
            ["rep_id", "period", "order_count", "territory_id"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        .drop_duplicates(["rep_id", "period"])
    )
    territory_sales = (
        orders.assign(_abs_sales=orders["net_sales"].abs())
        .groupby(["territory_id", "period"], observed=True)["_abs_sales"]
        .sum()
        .rename("period_sales")
        .reset_index()
        .sort_values(["territory_id", "period"], kind="mergesort")
    )
    territory_sales["prior_sales"] = territory_sales.groupby("territory_id", sort=False)[
        "period_sales"
    ].transform(lambda values: values.cumsum().shift(fill_value=0.0))
    territory_sales["prior_sales_rank"] = territory_sales.groupby("period", observed=True)[
        "prior_sales"
    ].rank(pct=True, method="average")
    no_prior_sales = territory_sales.groupby("period", observed=True)["prior_sales"].transform("max").le(0)
    territory_sales["territory_potential"] = (
        25.0 + 75.0 * territory_sales["prior_sales_rank"]
    ).where(~no_prior_sales, 50.0).clip(1.0, 100.0)
    base = base.merge(
        territory_counts[["rep_id", "period", "territory_id"]],
        on=["rep_id", "period"],
        how="left",
        validate="one_to_one",
    ).merge(
        territory_sales[["territory_id", "period", "territory_potential"]],
        on=["territory_id", "period"],
        how="left",
        validate="many_to_one",
    ).merge(
        rep_master[["rep_id", "hire_date"]], on="rep_id", how="left", validate="many_to_one"
    )
    base["territory_potential"] = base["territory_potential"].fillna(50.0)
    # Cold-start assumptions are explicit configuration values.  They avoid
    # filling the first target with that representative's all-time (future)
    # sales median.
    base["prior_sales_median"] = base["prior_sales_median"].fillna(
        float(config["synthetic"].get("cold_start_target_sales", 16_500_000.0))
    )
    base["prior_units_median"] = base["prior_units_median"].fillna(
        float(config["synthetic"].get("cold_start_target_units", 45_000.0))
    )
    monthly_sales = (
        orders.groupby("period", observed=True)["net_sales"].sum().sort_index()
    )
    seasonality_by_period: dict[pd.Timestamp, float] = {}
    for period, _ in monthly_sales.items():
        period = pd.Timestamp(period)
        history = monthly_sales.loc[monthly_sales.index < period]
        same_month = history.loc[history.index.month == period.month]
        if history.empty or same_month.empty or abs(float(history.mean())) < 1e-12:
            seasonality_by_period[period] = 1.0
        else:
            seasonality_by_period[period] = float(same_month.mean() / history.mean())
    season = base["period"].map(seasonality_by_period).fillna(1.0).clip(0.7, 1.3)
    tenure_months = (
        (base["period"].dt.year - base["hire_date"].dt.year) * 12
        + base["period"].dt.month
        - base["hire_date"].dt.month
    ).clip(lower=1)
    tenure_factor = np.where(tenure_months < 12, 0.88, np.where(tenure_months < 30, 0.96, 1.02))
    potential_factor = 0.85 + base["territory_potential"] / 330
    product_mix_factor = 0.94 + 0.12 * (
        base["prior_priority_product_mix"] / 0.25
    ).clip(0.5, 1.5)
    noise = rng.normal(1.0, float(config["synthetic"]["target_random_variation"]), len(base))
    base["target_sales"] = (
        base["prior_sales_median"].abs()
        * (0.72 + 0.18 * season + 0.10 * potential_factor)
        * product_mix_factor
        * tenure_factor
        * noise
    ).clip(lower=1)
    base["target_units"] = (
        base["prior_units_median"].abs() * (0.80 + 0.20 * season) * tenure_factor * noise
    ).clip(lower=1)
    base["target_priority_product_sales"] = (
        base["target_sales"]
        * base["prior_priority_product_mix"]
        * rng.uniform(0.95, 1.08, len(base))
    ).clip(lower=0.0)
    base["target_new_customer_sales"] = base["target_sales"] * rng.uniform(0.02, 0.08, len(base))
    customer_frequency = customer_master.groupby("primary_rep_id")["required_visit_frequency"].sum()
    base["target_visit_count"] = (
        base["rep_id"].map(customer_frequency).fillna(60) / 2.8
    ).round().clip(20, 110).astype(int)
    base["quota_difficulty_index"] = (
        base["target_sales"] / base["prior_sales_median"].abs().clip(lower=1)
    ).clip(0.5, 2.0)
    revision = rng.random(len(base)) < 0.05
    base["target_revision_flag"] = revision
    base["target_effective_date"] = base["period"] - pd.to_timedelta(
        np.where(revision, rng.integers(3, 13, len(base)), rng.integers(20, 46, len(base))), unit="D"
    )
    base["target_version"] = np.where(revision, 2, 1)
    base["currency_code"] = str(config["project"].get("currency_code", "UNK"))
    base["data_lineage"] = "synthetic_normal"
    columns = [
        "rep_id",
        "period",
        "target_sales",
        "target_units",
        "target_priority_product_sales",
        "target_new_customer_sales",
        "target_visit_count",
        "quota_difficulty_index",
        "target_revision_flag",
        "target_effective_date",
        "target_version",
        "currency_code",
        "data_lineage",
    ]
    return base[columns].sort_values(["period", "rep_id"]).reset_index(drop=True)


def build_visits_crm_expenses(
    source: pd.DataFrame,
    customer_master: pd.DataFrame,
    product_master: pd.DataFrame,
    territory_master: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate correlated but imperfect visits, CRM interactions, and expenses."""
    rng = np.random.default_rng(int(config["project"]["seed"]) + 37)
    currency = str(config["project"].get("currency_code", "UNK"))
    activity = (
        source.groupby(["rep_id", "customer_id", "territory_id", "period"], observed=True)
        .agg(
            order_date=("transaction_date", "min"),
            current_period_sales=("sales", "sum"),
            product_discussed=("product_name", _mode),
        )
        .reset_index()
    )
    next_period_sales = activity[
        ["rep_id", "customer_id", "territory_id", "period", "current_period_sales"]
    ].copy()
    next_period_sales["period"] = (
        pd.to_datetime(next_period_sales["period"]) - pd.offsets.MonthBegin(1)
    )
    next_period_sales = next_period_sales.rename(
        columns={"current_period_sales": "future_sales_opportunity"}
    )
    activity = activity.merge(
        next_period_sales,
        on=["rep_id", "customer_id", "territory_id", "period"],
        how="left",
        validate="one_to_one",
    )
    activity["future_sales_opportunity"] = activity[
        "future_sales_opportunity"
    ].fillna(0.0)
    activity = activity.merge(
        customer_master[
            ["customer_id", "customer_priority", "required_visit_frequency", "synthetic_latitude", "synthetic_longitude"]
        ],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )
    activity = activity.merge(
        territory_master[
            [
                "territory_id",
                "source_latitude_centroid",
                "source_longitude_centroid",
                "average_distance_between_customers",
            ]
        ],
        on="territory_id",
        how="left",
        validate="many_to_one",
    )
    latitude_km = (
        activity["synthetic_latitude"] - activity["source_latitude_centroid"]
    ).abs() * 111.0
    longitude_km = (
        activity["synthetic_longitude"] - activity["source_longitude_centroid"]
    ).abs() * 111.0 * np.cos(np.deg2rad(activity["synthetic_latitude"].fillna(50.0)))
    activity["expected_travel_distance_km"] = (
        0.40 * activity["average_distance_between_customers"].fillna(20.0)
        + np.hypot(latitude_km.fillna(0.0), longitude_km.fillna(0.0))
    ).clip(1.0, 180.0)

    rates = config["synthetic"]["visit_sampling_rate"]
    probability = activity["customer_priority"].map(rates).fillna(rates["medium"]).to_numpy(float)
    cadence_factor = 0.70 + 0.30 * activity["required_visit_frequency"].fillna(1).to_numpy(float)
    future_sales_rank = activity.groupby(["rep_id", "period"])[
        "future_sales_opportunity"
    ].rank(pct=True).fillna(0.5)
    distance_rank = activity.groupby(["rep_id", "period"])[
        "expected_travel_distance_km"
    ].rank(pct=True).fillna(0.5)
    future_sales_factor = 0.72 + 0.58 * future_sales_rank.to_numpy(float)
    distance_factor = 1.28 - 0.52 * distance_rank.to_numpy(float)
    probability = np.clip(
        probability * cadence_factor * future_sales_factor * distance_factor,
        0.03,
        0.92,
    )
    sampling_draw = rng.random(len(activity))
    activity["_visit_probability"] = probability
    activity["_sampling_draw"] = sampling_draw
    visits = activity.loc[sampling_draw < probability].copy()
    max_visits = int(config["synthetic"].get("max_normal_visits_per_rep_period", 84))
    if not 1 <= max_visits <= 84:
        raise ValueError("max_normal_visits_per_rep_period must be between 1 and 84")
    visits = (
        visits.sort_values(
            ["rep_id", "period", "_visit_probability", "_sampling_draw", "customer_id"],
            ascending=[True, True, False, True, True],
            kind="mergesort",
        )
        .groupby(["rep_id", "period"], observed=True, group_keys=False)
        .head(max_visits)
    )
    visits = visits.sort_values(["rep_id", "period", "customer_id"]).reset_index(drop=True)
    visits["slot"] = visits.groupby(["rep_id", "period"]).cumcount()
    visits["visit_date"] = visits["period"] + pd.to_timedelta(visits["slot"] // 4, unit="D")
    visits["scheduled_minute"] = 8 * 60 + visits["slot"].mod(4) * 150
    start_jitter = rng.integers(-5, 11, len(visits))
    duration = (
        rng.normal(float(config["synthetic"]["average_visit_minutes"]), 10, len(visits))
        + visits["customer_priority"].map({"high": 8, "medium": 2, "low": -4}).fillna(0)
    ).round().clip(18, 90).astype(int)
    visits["scheduled_start_time"] = pd.to_datetime(visits["visit_date"]) + pd.to_timedelta(
        visits["scheduled_minute"], unit="m"
    )
    visits["actual_start_time"] = visits["scheduled_start_time"] + pd.to_timedelta(start_jitter, unit="m")
    visits["visit_duration_minutes"] = duration
    visits["actual_end_time"] = visits["actual_start_time"] + pd.to_timedelta(duration, unit="m")
    visits["visit_type"] = np.where(rng.random(len(visits)) < 0.86, "in-person", "hybrid")
    visits["visit_outcome"] = rng.choice(
        ["follow-up requested", "product interest", "routine coverage", "no immediate action"],
        size=len(visits),
        p=[0.24, 0.30, 0.34, 0.12],
    )
    visits["check_in_latitude"] = visits["synthetic_latitude"] + rng.normal(0, 0.004, len(visits))
    visits["check_in_longitude"] = visits["synthetic_longitude"] + rng.normal(0, 0.004, len(visits))
    visits["check_out_latitude"] = visits["check_in_latitude"] + rng.normal(0, 0.002, len(visits))
    visits["check_out_longitude"] = visits["check_in_longitude"] + rng.normal(0, 0.002, len(visits))
    visits["estimated_travel_km"] = (
        visits["expected_travel_distance_km"].to_numpy(float)
        * rng.lognormal(mean=0.0, sigma=0.25, size=len(visits))
    ).clip(1, 30)
    visits["visit_completed_flag"] = rng.random(len(visits)) < float(
        config["synthetic"]["visit_completion_rate"]
    )
    visits["impossible_travel_flag"] = False
    visits["overlapping_visit_flag"] = False
    visits["data_lineage"] = "synthetic_normal"
    visits["visit_id"] = [f"VIS_{index:09d}" for index in range(1, len(visits) + 1)]
    visit_columns = [
        "visit_id",
        "rep_id",
        "customer_id",
        "territory_id",
        "period",
        "visit_date",
        "scheduled_start_time",
        "actual_start_time",
        "actual_end_time",
        "visit_duration_minutes",
        "visit_type",
        "product_discussed",
        "visit_outcome",
        "check_in_latitude",
        "check_in_longitude",
        "check_out_latitude",
        "check_out_longitude",
        "estimated_travel_km",
        "visit_completed_flag",
        "impossible_travel_flag",
        "overlapping_visit_flag",
        "data_lineage",
    ]
    visits = visits[visit_columns].reset_index(drop=True)

    physical = visits[
        ["visit_id", "rep_id", "customer_id", "period", "actual_start_time", "product_discussed", "visit_outcome"]
    ].copy()
    physical["interaction_type"] = "physical visit"
    extra = activity.loc[
        rng.random(len(activity)) < float(config["synthetic"]["crm_supplement_rate"])
    ].copy()
    extra["actual_start_time"] = pd.to_datetime(extra["period"]) + pd.to_timedelta(
        rng.integers(1, 25, len(extra)), unit="D"
    ) + pd.to_timedelta(rng.integers(9, 17, len(extra)), unit="h")
    extra["visit_id"] = ""
    extra["visit_outcome"] = rng.choice(
        ["follow-up requested", "information shared", "no response"], size=len(extra)
    )
    extra["interaction_type"] = rng.choice(
        ["call", "email", "virtual meeting"], size=len(extra), p=[0.45, 0.38, 0.17]
    )
    physical = physical.rename(columns={"actual_start_time": "interaction_datetime"})
    extra = extra.rename(columns={"actual_start_time": "interaction_datetime"})
    crm = pd.concat(
        [
            physical[
                ["visit_id", "rep_id", "customer_id", "period", "interaction_datetime", "interaction_type", "product_discussed", "visit_outcome"]
            ],
            extra[
                ["visit_id", "rep_id", "customer_id", "period", "interaction_datetime", "interaction_type", "product_discussed", "visit_outcome"]
            ],
        ],
        ignore_index=True,
    )
    crm = crm.rename(columns={"product_discussed": "product_focus", "visit_outcome": "interaction_outcome"})
    crm["follow_up_required_flag"] = crm["interaction_outcome"].astype(str).str.contains("follow-up")
    crm["next_action_date"] = pd.to_datetime(crm["interaction_datetime"]).dt.normalize() + pd.to_timedelta(
        rng.integers(3, 22, len(crm)), unit="D"
    )
    crm["sentiment_or_interest_score"] = rng.normal(0.62, 0.18, len(crm)).clip(0, 1)
    crm["data_lineage"] = "synthetic_normal"
    crm["interaction_id"] = [f"CRM_{index:09d}" for index in range(1, len(crm) + 1)]
    crm = crm[
        [
            "interaction_id",
            "visit_id",
            "rep_id",
            "customer_id",
            "period",
            "interaction_datetime",
            "interaction_type",
            "product_focus",
            "interaction_outcome",
            "follow_up_required_flag",
            "next_action_date",
            "sentiment_or_interest_score",
            "data_lineage",
        ]
    ]

    expense = visits.loc[visits["visit_completed_flag"]].copy()
    expense = expense.loc[rng.random(len(expense)) < 0.78].reset_index(drop=True)
    expense["expense_date"] = pd.to_datetime(expense["visit_date"])
    expense["expense_category"] = rng.choice(
        ["mileage", "local transit", "meal"], size=len(expense), p=[0.72, 0.18, 0.10]
    )
    expense["claimed_distance_km"] = (
        expense["estimated_travel_km"] * rng.normal(1.02, 0.08, len(expense))
    ).clip(lower=0)
    expense["estimated_distance_km"] = expense["estimated_travel_km"]
    rate = float(config["synthetic"]["expense_rate_per_km"])
    expense["expected_amount"] = np.where(
        expense["expense_category"].eq("meal"),
        rng.uniform(12, 32, len(expense)),
        expense["estimated_travel_km"] * rate,
    )
    expense["claimed_amount"] = (
        expense["expected_amount"] * rng.normal(1.01, 0.07, len(expense))
    ).clip(lower=0)
    expense["receipt_available_flag"] = rng.random(len(expense)) < 0.94
    expense["approval_status"] = np.where(
        expense["receipt_available_flag"], "approved", "pending documentation"
    )
    expense["deviation_pct"] = 100 * _safe_ratio(
        expense["claimed_amount"] - expense["expected_amount"], expense["expected_amount"]
    )
    expense["currency_code"] = currency
    expense["data_lineage"] = "synthetic_normal"
    expense["expense_id"] = [f"EXP_{index:09d}" for index in range(1, len(expense) + 1)]
    expense = expense[
        [
            "expense_id",
            "rep_id",
            "visit_id",
            "period",
            "expense_date",
            "expense_category",
            "claimed_distance_km",
            "estimated_distance_km",
            "claimed_amount",
            "expected_amount",
            "receipt_available_flag",
            "approval_status",
            "deviation_pct",
            "currency_code",
            "data_lineage",
        ]
    ]
    return visits, crm, expense


def generate_clean_datasets(
    normalized_source: pd.DataFrame, config: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    """Generate all relational clean datasets except the separately built capacity layer."""
    masters = build_masters(normalized_source, config)
    orders, discounts, returns = build_orders_discounts_returns(
        normalized_source,
        masters["rep_master"],
        masters["customer_master"],
        masters["product_master"],
        config,
    )
    targets = build_targets(
        orders,
        masters["rep_master"],
        masters["customer_master"],
        masters["product_master"],
        masters["territory_master"],
        config,
    )
    visits, crm, expenses = build_visits_crm_expenses(
        normalized_source,
        masters["customer_master"],
        masters["product_master"],
        masters["territory_master"],
        config,
    )
    policies = build_policy_rules(config)
    incentives = calculate_incentives(
        orders,
        discounts,
        returns,
        targets,
        masters["product_master"],
        masters["rep_master"],
        policies,
        str(config["project"].get("currency_code", "UNK")),
    )
    return {
        **masters,
        "rep_targets_quotas": targets,
        "incentive_policy_rules": policies,
        "orders": orders,
        "discount_detail": discounts,
        "returns_cancellations": returns,
        "field_visits": visits,
        "crm_interactions": crm,
        "travel_expenses": expenses,
        "incentive_calculations": incentives,
    }
