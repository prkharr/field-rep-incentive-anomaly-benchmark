"""Deterministic rep-period workload and field-capacity calculations.

The hours model in this module is deliberately additive to the repository's
existing normalized-workload planning method.  ``legacy_normalized_workload_index``
preserves that count-based formula for comparison, while the remaining fields
translate a synthetic service calendar into explicit hours and FTE quantities.

This module does not inject anomalies and does not make hiring decisions.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


LEGACY_WORKLOAD_METRICS = (
    "distinct_customers",
    "transaction_count",
    "distinct_cities",
    "distinct_products",
    "distributor_count",
)

DEFAULT_WORKLOAD_WEIGHTS = {
    "distinct_customers": 0.40,
    "transaction_count": 0.25,
    "distinct_cities": 0.15,
    "distinct_products": 0.10,
    "distributor_count": 0.10,
}

CAPACITY_KEY = ["rep_id", "period"]
CAPACITY_TERRITORY_KEY = ["rep_id", "period", "territory_id"]
TERRITORY_PERIOD_KEY = ["territory_id", "period"]

# These quantities are additive across an allocation.  Ratios, risk bands,
# distinct counts, and normalized workload indexes are deliberately excluded;
# they are recomputed (where meaningful) after territory-period aggregation.
CAPACITY_ALLOCATION_ADDITIVE_COLUMNS = (
    "transaction_count",
    "working_days",
    "leave_days",
    "holiday_days",
    "training_hours",
    "administrative_hours",
    "meeting_hours",
    "gross_rostered_field_hours",
    "non_field_hours",
    "available_field_hours",
    "planned_visit_count",
    "completed_visit_count",
    "credited_planned_visit_count",
    "credited_completed_visit_count",
    "observed_visit_count",
    "observed_completed_visit_count",
    "planned_visit_hours",
    "planned_travel_hours",
    "observed_visit_hours",
    "observed_travel_hours",
    "excess_service_visit_count",
    "excess_service_visit_hours",
    "excess_service_travel_hours",
    "excess_service_hours",
    "required_visit_count",
    "priority_required_visit_count",
    "required_customer_coverage_hours",
    "required_priority_customer_coverage_hours",
    "customer_coverage_gap",
    "priority_customer_coverage_gap",
    "core_required_hours",
    "workload_buffer_hours",
    "required_total_hours",
    "required_hours",
    "available_hours",
    "nominal_full_time_hours",
    "required_fte",
    "available_fte",
    "fte_gap",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _config_sections(config: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]:
    complete = _mapping(config)
    wrappers = {"capacity", "model", "synthetic", "project", "planning", "anomalies"}
    # A mapping without any known top-level section is the capacity subsection.
    # This also supports a valid one-setting call such as
    # {"administrative_buffer_pct": 0.05}.
    capacity = _mapping(complete.get("capacity")) if wrappers.intersection(complete) else complete
    model = _mapping(complete.get("model"))
    synthetic = _mapping(complete.get("synthetic"))
    project = _mapping(complete.get("project"))
    seed = int(complete.get("seed", project.get("seed", capacity.get("seed", 42))))
    return capacity, model, synthetic, seed


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _month(values: pd.Series, label: str) -> pd.Series:
    result = pd.to_datetime(values, errors="coerce").dt.to_period("M").dt.to_timestamp()
    if result.isna().any():
        raise ValueError(f"{label} contains invalid or missing dates")
    return result


def _stable_number(token: str, seed: int, salt: str, low: float, high: float) -> float:
    digest = hashlib.sha256(f"{seed}|{salt}|{token}".encode("utf-8")).hexdigest()
    fraction = int(digest[:15], 16) / float(16**15 - 1)
    return low + (high - low) * fraction


def _seeded_values(
    frame: pd.DataFrame,
    seed: int,
    salt: str,
    low: float,
    high: float,
    *,
    integers: bool = False,
) -> pd.Series:
    tokens = frame["rep_id"].astype(str) + "|" + frame["period"].dt.strftime("%Y-%m")
    values = tokens.map(lambda token: _stable_number(token, seed, salt, low, high))
    if integers:
        values = np.floor(values + 1e-12)
    return pd.Series(values, index=frame.index, dtype=float)


def _numeric_or_seeded(
    frame: pd.DataFrame,
    column: str,
    seed: int,
    low: float,
    high: float,
    *,
    integers: bool = False,
) -> pd.Series:
    fallback = _seeded_values(frame, seed, column, low, high, integers=integers)
    if column not in frame:
        return fallback
    actual = pd.to_numeric(frame[column], errors="coerce")
    return actual.where(actual.notna() & np.isfinite(actual), fallback).astype(float)


def _deterministic_mode(values: pd.Series) -> Any:
    present = values.dropna()
    if present.empty:
        return pd.NA
    modes = present.mode(dropna=True)
    return sorted(modes.astype(str).tolist())[0]


def _coerce_boolean(values: pd.Series, default: bool = False) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(default).astype(bool)
    normalized = values.astype("string").str.strip().str.casefold()
    truthy = normalized.isin(["true", "1", "1.0", "yes", "y", "completed"])
    falsey = normalized.isin(["false", "0", "0.0", "no", "n", "cancelled", "canceled"])
    result = pd.Series(default, index=values.index, dtype=bool)
    result.loc[truthy] = True
    result.loc[falsey] = False
    return result


def _prepare_transactions(normalized_transactions: pd.DataFrame) -> pd.DataFrame:
    transactions = normalized_transactions.copy(deep=True)
    _require_columns(transactions, ["rep_id", "customer_id"], "normalized transactions")
    if "period" not in transactions:
        _require_columns(transactions, ["transaction_date"], "normalized transactions")
        transactions["period"] = transactions["transaction_date"]
    transactions["period"] = _month(transactions["period"], "normalized transactions.period")
    if transactions.empty:
        raise ValueError("normalized transactions cannot be empty")
    if transactions[["rep_id", "customer_id"]].isna().any().any():
        raise ValueError("normalized transactions contain null representative or customer identifiers")
    optional = {
        "city": "__UNKNOWN_CITY__",
        "product_id": "__UNKNOWN_PRODUCT__",
        "product_name": "__UNKNOWN_PRODUCT__",
        "distributor": "__UNKNOWN_DISTRIBUTOR__",
    }
    for column, fallback in optional.items():
        if column not in transactions:
            transactions[column] = fallback
        else:
            transactions[column] = transactions[column].fillna(fallback)
    return transactions


def _prepare_visits(field_visits: pd.DataFrame | None, average_visit_minutes: float, average_speed_kmh: float) -> pd.DataFrame:
    if field_visits is None or field_visits.empty:
        return pd.DataFrame(
            columns=CAPACITY_KEY
            + [
                "customer_id",
                "territory_id",
                "territory_name",
                "planned_visit_count",
                "completed_visit_count",
                "planned_visit_hours",
                "planned_travel_hours",
            ]
        )
    visits = field_visits.copy(deep=True)
    _require_columns(visits, ["rep_id", "customer_id"], "field visits")
    if "period" in visits:
        visits["period"] = _month(visits["period"], "field visits.period")
    else:
        _require_columns(visits, ["visit_date"], "field visits")
        visits["period"] = _month(visits["visit_date"], "field visits.visit_date")

    duration = pd.to_numeric(visits.get("visit_duration_minutes"), errors="coerce")
    if duration is None:
        duration = pd.Series(np.nan, index=visits.index, dtype=float)
    if {"actual_start_time", "actual_end_time"}.issubset(visits):
        starts = pd.to_datetime(visits["actual_start_time"], errors="coerce")
        ends = pd.to_datetime(visits["actual_end_time"], errors="coerce")
        elapsed = (ends - starts).dt.total_seconds() / 60.0
        duration = duration.where(duration.gt(0), elapsed)
    visits["_visit_hours"] = duration.where(duration.gt(0), average_visit_minutes).fillna(average_visit_minutes) / 60.0

    if "planned_travel_hours" in visits:
        travel = pd.to_numeric(visits["planned_travel_hours"], errors="coerce")
    elif "estimated_travel_km" in visits:
        travel = pd.to_numeric(visits["estimated_travel_km"], errors="coerce") / average_speed_kmh
    else:
        travel = pd.Series(0.0, index=visits.index)
    visits["_travel_hours"] = travel.where(travel.ge(0), 0.0).fillna(0.0)
    completed_source = visits.get("visit_completed_flag", pd.Series(True, index=visits.index))
    visits["_completed"] = _coerce_boolean(completed_source, default=True).astype(int)
    # Retain the actual visited territory when it is available.  The compact
    # rep-period calendar does not need this dimension, but the dashboard's
    # territory workload fact does: a cross-territory call belongs to the place
    # actually served, not to a rep's dominant sales territory.
    visit_dimensions = [
        column for column in ["territory_id", "territory_name"] if column in visits
    ]
    result = (
        visits.groupby(
            CAPACITY_KEY + ["customer_id"] + visit_dimensions,
            observed=True,
            dropna=False,
        )
        .agg(
            planned_visit_count=("customer_id", "size"),
            completed_visit_count=("_completed", "sum"),
            planned_visit_hours=("_visit_hours", "sum"),
            planned_travel_hours=("_travel_hours", "sum"),
        )
        .reset_index()
    )
    return result


def _rep_identity(rep_master: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    reps = rep_master.copy(deep=True)
    _require_columns(reps, ["rep_id"], "rep master")
    if reps.empty or reps["rep_id"].isna().any() or reps["rep_id"].duplicated().any():
        raise ValueError("rep master must contain one non-null row per rep_id")
    missing = set(transactions["rep_id"]) - set(reps["rep_id"])
    if missing:
        raise ValueError(f"rep master is missing normalized-transaction representatives: {sorted(missing)}")

    identity_fields = [
        "rep_name",
        "manager_id",
        "manager_name",
        "team_id",
        "team_name",
        "territory_id",
        "territory_name",
    ]
    for field in identity_fields:
        if field not in reps:
            if field in transactions:
                lookup = transactions.groupby("rep_id", observed=True)[field].agg(_deterministic_mode)
                reps[field] = reps["rep_id"].map(lookup)
            else:
                reps[field] = pd.NA
    return reps


def _period_territory(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return dominant display territory plus the full fractional allocation."""
    columns = CAPACITY_KEY + [
        "dynamic_territory_id",
        "dynamic_territory_name",
        "active_territory_count",
        "dominant_territory_activity_share",
        "fractional_territory_allocation",
    ]
    if "territory_id" not in transactions:
        return pd.DataFrame(columns=columns)

    present = transactions.dropna(subset=["territory_id"]).copy()
    if present.empty:
        return pd.DataFrame(columns=columns)
    counts = (
        present.groupby(CAPACITY_KEY + ["territory_id"], observed=True, dropna=False)
        .size()
        .rename("activity_count")
        .reset_index()
    )
    counts["_territory_sort"] = counts["territory_id"].astype(str)
    counts["_total_activity"] = counts.groupby(CAPACITY_KEY, observed=True)[
        "activity_count"
    ].transform("sum")
    counts["_activity_share"] = counts["activity_count"] / counts[
        "_total_activity"
    ].clip(lower=1)
    allocations = (
        counts.sort_values(CAPACITY_KEY + ["_territory_sort"], kind="mergesort")
        .assign(
            _allocation_token=lambda frame: frame["territory_id"].astype(str)
            + ":"
            + frame["_activity_share"].map(lambda value: f"{value:.8f}")
        )
        .groupby(CAPACITY_KEY, observed=True)
        .agg(
            active_territory_count=("territory_id", "nunique"),
            fractional_territory_allocation=("_allocation_token", "|".join),
        )
        .reset_index()
    )
    dominant = (
        counts.sort_values(
            CAPACITY_KEY + ["activity_count", "_territory_sort"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        .drop_duplicates(CAPACITY_KEY)
        .rename(
            columns={
                "territory_id": "dynamic_territory_id",
                "_activity_share": "dominant_territory_activity_share",
            }
        )
    )
    dominant = dominant[
        CAPACITY_KEY
        + ["dynamic_territory_id", "dominant_territory_activity_share"]
    ].merge(allocations, on=CAPACITY_KEY, how="left", validate="one_to_one")

    if "territory_name" in present:
        names = (
            present.groupby(CAPACITY_KEY + ["territory_id"], observed=True, dropna=False)[
                "territory_name"
            ]
            .agg(_deterministic_mode)
            .rename("dynamic_territory_name")
            .reset_index()
            .rename(columns={"territory_id": "dynamic_territory_id"})
        )
        dominant = dominant.merge(
            names,
            on=CAPACITY_KEY + ["dynamic_territory_id"],
            how="left",
            validate="one_to_one",
        )
    else:
        dominant["dynamic_territory_name"] = dominant["dynamic_territory_id"].astype(str)
    return dominant[columns]


def _legacy_workload(
    transactions: pd.DataFrame,
    periods: pd.DatetimeIndex,
    reps: pd.DataFrame,
    capacity_config: dict[str, Any],
    model_config: dict[str, Any],
) -> pd.DataFrame:
    product_field = "product_id" if "product_id" in transactions else "product_name"
    loads = (
        transactions.groupby(CAPACITY_KEY, observed=True)
        .agg(
            distinct_customers=("customer_id", "nunique"),
            transaction_count=("customer_id", "size"),
            distinct_cities=("city", "nunique"),
            distinct_products=(product_field, "nunique"),
            distributor_count=("distributor", "nunique"),
        )
        .reset_index()
    )
    grid = reps[["rep_id"]].assign(_join=1).merge(
        pd.DataFrame({"period": periods, "_join": 1}), on="_join", how="inner"
    ).drop(columns="_join")
    grid = grid.merge(loads, on=CAPACITY_KEY, how="left", validate="one_to_one")
    grid[list(LEGACY_WORKLOAD_METRICS)] = grid[list(LEGACY_WORKLOAD_METRICS)].fillna(0.0)

    raw_weights = _mapping(capacity_config.get("workload_weights")) or DEFAULT_WORKLOAD_WEIGHTS
    weights = {metric: float(raw_weights.get(metric, 0.0)) for metric in LEGACY_WORKLOAD_METRICS}
    if not all(np.isfinite(value) and value >= 0 for value in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("capacity workload weights must be finite, nonnegative, and contain a positive weight")
    unknown = sorted(set(raw_weights).difference(LEGACY_WORKLOAD_METRICS))
    if unknown:
        raise ValueError(f"unsupported legacy workload metrics: {unknown}")

    explicit_scales = _mapping(
        capacity_config.get("training_medians", capacity_config.get("workload_training_medians"))
    )
    train_end_value = capacity_config.get("train_end", model_config.get("train_end"))
    training = loads
    if train_end_value is not None:
        train_end = pd.Timestamp(train_end_value).to_period("M").to_timestamp()
        candidate = loads.loc[loads["period"].le(train_end)]
        if not candidate.empty:
            training = candidate
    scales: dict[str, float] = {}
    for metric in LEGACY_WORKLOAD_METRICS:
        value = explicit_scales.get(metric, training[metric].median())
        value = float(value) if pd.notna(value) else 1.0
        if not np.isfinite(value):
            raise ValueError(f"legacy training median for {metric} must be finite")
        scales[metric] = max(value, 1.0)
        grid[f"legacy_{metric}_training_median"] = scales[metric]
    grid["legacy_normalized_workload_index"] = sum(
        weights[metric] * grid[metric] / scales[metric] for metric in LEGACY_WORKLOAD_METRICS
    )
    grid["legacy_workload_training_end"] = (
        pd.Timestamp(train_end_value).to_period("M").to_timestamp() if train_end_value is not None else training["period"].max()
    )
    grid["legacy_workload_formula"] = "; ".join(
        f"{metric}: weight={weights[metric]:g}, training_median={scales[metric]:g}"
        for metric in LEGACY_WORKLOAD_METRICS
    )
    return grid


def _visit_frequency(
    values: pd.Series,
    priorities: pd.Series,
    numeric_period_divisor: float,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce") / numeric_period_divisor
    labels = values.astype("string").str.strip().str.casefold()
    mapped = labels.map(
        {
            "weekly": 4.0,
            "biweekly": 2.0,
            "fortnightly": 2.0,
            "monthly": 1.0,
            "quarterly": 1.0 / 3.0,
            "high": 2.0,
            "medium": 1.0,
            "low": 0.5,
        }
    )
    priority_default = priorities.astype("string").str.casefold().map(
        {"high": 2.0, "medium": 1.0, "low": 0.5}
    ).fillna(1.0)
    result = numeric.fillna(mapped).fillna(priority_default)
    return result.clip(lower=0.0).astype(float)


def _customer_assignment(
    transactions: pd.DataFrame,
    visits: pd.DataFrame,
    customers: pd.DataFrame,
    reps: pd.DataFrame,
    periods: pd.DatetimeIndex,
    capacity_config: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    master = customers.copy(deep=True)
    _require_columns(master, ["customer_id"], "customer master")
    if master.empty or master["customer_id"].isna().any() or master["customer_id"].duplicated().any():
        raise ValueError("customer master must contain one non-null row per customer_id")
    missing = set(transactions["customer_id"]) - set(master["customer_id"])
    if missing:
        raise ValueError(f"customer master is missing normalized-transaction customers: {sorted(missing)}")

    rep_ids = set(reps["rep_id"])
    # Preserve an explicit/static portfolio owner only as the no-activity
    # fallback. Exact-period transaction and visit activity is authoritative.
    owner_fields = [
        field for field in ("primary_rep_id", "assigned_rep_id", "rep_id") if field in master
    ]
    master["_static_owner_rep_id"] = pd.Series(pd.NA, index=master.index, dtype="object")
    for owner_field in owner_fields:
        master["_static_owner_rep_id"] = master["_static_owner_rep_id"].where(
            master["_static_owner_rep_id"].notna(), master[owner_field]
        )
    # Avoid a duplicate rep_id when the final dynamic owner is materialized.
    master = master.drop(columns=[column for column in ["rep_id", "assigned_rep_id"] if column in master])

    territory_reps: dict[Any, list[Any]] = {}
    if "territory_id" in reps:
        for territory, group in reps.dropna(subset=["territory_id"]).groupby("territory_id", observed=True):
            territory_reps[territory] = sorted(group["rep_id"].tolist(), key=str)
    all_reps = sorted(rep_ids, key=str)
    for index in master.index[master["_static_owner_rep_id"].isna()]:
        territory = master.at[index, "territory_id"] if "territory_id" in master else None
        candidates = territory_reps.get(territory, all_reps)
        if not candidates:
            raise ValueError("customer assignment requires at least one representative")
        token = str(master.at[index, "customer_id"])
        selected = int(_stable_number(token, seed, "customer_owner", 0, len(candidates)))
        master.at[index, "_static_owner_rep_id"] = candidates[min(selected, len(candidates) - 1)]
    invalid = set(master["_static_owner_rep_id"]) - rep_ids
    if invalid:
        raise ValueError(f"customer master references unknown representatives: {sorted(invalid)}")

    transaction_activity = (
        transactions.groupby(["customer_id", "period", "rep_id"], observed=True)
        .size()
        .rename("transaction_activity_count")
        .reset_index()
    )
    if visits.empty:
        period_activity = transaction_activity
        period_activity["visit_activity_count"] = 0.0
    else:
        visit_activity = visits[
            ["customer_id", "period", "rep_id", "planned_visit_count"]
        ].rename(columns={"planned_visit_count": "visit_activity_count"})
        period_activity = transaction_activity.merge(
            visit_activity,
            on=["customer_id", "period", "rep_id"],
            how="outer",
            validate="one_to_one",
        )
    period_activity["transaction_activity_count"] = period_activity[
        "transaction_activity_count"
    ].fillna(0.0)
    period_activity["visit_activity_count"] = period_activity["visit_activity_count"].fillna(0.0)
    period_activity["activity_count"] = (
        period_activity["transaction_activity_count"] + period_activity["visit_activity_count"]
    )
    period_activity["_rep_sort"] = period_activity["rep_id"].astype(str)
    period_owners = (
        period_activity.sort_values(
            ["customer_id", "period", "activity_count", "_rep_sort"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        .drop_duplicates(["customer_id", "period"])[["customer_id", "period", "rep_id"]]
        .rename(columns={"rep_id": "_dynamic_owner_rep_id"})
    )

    start = transactions.groupby("customer_id", observed=True)["period"].min().rename("portfolio_start_period")
    if not visits.empty:
        visit_start = visits.groupby("customer_id", observed=True)["period"].min()
        start = pd.concat([start, visit_start], axis=1).min(axis=1)
        start.name = "portfolio_start_period"
    master = master.merge(start, on="customer_id", how="left")
    master["portfolio_start_period"] = master["portfolio_start_period"].fillna(periods.min())

    if "customer_priority" not in master:
        if "potential_score" in master:
            potential = pd.to_numeric(master["potential_score"], errors="coerce")
        else:
            potential = pd.Series(0.5, index=master.index, dtype=float)
        scale = 100.0 if potential.dropna().max() > 1.0 else 1.0
        master["customer_priority"] = np.select(
            [potential.ge(0.75 * scale), potential.ge(0.40 * scale)],
            ["High", "Medium"],
            default="Low",
        )
    master["customer_priority"] = (
        master["customer_priority"].fillna("medium").astype(str).str.strip().str.casefold()
    )
    # The benchmark's numeric frequency is a cadence score, not a literal monthly
    # call count.  The explicit 1.6-period service-cycle assumption is configurable
    # and emitted on every calendar row; text cadences such as "weekly" are already
    # expressed per month and are intentionally not divided.
    frequency_divisor = float(capacity_config.get("numeric_visit_frequency_period_divisor", 1.6))
    if not np.isfinite(frequency_divisor) or frequency_divisor <= 0:
        raise ValueError("numeric_visit_frequency_period_divisor must be finite and positive")
    frequency_source = master.get("required_visit_frequency", pd.Series(np.nan, index=master.index))
    master["required_visit_count"] = _visit_frequency(
        frequency_source,
        master["customer_priority"],
        frequency_divisor,
    )

    drill = master.assign(_join=1).merge(
        pd.DataFrame({"period": periods, "_join": 1}), on="_join", how="inner"
    ).drop(columns="_join")
    drill = drill.loc[drill["period"].ge(drill["portfolio_start_period"])].copy()
    drill = drill.merge(
        period_owners,
        on=["customer_id", "period"],
        how="left",
        validate="one_to_one",
    )
    # Never backfill from future activity: an inactive month falls directly back
    # to the explicit/static portfolio owner.
    drill["rep_id"] = drill["_dynamic_owner_rep_id"].where(
        drill["_dynamic_owner_rep_id"].notna(), drill["_static_owner_rep_id"]
    )
    drill = drill.drop(columns=["_dynamic_owner_rep_id", "_static_owner_rep_id"])
    return drill


def _coverage_drilldown(
    transactions: pd.DataFrame,
    visits: pd.DataFrame,
    customer_master: pd.DataFrame,
    reps: pd.DataFrame,
    periods: pd.DatetimeIndex,
    capacity_config: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    drill = _customer_assignment(
        transactions,
        visits,
        customer_master,
        reps,
        periods,
        capacity_config,
        seed,
    )
    # Coverage is a customer-level service obligation. A visit by a shared or
    # cross-territory rep therefore earns coverage credit for the customer, while
    # the rep calendar below attributes the consumed hours to the actual visitor.
    customer_visits = (
        visits.groupby(["period", "customer_id"], observed=True)
        .agg(
            servicing_rep_count=("rep_id", "nunique"),
            observed_visit_count=("planned_visit_count", "sum"),
            observed_completed_visit_count=("completed_visit_count", "sum"),
            observed_visit_hours=("planned_visit_hours", "sum"),
            observed_travel_hours=("planned_travel_hours", "sum"),
        )
        .reset_index()
    )
    drill = drill.merge(
        customer_visits,
        on=["period", "customer_id"],
        how="left",
        validate="one_to_one",
    )
    fill = [
        "servicing_rep_count",
        "observed_visit_count",
        "observed_completed_visit_count",
        "observed_visit_hours",
        "observed_travel_hours",
    ]
    drill[fill] = drill[fill].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    average_visit_hours = float(capacity_config.get("visit_hours_per_required_call", 0.70))
    default_travel_hours = float(capacity_config.get("default_travel_hours_per_visit", 0.50))
    if not np.isfinite(average_visit_hours) or average_visit_hours <= 0:
        raise ValueError("visit_hours_per_required_call must be finite and positive")
    if not np.isfinite(default_travel_hours) or default_travel_hours < 0:
        raise ValueError("default_travel_hours_per_visit must be finite and nonnegative")

    observed_travel = np.divide(
        drill["observed_travel_hours"].to_numpy(float),
        drill["observed_visit_count"].to_numpy(float),
        out=np.full(len(drill), np.nan),
        where=drill["observed_visit_count"].to_numpy(float) > 0,
    )
    drill["estimated_travel_hours_per_visit"] = observed_travel
    if "territory_id" in drill:
        territory_median = drill.groupby("territory_id", observed=True)["estimated_travel_hours_per_visit"].transform("median")
        drill["estimated_travel_hours_per_visit"] = drill["estimated_travel_hours_per_visit"].fillna(territory_median)
    drill["estimated_travel_hours_per_visit"] = drill["estimated_travel_hours_per_visit"].fillna(default_travel_hours)

    # Required workload is the owned-customer service cadence, not every observed
    # cross-servicing touch.  Retain the full observed activity, but cap the
    # cadence-aligned planned component at the customer's monthly requirement.
    drill["planned_visit_count"] = np.minimum(
        drill["required_visit_count"], drill["observed_visit_count"]
    )
    service_share = np.divide(
        drill["planned_visit_count"].to_numpy(float),
        drill["observed_visit_count"].to_numpy(float),
        out=np.zeros(len(drill), dtype=float),
        where=drill["observed_visit_count"].to_numpy(float) > 0,
    )
    drill["planned_visit_hours"] = drill["observed_visit_hours"] * service_share
    drill["planned_travel_hours"] = drill["observed_travel_hours"] * service_share
    drill["completed_visit_count"] = np.minimum(
        drill["required_visit_count"], drill["observed_completed_visit_count"]
    )
    drill["excess_service_visit_count"] = (
        drill["observed_visit_count"] - drill["planned_visit_count"]
    ).clip(lower=0.0)
    drill["excess_service_visit_hours"] = (
        drill["observed_visit_hours"] - drill["planned_visit_hours"]
    ).clip(lower=0.0)
    drill["excess_service_travel_hours"] = (
        drill["observed_travel_hours"] - drill["planned_travel_hours"]
    ).clip(lower=0.0)
    drill["excess_service_hours"] = (
        drill["excess_service_visit_hours"] + drill["excess_service_travel_hours"]
    )

    drill["planned_coverage_gap_count"] = (
        drill["required_visit_count"] - drill["planned_visit_count"]
    ).clip(lower=0.0)
    drill["customer_coverage_gap"] = (
        drill["required_visit_count"] - drill["completed_visit_count"]
    ).clip(lower=0.0)
    high_priority = drill["customer_priority"].astype(str).str.casefold().eq("high")
    drill["priority_required_visit_count"] = drill["required_visit_count"].where(high_priority, 0.0)
    drill["priority_customer_coverage_gap"] = drill["customer_coverage_gap"].where(high_priority, 0.0)
    drill["required_coverage_visit_hours"] = drill["planned_coverage_gap_count"] * average_visit_hours
    drill["required_coverage_travel_hours"] = (
        drill["planned_coverage_gap_count"] * drill["estimated_travel_hours_per_visit"]
    )
    drill["required_customer_coverage_hours"] = (
        drill["required_coverage_visit_hours"] + drill["required_coverage_travel_hours"]
    )
    drill["required_priority_customer_coverage_hours"] = drill["required_customer_coverage_hours"].where(high_priority, 0.0)
    required = drill["required_visit_count"].to_numpy(float)
    completed = drill["completed_visit_count"].to_numpy(float)
    drill["customer_coverage_pct"] = np.divide(
        completed,
        required,
        out=np.ones(len(drill), dtype=float),
        where=required > 0,
    ).clip(0.0, 1.0) * 100.0
    drill["coverage_status"] = np.select(
        [drill["customer_coverage_gap"].le(1e-12), drill["planned_coverage_gap_count"].le(1e-12)],
        ["Covered", "Planned / not completed"],
        default="Coverage gap",
    )
    drill["coverage_met_flag"] = drill["customer_coverage_gap"].le(1e-12)
    drill["priority_customer_flag"] = high_priority
    drill["coverage_visit_scope"] = "all servicing reps credited to portfolio customer"
    drill["data_lineage"] = "synthetic_derived"
    drill["synthetic_seed"] = seed

    identity = [
        "rep_id",
        "period",
        "customer_id",
        "customer_name",
        "territory_id",
        "geography",
        "channel",
        "customer_type",
        "customer_segment",
        "customer_priority",
        "potential_score",
    ]
    measures = [
        "required_visit_count",
        "priority_required_visit_count",
        "servicing_rep_count",
        "observed_visit_count",
        "observed_completed_visit_count",
        "observed_visit_hours",
        "observed_travel_hours",
        "planned_visit_count",
        "completed_visit_count",
        "planned_visit_hours",
        "planned_travel_hours",
        "excess_service_visit_count",
        "excess_service_visit_hours",
        "excess_service_travel_hours",
        "excess_service_hours",
        "planned_coverage_gap_count",
        "customer_coverage_gap",
        "priority_customer_coverage_gap",
        "estimated_travel_hours_per_visit",
        "required_coverage_visit_hours",
        "required_coverage_travel_hours",
        "required_customer_coverage_hours",
        "required_priority_customer_coverage_hours",
        "customer_coverage_pct",
        "coverage_met_flag",
        "priority_customer_flag",
        "coverage_visit_scope",
        "coverage_status",
        "data_lineage",
        "synthetic_seed",
    ]
    columns = [column for column in identity + measures if column in drill]
    result = drill[columns].sort_values(["period", "rep_id", "customer_id"], kind="mergesort").reset_index(drop=True)
    if result.duplicated(["rep_id", "period", "customer_id"]).any():
        raise RuntimeError("customer coverage drilldown is not unique at rep-period-customer grain")
    numeric = result.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError("customer coverage drilldown contains non-finite numeric values")
    return result


def _servicing_allocation(
    visits: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Allocate cadence-aligned workload to the reps who performed the visits.

    ``group_columns`` defaults to the compact rep-period grain.  Territory
    allocation supplies ``rep_id, period, territory_id`` so the same auditable
    cadence math is conserved at the actual visited geography.
    """
    group_columns = list(group_columns or CAPACITY_KEY)
    output_columns = group_columns + [
        "planned_visit_count",
        "completed_visit_count",
        "planned_visit_hours",
        "planned_travel_hours",
        "excess_service_visit_count",
        "excess_service_visit_hours",
        "excess_service_travel_hours",
        "excess_service_hours",
    ]
    if visits.empty:
        return pd.DataFrame(columns=output_columns)

    obligations = coverage[
        [
            "period",
            "customer_id",
            "required_visit_count",
            "observed_visit_count",
            "observed_completed_visit_count",
        ]
    ]
    if obligations.duplicated(["period", "customer_id"]).any():
        raise RuntimeError("coverage obligations are not unique at period-customer grain")
    allocated = visits.merge(
        obligations,
        on=["period", "customer_id"],
        how="left",
        validate="many_to_one",
    )
    if allocated["required_visit_count"].isna().any():
        raise RuntimeError("a servicing visit has no matching customer coverage obligation")

    required = allocated["required_visit_count"].to_numpy(float)
    observed = allocated["observed_visit_count"].to_numpy(float)
    observed_completed = allocated["observed_completed_visit_count"].to_numpy(float)
    planned_share = np.divide(
        np.minimum(required, observed),
        observed,
        out=np.zeros(len(allocated), dtype=float),
        where=observed > 0,
    )
    completed_share = np.divide(
        np.minimum(required, observed_completed),
        observed_completed,
        out=np.zeros(len(allocated), dtype=float),
        where=observed_completed > 0,
    )

    actual_count = allocated["planned_visit_count"].to_numpy(float)
    actual_completed = allocated["completed_visit_count"].to_numpy(float)
    actual_visit_hours = allocated["planned_visit_hours"].to_numpy(float)
    actual_travel_hours = allocated["planned_travel_hours"].to_numpy(float)
    allocated["planned_visit_count"] = actual_count * planned_share
    allocated["completed_visit_count"] = actual_completed * completed_share
    allocated["planned_visit_hours"] = actual_visit_hours * planned_share
    allocated["planned_travel_hours"] = actual_travel_hours * planned_share
    allocated["excess_service_visit_count"] = (
        actual_count - allocated["planned_visit_count"]
    ).clip(lower=0.0)
    allocated["excess_service_visit_hours"] = (
        actual_visit_hours - allocated["planned_visit_hours"]
    ).clip(lower=0.0)
    allocated["excess_service_travel_hours"] = (
        actual_travel_hours - allocated["planned_travel_hours"]
    ).clip(lower=0.0)
    allocated["excess_service_hours"] = (
        allocated["excess_service_visit_hours"]
        + allocated["excess_service_travel_hours"]
    )
    result = (
        allocated.groupby(group_columns, observed=True, dropna=False)[
            output_columns[len(group_columns) :]
        ]
        .sum()
        .reset_index()
    )
    return result[output_columns]


def _risk_band(utilization_pct: pd.Series, thresholds: dict[str, float]) -> pd.Series:
    medium = float(thresholds.get("medium", 0.85))
    high = float(thresholds.get("high", 1.00))
    critical = float(thresholds.get("critical", 1.20))
    if not (np.isfinite([medium, high, critical]).all() and 0 <= medium <= high <= critical):
        raise ValueError("capacity risk thresholds must be finite and ordered medium <= high <= critical")
    ratio = utilization_pct / 100.0
    return pd.Series(
        np.select(
            [ratio.ge(critical), ratio.ge(high), ratio.ge(medium)],
            ["critical", "high", "medium"],
            default="low",
        ),
        index=utilization_pct.index,
        dtype="object",
    )


def build_capacity_calendar(
    normalized_transactions: pd.DataFrame,
    rep_master: pd.DataFrame,
    customer_master: pd.DataFrame,
    field_visits: pd.DataFrame,
    config: Mapping[str, Any] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return deterministic rep-period capacity and rep-period-customer coverage.

    Required workload hours are calculated without double counting:

    ``planned visit hours + planned travel hours + hours for unplanned required``
    ``coverage + preparation/follow-up buffer``.

    Availability is gross rostered field time less leave, holidays, training,
    administration, and meetings.  Required and available FTE use the same
    nominal full-time hours denominator.
    """
    capacity_config, model_config, synthetic_config, seed = _config_sections(config)
    transactions = _prepare_transactions(normalized_transactions)
    reps = _rep_identity(rep_master, transactions)
    average_visit_minutes = float(synthetic_config.get("average_visit_minutes", capacity_config.get("average_visit_minutes", 42.0)))
    average_speed_kmh = float(synthetic_config.get("average_speed_kmh", capacity_config.get("average_speed_kmh", 42.0)))
    if not np.isfinite(average_visit_minutes) or average_visit_minutes <= 0:
        raise ValueError("average visit minutes must be finite and positive")
    if not np.isfinite(average_speed_kmh) or average_speed_kmh <= 0:
        raise ValueError("average speed must be finite and positive")
    visits = _prepare_visits(field_visits, average_visit_minutes, average_speed_kmh)
    _require_columns(customer_master, ["customer_id"], "customer master")
    if not visits.empty:
        unknown_reps = set(visits["rep_id"]) - set(reps["rep_id"])
        unknown_customers = set(visits["customer_id"]) - set(customer_master["customer_id"])
        if unknown_reps or unknown_customers:
            raise ValueError(
                "field visits violate master-table relationships: "
                f"unknown reps={sorted(unknown_reps)}, unknown customers={sorted(unknown_customers)}"
            )

    period_values = list(transactions["period"].unique())
    if not visits.empty:
        period_values.extend(visits["period"].unique())
    observed_periods = pd.to_datetime(pd.Series(period_values, dtype="datetime64[ns]")).drop_duplicates()
    periods = pd.date_range(observed_periods.min(), observed_periods.max(), freq="MS")
    legacy = _legacy_workload(transactions, periods, reps, capacity_config, model_config)

    identity_fields = [
        "rep_id",
        "rep_name",
        "manager_id",
        "manager_name",
        "team_id",
        "team_name",
        "territory_id",
        "territory_name",
        "hire_date",
        "employment_status",
        "role_grade",
        "standard_field_hours_per_day",
        "standard_working_days_per_month",
        "leave_days",
        "holiday_days",
        "training_hours",
        "administrative_hours",
        "meeting_hours",
    ]
    master_columns = [column for column in identity_fields if column in reps]
    calendar = legacy.merge(reps[master_columns], on="rep_id", how="left", validate="many_to_one")
    dynamic_territory = _period_territory(transactions)
    if not dynamic_territory.empty:
        calendar = calendar.merge(
            dynamic_territory,
            on=CAPACITY_KEY,
            how="left",
            validate="one_to_one",
        )
        calendar["territory_id"] = calendar["dynamic_territory_id"].where(
            calendar["dynamic_territory_id"].notna(), calendar["territory_id"]
        )
        calendar["territory_name"] = calendar["dynamic_territory_name"].where(
            calendar["dynamic_territory_id"].notna(), calendar["territory_name"]
        )
        dynamic_without_name = calendar["dynamic_territory_id"].notna() & calendar[
            "territory_name"
        ].isna()
        calendar.loc[dynamic_without_name, "territory_name"] = calendar.loc[
            dynamic_without_name, "dynamic_territory_id"
        ].astype(str)
        fallback_territory = calendar["dynamic_territory_id"].isna()
        calendar.loc[fallback_territory, "active_territory_count"] = 1
        calendar.loc[fallback_territory, "dominant_territory_activity_share"] = 1.0
        calendar.loc[fallback_territory, "fractional_territory_allocation"] = (
            calendar.loc[fallback_territory, "territory_id"].astype(str) + ":1.00000000"
        )
        calendar = calendar.drop(columns=["dynamic_territory_id", "dynamic_territory_name"])
    if "hire_date" in calendar:
        hire_period = pd.to_datetime(calendar["hire_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        calendar = calendar.loc[hire_period.isna() | calendar["period"].ge(hire_period)].copy()
    if calendar.empty:
        raise ValueError("capacity calendar has no active rep-period rows")

    default_hours = float(capacity_config.get("standard_field_hours_per_day", 8.0))
    default_days = float(capacity_config.get("standard_working_days_per_month", 20.0))
    calendar["standard_field_hours_per_day"] = pd.to_numeric(
        calendar.get(
            "standard_field_hours_per_day",
            pd.Series(default_hours, index=calendar.index, dtype=float),
        ),
        errors="coerce",
    ).fillna(default_hours)
    calendar["standard_working_days_per_month"] = pd.to_numeric(
        calendar.get(
            "standard_working_days_per_month",
            pd.Series(default_days, index=calendar.index, dtype=float),
        ),
        errors="coerce",
    ).fillna(default_days)
    if (
        calendar["standard_field_hours_per_day"].le(0).any()
        or calendar["standard_working_days_per_month"].le(0).any()
    ):
        raise ValueError("standard field hours and working days must be positive")

    calendar["working_days"] = _numeric_or_seeded(
        calendar, "working_days", seed, default_days, default_days + 1, integers=True
    )
    # If no explicit monthly working-days field exists, the roster standard is authoritative.
    if "working_days" not in reps:
        calendar["working_days"] = calendar["standard_working_days_per_month"]
    calendar["leave_days"] = _numeric_or_seeded(calendar, "leave_days", seed, 0.0, 2.999, integers=True)
    calendar["holiday_days"] = _numeric_or_seeded(calendar, "holiday_days", seed, 0.0, 2.999, integers=True)
    calendar["training_hours"] = _numeric_or_seeded(calendar, "training_hours", seed, 2.0, 7.0)
    calendar["administrative_hours"] = _numeric_or_seeded(calendar, "administrative_hours", seed, 8.0, 14.0)
    calendar["meeting_hours"] = _numeric_or_seeded(calendar, "meeting_hours", seed, 4.0, 9.0)

    calendar["working_days"] = calendar["working_days"].clip(lower=1.0)
    calendar["leave_days"] = calendar["leave_days"].clip(lower=0.0).clip(upper=calendar["working_days"])
    calendar["holiday_days"] = calendar["holiday_days"].clip(lower=0.0).clip(
        upper=(calendar["working_days"] - calendar["leave_days"]).clip(lower=0.0)
    )
    for column in ["training_hours", "administrative_hours", "meeting_hours"]:
        calendar[column] = calendar[column].clip(lower=0.0)
    calendar["gross_rostered_field_hours"] = (
        (calendar["working_days"] - calendar["leave_days"] - calendar["holiday_days"])
        * calendar["standard_field_hours_per_day"]
    )
    calendar["non_field_hours"] = calendar[["training_hours", "administrative_hours", "meeting_hours"]].sum(axis=1)
    calendar["available_field_hours"] = (
        calendar["gross_rostered_field_hours"] - calendar["non_field_hours"]
    ).clip(lower=0.0)

    coverage = _coverage_drilldown(
        transactions, visits, customer_master, reps, periods, capacity_config, seed
    )
    coverage_agg = (
        coverage.groupby(CAPACITY_KEY, observed=True)
        .agg(
            active_customer_count=("customer_id", "nunique"),
            priority_customer_count=("customer_priority", lambda values: values.astype(str).str.casefold().eq("high").sum()),
            required_visit_count=("required_visit_count", "sum"),
            priority_required_visit_count=("priority_required_visit_count", "sum"),
            credited_planned_visit_count=("planned_visit_count", "sum"),
            credited_completed_visit_count=("completed_visit_count", "sum"),
            required_customer_coverage_hours=("required_customer_coverage_hours", "sum"),
            required_priority_customer_coverage_hours=("required_priority_customer_coverage_hours", "sum"),
            customer_coverage_gap=("customer_coverage_gap", "sum"),
            priority_customer_coverage_gap=("priority_customer_coverage_gap", "sum"),
        )
        .reset_index()
    )
    rep_visit_agg = (
        visits.groupby(CAPACITY_KEY, observed=True)
        .agg(
            observed_visit_count=("planned_visit_count", "sum"),
            observed_completed_visit_count=("completed_visit_count", "sum"),
            observed_visit_hours=("planned_visit_hours", "sum"),
            observed_travel_hours=("planned_travel_hours", "sum"),
        )
        .reset_index()
    )
    servicing_agg = _servicing_allocation(visits, coverage)
    calendar = calendar.merge(coverage_agg, on=CAPACITY_KEY, how="left", validate="one_to_one")
    calendar = calendar.merge(servicing_agg, on=CAPACITY_KEY, how="left", validate="one_to_one")
    calendar = calendar.merge(rep_visit_agg, on=CAPACITY_KEY, how="left", validate="one_to_one")
    coverage_fields = [
        "active_customer_count",
        "priority_customer_count",
        "required_visit_count",
        "priority_required_visit_count",
        "credited_planned_visit_count",
        "credited_completed_visit_count",
        "required_customer_coverage_hours",
        "required_priority_customer_coverage_hours",
        "customer_coverage_gap",
        "priority_customer_coverage_gap",
    ]
    calendar[coverage_fields] = (
        calendar[coverage_fields].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    )
    servicing_fields = [
        "planned_visit_count",
        "completed_visit_count",
        "planned_visit_hours",
        "planned_travel_hours",
        "excess_service_visit_count",
        "excess_service_visit_hours",
        "excess_service_travel_hours",
        "excess_service_hours",
    ]
    calendar[servicing_fields] = (
        calendar[servicing_fields].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    )
    observed_fields = [
        "observed_visit_count",
        "observed_completed_visit_count",
        "observed_visit_hours",
        "observed_travel_hours",
    ]
    calendar[observed_fields] = (
        calendar[observed_fields].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    )
    required_visits = calendar["required_visit_count"].to_numpy(float)
    completed_visits = calendar["credited_completed_visit_count"].to_numpy(float)
    calendar["customer_coverage_pct"] = np.divide(
        completed_visits,
        required_visits,
        out=np.ones(len(calendar), dtype=float),
        where=required_visits > 0,
    ).clip(0.0, 1.0) * 100.0
    calendar["priority_customer_coverage_gap_pct"] = np.divide(
        calendar["priority_customer_coverage_gap"].to_numpy(float),
        np.maximum(calendar["priority_required_visit_count"].to_numpy(float), 1e-12),
    ) * 100.0

    buffer_pct = float(capacity_config.get("administrative_buffer_pct", 0.08))
    if not np.isfinite(buffer_pct) or buffer_pct < 0:
        raise ValueError("administrative_buffer_pct must be finite and nonnegative")
    calendar["core_required_hours"] = calendar[
        ["planned_visit_hours", "planned_travel_hours", "required_customer_coverage_hours"]
    ].sum(axis=1)
    calendar["workload_buffer_hours"] = calendar["core_required_hours"] * buffer_pct
    calendar["required_total_hours"] = calendar["core_required_hours"] + calendar["workload_buffer_hours"]
    calendar["nominal_full_time_hours"] = (
        calendar["standard_working_days_per_month"] * calendar["standard_field_hours_per_day"]
    )
    available = calendar["available_field_hours"].to_numpy(float)
    required_hours = calendar["required_total_hours"].to_numpy(float)
    zero_capacity_pct = float(capacity_config.get("zero_capacity_utilization_pct", 10000.0))
    if not np.isfinite(zero_capacity_pct) or zero_capacity_pct < 100:
        raise ValueError("zero_capacity_utilization_pct must be finite and at least 100")
    utilization = np.divide(
        required_hours * 100.0,
        available,
        out=np.zeros(len(calendar), dtype=float),
        where=available > 0,
    )
    utilization[(available <= 0) & (required_hours > 0)] = zero_capacity_pct
    calendar["capacity_zero_denominator_flag"] = (available <= 0) & (required_hours > 0)
    calendar["utilization_pct"] = utilization
    calendar["capacity_utilization_pct"] = utilization
    calendar["required_fte"] = calendar["required_total_hours"] / calendar["nominal_full_time_hours"]
    calendar["available_fte"] = calendar["available_field_hours"] / calendar["nominal_full_time_hours"]
    calendar["fte_gap"] = calendar["required_fte"] - calendar["available_fte"]
    calendar["required_hours"] = calendar["required_total_hours"]
    calendar["available_hours"] = calendar["available_field_hours"]
    calendar["average_travel_hours"] = np.divide(
        calendar["planned_travel_hours"].to_numpy(float),
        calendar["planned_visit_count"].to_numpy(float),
        out=np.zeros(len(calendar), dtype=float),
        where=calendar["planned_visit_count"].to_numpy(float) > 0,
    )
    calendar["workload_per_active_customer"] = np.divide(
        required_hours,
        calendar["active_customer_count"].to_numpy(float),
        out=np.zeros(len(calendar), dtype=float),
        where=calendar["active_customer_count"].to_numpy(float) > 0,
    )

    thresholds = _mapping(capacity_config.get("risk_thresholds"))
    calendar["workload_risk_band"] = _risk_band(calendar["utilization_pct"], thresholds)
    calendar["capacity_risk_band"] = calendar["workload_risk_band"]
    medium = float(thresholds.get("medium", 0.85)) * 100.0
    high = float(thresholds.get("high", 1.00)) * 100.0
    critical = float(thresholds.get("critical", 1.20)) * 100.0
    calendar["risk_medium_threshold_pct"] = medium
    calendar["risk_high_threshold_pct"] = high
    calendar["risk_critical_threshold_pct"] = critical
    calendar["overload_threshold_pct"] = high
    calendar["overload_flag"] = calendar["utilization_pct"].ge(high)
    calendar["capacity_overload_flag"] = calendar["overload_flag"]
    calendar["required_hours_formula"] = (
        "cadence_aligned_planned_visit_hours + cadence_aligned_planned_travel_hours "
        "+ unplanned_required_coverage_hours + preparation_follow_up_buffer"
    )
    calendar["required_workload_scope"] = (
        "cadence-aligned service attributed to servicing reps; uncovered obligations "
        "attributed to dynamic portfolio owners; observed excess retained as diagnostics; "
        "all served territories retained as fractional transaction allocations"
    )
    calendar["numeric_visit_frequency_period_divisor"] = float(
        capacity_config.get("numeric_visit_frequency_period_divisor", 1.6)
    )
    calendar["capacity_methodology"] = "deterministic_hours_plus_legacy_normalized_workload"
    calendar["data_lineage"] = "synthetic_derived"
    calendar["synthetic_seed"] = seed
    calendar["capacity_record_id"] = [
        "CAPCAL_"
        + hashlib.sha256(f"{rep_id}|{pd.Timestamp(period).date()}".encode("utf-8")).hexdigest()[:16]
        for rep_id, period in calendar[CAPACITY_KEY].itertuples(index=False, name=None)
    ]

    preferred = [
        "capacity_record_id",
        "rep_id",
        "rep_name",
        "manager_id",
        "manager_name",
        "team_id",
        "team_name",
        "territory_id",
        "territory_name",
        "active_territory_count",
        "dominant_territory_activity_share",
        "fractional_territory_allocation",
        "period",
        "working_days",
        "leave_days",
        "holiday_days",
        "training_hours",
        "administrative_hours",
        "meeting_hours",
        "standard_field_hours_per_day",
        "standard_working_days_per_month",
        "gross_rostered_field_hours",
        "non_field_hours",
        "available_field_hours",
        "planned_visit_hours",
        "planned_travel_hours",
        "observed_visit_hours",
        "observed_travel_hours",
        "excess_service_visit_hours",
        "excess_service_travel_hours",
        "excess_service_hours",
        "required_customer_coverage_hours",
        "required_priority_customer_coverage_hours",
        "required_total_hours",
        "utilization_pct",
        "capacity_utilization_pct",
        "required_fte",
        "available_fte",
        "fte_gap",
        "customer_coverage_gap",
        "priority_customer_coverage_gap",
        "customer_coverage_pct",
        "average_travel_hours",
        "workload_per_active_customer",
        "legacy_normalized_workload_index",
        "workload_risk_band",
        "capacity_risk_band",
        "overload_flag",
        "capacity_overload_flag",
        "data_lineage",
        "synthetic_seed",
    ]
    ordered = [column for column in preferred if column in calendar]
    ordered += [column for column in calendar if column not in ordered]
    calendar = calendar[ordered].sort_values(["period", "rep_id"], kind="mergesort").reset_index(drop=True)
    if calendar.duplicated(CAPACITY_KEY).any():
        raise RuntimeError("capacity calendar is not unique at rep-period grain")
    numeric = calendar.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError("capacity calendar contains non-finite numeric values")
    nonnegative = [
        "working_days",
        "leave_days",
        "holiday_days",
        "training_hours",
        "administrative_hours",
        "meeting_hours",
        "available_field_hours",
        "planned_visit_hours",
        "planned_travel_hours",
        "excess_service_visit_count",
        "excess_service_visit_hours",
        "excess_service_travel_hours",
        "excess_service_hours",
        "required_customer_coverage_hours",
        "required_total_hours",
        "utilization_pct",
        "required_fte",
        "available_fte",
        "customer_coverage_gap",
        "priority_customer_coverage_gap",
    ]
    if calendar[nonnegative].lt(-1e-12).any().any():
        raise RuntimeError("capacity calendar contains negative hour, FTE, utilization, or coverage values")
    return calendar, coverage


def _calendar_allocation_shares(calendar: pd.DataFrame) -> pd.DataFrame:
    """Recover normalized allocation shares from the compact audit field."""
    rows: list[dict[str, Any]] = []
    for record in calendar.to_dict("records"):
        rep_id = record["rep_id"]
        period = pd.Timestamp(record["period"])
        parsed: list[tuple[Any, float]] = []
        raw = record.get("fractional_territory_allocation")
        if pd.notna(raw):
            for token in str(raw).split("|"):
                try:
                    territory_id, raw_share = token.rsplit(":", 1)
                    share = float(raw_share)
                except (TypeError, ValueError):
                    parsed = []
                    break
                if territory_id and np.isfinite(share) and share > 0:
                    parsed.append((territory_id, share))
        if not parsed:
            parsed = [(record.get("territory_id"), 1.0)]
        total = sum(share for _, share in parsed)
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"invalid territory allocation for rep-period {rep_id}|{period.date()}")
        for territory_id, share in parsed:
            rows.append(
                {
                    "rep_id": rep_id,
                    "period": period,
                    "territory_id": territory_id,
                    "territory_activity_count": np.nan,
                    "territory_allocation_share": share / total,
                    "allocation_basis": "capacity_calendar_fractional_allocation",
                }
            )
    return pd.DataFrame(rows)


def _transaction_allocation_shares(
    calendar: pd.DataFrame, normalized_transactions: pd.DataFrame | None
) -> pd.DataFrame:
    """Return exact transaction-count shares, with roster fallback for idle rows."""
    if normalized_transactions is None or "territory_id" not in normalized_transactions:
        return _calendar_allocation_shares(calendar)
    transactions = _prepare_transactions(normalized_transactions)
    present = transactions.dropna(subset=["territory_id"]).copy()
    calendar_keys = pd.MultiIndex.from_frame(calendar[CAPACITY_KEY])
    if not present.empty:
        present = present.loc[
            pd.MultiIndex.from_frame(present[CAPACITY_KEY]).isin(calendar_keys)
        ].copy()
    if present.empty:
        return _calendar_allocation_shares(calendar)

    shares = (
        present.groupby(CAPACITY_TERRITORY_KEY, observed=True, dropna=False)
        .size()
        .rename("territory_activity_count")
        .reset_index()
    )
    totals = shares.groupby(CAPACITY_KEY, observed=True)["territory_activity_count"].transform("sum")
    shares["territory_allocation_share"] = shares["territory_activity_count"] / totals.clip(lower=1)
    shares["allocation_basis"] = "rep_period_transaction_count"
    if "territory_name" in present:
        names = (
            present.groupby(CAPACITY_TERRITORY_KEY, observed=True, dropna=False)["territory_name"]
            .agg(_deterministic_mode)
            .reset_index()
        )
        shares = shares.merge(names, on=CAPACITY_TERRITORY_KEY, how="left", validate="one_to_one")

    observed_keys = pd.MultiIndex.from_frame(shares[CAPACITY_KEY].drop_duplicates())
    fallback = calendar.loc[~calendar_keys.isin(observed_keys), CAPACITY_KEY + ["territory_id"]].copy()
    if not fallback.empty:
        fallback["territory_activity_count"] = 0.0
        fallback["territory_allocation_share"] = 1.0
        fallback["allocation_basis"] = "roster_territory_idle_period_fallback"
        if "territory_name" in calendar:
            fallback = fallback.merge(
                calendar[CAPACITY_KEY + ["territory_name"]],
                on=CAPACITY_KEY,
                how="left",
                validate="one_to_one",
            )
        shares = pd.concat([shares, fallback], ignore_index=True, sort=False)
    return shares


def _build_proxy_capacity_territory_allocation(
    capacity_calendar: pd.DataFrame,
    normalized_transactions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Explode a calendar using transaction shares when service facts are absent.

    The compact ``capacity_calendar`` remains unique by rep and period.  Every
    additive hour/FTE quantity in this fact is the source rep-period value
    multiplied by a normalized transaction-activity share.  Idle rep-periods
    receive a deterministic 100% roster-territory fallback.
    """
    calendar = capacity_calendar.copy(deep=True)
    _require_columns(
        calendar,
        CAPACITY_KEY
        + ["territory_id", "required_total_hours", "available_field_hours", "required_fte", "available_fte"],
        "capacity calendar",
    )
    calendar["period"] = _month(calendar["period"], "capacity calendar.period")
    if calendar.empty or calendar.duplicated(CAPACITY_KEY).any():
        raise ValueError("capacity calendar must contain one row per rep-period")
    if "fte_gap" not in calendar:
        calendar["fte_gap"] = (
            pd.to_numeric(calendar["required_fte"], errors="raise")
            - pd.to_numeric(calendar["available_fte"], errors="raise")
        )

    shares = _transaction_allocation_shares(calendar, normalized_transactions)
    if shares.empty:
        raise RuntimeError("territory allocation produced no rows")
    shares["period"] = _month(shares["period"], "territory allocation.period")
    shares["territory_allocation_share"] = pd.to_numeric(
        shares["territory_allocation_share"], errors="raise"
    )
    if (
        ~np.isfinite(shares["territory_allocation_share"].to_numpy(float))
    ).any() or shares["territory_allocation_share"].le(0).any():
        raise ValueError("territory allocation shares must be finite and positive")
    shares = (
        shares.groupby(CAPACITY_TERRITORY_KEY, observed=True, dropna=False)
        .agg(
            territory_activity_count=("territory_activity_count", "sum"),
            territory_allocation_share=("territory_allocation_share", "sum"),
            allocation_basis=("allocation_basis", _deterministic_mode),
            **(
                {"territory_name": ("territory_name", _deterministic_mode)}
                if "territory_name" in shares
                else {}
            ),
        )
        .reset_index()
    )
    share_total = shares.groupby(CAPACITY_KEY, observed=True)["territory_allocation_share"].transform("sum")
    if share_total.le(0).any():
        raise ValueError("territory allocation has a rep-period with no positive share")
    shares["territory_allocation_share"] /= share_total

    context_columns = [
        "capacity_record_id",
        "rep_name",
        "manager_id",
        "manager_name",
        "team_id",
        "team_name",
        "territory_id",
        "territory_name",
        "risk_medium_threshold_pct",
        "risk_high_threshold_pct",
        "risk_critical_threshold_pct",
        "overload_threshold_pct",
        "capacity_utilization_pct",
        "capacity_risk_band",
        "capacity_overload_flag",
        "data_lineage",
        "synthetic_seed",
    ]
    additive_columns = [
        column for column in CAPACITY_ALLOCATION_ADDITIVE_COLUMNS if column in calendar
    ]
    source_columns = CAPACITY_KEY + [
        column for column in context_columns + additive_columns if column in calendar and column not in CAPACITY_KEY
    ]
    source = calendar[source_columns].copy()
    source = source.rename(
        columns={
            "territory_id": "dominant_territory_id",
            "territory_name": "dominant_territory_name",
            "capacity_utilization_pct": "source_rep_period_utilization_pct",
            "capacity_risk_band": "source_rep_period_risk_band",
            "capacity_overload_flag": "source_rep_period_overload_flag",
        }
    )
    allocation = shares.merge(source, on=CAPACITY_KEY, how="left", validate="many_to_one")
    if "territory_name" not in allocation:
        allocation["territory_name"] = allocation["territory_id"].astype("string")
    else:
        allocation["territory_name"] = allocation["territory_name"].where(
            allocation["territory_name"].notna(), allocation["territory_id"].astype("string")
        )
    for column in additive_columns:
        values = pd.to_numeric(allocation[column], errors="raise")
        allocation[column] = values * allocation["territory_allocation_share"]

    required = pd.to_numeric(allocation["required_total_hours"], errors="raise").to_numpy(float)
    available = pd.to_numeric(allocation["available_field_hours"], errors="raise").to_numpy(float)
    allocation["utilization_pct"] = np.divide(
        required * 100.0,
        available,
        out=np.zeros(len(allocation), dtype=float),
        where=available > 0,
    )
    allocation.loc[(available <= 0) & (required > 0), "utilization_pct"] = 10000.0
    allocation["capacity_utilization_pct"] = allocation["utilization_pct"]
    if "source_rep_period_risk_band" in allocation:
        allocation["capacity_risk_band"] = allocation["source_rep_period_risk_band"]
        allocation["workload_risk_band"] = allocation["source_rep_period_risk_band"]
    if "source_rep_period_overload_flag" in allocation:
        allocation["capacity_overload_flag"] = allocation[
            "source_rep_period_overload_flag"
        ].astype(bool)
        allocation["overload_flag"] = allocation["capacity_overload_flag"]
    allocation["dominant_territory_flag"] = allocation["territory_id"].astype("string").eq(
        allocation["dominant_territory_id"].astype("string")
    )
    allocation["geographic_workload_attribution_flag"] = False
    allocation["allocation_scope"] = (
        "planning proxy: transaction-activity shares allocate additive rep-period workload "
        "and capacity; not observed territory-specific time attribution"
    )
    allocation["capacity_territory_allocation_id"] = [
        "CAPALLOC_"
        + hashlib.sha256(
            f"{rep_id}|{pd.Timestamp(period).date()}|{territory_id}".encode("utf-8")
        ).hexdigest()[:16]
        for rep_id, period, territory_id in allocation[CAPACITY_TERRITORY_KEY].itertuples(
            index=False, name=None
        )
    ]

    if allocation.duplicated(CAPACITY_TERRITORY_KEY).any():
        raise RuntimeError("capacity territory allocation is not unique at rep-territory-period grain")
    normalized = allocation.groupby(CAPACITY_KEY, observed=True)["territory_allocation_share"].sum()
    if not np.allclose(normalized.to_numpy(float), 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("capacity territory allocation shares do not sum to one")
    source_indexed = calendar.set_index(CAPACITY_KEY)
    allocated_indexed = allocation.groupby(CAPACITY_KEY, observed=True)[additive_columns].sum()
    for column in additive_columns:
        expected = pd.to_numeric(source_indexed.loc[allocated_indexed.index, column], errors="raise")
        if not np.allclose(
            allocated_indexed[column].to_numpy(float),
            expected.to_numpy(float),
            rtol=1e-10,
            atol=1e-10,
        ):
            raise RuntimeError(f"capacity territory allocation does not conserve {column}")
    numeric = allocation.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise RuntimeError("capacity territory allocation contains non-finite numeric values")

    preferred = [
        "capacity_territory_allocation_id",
        "capacity_record_id",
        "rep_id",
        "rep_name",
        "manager_id",
        "manager_name",
        "team_id",
        "team_name",
        "territory_id",
        "territory_name",
        "period",
        "territory_activity_count",
        "territory_allocation_share",
        "dominant_territory_id",
        "dominant_territory_name",
        "dominant_territory_flag",
        "allocation_basis",
    ]
    ordered = [column for column in preferred if column in allocation]
    ordered += [column for column in allocation if column not in ordered]
    return allocation[ordered].sort_values(
        ["period", "rep_id", "territory_id"], kind="mergesort"
    ).reset_index(drop=True)


def _territory_visit_workload(
    field_visits: pd.DataFrame,
    coverage: pd.DataFrame,
    config: Mapping[str, Any] | None,
) -> pd.DataFrame:
    """Return actual-visitor workload at rep-period-territory grain."""
    capacity_config, _, synthetic_config, _ = _config_sections(config)
    average_visit_minutes = float(
        synthetic_config.get(
            "average_visit_minutes",
            capacity_config.get("average_visit_minutes", 42.0),
        )
    )
    average_speed_kmh = float(
        synthetic_config.get(
            "average_speed_kmh", capacity_config.get("average_speed_kmh", 42.0)
        )
    )
    visits = _prepare_visits(field_visits, average_visit_minutes, average_speed_kmh)
    visit_measures = [
        "planned_visit_count",
        "completed_visit_count",
        "planned_visit_hours",
        "planned_travel_hours",
        "excess_service_visit_count",
        "excess_service_visit_hours",
        "excess_service_travel_hours",
        "excess_service_hours",
        "observed_visit_count",
        "observed_completed_visit_count",
        "observed_visit_hours",
        "observed_travel_hours",
    ]
    if visits.empty:
        return pd.DataFrame(columns=CAPACITY_TERRITORY_KEY + visit_measures)

    owner_geography = coverage[
        [
            column
            for column in ["period", "customer_id", "territory_id", "territory_name"]
            if column in coverage
        ]
    ].drop_duplicates(["period", "customer_id"])
    if owner_geography.duplicated(["period", "customer_id"]).any():
        raise RuntimeError("customer coverage geography is not unique at period-customer grain")
    owner_geography = owner_geography.rename(
        columns={
            "territory_id": "_owner_territory_id",
            "territory_name": "_owner_territory_name",
        }
    )
    visits = visits.merge(
        owner_geography,
        on=["period", "customer_id"],
        how="left",
        validate="many_to_one",
    )
    if "territory_id" not in visits:
        visits["territory_id"] = visits["_owner_territory_id"]
    else:
        visits["territory_id"] = visits["territory_id"].where(
            visits["territory_id"].notna(), visits["_owner_territory_id"]
        )
    if visits["territory_id"].isna().any():
        raise RuntimeError("an actual field visit could not be assigned to a territory")
    if "territory_name" not in visits:
        visits["territory_name"] = visits.get("_owner_territory_name")
    elif "_owner_territory_name" in visits:
        visits["territory_name"] = visits["territory_name"].where(
            visits["territory_name"].notna(), visits["_owner_territory_name"]
        )

    servicing = _servicing_allocation(
        visits,
        coverage,
        group_columns=CAPACITY_TERRITORY_KEY,
    )
    observed = (
        visits.groupby(CAPACITY_TERRITORY_KEY, observed=True, dropna=False)
        .agg(
            observed_visit_count=("planned_visit_count", "sum"),
            observed_completed_visit_count=("completed_visit_count", "sum"),
            observed_visit_hours=("planned_visit_hours", "sum"),
            observed_travel_hours=("planned_travel_hours", "sum"),
        )
        .reset_index()
    )
    result = servicing.merge(
        observed,
        on=CAPACITY_TERRITORY_KEY,
        how="outer",
        validate="one_to_one",
    )
    result[visit_measures] = result[visit_measures].fillna(0.0)
    if "territory_name" in visits:
        names = (
            visits.groupby(CAPACITY_TERRITORY_KEY, observed=True, dropna=False)[
                "territory_name"
            ]
            .agg(_deterministic_mode)
            .reset_index()
        )
        result = result.merge(
            names, on=CAPACITY_TERRITORY_KEY, how="left", validate="one_to_one"
        )
    return result


def _territory_coverage_workload(coverage: pd.DataFrame) -> pd.DataFrame:
    """Return owned-customer cadence and coverage workload by territory."""
    drill = coverage.copy(deep=True)
    _require_columns(
        drill,
        CAPACITY_TERRITORY_KEY + ["customer_id", "required_visit_count"],
        "capacity customer drilldown",
    )
    drill["period"] = _month(drill["period"], "capacity customer drilldown.period")
    specifications = {
        "required_visit_count": ("required_visit_count", "sum"),
        "priority_required_visit_count": ("priority_required_visit_count", "sum"),
        "credited_planned_visit_count": ("planned_visit_count", "sum"),
        "credited_completed_visit_count": ("completed_visit_count", "sum"),
        "required_customer_coverage_hours": ("required_customer_coverage_hours", "sum"),
        "required_priority_customer_coverage_hours": (
            "required_priority_customer_coverage_hours",
            "sum",
        ),
        "customer_coverage_gap": ("customer_coverage_gap", "sum"),
        "priority_customer_coverage_gap": ("priority_customer_coverage_gap", "sum"),
    }
    specifications = {
        output: specification
        for output, specification in specifications.items()
        if specification[0] in drill
    }
    result = (
        drill.groupby(CAPACITY_TERRITORY_KEY, observed=True, dropna=False)
        .agg(**specifications)
        .reset_index()
    )
    if "territory_name" in drill:
        names = (
            drill.groupby(CAPACITY_TERRITORY_KEY, observed=True, dropna=False)[
                "territory_name"
            ]
            .agg(_deterministic_mode)
            .reset_index()
        )
        result = result.merge(
            names, on=CAPACITY_TERRITORY_KEY, how="left", validate="one_to_one"
        )
    return result


def _territory_transaction_counts(
    normalized_transactions: pd.DataFrame | None,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Return exact transaction counts, assigning null geography to the roster fallback."""
    columns = CAPACITY_TERRITORY_KEY + ["transaction_count", "territory_name"]
    if normalized_transactions is None:
        return pd.DataFrame(columns=columns)
    transactions = _prepare_transactions(normalized_transactions)
    if "territory_id" not in transactions:
        return pd.DataFrame(columns=columns)
    fallback = calendar[CAPACITY_KEY + ["territory_id"]].rename(
        columns={"territory_id": "_fallback_territory_id"}
    )
    transactions = transactions.merge(
        fallback, on=CAPACITY_KEY, how="inner", validate="many_to_one"
    )
    transactions["territory_id"] = transactions["territory_id"].where(
        transactions["territory_id"].notna(), transactions["_fallback_territory_id"]
    )
    result = (
        transactions.groupby(CAPACITY_TERRITORY_KEY, observed=True, dropna=False)
        .size()
        .rename("transaction_count")
        .reset_index()
    )
    if "territory_name" in transactions:
        names = (
            transactions.groupby(CAPACITY_TERRITORY_KEY, observed=True, dropna=False)[
                "territory_name"
            ]
            .agg(_deterministic_mode)
            .reset_index()
        )
        result = result.merge(
            names, on=CAPACITY_TERRITORY_KEY, how="left", validate="one_to_one"
        )
    return result


def _append_residual_column(existing: Any, column: str) -> str:
    tokens = set(str(existing).split("|")) if pd.notna(existing) and str(existing) else set()
    tokens.add(column)
    return "|".join(sorted(token for token in tokens if token))


def _build_exact_capacity_territory_allocation(
    capacity_calendar: pd.DataFrame,
    normalized_transactions: pd.DataFrame | None,
    field_visits: pd.DataFrame,
    capacity_customer_drilldown: pd.DataFrame,
    config: Mapping[str, Any] | None,
) -> pd.DataFrame:
    """Build a conserved fact whose core workload has observed geography."""
    calendar = capacity_calendar.copy(deep=True)
    _require_columns(
        calendar,
        CAPACITY_KEY
        + [
            "territory_id",
            "required_total_hours",
            "available_field_hours",
            "required_fte",
            "available_fte",
        ],
        "capacity calendar",
    )
    calendar["period"] = _month(calendar["period"], "capacity calendar.period")
    if calendar.empty or calendar.duplicated(CAPACITY_KEY).any():
        raise ValueError("capacity calendar must contain one row per rep-period")

    visits = _territory_visit_workload(
        field_visits, capacity_customer_drilldown, config
    )
    coverage = _territory_coverage_workload(capacity_customer_drilldown)
    transactions = _territory_transaction_counts(normalized_transactions, calendar)
    fact_frames = [frame for frame in [visits, coverage, transactions] if not frame.empty]
    keys = pd.concat(
        [frame[CAPACITY_TERRITORY_KEY] for frame in fact_frames]
        + [calendar[CAPACITY_KEY + ["territory_id"]]],
        ignore_index=True,
    ).drop_duplicates(CAPACITY_TERRITORY_KEY)
    allocation = keys.copy()

    name_frames: list[pd.DataFrame] = []
    for frame in fact_frames:
        measures = [
            column
            for column in frame
            if column not in CAPACITY_TERRITORY_KEY + ["territory_name"]
        ]
        allocation = allocation.merge(
            frame[CAPACITY_TERRITORY_KEY + measures],
            on=CAPACITY_TERRITORY_KEY,
            how="left",
            validate="one_to_one",
        )
        if "territory_name" in frame:
            name_frames.append(frame[CAPACITY_TERRITORY_KEY + ["territory_name"]])

    if name_frames:
        names = (
            pd.concat(name_frames, ignore_index=True)
            .groupby(CAPACITY_TERRITORY_KEY, observed=True, dropna=False)["territory_name"]
            .agg(_deterministic_mode)
            .reset_index()
        )
        allocation = allocation.merge(
            names, on=CAPACITY_TERRITORY_KEY, how="left", validate="one_to_one"
        )
    else:
        allocation["territory_name"] = pd.NA

    visit_columns = [
        "planned_visit_count",
        "completed_visit_count",
        "planned_visit_hours",
        "planned_travel_hours",
        "excess_service_visit_count",
        "excess_service_visit_hours",
        "excess_service_travel_hours",
        "excess_service_hours",
        "observed_visit_count",
        "observed_completed_visit_count",
        "observed_visit_hours",
        "observed_travel_hours",
    ]
    coverage_columns = [
        "required_visit_count",
        "priority_required_visit_count",
        "credited_planned_visit_count",
        "credited_completed_visit_count",
        "required_customer_coverage_hours",
        "required_priority_customer_coverage_hours",
        "customer_coverage_gap",
        "priority_customer_coverage_gap",
    ]
    exact_columns = [
        column
        for column in ["transaction_count"] + visit_columns + coverage_columns
        if column in calendar
    ]
    for column in exact_columns:
        if column not in allocation:
            allocation[column] = 0.0
        allocation[column] = pd.to_numeric(allocation[column], errors="coerce").fillna(0.0)

    context_columns = [
        "capacity_record_id",
        "rep_name",
        "manager_id",
        "manager_name",
        "team_id",
        "team_name",
        "territory_id",
        "territory_name",
        "risk_medium_threshold_pct",
        "risk_high_threshold_pct",
        "risk_critical_threshold_pct",
        "overload_threshold_pct",
        "capacity_utilization_pct",
        "capacity_risk_band",
        "capacity_overload_flag",
        "data_lineage",
        "synthetic_seed",
    ]
    source = calendar[
        CAPACITY_KEY
        + [
            column
            for column in context_columns
            if column in calendar and column not in CAPACITY_KEY
        ]
    ].rename(
        columns={
            "territory_id": "dominant_territory_id",
            "territory_name": "dominant_territory_name",
            "capacity_utilization_pct": "source_rep_period_utilization_pct",
            "capacity_risk_band": "source_rep_period_risk_band",
            "capacity_overload_flag": "source_rep_period_overload_flag",
        }
    )
    allocation = allocation.merge(source, on=CAPACITY_KEY, how="left", validate="many_to_one")
    allocation["territory_name"] = allocation["territory_name"].where(
        allocation["territory_name"].notna(),
        allocation["territory_id"].astype("string"),
    )
    dominant_name = allocation["dominant_territory_name"].where(
        allocation["dominant_territory_name"].notna(),
        allocation["dominant_territory_id"].astype("string"),
    )
    dominant_rows = allocation["territory_id"].astype("string").eq(
        allocation["dominant_territory_id"].astype("string")
    )
    allocation.loc[dominant_rows, "territory_name"] = allocation.loc[
        dominant_rows, "territory_name"
    ].where(allocation.loc[dominant_rows, "territory_name"].notna(), dominant_name[dominant_rows])

    allocation["residual_allocation_flag"] = False
    allocation["injected_residual_allocation_flag"] = False
    allocation["residual_allocation_columns"] = ""
    allocation["unrepresented_workload_residual_hours"] = 0.0
    source_indexed = calendar.set_index(CAPACITY_KEY)

    def reconcile(columns: list[str], *, workload_hours: set[str] | None = None) -> None:
        workload_hours = workload_hours or set()
        for key, source_row in source_indexed.iterrows():
            mask = allocation["rep_id"].eq(key[0]) & allocation["period"].eq(key[1])
            dominant_mask = mask & allocation["territory_id"].astype("string").eq(
                str(source_row["territory_id"])
            )
            target_index = allocation.index[dominant_mask]
            if len(target_index) != 1:
                raise RuntimeError(f"no unique dominant territory allocation row for {key}")
            target = target_index[0]
            for column in columns:
                if column not in source_row.index or column not in allocation:
                    continue
                expected = float(source_row[column])
                difference = expected - float(allocation.loc[mask, column].sum())
                if abs(difference) <= 1e-9:
                    continue
                allocation.at[target, column] = float(allocation.at[target, column]) + difference
                allocation.at[target, "residual_allocation_flag"] = True
                if str(source_row.get("data_lineage", "")) == "synthetic_injected":
                    allocation.at[target, "injected_residual_allocation_flag"] = True
                allocation.at[target, "residual_allocation_columns"] = _append_residual_column(
                    allocation.at[target, "residual_allocation_columns"], column
                )
                if column in workload_hours:
                    allocation.at[target, "unrepresented_workload_residual_hours"] += difference

    reconcile(
        exact_columns,
        workload_hours={
            "planned_visit_hours",
            "planned_travel_hours",
            "required_customer_coverage_hours",
        },
    )
    if allocation[exact_columns].lt(-1e-8).any().any():
        raise RuntimeError("exact territory workload reconciliation produced a negative value")
    allocation[exact_columns] = allocation[exact_columns].clip(lower=0.0)

    allocation["core_required_hours"] = allocation[
        [
            column
            for column in [
                "planned_visit_hours",
                "planned_travel_hours",
                "required_customer_coverage_hours",
            ]
            if column in allocation
        ]
    ].sum(axis=1)

    transaction_total = allocation.groupby(CAPACITY_KEY, observed=True)[
        "transaction_count"
    ].transform("sum")
    workload_total = allocation.groupby(CAPACITY_KEY, observed=True)[
        "core_required_hours"
    ].transform("sum")
    transaction_share = np.divide(
        allocation["transaction_count"].to_numpy(float),
        transaction_total.to_numpy(float),
        out=np.zeros(len(allocation), dtype=float),
        where=transaction_total.to_numpy(float) > 0,
    )
    workload_share = np.divide(
        allocation["core_required_hours"].to_numpy(float),
        workload_total.to_numpy(float),
        out=np.zeros(len(allocation), dtype=float),
        where=workload_total.to_numpy(float) > 0,
    )
    allocation["territory_allocation_share"] = np.where(
        transaction_total.to_numpy(float) > 0, transaction_share, workload_share
    )
    empty_share = allocation.groupby(CAPACITY_KEY, observed=True)[
        "territory_allocation_share"
    ].transform("sum").le(0)
    allocation.loc[empty_share & dominant_rows, "territory_allocation_share"] = 1.0
    allocation["territory_activity_count"] = allocation["transaction_count"]

    additive_columns = [
        column for column in CAPACITY_ALLOCATION_ADDITIVE_COLUMNS if column in calendar
    ]
    derived_columns = {
        *exact_columns,
        "core_required_hours",
        "workload_buffer_hours",
        "required_total_hours",
        "required_hours",
        "required_fte",
        "available_fte",
        "fte_gap",
    }
    proportional_columns = [
        column for column in additive_columns if column not in derived_columns
    ]
    for column in proportional_columns:
        source_values = allocation[CAPACITY_KEY].merge(
            calendar[CAPACITY_KEY + [column]],
            on=CAPACITY_KEY,
            how="left",
            validate="many_to_one",
        )[column]
        allocation[column] = (
            pd.to_numeric(source_values, errors="raise").to_numpy(float)
            * allocation["territory_allocation_share"].to_numpy(float)
        )

    source_core = allocation[CAPACITY_KEY].merge(
        calendar[CAPACITY_KEY + ["core_required_hours", "workload_buffer_hours"]],
        on=CAPACITY_KEY,
        how="left",
        validate="many_to_one",
    )
    buffer_rate = np.divide(
        source_core["workload_buffer_hours"].to_numpy(float),
        source_core["core_required_hours"].to_numpy(float),
        out=np.zeros(len(allocation), dtype=float),
        where=source_core["core_required_hours"].to_numpy(float) > 0,
    )
    allocation["workload_buffer_hours"] = allocation["core_required_hours"] * buffer_rate
    allocation["required_total_hours"] = (
        allocation["core_required_hours"] + allocation["workload_buffer_hours"]
    )
    reconcile(["core_required_hours", "workload_buffer_hours", "required_total_hours"])
    allocation["required_hours"] = allocation["required_total_hours"]

    nominal = allocation[CAPACITY_KEY].merge(
        calendar[CAPACITY_KEY + ["nominal_full_time_hours"]],
        on=CAPACITY_KEY,
        how="left",
        validate="many_to_one",
    )["nominal_full_time_hours"].to_numpy(float)
    allocation["required_fte"] = np.divide(
        allocation["required_total_hours"].to_numpy(float),
        nominal,
        out=np.zeros(len(allocation), dtype=float),
        where=nominal > 0,
    )
    allocation["available_fte"] = np.divide(
        allocation["available_field_hours"].to_numpy(float),
        nominal,
        out=np.zeros(len(allocation), dtype=float),
        where=nominal > 0,
    )
    allocation["fte_gap"] = allocation["required_fte"] - allocation["available_fte"]
    if "available_hours" in calendar:
        allocation["available_hours"] = allocation["available_field_hours"]

    required = allocation["required_total_hours"].to_numpy(float)
    available = allocation["available_field_hours"].to_numpy(float)
    utilization = np.divide(
        required * 100.0,
        available,
        out=np.zeros(len(allocation), dtype=float),
        where=available > 0,
    )
    utilization[(available <= 0) & (required > 0)] = 10000.0
    allocation["utilization_pct"] = utilization
    allocation["capacity_utilization_pct"] = utilization
    medium = pd.to_numeric(
        allocation.get("risk_medium_threshold_pct", pd.Series(85.0, index=allocation.index)),
        errors="coerce",
    ).fillna(85.0)
    high = pd.to_numeric(
        allocation.get("overload_threshold_pct", pd.Series(100.0, index=allocation.index)),
        errors="coerce",
    ).fillna(100.0)
    critical = pd.to_numeric(
        allocation.get("risk_critical_threshold_pct", pd.Series(120.0, index=allocation.index)),
        errors="coerce",
    ).fillna(120.0)
    allocation["capacity_risk_band"] = np.select(
        [utilization >= critical.to_numpy(float), utilization >= high.to_numpy(float), utilization >= medium.to_numpy(float)],
        ["critical", "high", "medium"],
        default="low",
    )
    allocation["workload_risk_band"] = allocation["capacity_risk_band"]
    allocation["capacity_overload_flag"] = utilization >= high.to_numpy(float)
    allocation["overload_flag"] = allocation["capacity_overload_flag"]
    allocation["dominant_territory_flag"] = dominant_rows
    allocation["geographic_workload_attribution_flag"] = True
    allocation["core_workload_geographic_attribution_flag"] = True
    allocation["availability_geographic_attribution_flag"] = False
    allocation["allocation_basis"] = np.where(
        transaction_total.to_numpy(float) > 0,
        "actual visits + owned-customer coverage; availability by rep-period transaction share",
        "actual visits + owned-customer coverage; availability by workload/roster fallback",
    )
    allocation["allocation_scope"] = (
        "core workload is attributed to actual visited territory and owned-customer coverage "
        "territory; common availability/non-workload quantities are proportional; calendar-only "
        "injected workload residuals are assigned deterministically to the dominant territory"
    )
    allocation["capacity_territory_allocation_id"] = [
        "CAPALLOC_"
        + hashlib.sha256(
            f"{rep_id}|{pd.Timestamp(period).date()}|{territory_id}".encode("utf-8")
        ).hexdigest()[:16]
        for rep_id, period, territory_id in allocation[CAPACITY_TERRITORY_KEY].itertuples(
            index=False, name=None
        )
    ]

    if allocation.duplicated(CAPACITY_TERRITORY_KEY).any():
        raise RuntimeError("capacity territory allocation is not unique at rep-territory-period grain")
    share_total = allocation.groupby(CAPACITY_KEY, observed=True)[
        "territory_allocation_share"
    ].sum()
    if not np.allclose(share_total.to_numpy(float), 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("capacity territory allocation shares do not sum to one")
    allocated_indexed = allocation.groupby(CAPACITY_KEY, observed=True)[additive_columns].sum()
    for column in additive_columns:
        expected = pd.to_numeric(
            source_indexed.loc[allocated_indexed.index, column], errors="raise"
        )
        if not np.allclose(
            allocated_indexed[column].to_numpy(float),
            expected.to_numpy(float),
            rtol=1e-9,
            atol=1e-9,
        ):
            raise RuntimeError(f"capacity territory allocation does not conserve {column}")
    numeric = allocation.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise RuntimeError("capacity territory allocation contains non-finite numeric values")

    preferred = [
        "capacity_territory_allocation_id",
        "capacity_record_id",
        "rep_id",
        "rep_name",
        "manager_id",
        "manager_name",
        "team_id",
        "team_name",
        "territory_id",
        "territory_name",
        "period",
        "transaction_count",
        "territory_allocation_share",
        "dominant_territory_id",
        "dominant_territory_name",
        "dominant_territory_flag",
        "allocation_basis",
        "geographic_workload_attribution_flag",
        "residual_allocation_flag",
        "injected_residual_allocation_flag",
        "unrepresented_workload_residual_hours",
        "residual_allocation_columns",
    ]
    ordered = [column for column in preferred if column in allocation]
    ordered += [column for column in allocation if column not in ordered]
    return allocation[ordered].sort_values(
        ["period", "rep_id", "territory_id"], kind="mergesort"
    ).reset_index(drop=True)


def build_capacity_territory_allocation(
    capacity_calendar: pd.DataFrame,
    normalized_transactions: pd.DataFrame | None = None,
    field_visits: pd.DataFrame | None = None,
    capacity_customer_drilldown: pd.DataFrame | None = None,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Return a conserved rep-period-territory workload and availability fact.

    When visits and the customer coverage drilldown are supplied, core workload
    is geographically attributed from those service facts.  The two-argument
    form remains a documented planning proxy for callers that only have the
    compact calendar and transaction activity.
    """
    if field_visits is None or capacity_customer_drilldown is None:
        return _build_proxy_capacity_territory_allocation(
            capacity_calendar, normalized_transactions
        )
    return _build_exact_capacity_territory_allocation(
        capacity_calendar,
        normalized_transactions,
        field_visits,
        capacity_customer_drilldown,
        config,
    )


def _joined_tokens(values: pd.Series) -> str:
    tokens = sorted(set(values.dropna().astype(str)))
    return "|".join(tokens)


def _aggregate_lineage(values: pd.Series) -> str:
    tokens = set(values.dropna().astype(str))
    if "synthetic_injected" in tokens:
        return "synthetic_injected"
    return _deterministic_mode(values) if tokens else "synthetic_derived"


def build_capacity_territory_summary(allocation: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a conserved allocation fact to one territory-period row."""
    fact = allocation.copy(deep=True)
    _require_columns(
        fact,
        CAPACITY_TERRITORY_KEY
        + ["territory_allocation_share", "required_total_hours", "available_field_hours", "required_fte", "available_fte"],
        "capacity territory allocation",
    )
    fact["period"] = _month(fact["period"], "capacity territory allocation.period")
    if fact.empty or fact.duplicated(CAPACITY_TERRITORY_KEY).any():
        raise ValueError("capacity territory allocation must be unique at rep-territory-period grain")
    additive_columns = [
        column for column in CAPACITY_ALLOCATION_ADDITIVE_COLUMNS if column in fact
    ]
    aggregations: dict[str, Any] = {
        column: (column, "sum") for column in additive_columns
    }
    aggregations.update(
        {
            "territory_name": (
                "territory_name" if "territory_name" in fact else "territory_id",
                _deterministic_mode,
            ),
            "rep_count": ("rep_id", "nunique"),
            "fractional_rep_equivalent": ("territory_allocation_share", "sum"),
            "allocation_row_count": ("rep_id", "size"),
            "rep_ids": ("rep_id", _joined_tokens),
        }
    )
    if "allocation_basis" in fact:
        aggregations["allocation_basis"] = ("allocation_basis", _joined_tokens)
    if "allocation_scope" in fact:
        aggregations["allocation_scope"] = ("allocation_scope", _deterministic_mode)
    for column in [
        "geographic_workload_attribution_flag",
        "core_workload_geographic_attribution_flag",
        "availability_geographic_attribution_flag",
        "residual_allocation_flag",
        "injected_residual_allocation_flag",
    ]:
        if column in fact:
            aggregations[column] = (column, "max")
    if "unrepresented_workload_residual_hours" in fact:
        aggregations["unrepresented_workload_residual_hours"] = (
            "unrepresented_workload_residual_hours",
            "sum",
        )
    for column in ["manager_id", "manager_name", "team_id", "team_name"]:
        if column in fact:
            aggregations[f"{column}s"] = (column, _joined_tokens)
    for column in [
        "risk_medium_threshold_pct",
        "risk_high_threshold_pct",
        "risk_critical_threshold_pct",
        "overload_threshold_pct",
        "synthetic_seed",
    ]:
        if column in fact:
            aggregations[column] = (column, _deterministic_mode)
    if "data_lineage" in fact:
        aggregations["data_lineage"] = ("data_lineage", _aggregate_lineage)

    summary = (
        fact.groupby(TERRITORY_PERIOD_KEY, observed=True, dropna=False)
        .agg(**aggregations)
        .reset_index()
    )
    required = pd.to_numeric(summary["required_total_hours"], errors="raise").to_numpy(float)
    available = pd.to_numeric(summary["available_field_hours"], errors="raise").to_numpy(float)
    utilization = np.divide(
        required * 100.0,
        available,
        out=np.zeros(len(summary), dtype=float),
        where=available > 0,
    )
    utilization[(available <= 0) & (required > 0)] = 10000.0
    summary["utilization_pct"] = utilization
    summary["capacity_utilization_pct"] = utilization
    summary["fte_gap"] = summary["required_fte"] - summary["available_fte"]
    summary["positive_fte_gap"] = summary["fte_gap"].clip(lower=0.0)
    medium = pd.to_numeric(
        summary.get("risk_medium_threshold_pct", pd.Series(85.0, index=summary.index)),
        errors="coerce",
    ).fillna(85.0)
    high = pd.to_numeric(
        summary.get("overload_threshold_pct", summary.get("risk_high_threshold_pct", pd.Series(100.0, index=summary.index))),
        errors="coerce",
    ).fillna(100.0)
    critical = pd.to_numeric(
        summary.get("risk_critical_threshold_pct", pd.Series(120.0, index=summary.index)),
        errors="coerce",
    ).fillna(120.0)
    summary["capacity_risk_band"] = np.select(
        [utilization >= critical.to_numpy(float), utilization >= high.to_numpy(float), utilization >= medium.to_numpy(float)],
        ["critical", "high", "medium"],
        default="low",
    )
    summary["workload_risk_band"] = summary["capacity_risk_band"]
    summary["capacity_overload_flag"] = utilization >= high.to_numpy(float)
    summary["overload_flag"] = summary["capacity_overload_flag"]
    exact = summary.get(
        "geographic_workload_attribution_flag",
        pd.Series(False, index=summary.index),
    ).astype(bool)
    summary["capacity_methodology"] = np.where(
        exact,
        "actual_geographic_workload_plus_proportional_rep_availability",
        "fractionally_allocated_rep_capacity_to_territory_period",
    )
    if "allocation_basis" not in summary:
        summary["allocation_basis"] = "normalized rep-period transaction activity shares"
    if "geographic_workload_attribution_flag" not in summary:
        summary["geographic_workload_attribution_flag"] = False
    if "allocation_scope" not in summary:
        summary["allocation_scope"] = (
            "territory planning proxy from conserved transaction-share allocation; not independent "
            "observed territory workload or coverage truth"
        )
    if "data_lineage" not in summary:
        summary["data_lineage"] = "synthetic_derived"
    summary["capacity_territory_record_id"] = [
        "CAPTERR_"
        + hashlib.sha256(f"{territory_id}|{pd.Timestamp(period).date()}".encode("utf-8")).hexdigest()[:16]
        for territory_id, period in summary[TERRITORY_PERIOD_KEY].itertuples(index=False, name=None)
    ]
    if summary.duplicated(TERRITORY_PERIOD_KEY).any():
        raise RuntimeError("capacity territory summary is not unique at territory-period grain")
    numeric = summary.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise RuntimeError("capacity territory summary contains non-finite numeric values")
    preferred = [
        "capacity_territory_record_id",
        "territory_id",
        "territory_name",
        "period",
        "rep_count",
        "fractional_rep_equivalent",
        "allocation_row_count",
        "required_total_hours",
        "available_field_hours",
        "capacity_utilization_pct",
        "required_fte",
        "available_fte",
        "fte_gap",
        "positive_fte_gap",
        "capacity_risk_band",
        "capacity_overload_flag",
    ]
    ordered = [column for column in preferred if column in summary]
    ordered += [column for column in summary if column not in ordered]
    return summary[ordered].sort_values(["period", "territory_id"], kind="mergesort").reset_index(drop=True)


def _positive_ground_truth(ground_truth: pd.DataFrame) -> pd.DataFrame:
    truth = ground_truth.copy(deep=True)
    if truth.empty:
        return truth
    capacity_text = pd.Series("", index=truth.index, dtype="string")
    for column in ["anomaly_type", "anomaly_category", "affected_dataset", "injection_description"]:
        if column in truth:
            capacity_text = capacity_text.str.cat(truth[column].astype("string").fillna(""), sep=" ")
    capacity_mask = capacity_text.str.contains(
        "capacity|workload|overload|undercoverage|coverage gap", case=False, regex=True, na=False
    )
    if "ground_truth_label" in truth:
        labels = truth["ground_truth_label"]
        numeric = pd.to_numeric(labels, errors="coerce")
        positive = numeric.gt(0) | labels.astype("string").str.strip().str.casefold().isin(
            ["true", "yes", "y", "positive", "anomaly", "injected"]
        )
    else:
        positive = pd.Series(True, index=truth.index)
    # If descriptive fields exist, never reinterpret unrelated commercial truth as
    # capacity truth merely because no capacity case happened to be generated.
    if any(
        column in truth
        for column in ["anomaly_type", "anomaly_category", "affected_dataset", "injection_description"]
    ):
        positive &= capacity_mask
    truth = truth.loc[positive].copy()
    if "period" in truth:
        supplied = truth["period"].notna()
        parsed = pd.to_datetime(truth["period"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        if (supplied & parsed.isna()).any():
            raise ValueError("capacity ground truth.period contains invalid dates")
        truth["period"] = parsed
    return truth


def _overload_ground_truth(capacity_truth: pd.DataFrame) -> pd.DataFrame:
    """Separate workload-overload truth from priority-undercoverage truth."""
    if capacity_truth.empty or "anomaly_type" not in capacity_truth:
        return capacity_truth.copy()
    kinds = capacity_truth["anomaly_type"].astype("string")
    explicit = kinds.str.contains(
        r"overload|workload.*(?:exceed|capacity)|capacity.*exceed",
        case=False,
        regex=True,
        na=False,
    )
    return capacity_truth.loc[explicit].copy()


def _parsed_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _first_mapping(value: Any) -> dict[str, Any]:
    parsed = _parsed_json(value)
    if isinstance(parsed, Mapping):
        return dict(parsed)
    if isinstance(parsed, list):
        return next((dict(item) for item in parsed if isinstance(item, Mapping)), {})
    return {}


def _numeric_capacity_truth(capacity_truth: pd.DataFrame) -> pd.DataFrame:
    """Expose numeric truth embedded in the benchmark JSON audit fields."""
    truth = capacity_truth.copy(deep=True)
    if truth.empty:
        return truth
    payloads = (
        truth["injected_value"].map(_first_mapping)
        if "injected_value" in truth
        else pd.Series([{} for _ in range(len(truth))], index=truth.index)
    )
    truth["injected_required_total_hours"] = payloads.map(
        lambda value: value.get("required_total_hours", np.nan)
    )
    truth["injected_available_field_hours"] = payloads.map(
        lambda value: value.get("available_field_hours", np.nan)
    )
    injected_required = pd.to_numeric(truth["injected_required_total_hours"], errors="coerce")
    injected_available = pd.to_numeric(truth["injected_available_field_hours"], errors="coerce")
    truth["injected_utilization_pct"] = np.divide(
        injected_required * 100.0,
        injected_available,
        out=np.full(len(truth), np.nan),
        where=injected_available.to_numpy(float) > 0,
    )

    def record_id(row: pd.Series) -> Any:
        if str(row.get("entity_type", "")).casefold() == "capacity_record" and pd.notna(
            row.get("entity_id", pd.NA)
        ):
            return row["entity_id"]
        affected = _parsed_json(row.get("affected_record_ids", None))
        return affected[0] if isinstance(affected, list) and len(affected) == 1 else pd.NA

    if "capacity_record_id" not in truth:
        truth["capacity_record_id"] = truth.apply(record_id, axis=1)
    return truth


def _truth_row_mask(row: pd.Series, predictions: pd.DataFrame) -> pd.Series | None:
    """Resolve one truth row to a capacity record without period-wide fallthrough."""
    if "capacity_record_id" in predictions:
        identifiers: list[str] = []
        if str(row.get("entity_type", "")).casefold() == "capacity_record" and pd.notna(
            row.get("entity_id", pd.NA)
        ):
            identifiers.append(str(row["entity_id"]))
        affected = _parsed_json(row.get("affected_record_ids", None))
        if isinstance(affected, list):
            identifiers.extend(str(value) for value in affected)
        if identifiers:
            matched = predictions["capacity_record_id"].astype(str).isin(set(identifiers))
            return matched

    period = row.get("period", pd.NaT)
    rep_id = row.get("rep_id", pd.NA)
    if pd.notna(period) and pd.notna(rep_id):
        return predictions["period"].eq(pd.Timestamp(period)) & predictions["rep_id"].eq(rep_id)

    territory = row.get("territory_id", pd.NA)
    if pd.isna(territory) and str(row.get("entity_type", "")).casefold() == "territory":
        territory = row.get("entity_id", pd.NA)
    if pd.notna(period) and pd.notna(territory) and "territory_id" in predictions:
        return predictions["period"].eq(pd.Timestamp(period)) & predictions["territory_id"].eq(territory)
    return None


def _safe_metric(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _spearman(left: pd.Series, right: pd.Series) -> float:
    valid = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(valid) < 2 or valid.left.nunique() < 2 or valid.right.nunique() < 2:
        return float("nan")
    return float(valid.left.rank(method="average").corr(valid.right.rank(method="average")))


def _numeric_truth_mae(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    prediction_column: str,
    candidates: list[str],
) -> tuple[float, int, str]:
    truth_column = next(
        (
            column
            for column in candidates
            if column in truth
            and pd.to_numeric(truth[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .notna()
            .any()
        ),
        None,
    )
    if truth_column is None:
        return float("nan"), 0, "not_available"
    if (
        "capacity_record_id" in predictions
        and "capacity_record_id" in truth
        and truth["capacity_record_id"].notna().any()
    ):
        keys = ["capacity_record_id"]
    elif set(CAPACITY_KEY).issubset(truth):
        keys = CAPACITY_KEY
    else:
        return float("nan"), 0, "not_available"
    values = truth[keys + [truth_column]].copy()
    values[truth_column] = pd.to_numeric(values[truth_column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    values = values.dropna(subset=keys + [truth_column]).groupby(keys, as_index=False)[truth_column].mean()
    matched = predictions[keys + [prediction_column]].merge(values, on=keys, how="inner")
    matched[prediction_column] = pd.to_numeric(matched[prediction_column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    matched = matched.dropna(subset=[prediction_column, truth_column])
    if matched.empty:
        return float("nan"), 0, f"ground_truth.{truth_column}"
    mae = float((matched[prediction_column] - matched[truth_column]).abs().mean())
    return mae, len(matched), f"ground_truth.{truth_column}"


def evaluate_capacity(
    clean: pd.DataFrame,
    injected: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    clean_territory_allocation: pd.DataFrame | None = None,
    injected_territory_allocation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Evaluate overload flags and numeric/ranking evidence against separate truth.

    The returned one-row frame is CSV-ready.  Numeric MAE fields remain ``NaN``
    with an explicit ``*_basis=not_available`` when no numeric truth was generated.
    No labels are copied into either capacity input.
    """
    clean_frame = clean.copy(deep=True)
    injected_frame = injected.copy(deep=True)
    for label, frame in [("clean capacity", clean_frame), ("injected capacity", injected_frame)]:
        _require_columns(
            frame,
            CAPACITY_KEY
            + ["utilization_pct", "required_total_hours", "required_fte", "available_fte"],
            label,
        )
        frame["period"] = _month(frame["period"], f"{label}.period")
        if frame.duplicated(CAPACITY_KEY).any():
            raise ValueError(f"{label} is not unique at rep-period grain")
    clean_keys = pd.MultiIndex.from_frame(clean_frame[CAPACITY_KEY])
    injected_keys = pd.MultiIndex.from_frame(injected_frame[CAPACITY_KEY])
    if not clean_keys.equals(injected_keys) and set(clean_keys) != set(injected_keys):
        raise ValueError("clean and injected capacity must contain identical rep-period keys")
    capacity_truth = _positive_ground_truth(ground_truth)
    overload_truth = _overload_ground_truth(capacity_truth)
    numeric_truth = _numeric_capacity_truth(capacity_truth)

    threshold = 100.0
    for column in ["overload_threshold_pct", "risk_high_threshold_pct"]:
        if column in injected_frame:
            values = pd.to_numeric(injected_frame[column], errors="coerce").dropna().unique()
            if len(values) == 1 and np.isfinite(values[0]):
                threshold = float(values[0])
                break
    predicted_utilization = pd.to_numeric(injected_frame["utilization_pct"], errors="raise")
    if not np.isfinite(predicted_utilization.to_numpy(float)).all():
        raise ValueError("injected capacity utilization_pct must be finite")
    # The injector may set overload_flag as part of the label construction.  The
    # evaluated rule is independently recomputed from numeric utilization.
    predicted = predicted_utilization.ge(threshold)

    truth_labels = pd.Series(False, index=injected_frame.index, dtype=bool)
    unmatched_truth_rows = 0
    for _, row in overload_truth.iterrows():
        mask = _truth_row_mask(row, injected_frame)
        if mask is None or not mask.any():
            unmatched_truth_rows += 1
            continue
        truth_labels |= mask

    tp = int((predicted & truth_labels).sum())
    fp = int((predicted & ~truth_labels).sum())
    fn = int((~predicted & truth_labels).sum()) + unmatched_truth_rows
    tn = int((~predicted & ~truth_labels).sum())
    workload_mae, workload_n, workload_basis = _numeric_truth_mae(
        injected_frame,
        numeric_truth,
        "required_total_hours",
        [
            "true_required_total_hours",
            "required_total_hours_truth",
            "expected_required_total_hours",
            "truth_required_total_hours",
            "true_workload_hours",
            "injected_required_total_hours",
        ],
    )
    utilization_mae, utilization_n, utilization_basis = _numeric_truth_mae(
        injected_frame,
        numeric_truth,
        "utilization_pct",
        [
            "true_utilization_pct",
            "utilization_pct_truth",
            "expected_utilization_pct",
            "truth_utilization_pct",
            "injected_utilization_pct",
        ],
    )

    paired = clean_frame[CAPACITY_KEY + ["required_total_hours", "utilization_pct"]].merge(
        injected_frame[CAPACITY_KEY + ["required_total_hours", "utilization_pct"]],
        on=CAPACITY_KEY,
        how="inner",
        suffixes=("_clean", "_injected"),
        validate="one_to_one",
    )
    clean_injected_workload_mae = float(
        (paired.required_total_hours_injected - paired.required_total_hours_clean).abs().mean()
    ) if not paired.empty else float("nan")
    clean_injected_utilization_mae = float(
        (paired.utilization_pct_injected - paired.utilization_pct_clean).abs().mean()
    ) if not paired.empty else float("nan")

    clean_allocation = (
        build_capacity_territory_allocation(clean_frame)
        if clean_territory_allocation is None
        else clean_territory_allocation.copy(deep=True)
    )
    injected_allocation = (
        build_capacity_territory_allocation(injected_frame)
        if injected_territory_allocation is None
        else injected_territory_allocation.copy(deep=True)
    )
    clean_territory = build_capacity_territory_summary(clean_allocation)
    injected_territory = build_capacity_territory_summary(injected_allocation)

    ranking = float("nan")
    ranking_basis = "not_available"
    territory_count = int(injected_territory["territory_id"].nunique(dropna=False))
    if territory_count:
        truth_by_key = pd.Series(
            truth_labels.to_numpy(bool),
            index=pd.MultiIndex.from_frame(injected_frame[CAPACITY_KEY]),
        )
        allocation_truth_keys = pd.MultiIndex.from_frame(injected_allocation[CAPACITY_KEY])
        injected_allocation["truth_overload_weight"] = (
            truth_by_key.reindex(allocation_truth_keys, fill_value=False).to_numpy(bool).astype(float)
            * injected_allocation["territory_allocation_share"].to_numpy(float)
        )
        truth_pressure = (
            injected_allocation.groupby(TERRITORY_PERIOD_KEY, observed=True, dropna=False)[
                "truth_overload_weight"
            ]
            .sum()
            .rename("allocated_rep_period_truth_weight")
            .reset_index()
        )
        diagnostic = injected_territory[
            TERRITORY_PERIOD_KEY + ["utilization_pct"]
        ].merge(
            truth_pressure,
            on=TERRITORY_PERIOD_KEY,
            how="left",
            validate="one_to_one",
        )
        diagnostic["allocated_rep_period_truth_weight"] = diagnostic[
            "allocated_rep_period_truth_weight"
        ].fillna(0.0)
        ranking = _spearman(
            diagnostic["utilization_pct"],
            diagnostic["allocated_rep_period_truth_weight"],
        )
        ranking_basis = (
            "allocation-sensitivity diagnostic at territory-period grain: utilization versus "
            "fractionally propagated rep-period truth; not independent territory truth"
        )

    risk = injected_frame.get("capacity_risk_band", injected_frame.get("workload_risk_band", pd.Series("Unknown", index=injected_frame.index)))
    risk = risk.astype(str).str.title()
    def finite_fte(column: str) -> pd.Series:
        if column not in injected_frame:
            raise ValueError(f"injected capacity is missing required column: {column}")
        values = pd.to_numeric(injected_frame[column], errors="coerce")
        if not np.isfinite(values.to_numpy(float)).all():
            raise ValueError(f"injected capacity {column} must be finite")
        return values

    required_fte_values = finite_fte("required_fte")
    available_fte_values = finite_fte("available_fte")
    fte_gap_values = required_fte_values - available_fte_values
    total_required_fte = float(required_fte_values.sum())
    total_available_fte = float(available_fte_values.sum())

    def configured_threshold(column: str, default: float) -> float:
        if column not in injected_frame:
            return default
        values = pd.to_numeric(injected_frame[column], errors="coerce").dropna().unique()
        return float(values[0]) if len(values) == 1 and np.isfinite(values[0]) else default

    medium_threshold = configured_threshold("risk_medium_threshold_pct", 85.0)
    critical_threshold = configured_threshold("risk_critical_threshold_pct", 120.0)
    above_medium = predicted_utilization.ge(medium_threshold)
    above_critical = predicted_utilization.ge(critical_threshold)
    territory_utilization = pd.to_numeric(
        injected_territory["utilization_pct"], errors="raise"
    )
    territory_above_medium = territory_utilization.ge(medium_threshold)
    territory_above_high = territory_utilization.ge(threshold)
    territory_above_critical = territory_utilization.ge(critical_threshold)
    undercoverage_truth_count = int(
        capacity_truth.get("anomaly_type", pd.Series(dtype="string"))
        .astype("string")
        .str.contains("undercoverage|coverage_gap", case=False, regex=True, na=False)
        .sum()
    )
    result = {
        "risk_medium_threshold_pct": medium_threshold,
        "overload_threshold_pct": threshold,
        "risk_critical_threshold_pct": critical_threshold,
        "precision": _safe_metric(tp, tp + fp),
        "recall": _safe_metric(tp, tp + fn),
        "overload_precision": _safe_metric(tp, tp + fp),
        "overload_recall": _safe_metric(tp, tp + fn),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "ground_truth_overload_count": int(truth_labels.sum()) + unmatched_truth_rows,
        "capacity_ground_truth_row_count": len(capacity_truth),
        "undercoverage_ground_truth_row_count": undercoverage_truth_count,
        "unmatched_overload_truth_row_count": unmatched_truth_rows,
        "predicted_overload_count": int(predicted.sum()),
        "above_medium_threshold_count": int(above_medium.sum()),
        "above_high_threshold_count": int(predicted.sum()),
        "above_critical_threshold_count": int(above_critical.sum()),
        "overloaded_rep_period_count": int(predicted.sum()),
        "reps_above_medium_threshold": int(injected_frame.loc[above_medium, "rep_id"].nunique()),
        "reps_above_high_threshold": int(injected_frame.loc[predicted, "rep_id"].nunique()),
        "reps_above_critical_threshold": int(injected_frame.loc[above_critical, "rep_id"].nunique()),
        "overloaded_territory_count": int(
            injected_territory.loc[territory_above_high, "territory_id"].nunique(dropna=False)
        ),
        "territories_above_medium_threshold": int(
            injected_territory.loc[territory_above_medium, "territory_id"].nunique(dropna=False)
        ),
        "territories_above_critical_threshold": int(
            injected_territory.loc[territory_above_critical, "territory_id"].nunique(dropna=False)
        ),
        "overloaded_territory_period_count": int(territory_above_high.sum()),
        "territory_period_count": len(injected_territory),
        "clean_row_count": len(clean_frame),
        "injected_row_count": len(injected_frame),
        "territory_count": territory_count,
        "mae_required_total_hours": workload_mae,
        "workload_mae": workload_mae,
        "workload_mae_observations": workload_n,
        "workload_mae_basis": workload_basis,
        "mae_utilization_pct": utilization_mae,
        "utilization_mae": utilization_mae,
        "utilization_mae_observations": utilization_n,
        "utilization_mae_basis": utilization_basis,
        "numeric_truth_independent_flag": False,
        "numeric_mae_interpretation": (
            "Deterministic reconciliation to controlled injected values; "
            "not an independent predictive-accuracy estimate"
        ),
        "clean_injected_required_hours_mae": clean_injected_workload_mae,
        "clean_injected_utilization_pct_mae": clean_injected_utilization_mae,
        "territory_ranking_agreement": float("nan"),
        "territory_rank_spearman": float("nan"),
        "territory_ranking_basis": "not_available: no independently targeted territory-period truth",
        "territory_allocation_sensitivity_spearman": ranking,
        "territory_allocation_sensitivity_basis": ranking_basis,
        "territory_truth_independent_flag": False,
        "territory_capacity_basis": (
            "conserved rep-territory-period allocation aggregated to territory-period"
        ),
        "low_risk_count": int(risk.eq("Low").sum()),
        "medium_risk_count": int(risk.eq("Medium").sum()),
        "high_risk_count": int(risk.eq("High").sum()),
        "critical_risk_count": int(risk.eq("Critical").sum()),
        "total_required_fte": total_required_fte,
        "total_available_fte": total_available_fte,
        "total_fte_gap": total_required_fte - total_available_fte,
        "positive_fte_gap_fte_months": float(fte_gap_values.clip(lower=0.0).sum()),
        "fte_aggregate_basis": "sum across rep-periods (FTE-months), not a point-in-time headcount",
        "data_lineage": "synthetic_derived",
    }
    return pd.DataFrame([result])
