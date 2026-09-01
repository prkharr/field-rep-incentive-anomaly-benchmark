"""Controlled relational anomaly injection with separate ground truth."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from .policy import calculate_incentives


SCENARIOS = [
    ("peer_incentive_outlier", "incentive", "Payout substantially above comparable peers at similar attainment"),
    ("incorrect_accelerator_tier", "incentive", "Incorrect accelerator tier applied"),
    ("duplicate_incentive_adjustment", "incentive", "Duplicate or repeated incentive adjustment"),
    ("unsupported_manual_override", "incentive", "Manual incentive override without supporting performance"),
    ("end_of_period_sales_spike", "order_timing", "Large end-of-period sales spike"),
    ("post_payout_returns", "returns", "High end-of-period sales followed by returns after payout"),
    ("threshold_crossing_discount", "discount", "High discounts used to cross an incentive threshold"),
    ("low_volume_customer_spike", "customer", "Sudden high-volume purchase by historically low-volume customer"),
    ("customer_sales_concentration", "customer", "Unusual concentration of sales in one customer"),
    ("incentivized_product_mix_shift", "product", "Sudden mix shift toward a highly incentivized low-volume product"),
    ("extremely_short_visits", "activity", "Excessive number of extremely short customer visits"),
    ("overlap_impossible_travel", "activity", "Overlapping visits or physically impossible travel"),
    ("sales_without_supporting_activity", "activity", "Sales reported without reasonable supporting customer activity"),
    ("high_activity_low_engagement", "activity", "High activity claims with very low customer engagement"),
    ("inflated_travel_distance", "expense", "Expense distance materially above simulated travel distance"),
    ("duplicate_expense_claim", "expense", "Duplicate expense claim"),
    ("unusual_return_rate", "returns", "Unusual return or cancellation rate"),
    ("late_repeated_target_revision", "quota", "Repeated target revision shortly before period close"),
    ("territory_workload_exceeds_capacity", "capacity", "Territory workload exceeds available field capacity"),
    ("persistent_priority_undercoverage", "capacity", "Persistent undercoverage of high-priority customers"),
    ("territory_potential_explained_performance", "territory", "High performance mainly explained by territory potential"),
    ("multi_signal_sales_discount_returns", "multi_signal", "Combined sales spike, high discount, and post-payout returns"),
]

EXPECTED_SIGNALS = {
    "peer_incentive_outlier": ["payout_to_peer_median_ratio", "incentive_calculation_residual"],
    "incorrect_accelerator_tier": ["accelerator_cliff_distance", "incentive_calculation_residual"],
    "duplicate_incentive_adjustment": ["manual_adjustment_ratio", "incentive_calculation_residual"],
    "unsupported_manual_override": ["manual_adjustment_ratio", "territory_adjusted_incentive_residual"],
    "end_of_period_sales_spike": ["end_of_period_sales_share", "sales_growth"],
    "post_payout_returns": ["post_incentive_return_rate", "end_of_period_sales_share"],
    "threshold_crossing_discount": ["threshold_crossing_discount_signal", "average_discount_pct"],
    "low_volume_customer_spike": ["low_potential_customer_sales_share", "customer_mix_shift"],
    "customer_sales_concentration": ["customer_concentration_hhi", "top_customer_sales_share"],
    "incentivized_product_mix_shift": ["highly_incentivized_product_share", "product_mix_shift"],
    "extremely_short_visits": ["extremely_short_visit_rate", "average_visit_duration"],
    "overlap_impossible_travel": ["impossible_travel_count", "overlapping_visit_count"],
    "sales_without_supporting_activity": ["sales_per_visit", "completed_visit_count"],
    "high_activity_low_engagement": ["completed_visit_count", "visit_to_sales_conversion"],
    "inflated_travel_distance": ["distance_claim_ratio", "expense_vs_peer"],
    "duplicate_expense_claim": ["duplicate_expense_signal", "expense_per_visit"],
    "unusual_return_rate": ["return_rate", "cancelled_order_rate"],
    "late_repeated_target_revision": ["target_revision_flag", "quota_difficulty_index"],
    "territory_workload_exceeds_capacity": ["capacity_utilization_pct", "fte_gap"],
    "persistent_priority_undercoverage": ["priority_customer_coverage_gap", "missed_priority_visit_count"],
    "territory_potential_explained_performance": ["sales_vs_territory_potential", "territory_adjusted_sales_residual"],
    "multi_signal_sales_discount_returns": ["sales_growth", "average_discount_pct", "post_incentive_return_rate"],
}

ORDER_RELATED_CATEGORIES = {
    "order_timing",
    "customer",
    "product",
    "territory",
    "multi_signal",
    "discount",
    "returns",
}


# Stable keys for facts that the controlled injector is allowed to mutate or
# add. Territory allocation and summary outputs are rebuilt later by the
# pipeline and are deliberately outside this injection-stage audit.
INJECTION_PROVENANCE_KEYS = {
    "rep_targets_quotas": ["rep_id", "period"],
    "orders": ["order_line_id"],
    "discount_detail": ["discount_id"],
    "returns_cancellations": ["return_id"],
    "field_visits": ["visit_id"],
    "crm_interactions": ["interaction_id"],
    "travel_expenses": ["expense_id"],
    "incentive_calculations": ["incentive_record_id"],
    "capacity_calendar": ["capacity_record_id"],
    "capacity_customer_drilldown": ["rep_id", "period", "customer_id"],
}

PROVENANCE_AUDIT_COLUMNS = [
    "dataset",
    "record_id",
    "rep_id",
    "period",
    "change_type",
    "data_lineage",
]


def _trace_component(column: str, value: Any) -> str:
    if pd.isna(value):
        raise ValueError(f"Stable provenance key {column!r} cannot be null")
    if column == "period":
        return str(pd.Timestamp(value).date())
    return str(value)


def _stable_record_ids(frame: pd.DataFrame, key_columns: list[str]) -> pd.Series:
    missing = [column for column in key_columns if column not in frame]
    if missing:
        raise ValueError(f"Missing stable provenance key columns: {missing}")
    return pd.Series(
        [
            "|".join(
                _trace_component(column, value)
                for column, value in zip(key_columns, values)
            )
            for values in frame[key_columns].itertuples(index=False, name=None)
        ],
        index=frame.index,
        dtype="string",
    )


def _provenance_context(
    dataset: str,
    row: pd.Series,
    injected_tables: dict[str, pd.DataFrame],
) -> tuple[str | None, pd.Timestamp | None]:
    rep_id = row.get("rep_id")
    period = row.get("period")
    if dataset == "returns_cancellations":
        period = row.get("payout_period")
    elif dataset == "discount_detail":
        orders = injected_tables.get("orders")
        if orders is not None and "order_line_id" in row:
            matches = orders.loc[
                orders["order_line_id"].astype(str).eq(str(row["order_line_id"])),
                ["rep_id", "period"],
            ]
            if len(matches) == 1:
                rep_id, period = matches.iloc[0][["rep_id", "period"]]
    normalized_rep = None if pd.isna(rep_id) else str(rep_id)
    normalized_period = (
        None
        if pd.isna(period)
        else pd.Timestamp(period).to_period("M").to_timestamp()
    )
    return normalized_rep, normalized_period


def audit_injected_record_changes(
    clean_tables: dict[str, pd.DataFrame],
    injected_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Diff mutable injection facts by stable primary key, excluding lineage.

    The returned rows are the added or value-modified records for which the
    injected layer must carry both ``synthetic_injected`` lineage and a
    ground-truth audit reference.
    """
    changes: list[dict[str, Any]] = []
    for dataset, key_columns in INJECTION_PROVENANCE_KEYS.items():
        if dataset not in clean_tables or dataset not in injected_tables:
            continue
        clean = clean_tables[dataset]
        injected = injected_tables[dataset]
        clean_ids = _stable_record_ids(clean, key_columns)
        injected_ids = _stable_record_ids(injected, key_columns)
        if clean_ids.duplicated().any() or injected_ids.duplicated().any():
            raise ValueError(f"Stable provenance keys are not unique for {dataset}")

        clean_columns = set(clean.columns) - {"data_lineage"}
        injected_columns = set(injected.columns) - {"data_lineage"}
        if clean_columns != injected_columns:
            raise ValueError(
                f"Clean/injected schema mismatch for {dataset}: "
                f"clean_only={sorted(clean_columns - injected_columns)}, "
                f"injected_only={sorted(injected_columns - clean_columns)}"
            )
        compare_columns = sorted(clean_columns)
        clean_aligned = clean[compare_columns].copy()
        clean_aligned.index = pd.Index(clean_ids.astype(str), name="_record_id")
        injected_values = injected[compare_columns].reset_index(drop=True)
        baseline_values = clean_aligned.reindex(
            pd.Index(injected_ids.astype(str), name="_record_id")
        ).reset_index(drop=True)
        same_values = (
            injected_values.eq(baseline_values)
            | (injected_values.isna() & baseline_values.isna())
        ).fillna(False).all(axis=1)
        existing = injected_ids.astype(str).isin(set(clean_ids.astype(str)))
        changed_mask = (~existing.to_numpy(dtype=bool)) | (~same_values.to_numpy(dtype=bool))
        if not changed_mask.any():
            continue

        changed = injected.loc[changed_mask]
        changed_ids = injected_ids.loc[changed.index]
        existing_by_index = existing.loc[changed.index]
        for index, row in changed.iterrows():
            rep_id, period = _provenance_context(dataset, row, injected_tables)
            changes.append(
                {
                    "dataset": dataset,
                    "record_id": str(changed_ids.loc[index]),
                    "rep_id": rep_id,
                    "period": period,
                    "change_type": (
                        "modified" if bool(existing_by_index.loc[index]) else "added"
                    ),
                    "data_lineage": row.get("data_lineage"),
                }
            )
    if not changes:
        return pd.DataFrame(columns=PROVENANCE_AUDIT_COLUMNS)
    return (
        pd.DataFrame(changes, columns=PROVENANCE_AUDIT_COLUMNS)
        .sort_values(["period", "rep_id", "dataset", "record_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def ground_truth_trace_pairs(
    ground_truth: list[dict[str, Any]] | pd.DataFrame,
    injected_tables: dict[str, pd.DataFrame],
) -> set[tuple[str, str]]:
    """Return dataset/record-ID pairs that resolve against stable fact keys."""
    records = (
        ground_truth.to_dict("records")
        if isinstance(ground_truth, pd.DataFrame)
        else ground_truth
    )
    id_registry = {
        dataset: set(_stable_record_ids(injected_tables[dataset], key_columns).astype(str))
        for dataset, key_columns in INJECTION_PROVENANCE_KEYS.items()
        if dataset in injected_tables
    }
    pairs: set[tuple[str, str]] = set()
    for record in records:
        datasets = [
            value for value in str(record.get("affected_dataset", "")).split("|")
            if value
        ]
        raw_ids = record.get("affected_record_ids", "[]")
        record_ids = json.loads(raw_ids) if isinstance(raw_ids, str) else raw_ids
        if not isinstance(record_ids, list):
            raise ValueError("affected_record_ids must encode a JSON list")
        for dataset in datasets:
            registered_ids = id_registry.get(dataset, set())
            pairs.update(
                (dataset, str(record_id))
                for record_id in record_ids
                if str(record_id) in registered_ids
            )
    return pairs


def _repair_injected_provenance(
    clean_tables: dict[str, pd.DataFrame],
    injected_tables: dict[str, pd.DataFrame],
    records: list[dict[str, Any]],
) -> None:
    """Mark and trace every added/modified injection-stage fact in place."""
    audit = audit_injected_record_changes(clean_tables, injected_tables)
    if audit.empty:
        return

    for dataset, group in audit.groupby("dataset", observed=True):
        frame = injected_tables[str(dataset)]
        if "data_lineage" not in frame:
            raise RuntimeError(f"Changed injected dataset lacks data_lineage: {dataset}")
        row_ids = _stable_record_ids(frame, INJECTION_PROVENANCE_KEYS[str(dataset)])
        mask = row_ids.astype(str).isin(set(group["record_id"].astype(str)))
        frame.loc[mask, "data_lineage"] = "synthetic_injected"

    traced_pairs = ground_truth_trace_pairs(records, injected_tables)
    for change in audit.itertuples(index=False):
        pair = (str(change.dataset), str(change.record_id))
        if pair in traced_pairs:
            continue
        if change.rep_id is None or pd.isna(change.period):
            raise RuntimeError(f"Changed record lacks rep-period provenance context: {pair}")
        period = pd.Timestamp(change.period).to_period("M").to_timestamp()
        matches = [
            record
            for record in records
            if str(record.get("rep_id")) == str(change.rep_id)
            and pd.Timestamp(record.get("period")).to_period("M").to_timestamp()
            == period
        ]
        if not matches:
            raise RuntimeError(
                "Changed record has no same-period ground-truth record: "
                f"{pair} rep_id={change.rep_id} period={period.date()}"
            )
        for record in matches:
            raw_ids = record.get("affected_record_ids", "[]")
            record_ids = json.loads(raw_ids) if isinstance(raw_ids, str) else raw_ids
            if not isinstance(record_ids, list):
                raise ValueError("affected_record_ids must encode a JSON list")
            record["affected_record_ids"] = _json(
                list(dict.fromkeys([*[str(value) for value in record_ids], str(change.record_id)]))
            )
            datasets = [
                value
                for value in str(record.get("affected_dataset", "")).split("|")
                if value
            ]
            record["affected_dataset"] = "|".join(
                dict.fromkeys([*datasets, str(change.dataset)])
            )
        traced_pairs.add(pair)

    verified = audit_injected_record_changes(clean_tables, injected_tables)
    bad_lineage = verified.loc[
        ~verified["data_lineage"].eq("synthetic_injected"),
        ["dataset", "record_id", "data_lineage"],
    ]
    if not bad_lineage.empty:
        raise RuntimeError(
            "Changed records lack synthetic_injected lineage: "
            f"{bad_lineage.head(10).to_dict('records')}"
        )
    traced_pairs = ground_truth_trace_pairs(records, injected_tables)
    missing_trace = [
        (str(row.dataset), str(row.record_id))
        for row in verified.itertuples(index=False)
        if (str(row.dataset), str(row.record_id)) not in traced_pairs
    ]
    if missing_trace:
        raise RuntimeError(
            f"Changed records lack ground-truth traceability: {missing_trace[:10]}"
        )


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _severities(count: int, mix: dict[str, float], rng: np.random.Generator) -> list[str]:
    labels = list(mix)
    probabilities = np.asarray([mix[label] for label in labels], dtype=float)
    probabilities /= probabilities.sum()
    raw_counts = probabilities * count
    allocated = np.floor(raw_counts).astype(int)
    remainder = count - int(allocated.sum())
    order = np.argsort(-(raw_counts - allocated), kind="stable")
    for index in order[:remainder]:
        allocated[index] += 1
    if count >= len(labels):
        missing = np.flatnonzero(allocated == 0)
        for index in missing:
            donor = int(np.argmax(allocated))
            allocated[donor] -= 1
            allocated[index] += 1
    result = [
        label for label, label_count in zip(labels, allocated)
        for _ in range(int(label_count))
    ]
    rng.shuffle(result)
    return result


def _factor(severity: str) -> float:
    return {"low": 1.22, "medium": 1.55, "high": 2.05}[severity]


def _record(
    records: list[dict[str, Any]],
    injection_id: str,
    anomaly_type: str,
    category: str,
    description: str,
    severity: str,
    rep_id: str,
    period: pd.Timestamp,
    dataset: str,
    record_ids: list[str],
    original: Any,
    injected: Any,
    entity_type: str = "rep_period",
    entity_id: str | None = None,
) -> None:
    original_json = _json(original)
    injected_json = _json(injected)
    if original_json == injected_json:
        raise RuntimeError(
            f"Controlled injection {injection_id} ({anomaly_type}) recorded no value change"
        )
    records.append(
        {
            "injection_id": injection_id,
            "entity_type": entity_type,
            "entity_id": entity_id or f"{rep_id}|{pd.Timestamp(period).date()}",
            "rep_id": rep_id,
            "period": pd.Timestamp(period),
            "anomaly_type": anomaly_type,
            "anomaly_category": category,
            "severity": severity,
            "affected_dataset": dataset,
            "affected_record_ids": _json(record_ids),
            "injection_description": description,
            "original_value": original_json,
            "injected_value": injected_json,
            "expected_detection_signals": _json(EXPECTED_SIGNALS[anomaly_type]),
            "ground_truth_label": 1,
            "data_lineage": "synthetic_injected",
        }
    )


def _extend_truth_with_order_calibration(
    records: list[dict[str, Any]],
    rep_id: str,
    period: pd.Timestamp,
    record_ids: list[str],
    severity: str = "low",
    value_factor: float = 1.08,
) -> None:
    """Attach subtle order-prevalence lines to an existing case audit.

    The extra order lines calibrate line-level prevalence but are not separate
    rep-period scenarios.  Recording them on the existing positive case keeps
    the configured correlated-case share meaningful and preserves complete
    affected-record traceability.
    """
    matches = [
        record
        for record in records
        if record["rep_id"] == rep_id
        and pd.Timestamp(record["period"]) == pd.Timestamp(period)
        and record["anomaly_category"] != "capacity"
    ]
    match = next(
        (
            record
            for record in matches
            if record["anomaly_category"] in ORDER_RELATED_CATEGORIES
        ),
        matches[0] if matches else None,
    )
    if match is None:
        _record(
            records,
            f"ORD_CASE_{len(records) + 1:04d}",
            "end_of_period_sales_spike",
            "order_timing",
            "Subtle controlled timing/value case used to complete configured rep-period prevalence",
            severity,
            rep_id,
            pd.Timestamp(period),
            "orders",
            record_ids,
            {"line_count": len(record_ids), "value_factor": 1.0},
            {"line_count": len(record_ids), "value_factor": value_factor, "order_day": "period_end"},
        )
        return
    existing_ids = json.loads(match["affected_record_ids"])
    match["affected_record_ids"] = _json(list(dict.fromkeys([*existing_ids, *record_ids])))
    datasets = [value for value in str(match["affected_dataset"]).split("|") if value]
    match["affected_dataset"] = "|".join(dict.fromkeys([*datasets, "orders"]))
    original = json.loads(match["original_value"])
    injected = json.loads(match["injected_value"])
    audit = {
        "line_count": len(record_ids),
        "value_factor": value_factor,
        "order_day": "period_end",
    }
    match["original_value"] = _json(
        {"scenario_value": original, "order_prevalence_calibration": {"line_count": len(record_ids), "value_factor": 1.0}}
    )
    match["injected_value"] = _json(
        {"scenario_value": injected, "order_prevalence_calibration": audit}
    )
    signals = json.loads(match["expected_detection_signals"])
    match["expected_detection_signals"] = _json(
        list(dict.fromkeys([*signals, "end_of_period_sales_share", "sales_growth"]))
    )
    match["injection_description"] += "; includes subtle order-line timing/value calibration"


def _order_rows(tables: dict[str, pd.DataFrame], rep_id: str, period: pd.Timestamp) -> pd.Index:
    orders = tables["orders"]
    return orders.index[orders["rep_id"].eq(rep_id) & pd.to_datetime(orders["period"]).eq(period)]


def _reconcile_order_discount_arithmetic(tables: dict[str, pd.DataFrame]) -> None:
    """Keep order and discount-detail arithmetic exact after perturbations."""
    orders = tables["orders"]
    orders["discount_amount"] = orders["gross_sales"].abs() * orders["discount_pct"]
    orders["net_sales"] = (
        orders["gross_sales"]
        - np.sign(orders["gross_sales"]) * orders["discount_amount"]
    )
    order_values = orders.set_index("order_line_id")[["discount_pct", "discount_amount"]]
    detail = tables["discount_detail"]
    detail["discount_pct"] = detail["order_line_id"].map(order_values["discount_pct"]).to_numpy(float)
    detail["discount_amount"] = detail["order_line_id"].map(
        order_values["discount_amount"]
    ).to_numpy(float)


def _set_order_period_end(
    tables: dict[str, pd.DataFrame], indices: Any, period: pd.Timestamp
) -> None:
    """Move controlled order lines to period close without invalid date order."""
    orders = tables["orders"]
    indices = pd.Index(indices)
    new_order_date = pd.Timestamp(period) + pd.offsets.MonthEnd(1)
    old_order = pd.to_datetime(orders.loc[indices, "order_date"])
    invoice = pd.to_datetime(orders.loc[indices, "invoice_date"])
    fulfillment = pd.to_datetime(orders.loc[indices, "fulfillment_date"])
    invoice_lag = (invoice - old_order).clip(lower=pd.Timedelta(0))
    fulfillment_lag = (fulfillment - invoice).clip(lower=pd.Timedelta(0))
    orders.loc[indices, "order_date"] = new_order_date
    orders.loc[indices, "invoice_date"] = (
        pd.Series(new_order_date, index=indices) + invoice_lag
    ).to_numpy()
    orders.loc[indices, "fulfillment_date"] = (
        pd.to_datetime(orders.loc[indices, "invoice_date"]) + fulfillment_lag
    ).to_numpy()


def _reconcile_order_and_return_chronology(tables: dict[str, pd.DataFrame]) -> None:
    """Reconcile dates and cascaded return linkage after order timing changes."""
    orders = tables["orders"]
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    orders["invoice_date"] = pd.to_datetime(orders["invoice_date"])
    orders["fulfillment_date"] = pd.to_datetime(orders["fulfillment_date"])
    orders["invoice_date"] = orders[["order_date", "invoice_date"]].max(axis=1)
    orders["fulfillment_date"] = orders[["invoice_date", "fulfillment_date"]].max(axis=1)

    returns = tables["returns_cancellations"]
    linked_order_dates = returns["order_line_id"].map(
        orders.set_index("order_line_id")["order_date"]
    )
    returns["original_order_date"] = pd.to_datetime(linked_order_dates)
    returns["return_date"] = pd.to_datetime(returns["return_date"])
    early = returns["return_date"].lt(returns["original_order_date"])
    returns.loc[early, "return_date"] = (
        returns.loc[early, "original_order_date"] + pd.Timedelta(days=1)
    )
    returns["days_after_order"] = (
        returns["return_date"] - returns["original_order_date"]
    ).dt.days.astype(int)
    payout_dates = tables["incentive_calculations"].set_index(["rep_id", "period"])[
        "payout_date"
    ]
    return_keys = pd.MultiIndex.from_arrays(
        [returns["rep_id"], pd.to_datetime(returns["payout_period"])]
    )
    effective_payout = pd.Series(return_keys.map(payout_dates), index=returns.index)
    returns["after_incentive_payout_flag"] = returns["return_date"].gt(
        pd.to_datetime(effective_payout)
    )


def _capacity_buffer_rates(
    capacity: pd.DataFrame, indices: pd.Index, config: dict[str, Any]
) -> pd.Series:
    core = pd.to_numeric(capacity.loc[indices, "core_required_hours"], errors="coerce")
    buffer = pd.to_numeric(capacity.loc[indices, "workload_buffer_hours"], errors="coerce")
    fallback = float(config.get("capacity", {}).get("administrative_buffer_pct", 0.08))
    rates = pd.Series(
        np.divide(
            buffer.to_numpy(float),
            core.to_numpy(float),
            out=np.full(len(indices), fallback, dtype=float),
            where=core.to_numpy(float) > 1e-12,
        ),
        index=indices,
    )
    return rates.replace([np.inf, -np.inf], fallback).fillna(fallback).clip(lower=0.0)


def _recalculate_capacity_rows(
    capacity: pd.DataFrame,
    indices: pd.Index,
    config: dict[str, Any],
    buffer_rates: pd.Series | None = None,
) -> None:
    """Recompute every capacity alias from auditable workload components."""
    indices = pd.Index(indices)
    if len(indices) == 0:
        return
    if buffer_rates is None:
        buffer_rates = _capacity_buffer_rates(capacity, indices, config)
    core = capacity.loc[
        indices,
        ["planned_visit_hours", "planned_travel_hours", "required_customer_coverage_hours"],
    ].sum(axis=1)
    capacity.loc[indices, "core_required_hours"] = core.to_numpy(float)
    capacity.loc[indices, "workload_buffer_hours"] = (
        core * buffer_rates.reindex(indices).to_numpy(float)
    ).to_numpy(float)
    capacity.loc[indices, "required_total_hours"] = (
        capacity.loc[indices, "core_required_hours"]
        + capacity.loc[indices, "workload_buffer_hours"]
    ).to_numpy(float)
    available = capacity.loc[indices, "available_field_hours"].clip(lower=1e-12)
    utilization = 100.0 * capacity.loc[indices, "required_total_hours"] / available
    capacity.loc[indices, "utilization_pct"] = utilization.to_numpy(float)
    capacity.loc[indices, "capacity_utilization_pct"] = utilization.to_numpy(float)
    capacity.loc[indices, "required_hours"] = capacity.loc[
        indices, "required_total_hours"
    ].to_numpy(float)
    nominal = capacity.loc[indices, "nominal_full_time_hours"].clip(lower=1e-12)
    capacity.loc[indices, "required_fte"] = (
        capacity.loc[indices, "required_total_hours"] / nominal
    ).to_numpy(float)
    capacity.loc[indices, "fte_gap"] = (
        capacity.loc[indices, "required_fte"] - capacity.loc[indices, "available_fte"]
    ).to_numpy(float)
    thresholds = config.get("capacity", {}).get("risk_thresholds", {})
    medium_pct = 100.0 * float(thresholds.get("medium", 0.85))
    high_pct = 100.0 * float(thresholds.get("high", 1.00))
    critical_pct = 100.0 * float(thresholds.get("critical", 1.20))
    bands = np.select(
        [utilization.ge(critical_pct), utilization.ge(high_pct), utilization.ge(medium_pct)],
        ["critical", "high", "medium"],
        default="low",
    )
    capacity.loc[indices, "capacity_risk_band"] = bands
    capacity.loc[indices, "workload_risk_band"] = bands
    overload = utilization.ge(high_pct)
    capacity.loc[indices, "overload_flag"] = overload.to_numpy(bool)
    capacity.loc[indices, "capacity_overload_flag"] = overload.to_numpy(bool)


def _raise_capacity_required_total(
    capacity: pd.DataFrame,
    indices: pd.Index,
    desired_total: pd.Series,
    config: dict[str, Any],
) -> None:
    """Raise an auditable servicing component, then reconcile the capacity row."""
    indices = pd.Index(indices)
    rates = _capacity_buffer_rates(capacity, indices, config)
    desired = pd.Series(desired_total, index=indices, dtype=float)
    desired_core = desired / (1.0 + rates)
    current_core = capacity.loc[
        indices,
        ["planned_visit_hours", "planned_travel_hours", "required_customer_coverage_hours"],
    ].sum(axis=1)
    delta = (desired_core - current_core).clip(lower=0.0)
    capacity.loc[indices, "planned_visit_hours"] = (
        capacity.loc[indices, "planned_visit_hours"] + delta
    ).to_numpy(float)
    _recalculate_capacity_rows(capacity, indices, config, rates)


def inject_controlled_anomalies(
    clean_tables: dict[str, pd.DataFrame], config: dict[str, Any]
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Return an independent anomaly-injected layer plus separate ground truth."""
    rng = np.random.default_rng(int(config["project"]["seed"]) + 101)
    tables = {name: frame.copy(deep=True) for name, frame in clean_tables.items()}
    # Controlled injections can introduce fractional quantities even when the
    # observed source column was integral.  Promote the mutable measures once
    # up front so pandas does not silently depend on incompatible-dtype writes.
    for column in ("quantity", "gross_sales", "discount_pct", "discount_amount", "net_sales"):
        if column in tables["orders"]:
            tables["orders"][column] = pd.to_numeric(tables["orders"][column], errors="coerce").astype(float)
    truth: list[dict[str, Any]] = []
    injected_overload_record_ids: set[str] = set()
    incentives = tables["incentive_calculations"]
    total_rep_periods = len(incentives)
    unique_target_count = max(1, int(round(total_rep_periods * float(config["anomalies"]["rep_period_prevalence"]))))
    validation_start = pd.Timestamp(config["model"]["train_end"]) + pd.offsets.MonthBegin(1)
    eligible = incentives.loc[pd.to_datetime(incentives["period"]).ge(validation_start), ["rep_id", "period"]]
    if eligible.empty:
        eligible = incentives[["rep_id", "period"]]
    unique_target_count = min(unique_target_count, len(eligible))
    configured_correlated_share = float(
        config["anomalies"].get("correlated_case_share", 0.20)
    )
    if not 0.0 <= configured_correlated_share <= 1.0:
        raise ValueError("correlated_case_share must be between zero and one")
    validation_end = pd.Timestamp(config["model"]["validation_end"])
    eligible_period = pd.to_datetime(eligible["period"])
    validation_pool = eligible.loc[eligible_period.le(validation_end)]
    test_pool = eligible.loc[eligible_period.gt(validation_end)]
    if test_pool.empty:
        test_pool = eligible

    # Every named scenario is assigned to the final holdout so detection by
    # anomaly type and severity has real support. Keep at least one mixed
    # validation observation as well; a single extra case is permitted only
    # when rounding the configured prevalence would otherwise consume every
    # target needed by the named final-test scenarios.
    selected_count = unique_target_count
    requested_correlated = int(round(selected_count * configured_correlated_share))
    unavoidable_overlap = max(0, len(SCENARIOS) - selected_count)
    desired_correlated = min(
        selected_count, max(requested_correlated, unavoidable_overlap)
    )
    # Use fewer unique primary targets when necessary so the configured share
    # represents real overlapping scenarios, even when prevalence selects more
    # rep-periods than the 22 named scenarios. Remaining selected periods receive
    # a subtle recorded order-timing case during prevalence calibration below.
    unique_primary_count = max(
        1, min(selected_count, len(SCENARIOS) - desired_correlated)
    )
    if (
        not validation_pool.empty
        and selected_count - 1 < unique_primary_count
        and selected_count < len(eligible)
    ):
        selected_count += 1
        requested_correlated = int(
            round(selected_count * configured_correlated_share)
        )
        unavoidable_overlap = max(0, len(SCENARIOS) - selected_count)
        desired_correlated = min(
            selected_count, max(requested_correlated, unavoidable_overlap)
        )
        unique_primary_count = max(
            1, min(selected_count, len(SCENARIOS) - desired_correlated)
        )
    unique_primary_count = min(unique_primary_count, len(test_pool))
    primary_indices = rng.choice(
        test_pool.index.to_numpy(), size=unique_primary_count, replace=False
    )
    primary_frame = eligible.loc[primary_indices].sort_values(
        ["period", "rep_id"], kind="mergesort"
    )
    selected_indices = list(primary_indices)
    remaining_count = selected_count - len(selected_indices)
    if remaining_count > 0 and not validation_pool.empty:
        validation_candidates = validation_pool.index.difference(selected_indices)
        if len(validation_candidates):
            selected_indices.append(int(rng.choice(validation_candidates.to_numpy())))
            remaining_count -= 1
    if remaining_count > 0:
        remaining_candidates = eligible.index.difference(selected_indices)
        selected_indices.extend(
            rng.choice(
                remaining_candidates.to_numpy(),
                size=min(remaining_count, len(remaining_candidates)),
                replace=False,
            ).tolist()
        )
    selected = eligible.loc[selected_indices].sort_values(
        ["period", "rep_id"], kind="mergesort"
    ).reset_index(drop=True)
    scenario_selected = primary_frame.reset_index(drop=True)
    selected_key_index = pd.MultiIndex.from_frame(selected[["rep_id", "period"]])
    primary_targets = [
        scenario_selected.iloc[index] for index in range(len(scenario_selected))
    ]
    overlap_target_count = max(1, min(desired_correlated, len(primary_targets)))
    assignments = [
        primary_targets[index]
        if index < len(primary_targets)
        else primary_targets[(index - len(primary_targets)) % overlap_target_count]
        for index in range(len(SCENARIOS))
    ]
    severities = _severities(len(SCENARIOS), config["anomalies"]["severity_mix"], rng)

    # Select the territory-context case from the same leakage-safe feature view
    # used by the model. Prefer an already high-performing representative in a
    # high-potential territory, then move performance *toward* the potential-
    # explained expectation in the injection branch below.
    from .features import build_feature_store

    clean_feature_context, _, _ = build_feature_store(tables, config)
    selected_potential = scenario_selected.merge(
        clean_feature_context[
            [
                "rep_id",
                "period",
                "net_sales",
                "territory_potential",
                "territory_adjusted_sales_residual",
                "sales_peer_percentile",
            ]
        ],
        on=["rep_id", "period"],
        how="left",
        validate="one_to_one",
    )
    high_performance = selected_potential.loc[
        selected_potential["sales_peer_percentile"].ge(0.60)
    ]
    potential_pool = high_performance if not high_performance.empty else selected_potential
    sales_mutating_categories = {
        "order_timing",
        "returns",
        "discount",
        "customer",
        "product",
        "multi_signal",
    }
    other_sales_case_keys = {
        (str(target.rep_id), pd.Timestamp(target.period))
        for (anomaly_type, category, _), target in zip(SCENARIOS, assignments)
        if anomaly_type != "territory_potential_explained_performance"
        and category in sales_mutating_categories
    }
    unoccupied_potential = potential_pool.loc[
        [
            (str(rep_id), pd.Timestamp(period)) not in other_sales_case_keys
            for rep_id, period in potential_pool[["rep_id", "period"]].itertuples(
                index=False, name=None
            )
        ]
    ]
    if not unoccupied_potential.empty:
        potential_pool = unoccupied_potential
    highest_potential_target = potential_pool.sort_values(
        ["territory_potential", "sales_peer_percentile", "period", "rep_id"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).iloc[0]
    selected_incentive_context = scenario_selected.merge(
        incentives[["rep_id", "period", "attainment_pct"]],
        on=["rep_id", "period"],
        how="left",
        validate="one_to_one",
    )
    unsupported_override_target = selected_incentive_context.sort_values(
        ["attainment_pct", "period", "rep_id"], kind="mergesort"
    ).iloc[0]
    occupied_by_other_scenarios = {
        (str(assigned.rep_id), pd.Timestamp(assigned.period))
        for scenario, assigned in zip(SCENARIOS, assignments)
        if scenario[0] != "threshold_crossing_discount"
    }
    threshold_candidate_mask = [
        (str(rep_id), pd.Timestamp(period)) not in occupied_by_other_scenarios
        for rep_id, period in selected_incentive_context[
            ["rep_id", "period"]
        ].itertuples(index=False, name=None)
    ]
    threshold_candidates = selected_incentive_context.loc[
        threshold_candidate_mask
    ]
    if threshold_candidates.empty:
        threshold_candidates = selected_incentive_context
    threshold_candidates = threshold_candidates.loc[
        ~(
            threshold_candidates["rep_id"].astype(str).eq(
                str(unsupported_override_target.rep_id)
            )
            & pd.to_datetime(threshold_candidates["period"]).eq(
                pd.Timestamp(unsupported_override_target.period)
            )
        )
    ]
    threshold_candidates = threshold_candidates.loc[
        ~(
            threshold_candidates["rep_id"].astype(str).eq(
                str(highest_potential_target.rep_id)
            )
            & pd.to_datetime(threshold_candidates["period"]).eq(
                pd.Timestamp(highest_potential_target.period)
            )
        )
    ]
    if threshold_candidates.empty:
        threshold_candidates = selected_incentive_context
    below_threshold = threshold_candidates.loc[
        threshold_candidates["attainment_pct"].lt(100.0)
    ].copy()
    threshold_pool = below_threshold if not below_threshold.empty else threshold_candidates
    threshold_discount_target = (
        threshold_pool.assign(
            _threshold_distance=(threshold_pool["attainment_pct"] - 100.0).abs()
        )
        .sort_values(["_threshold_distance", "period", "rep_id"], kind="mergesort")
        .iloc[0]
    )

    incentive_actions: list[tuple[int, str, str, str, str, pd.Series]] = []
    for number, ((anomaly_type, category, description), severity, target) in enumerate(
        zip(SCENARIOS, severities, assignments), start=1
    ):
        if anomaly_type == "territory_potential_explained_performance":
            target = highest_potential_target
        elif anomaly_type == "unsupported_manual_override":
            target = unsupported_override_target
        elif anomaly_type == "threshold_crossing_discount":
            target = threshold_discount_target
        injection_id = f"INJ_{number:04d}"
        rep_id = str(target.rep_id)
        period = pd.Timestamp(target.period)
        factor = _factor(severity)
        if category == "incentive":
            incentive_actions.append((number, anomaly_type, category, description, severity, target))
            continue

        order_idx = _order_rows(tables, rep_id, period)
        if category in {"order_timing", "customer", "product", "territory", "multi_signal", "discount", "returns"} and len(order_idx) == 0:
            continue
        if anomaly_type == "end_of_period_sales_spike":
            chosen = rng.choice(order_idx, size=max(1, min(len(order_idx), int(np.ceil(len(order_idx) * 0.18)))), replace=False)
            original = tables["orders"].loc[chosen, ["net_sales", "gross_sales", "order_date"]].to_dict("records")
            tables["orders"].loc[chosen, ["net_sales", "gross_sales", "quantity"]] *= factor
            _set_order_period_end(tables, chosen, period)
            tables["orders"].loc[chosen, "end_of_period_flag"] = True
            tables["orders"].loc[chosen, "data_lineage"] = "synthetic_injected"
            _record(truth, injection_id, anomaly_type, category, description, severity, rep_id, period, "orders", tables["orders"].loc[chosen, "order_line_id"].tolist(), original, {"sales_factor": factor, "order_day": "period_end"})
        elif anomaly_type in {"post_payout_returns", "unusual_return_rate"}:
            return_selection_pool = order_idx
            if anomaly_type == "unusual_return_rate":
                period_end = period + pd.offsets.MonthEnd(1)
                same_month_eligible = order_idx[
                    pd.to_datetime(tables["orders"].loc[order_idx, "order_date"])
                    .lt(period_end)
                    .to_numpy()
                ]
                if len(same_month_eligible):
                    return_selection_pool = same_month_eligible
            chosen = rng.choice(return_selection_pool, size=max(1, min(len(return_selection_pool), int(np.ceil(len(return_selection_pool) * 0.15)))), replace=False)
            original_orders = tables["orders"].loc[
                chosen, ["order_line_id", "gross_sales", "net_sales", "quantity", "order_date"]
            ].to_dict("records")
            if anomaly_type == "post_payout_returns":
                tables["orders"].loc[
                    chosen, ["gross_sales", "net_sales", "quantity"]
                ] *= factor * 1.08
                _set_order_period_end(tables, chosen, period)
                tables["orders"].loc[
                    chosen, ["end_of_period_flag", "data_lineage"]
                ] = [True, "synthetic_injected"]
            additions = []
            for offset, row in enumerate(tables["orders"].loc[chosen].itertuples(index=False), start=1):
                if anomaly_type == "unusual_return_rate":
                    return_date = min(
                        period + pd.offsets.MonthEnd(1),
                        max(
                            pd.Timestamp(row.order_date) + pd.Timedelta(days=1),
                            period + pd.Timedelta(days=20 + offset % 7),
                        ),
                    )
                else:
                    return_date = period + pd.offsets.MonthEnd(1) + pd.Timedelta(days=20 + offset % 10)
                additions.append(
                    {
                        "return_id": f"RET_INJ_{number:04d}_{offset:04d}", "order_id": row.order_id,
                        "order_line_id": row.order_line_id, "rep_id": rep_id, "customer_id": row.customer_id,
                        "product_id": row.product_id, "original_order_date": row.order_date,
                        "return_date": return_date, "return_quantity": abs(float(row.quantity)) * min(factor / 2, 1),
                        "return_amount": abs(float(row.net_sales)) * min(factor / 2, 1),
                        "cancellation_flag": anomaly_type == "unusual_return_rate" and offset % 3 == 0,
                        "return_reason": "controlled injected post-period reversal",
                        "payout_period": period,
                        "after_incentive_payout_flag": anomaly_type == "post_payout_returns",
                        "days_after_order": int((return_date - pd.Timestamp(row.order_date)).days),
                        "currency_code": config["project"].get("currency_code", "UNK"), "data_lineage": "synthetic_injected",
                    }
                )
            tables["returns_cancellations"] = pd.concat([tables["returns_cancellations"], pd.DataFrame(additions)], ignore_index=True)
            affected_dataset = (
                "orders|returns_cancellations"
                if anomaly_type == "post_payout_returns"
                else "returns_cancellations"
            )
            affected_ids = [row["return_id"] for row in additions]
            if anomaly_type == "post_payout_returns":
                affected_ids = (
                    tables["orders"].loc[chosen, "order_line_id"].astype(str).tolist()
                    + affected_ids
                )
            _record(truth, injection_id, anomaly_type, category, description, severity, rep_id, period, affected_dataset, affected_ids, {"orders": original_orders, "return_count": 0}, {"return_count": len(additions), "after_payout": anomaly_type == "post_payout_returns", "period_end_sales_factor": factor * 1.08 if anomaly_type == "post_payout_returns" else 1.0})
        elif anomaly_type == "threshold_crossing_discount":
            positive_order_idx = order_idx[
                tables["orders"].loc[order_idx, "gross_sales"].gt(0).to_numpy()
            ]
            selection_pool = positive_order_idx if len(positive_order_idx) else order_idx
            chosen = rng.choice(selection_pool, size=max(1, min(len(selection_pool), int(np.ceil(len(selection_pool) * 0.22)))), replace=False)
            line_ids = tables["orders"].loc[chosen, "order_line_id"]
            disc_idx = tables["discount_detail"].index[tables["discount_detail"]["order_line_id"].isin(line_ids)]
            original = tables["discount_detail"].loc[disc_idx, "discount_pct"].tolist()
            new_pct = (tables["discount_detail"].loc[disc_idx, "discount_pct"] + (factor - 1) * 0.13).clip(0, 0.42)
            tables["discount_detail"].loc[disc_idx, "discount_pct"] = new_pct
            tables["discount_detail"].loc[disc_idx, ["approved_flag", "exception_flag"]] = [False, True]
            tables["discount_detail"].loc[disc_idx, "data_lineage"] = "synthetic_injected"
            pct_map = tables["discount_detail"].set_index("order_line_id")["discount_pct"]
            tables["orders"].loc[chosen, "discount_pct"] = line_ids.map(pct_map).to_numpy()
            target_sales = float(
                tables["rep_targets_quotas"].loc[
                    tables["rep_targets_quotas"]["rep_id"].eq(rep_id)
                    & pd.to_datetime(tables["rep_targets_quotas"]["period"]).eq(period),
                    "target_sales",
                ].iloc[0]
            )
            product_policy = tables["product_master"].set_index("product_id")
            eligible_weights = tables["orders"].loc[order_idx, "product_id"].map(
                product_policy["incentive_weight"]
            ).fillna(0.0)
            eligible_flags = tables["orders"].loc[order_idx, "product_id"].map(
                product_policy["incentive_eligible_flag"]
            ).fillna(False)
            eligible_weights = eligible_weights.where(eligible_flags, 0.0)
            period_eligible_net = (
                tables["orders"].loc[order_idx, "net_sales"].to_numpy(float)
                * eligible_weights.to_numpy(float)
            )
            chosen_positions = pd.Index(order_idx).get_indexer(pd.Index(chosen))
            other_eligible_net = float(
                period_eligible_net.sum() - period_eligible_net[chosen_positions].sum()
            )
            chosen_weights = eligible_weights.iloc[chosen_positions].to_numpy(float)
            selected_eligible_net_before_scale = float(
                (
                    tables["orders"].loc[chosen, "gross_sales"].to_numpy(float)
                    * (1.0 - tables["orders"].loc[chosen, "discount_pct"].to_numpy(float))
                    * chosen_weights
                ).sum()
            )
            required_selected_net = max(
                target_sales * 1.015 - other_eligible_net, 0.0
            )
            volume_scale = max(
                1.0,
                required_selected_net
                / max(selected_eligible_net_before_scale, 1e-12),
            )
            tables["orders"].loc[chosen, ["gross_sales", "quantity"]] *= volume_scale
            tables["orders"].loc[chosen, "discount_amount"] = tables["orders"].loc[chosen, "gross_sales"].abs() * tables["orders"].loc[chosen, "discount_pct"]
            tables["orders"].loc[chosen, "net_sales"] = tables["orders"].loc[chosen, "gross_sales"] - np.sign(tables["orders"].loc[chosen, "gross_sales"]) * tables["orders"].loc[chosen, "discount_amount"]
            tables["orders"].loc[chosen, "data_lineage"] = "synthetic_injected"
            _record(truth, injection_id, anomaly_type, category, description, severity, rep_id, period, "orders|discount_detail", tables["orders"].loc[chosen, "order_line_id"].astype(str).tolist() + tables["discount_detail"].loc[disc_idx, "discount_id"].astype(str).tolist(), {"discount_pct": original, "attainment_pct": float(target.attainment_pct)}, {"discount_pct": new_pct.tolist(), "volume_scale": volume_scale, "intended_post_injection_attainment_pct": 101.5})
        elif anomaly_type in {"low_volume_customer_spike", "customer_sales_concentration"}:
            sub = tables["orders"].loc[order_idx]
            if anomaly_type == "low_volume_customer_spike":
                historical = (
                    tables["orders"].loc[
                        tables["orders"]["rep_id"].eq(rep_id)
                        & pd.to_datetime(tables["orders"]["period"]).lt(period)
                    ]
                    .groupby("customer_id", observed=True)["net_sales"]
                    .sum()
                    .abs()
                )
                customer_sales = pd.Series(
                    {
                        customer_id: float(historical.get(customer_id, 0.0))
                        for customer_id in sub["customer_id"].unique()
                    }
                ).sort_values(kind="mergesort")
                customer = customer_sales.index[0]
            else:
                customer_sales = sub.groupby("customer_id")["net_sales"].sum().abs().sort_values()
                customer = customer_sales.index[-1]
            chosen = sub.index[sub["customer_id"].eq(customer)]
            original = tables["orders"].loc[chosen, "net_sales"].tolist()
            tables["orders"].loc[chosen, ["net_sales", "gross_sales", "quantity"]] *= factor * (1.3 if anomaly_type.endswith("concentration") else 1.0)
            tables["orders"].loc[chosen, "data_lineage"] = "synthetic_injected"
            _record(truth, injection_id, anomaly_type, category, description, severity, rep_id, period, "orders", tables["orders"].loc[chosen, "order_line_id"].tolist(), original, tables["orders"].loc[chosen, "net_sales"].tolist(), entity_type="customer_period", entity_id=f"{customer}|{period.date()}")
        elif anomaly_type == "incentivized_product_mix_shift":
            weights = tables["product_master"].set_index("product_id")["incentive_weight"]
            sub = tables["orders"].loc[order_idx]
            product = sub.assign(weight=sub["product_id"].map(weights)).sort_values(["weight", "net_sales"], ascending=[False, True]).iloc[0]["product_id"]
            chosen = sub.index[sub["product_id"].eq(product)]
            original = tables["orders"].loc[chosen, "net_sales"].tolist()
            tables["orders"].loc[chosen, ["net_sales", "gross_sales", "quantity"]] *= factor
            tables["orders"].loc[chosen, "data_lineage"] = "synthetic_injected"
            _record(truth, injection_id, anomaly_type, category, description, severity, rep_id, period, "orders", tables["orders"].loc[chosen, "order_line_id"].tolist(), original, tables["orders"].loc[chosen, "net_sales"].tolist(), entity_type="product_period", entity_id=f"{product}|{period.date()}")
        elif anomaly_type in {"extremely_short_visits", "overlap_impossible_travel", "sales_without_supporting_activity", "high_activity_low_engagement"}:
            visits = tables["field_visits"]
            visit_idx = visits.index[visits["rep_id"].eq(rep_id) & pd.to_datetime(visits["period"]).eq(period)]
            if len(visit_idx):
                chosen = rng.choice(visit_idx, size=max(1, min(len(visit_idx), int(np.ceil(len(visit_idx) * 0.45)))), replace=False)
                original = visits.loc[
                    chosen,
                    ["visit_duration_minutes", "visit_completed_flag", "visit_outcome"],
                ].to_dict("records")
                record_ids = visits.loc[chosen, "visit_id"].astype(str).tolist()
                affected_dataset = "field_visits"
                injected_payload: Any | None = None
                if anomaly_type == "extremely_short_visits":
                    visits.loc[chosen, "visit_duration_minutes"] = rng.integers(3, 9, len(chosen))
                    visits.loc[chosen, "actual_end_time"] = pd.to_datetime(visits.loc[chosen, "actual_start_time"]) + pd.to_timedelta(visits.loc[chosen, "visit_duration_minutes"], unit="m")
                elif anomaly_type == "overlap_impossible_travel":
                    start = pd.Timestamp(period) + pd.Timedelta(days=10, hours=10)
                    visits.loc[chosen, "actual_start_time"] = start
                    visits.loc[chosen, "actual_end_time"] = start + pd.Timedelta(minutes=35)
                    visits.loc[chosen, "visit_duration_minutes"] = 35.0
                    visits.loc[chosen, ["overlapping_visit_flag", "impossible_travel_flag"]] = True
                    visits.loc[chosen, "estimated_travel_km"] *= factor * 4
                elif anomaly_type == "sales_without_supporting_activity":
                    visits.loc[chosen, "visit_completed_flag"] = False
                else:
                    # The activity anomaly is represented by new claimed visits;
                    # the clean source visits used as templates remain untouched.
                    # This keeps every changed row both injected-lineage and
                    # traceable through the ground-truth affected-record list.
                    copies = visits.loc[chosen].copy()
                    copies["visit_id"] = [
                        f"VIS_ACTIVITY_{number:04d}_{offset:04d}"
                        for offset in range(1, len(copies) + 1)
                    ]
                    copies["visit_outcome"] = "no immediate action"
                    copies["visit_completed_flag"] = True
                    copy_offsets = np.arange(len(copies))
                    copy_dates = period + pd.to_timedelta(copy_offsets % 21, unit="D")
                    copy_hours = np.where(copy_offsets < 21, 6, 19)
                    copy_starts = pd.to_datetime(copy_dates) + pd.to_timedelta(
                        copy_hours, unit="h"
                    )
                    copy_durations = pd.to_numeric(
                        copies["visit_duration_minutes"], errors="coerce"
                    ).clip(18, 90)
                    copies["visit_date"] = pd.to_datetime(copy_dates)
                    copies["scheduled_start_time"] = copy_starts
                    copies["actual_start_time"] = copy_starts
                    copies["actual_end_time"] = copy_starts + pd.to_timedelta(
                        copy_durations, unit="m"
                    )
                    copies["overlapping_visit_flag"] = False
                    copies["impossible_travel_flag"] = False
                    copies["data_lineage"] = "synthetic_injected"
                    tables["field_visits"] = pd.concat(
                        [visits, copies], ignore_index=True
                    )
                    record_ids = copies["visit_id"].astype(str).tolist()
                    crm_idx = tables["crm_interactions"].index[tables["crm_interactions"]["rep_id"].eq(rep_id) & pd.to_datetime(tables["crm_interactions"]["period"]).eq(period)]
                    crm_columns = [
                        "interaction_id",
                        "sentiment_or_interest_score",
                        "interaction_outcome",
                        "data_lineage",
                    ]
                    original = {
                        "template_visit_context": original,
                        "crm_interactions": tables["crm_interactions"].loc[
                            crm_idx, crm_columns
                        ].to_dict("records"),
                    }
                    tables["crm_interactions"].loc[crm_idx, "sentiment_or_interest_score"] = 0.05
                    tables["crm_interactions"].loc[crm_idx, "interaction_outcome"] = "no response"
                    tables["crm_interactions"].loc[crm_idx, "data_lineage"] = "synthetic_injected"
                    affected_dataset = "field_visits|crm_interactions"
                    record_ids.extend(
                        tables["crm_interactions"].loc[
                            crm_idx, "interaction_id"
                        ].astype(str).tolist()
                    )
                    injected_payload = {
                        "new_claimed_visits": copies[
                            [
                                "visit_id",
                                "visit_outcome",
                                "visit_completed_flag",
                                "data_lineage",
                            ]
                        ].to_dict("records"),
                        "crm_interactions": tables["crm_interactions"].loc[
                            crm_idx, crm_columns
                        ].to_dict("records"),
                        "activity_multiplier": 2.0,
                        "low_engagement": True,
                    }
                if anomaly_type != "high_activity_low_engagement":
                    visits.loc[chosen, "data_lineage"] = "synthetic_injected"
                if injected_payload is None:
                    injected_payload = {
                        "affected_record_count": len(record_ids),
                        "activity_multiplier": 1.0,
                        "low_engagement": False,
                    }
                _record(truth, injection_id, anomaly_type, category, description, severity, rep_id, period, affected_dataset, record_ids, original, injected_payload)
        elif anomaly_type in {"inflated_travel_distance", "duplicate_expense_claim"}:
            expenses = tables["travel_expenses"]
            expense_idx = expenses.index[expenses["rep_id"].eq(rep_id) & pd.to_datetime(expenses["period"]).eq(period)]
            if len(expense_idx):
                chosen = rng.choice(expense_idx, size=max(1, min(len(expense_idx), int(np.ceil(len(expense_idx) * 0.30)))), replace=False)
                original = expenses.loc[chosen, ["claimed_distance_km", "claimed_amount"]].to_dict("records")
                if anomaly_type == "inflated_travel_distance":
                    expenses.loc[chosen, "claimed_distance_km"] *= factor * 1.6
                    expenses.loc[chosen, "claimed_amount"] *= factor * 1.6
                    expenses.loc[chosen, "deviation_pct"] = 100.0 * (
                        expenses.loc[chosen, "claimed_amount"]
                        - expenses.loc[chosen, "expected_amount"]
                    ) / expenses.loc[chosen, "expected_amount"].clip(lower=1e-12)
                    record_ids = expenses.loc[chosen, "expense_id"].tolist()
                else:
                    copies = expenses.loc[chosen].copy()
                    copies["expense_id"] = [f"EXP_DUP_{number:04d}_{i:04d}" for i in range(len(copies))]
                    expenses = pd.concat([expenses, copies], ignore_index=True)
                    tables["travel_expenses"] = expenses
                    record_ids = copies["expense_id"].tolist()
                expenses.loc[expenses["expense_id"].isin(record_ids), "data_lineage"] = "synthetic_injected"
                _record(truth, injection_id, anomaly_type, category, description, severity, rep_id, period, "travel_expenses", record_ids, original, {"factor_or_duplicate": factor})
        elif anomaly_type == "late_repeated_target_revision":
            targets = tables["rep_targets_quotas"]
            target_idx = targets.index[targets["rep_id"].eq(rep_id) & pd.to_datetime(targets["period"]).eq(period)]
            original = targets.loc[target_idx, ["target_sales", "target_effective_date", "target_version"]].to_dict("records")
            targets.loc[target_idx, "target_revision_flag"] = True
            targets.loc[target_idx, "target_version"] = targets.loc[target_idx, "target_version"] + 2
            targets.loc[target_idx, "target_effective_date"] = period + pd.offsets.MonthEnd(1) - pd.Timedelta(days=2)
            targets.loc[target_idx, "target_sales"] *= 0.88
            targets.loc[target_idx, "data_lineage"] = "synthetic_injected"
            _record(truth, injection_id, anomaly_type, category, description, severity, rep_id, period, "rep_targets_quotas", [f"{rep_id}|{period.date()}"], original, targets.loc[target_idx, ["target_sales", "target_effective_date", "target_version"]].to_dict("records"))
        elif anomaly_type in {"territory_workload_exceeds_capacity", "persistent_priority_undercoverage"}:
            capacity = tables["capacity_calendar"]
            cap_idx = capacity.index[capacity["rep_id"].eq(rep_id) & pd.to_datetime(capacity["period"]).eq(period)]
            if len(cap_idx):
                if anomaly_type == "territory_workload_exceeds_capacity":
                    affected_idx = pd.Index(cap_idx)
                    original = capacity.loc[affected_idx, ["required_total_hours", "available_field_hours", "priority_customer_coverage_gap"]].to_dict("records")
                    risk_thresholds = config.get("capacity", {}).get("risk_thresholds", {})
                    high_ratio = float(risk_thresholds.get("high", 1.00))
                    critical_ratio = float(risk_thresholds.get("critical", 1.20))
                    minimum_ratio = {
                        "low": high_ratio * 1.02,
                        "medium": high_ratio * 1.18,
                        "high": max(high_ratio * 1.35, critical_ratio * 1.02),
                    }[severity]
                    desired_total = pd.Series(
                        np.maximum(
                            capacity.loc[affected_idx, "required_total_hours"].to_numpy(float) * factor,
                            capacity.loc[affected_idx, "available_field_hours"].to_numpy(float) * minimum_ratio,
                        ),
                        index=affected_idx,
                    )
                    _raise_capacity_required_total(
                        capacity, affected_idx, desired_total, config
                    )
                    affected_dataset = "capacity_calendar"
                    affected_ids = capacity.loc[
                        affected_idx, "capacity_record_id"
                    ].astype(str).tolist()
                    injected_overload_record_ids.update(affected_ids)
                else:
                    rep_rows = capacity.loc[capacity["rep_id"].eq(rep_id)].sort_values(
                        "period", kind="mergesort"
                    )
                    positions = np.flatnonzero(
                        pd.to_datetime(rep_rows["period"]).eq(period).to_numpy()
                    )
                    position = int(positions[0])
                    start = min(position, max(len(rep_rows) - 3, 0))
                    if start > position:
                        start = max(0, position - 2)
                    affected_idx = pd.Index(rep_rows.iloc[start : start + 3].index)
                    original = capacity.loc[affected_idx, ["required_total_hours", "available_field_hours", "priority_customer_coverage_gap"]].to_dict("records")
                    rates = _capacity_buffer_rates(capacity, affected_idx, config)
                    added_gap = float(np.ceil(3.0 * factor))
                    hours_per_gap = float(
                        config.get("capacity", {}).get("visit_hours_per_required_call", 0.70)
                    ) + float(
                        config.get("capacity", {}).get("default_travel_hours_per_visit", 0.50)
                    )
                    capacity.loc[affected_idx, "priority_customer_coverage_gap"] += added_gap
                    if "customer_coverage_gap" in capacity:
                        capacity.loc[affected_idx, "customer_coverage_gap"] += added_gap
                    capacity.loc[affected_idx, "required_customer_coverage_hours"] += (
                        added_gap * hours_per_gap
                    )
                    if "required_priority_customer_coverage_hours" in capacity:
                        capacity.loc[
                            affected_idx, "required_priority_customer_coverage_hours"
                        ] += added_gap * hours_per_gap
                    _recalculate_capacity_rows(capacity, affected_idx, config, rates)
                    affected_dataset = "capacity_calendar"
                    affected_ids = capacity.loc[
                        affected_idx, "capacity_record_id"
                    ].astype(str).tolist()
                    coverage = tables.get("capacity_customer_drilldown")
                    if coverage is not None and not coverage.empty:
                        coverage_ids: list[str] = []
                        for cap_row in capacity.loc[affected_idx].itertuples(index=False):
                            priority_mask = (
                                coverage["rep_id"].astype(str).eq(str(cap_row.rep_id))
                                & pd.to_datetime(coverage["period"]).eq(pd.Timestamp(cap_row.period))
                                & coverage.get(
                                    "priority_customer_flag",
                                    coverage["customer_priority"].astype(str).str.casefold().eq("high"),
                                ).astype(bool)
                            )
                            candidates = coverage.index[priority_mask]
                            if len(candidates) == 0:
                                continue
                            drill_idx = candidates[:1]
                            for column in (
                                "required_visit_count",
                                "priority_required_visit_count",
                                "planned_coverage_gap_count",
                                "customer_coverage_gap",
                                "priority_customer_coverage_gap",
                            ):
                                if column in coverage:
                                    coverage.loc[drill_idx, column] += added_gap
                            for column in (
                                "required_customer_coverage_hours",
                                "required_priority_customer_coverage_hours",
                            ):
                                if column in coverage:
                                    coverage.loc[drill_idx, column] += added_gap * hours_per_gap
                            coverage.loc[drill_idx, "coverage_status"] = "Coverage gap"
                            coverage.loc[drill_idx, "coverage_met_flag"] = False
                            coverage.loc[drill_idx, "data_lineage"] = "synthetic_injected"
                            coverage_ids.extend(
                                f"{row.rep_id}|{pd.Timestamp(row.period).date()}|{row.customer_id}"
                                for row in coverage.loc[drill_idx].itertuples(index=False)
                            )
                        if coverage_ids:
                            affected_dataset += "|capacity_customer_drilldown"
                            affected_ids.extend(coverage_ids)
                    description = (
                        f"{description}; sustained across {len(affected_idx)} consecutive periods"
                    )
                capacity.loc[affected_idx, "data_lineage"] = "synthetic_injected"
                _record(
                    truth,
                    injection_id,
                    anomaly_type,
                    category,
                    description,
                    severity,
                    rep_id,
                    period,
                    affected_dataset,
                    affected_ids,
                    original,
                    capacity.loc[
                        affected_idx,
                        ["required_total_hours", "available_field_hours", "priority_customer_coverage_gap"],
                    ].to_dict("records"),
                )
        elif anomaly_type == "territory_potential_explained_performance":
            context = clean_feature_context.loc[
                clean_feature_context["rep_id"].astype(str).eq(rep_id)
                & pd.to_datetime(clean_feature_context["period"]).eq(period)
            ].iloc[0]
            period_context = clean_feature_context.loc[
                pd.to_datetime(clean_feature_context["period"]).eq(period)
            ]
            current_sales = float(context["net_sales"])
            other_sales = float(period_context["net_sales"].sum() - current_sales)
            potential_share = float(context["territory_potential"]) / max(
                float(period_context["territory_potential"].sum()), 1e-12
            )
            original_residual = float(context["territory_adjusted_sales_residual"])
            retained_residual_share = {
                "low": 0.45,
                "medium": 0.25,
                "high": 0.10,
            }[severity]
            desired_residual = original_residual * retained_residual_share
            desired_sales = (
                potential_share * other_sales + desired_residual
            ) / max(1.0 - potential_share, 1e-12)
            desired_sales = max(desired_sales, max(current_sales, 1.0) * 0.10)
            sales_scale = desired_sales / max(current_sales, 1e-12)
            chosen = order_idx.to_numpy()
            original = {
                "net_sales": current_sales,
                "territory_potential": float(context["territory_potential"]),
                "sales_peer_percentile": float(context["sales_peer_percentile"]),
                "territory_adjusted_sales_residual": original_residual,
            }
            tables["orders"].loc[chosen, ["gross_sales", "quantity"]] *= sales_scale
            tables["orders"].loc[chosen, "discount_amount"] = (
                tables["orders"].loc[chosen, "gross_sales"].abs()
                * tables["orders"].loc[chosen, "discount_pct"]
            )
            tables["orders"].loc[chosen, "net_sales"] = (
                tables["orders"].loc[chosen, "gross_sales"]
                - np.sign(tables["orders"].loc[chosen, "gross_sales"])
                * tables["orders"].loc[chosen, "discount_amount"]
            )
            tables["orders"].loc[chosen, "data_lineage"] = "synthetic_injected"
            _record(
                truth,
                injection_id,
                anomaly_type,
                category,
                (
                    f"{description}; sales were moved toward the expectation implied "
                    f"by leakage-safe territory potential={float(context['territory_potential']):.2f}"
                ),
                severity,
                rep_id,
                period,
                "orders",
                tables["orders"].loc[chosen, "order_line_id"].astype(str).tolist(),
                original,
                {
                    "net_sales": float(tables["orders"].loc[chosen, "net_sales"].sum()),
                    "sales_scale": sales_scale,
                    "target_residual": desired_residual,
                    "retained_residual_share": retained_residual_share,
                },
            )
        elif anomaly_type == "multi_signal_sales_discount_returns":
            chosen = rng.choice(order_idx, size=max(1, min(len(order_idx), int(np.ceil(len(order_idx) * 0.20)))), replace=False)
            line_ids = tables["orders"].loc[chosen, "order_line_id"]
            discount_idx = tables["discount_detail"].index[
                tables["discount_detail"]["order_line_id"].isin(line_ids)
            ]
            original = {
                "orders": tables["orders"].loc[chosen, ["net_sales", "discount_pct"]].to_dict("records"),
                "discount_detail": tables["discount_detail"].loc[
                    discount_idx, ["discount_id", "discount_pct", "discount_amount"]
                ].to_dict("records"),
            }
            tables["orders"].loc[chosen, ["gross_sales", "quantity"]] *= factor
            tables["orders"].loc[chosen, "discount_pct"] = (tables["orders"].loc[chosen, "discount_pct"] + 0.12).clip(upper=0.45)
            tables["orders"].loc[chosen, "discount_amount"] = tables["orders"].loc[chosen, "gross_sales"].abs() * tables["orders"].loc[chosen, "discount_pct"]
            tables["orders"].loc[chosen, "net_sales"] = tables["orders"].loc[chosen, "gross_sales"] - tables["orders"].loc[chosen, "discount_amount"]
            _set_order_period_end(tables, chosen, period)
            tables["orders"].loc[chosen, ["end_of_period_flag", "data_lineage"]] = [True, "synthetic_injected"]
            order_discount = tables["orders"].set_index("order_line_id")
            mapped_pct = tables["discount_detail"].loc[discount_idx, "order_line_id"].map(
                order_discount["discount_pct"]
            )
            mapped_amount = tables["discount_detail"].loc[discount_idx, "order_line_id"].map(
                order_discount["discount_amount"]
            )
            tables["discount_detail"].loc[discount_idx, "discount_pct"] = mapped_pct.to_numpy(float)
            tables["discount_detail"].loc[discount_idx, "discount_amount"] = mapped_amount.to_numpy(float)
            tables["discount_detail"].loc[discount_idx, "approved_flag"] = False
            tables["discount_detail"].loc[discount_idx, "exception_flag"] = True
            tables["discount_detail"].loc[discount_idx, "data_lineage"] = "synthetic_injected"
            additions = []
            for offset, row in enumerate(tables["orders"].loc[chosen].itertuples(index=False), start=1):
                ret_date = period + pd.offsets.MonthEnd(1) + pd.Timedelta(days=24 + offset % 5)
                additions.append({"return_id": f"RET_MULTI_{number:04d}_{offset:04d}", "order_id": row.order_id, "order_line_id": row.order_line_id, "rep_id": rep_id, "customer_id": row.customer_id, "product_id": row.product_id, "original_order_date": row.order_date, "return_date": ret_date, "return_quantity": abs(float(row.quantity)) * 0.65, "return_amount": abs(float(row.net_sales)) * 0.65, "cancellation_flag": False, "return_reason": "controlled multi-signal reversal", "payout_period": period, "after_incentive_payout_flag": True, "days_after_order": int((ret_date - pd.Timestamp(row.order_date)).days), "currency_code": config["project"].get("currency_code", "UNK"), "data_lineage": "synthetic_injected"})
            tables["returns_cancellations"] = pd.concat([tables["returns_cancellations"], pd.DataFrame(additions)], ignore_index=True)
            _record(
                truth,
                injection_id,
                anomaly_type,
                category,
                description,
                severity,
                rep_id,
                period,
                "orders|discount_detail|returns_cancellations",
                tables["orders"].loc[chosen, "order_line_id"].tolist()
                + tables["discount_detail"].loc[discount_idx, "discount_id"].tolist()
                + [row["return_id"] for row in additions],
                original,
                {"sales_factor": factor, "discount_add": 0.12, "return_count": len(additions)},
            )

    # Hold the affected order-line prevalence near the configured 1–2% without
    # introducing additional anomalous rep-periods. Extra cases are subtle timing
    # and value perturbations distributed across the already selected periods.
    desired_order_rows = int(round(len(tables["orders"]) * float(config["anomalies"]["order_level_prevalence"])))
    currently_injected = tables["orders"]["data_lineage"].eq("synthetic_injected")
    needed_order_rows = max(0, desired_order_rows - int(currently_injected.sum()))
    if needed_order_rows:
        order_keys = pd.MultiIndex.from_frame(tables["orders"][["rep_id", "period"]])
        recorded_keys = {
            (str(record["rep_id"]), pd.Timestamp(record["period"]))
            for record in truth
            if record["anomaly_category"] != "capacity"
        }
        order_related_keys = {
            (str(record["rep_id"]), pd.Timestamp(record["period"]))
            for record in truth
            if record["anomaly_category"] in ORDER_RELATED_CATEGORIES
        }
        # If configured prevalence requests more rep-period cases than the 22
        # named scenarios, the otherwise-unassigned periods become subtle order
        # timing cases.  Other incentive/activity/expense cases are not given a
        # generic order signal merely to hit the line-level budget.
        unassigned_keys = {
            (str(rep_id), pd.Timestamp(period))
            for rep_id, period in selected[["rep_id", "period"]].itertuples(index=False, name=None)
            if (str(rep_id), pd.Timestamp(period)) not in recorded_keys
        }
        calibration_keys = sorted(
            unassigned_keys, key=lambda value: (value[1], value[0])
        )
        calibration_severity = dict(
            zip(
                calibration_keys,
                _severities(
                    len(calibration_keys),
                    config["anomalies"]["severity_mix"],
                    rng,
                ),
            )
        )
        allowed_order_keys = order_related_keys | unassigned_keys
        allowed_key_index = pd.MultiIndex.from_tuples(
            sorted(allowed_order_keys, key=lambda value: (value[1], value[0])),
            names=["rep_id", "period"],
        )
        calibration_protected_keys = {
            (str(record["rep_id"]), pd.Timestamp(record["period"]))
            for record in truth
            if record["anomaly_type"] == "threshold_crossing_discount"
        }
        protected_key_index = pd.MultiIndex.from_tuples(
            sorted(calibration_protected_keys, key=lambda value: (value[1], value[0])),
            names=["rep_id", "period"],
        )
        candidates = tables["orders"].index[
            order_keys.isin(allowed_key_index) & ~currently_injected
            & ~order_keys.isin(protected_key_index)
        ].to_numpy()
        mandatory: list[int] = []
        for rep_id, period in selected[["rep_id", "period"]].itertuples(index=False, name=None):
            if (str(rep_id), pd.Timestamp(period)) in recorded_keys:
                continue
            key_mask = (
                tables["orders"].loc[candidates, "rep_id"].astype(str).eq(str(rep_id))
                & pd.to_datetime(tables["orders"].loc[candidates, "period"]).eq(pd.Timestamp(period))
            )
            eligible_indices = candidates[key_mask.to_numpy()]
            if len(eligible_indices):
                mandatory.append(int(rng.choice(eligible_indices)))
        mandatory = list(dict.fromkeys(mandatory))
        remaining_candidates = np.asarray(
            [index for index in candidates if int(index) not in set(mandatory)], dtype=int
        )
        remaining_needed = max(0, min(needed_order_rows - len(mandatory), len(remaining_candidates)))
        additional = (
            rng.choice(remaining_candidates, size=remaining_needed, replace=False)
            if remaining_needed
            else np.asarray([], dtype=int)
        )
        chosen = np.asarray([*mandatory, *additional.tolist()], dtype=int)
        for (rep_id, period), group in tables["orders"].loc[chosen].groupby(
            ["rep_id", "period"], observed=True
        ):
            key = (str(rep_id), pd.Timestamp(period))
            severity = calibration_severity.get(key, "low")
            value_factor = (
                {"low": 1.08, "medium": 1.12, "high": 1.18}[severity]
                if key in unassigned_keys
                else 1.08
            )
            period_indices = group.index
            tables["orders"].loc[
                period_indices, ["gross_sales", "net_sales", "quantity"]
            ] *= value_factor
            _set_order_period_end(tables, period_indices, pd.Timestamp(period))
            tables["orders"].loc[period_indices, "end_of_period_flag"] = True
            tables["orders"].loc[
                period_indices, "data_lineage"
            ] = "synthetic_injected"
            _extend_truth_with_order_calibration(
                truth,
                str(rep_id),
                pd.Timestamp(period),
                group["order_line_id"].astype(str).tolist(),
                severity=severity,
                value_factor=value_factor,
            )

    _reconcile_order_and_return_chronology(tables)
    _reconcile_order_discount_arithmetic(tables)

    # Recalculate reported incentives from the injected commercial facts, then add
    # deliberately incorrect calculation/payout scenarios. Clean tables remain untouched.
    tables["incentive_calculations"] = calculate_incentives(
        tables["orders"], tables["discount_detail"], tables["returns_cancellations"],
        tables["rep_targets_quotas"], tables["product_master"], tables["rep_master"],
        tables["incentive_policy_rules"], str(config["project"].get("currency_code", "UNK")),
    )
    for number, anomaly_type, category, description, severity, target in incentive_actions:
        injection_id = f"INJ_{number:04d}"
        rep_id, period = str(target.rep_id), pd.Timestamp(target.period)
        frame = tables["incentive_calculations"]
        idx = frame.index[frame["rep_id"].eq(rep_id) & pd.to_datetime(frame["period"]).eq(period)]
        if len(idx) == 0:
            continue
        original = frame.loc[idx, ["accelerator_amount", "manual_adjustment", "calculated_incentive", "final_incentive_paid"]].to_dict("records")
        factor = _factor(severity)
        if anomaly_type == "peer_incentive_outlier":
            delta = np.maximum(
                frame.loc[idx, "final_incentive_paid"].abs().to_numpy(float)
                * (factor - 1.0),
                250.0 * factor,
            )
            frame.loc[idx, "final_incentive_paid"] += delta
        elif anomaly_type == "incorrect_accelerator_tier":
            delta = np.maximum(
                np.maximum(
                    frame.loc[idx, "accelerator_amount"].abs().to_numpy(float),
                    frame.loc[idx, "base_incentive"].abs().to_numpy(float) * 0.25,
                ) * factor,
                250.0 * factor,
            )
            frame.loc[idx, "accelerator_amount"] += delta
            frame.loc[idx, "final_incentive_paid"] += delta
        elif anomaly_type == "duplicate_incentive_adjustment":
            adjustment = np.maximum(
                frame.loc[idx, "base_incentive"].abs().to_numpy(float)
                * 0.35
                * factor,
                250.0 * factor,
            )
            frame.loc[idx, "manual_adjustment"] = adjustment * 2
            frame.loc[idx, "final_incentive_paid"] += adjustment * 2
        else:
            adjustment = frame.loc[idx, "base_incentive"].clip(lower=100) * 0.65 * factor
            frame.loc[idx, "manual_adjustment"] = adjustment
            frame.loc[idx, "final_incentive_paid"] += adjustment
        frame.loc[idx, "payout_to_sales_ratio"] = frame.loc[idx, "final_incentive_paid"] / frame.loc[idx, "eligible_net_sales"].abs().clip(lower=1)
        frame.loc[idx, "data_lineage"] = "synthetic_injected"
        _record(truth, injection_id, anomaly_type, category, description, severity, rep_id, period, "incentive_calculations", frame.loc[idx, "incentive_record_id"].tolist(), original, frame.loc[idx, ["accelerator_amount", "manual_adjustment", "calculated_incentive", "final_incentive_paid"]].to_dict("records"))

    # Expand capacity overload to the configured 10–12% prevalence independently
    # of the ~5% commercial review cases.
    capacity = tables.get("capacity_calendar")
    if capacity is not None and len(capacity):
        desired = int(round(len(capacity) * float(config["anomalies"]["capacity_overload_prevalence"])))
        existing = capacity.index[
            capacity["capacity_record_id"].astype(str).isin(injected_overload_record_ids)
        ].tolist()
        # Keep the overload benchmark at unique rep-period grain and avoid
        # stacking it onto the separately controlled commercial/correlated
        # cases.  The two explicit capacity scenarios already occupy selected
        # commercial cases by design.
        capacity_keys = pd.MultiIndex.from_frame(capacity[["rep_id", "period"]])
        candidates = capacity.index[
            ~capacity.index.isin(existing)
            & ~capacity["data_lineage"].eq("synthetic_injected")
            & ~capacity_keys.isin(selected_key_index)
        ].to_numpy()
        needed = max(0, min(desired - len(injected_overload_record_ids), len(candidates)))
        chosen_capacity = (
            rng.choice(candidates, size=needed, replace=False) if needed else []
        )
        capacity_severities = _severities(
            needed, config["anomalies"]["severity_mix"], rng
        )
        for sequence, (idx, severity) in enumerate(
            zip(chosen_capacity, capacity_severities), start=1
        ):
            row = capacity.loc[idx]
            original = {"required_total_hours": row.required_total_hours, "available_field_hours": row.available_field_hours}
            multiplier = 1.08 if severity == "low" else (1.25 if severity == "medium" else 1.48)
            risk_thresholds = config.get("capacity", {}).get("risk_thresholds", {})
            high_ratio = float(risk_thresholds.get("high", 1.00))
            critical_ratio = float(risk_thresholds.get("critical", 1.20))
            minimum_ratio = {
                "low": high_ratio * 1.02,
                "medium": high_ratio * 1.18,
                "high": max(high_ratio * 1.35, critical_ratio * 1.02),
            }[str(severity)]
            target_total = max(
                float(row.required_total_hours) * multiplier,
                float(row.available_field_hours) * minimum_ratio,
            )
            _raise_capacity_required_total(
                capacity,
                pd.Index([idx]),
                pd.Series([target_total], index=[idx]),
                config,
            )
            capacity.loc[idx, "data_lineage"] = "synthetic_injected"
            injected_overload_record_ids.add(str(row.capacity_record_id))
            _record(truth, f"CAP_{sequence:04d}", "territory_workload_exceeds_capacity", "capacity", "Configured controlled overload case", str(severity), str(row.rep_id), pd.Timestamp(row.period), "capacity_calendar", [str(row.capacity_record_id)], original, {"required_total_hours": float(capacity.loc[idx, "required_total_hours"]), "available_field_hours": float(row.available_field_hours)}, entity_type="capacity_record", entity_id=str(row.capacity_record_id))

    _repair_injected_provenance(clean_tables, tables, truth)
    ground_truth = pd.DataFrame(truth).sort_values(["period", "injection_id"]).reset_index(drop=True)
    desired_correlated_count = min(desired_correlated, len(selected))
    truth_key_groups = ground_truth.groupby(["rep_id", "period"], observed=True)
    multi_type_keys = [
        (str(rep_id), pd.Timestamp(period))
        for (rep_id, period), group in truth_key_groups
        if group["anomaly_type"].nunique() > 1
    ]
    intrinsically_correlated_types = {
        "multi_signal_sales_discount_returns",
        "post_payout_returns",
        "threshold_crossing_discount",
        "overlap_impossible_travel",
        "end_of_period_sales_spike",
    }
    intrinsic_keys = [
        (str(row.rep_id), pd.Timestamp(row.period))
        for row in ground_truth.loc[
            ground_truth["anomaly_type"].isin(intrinsically_correlated_types),
            ["rep_id", "period"],
        ].drop_duplicates().itertuples(index=False)
    ]
    correlated_candidates = list(
        dict.fromkeys(
            sorted(multi_type_keys, key=lambda value: (value[1], value[0]))
            + sorted(intrinsic_keys, key=lambda value: (value[1], value[0]))
        )
    )
    correlated_keys = set(correlated_candidates[:desired_correlated_count])
    ground_truth["correlated_case_flag"] = [
        (str(rep_id), pd.Timestamp(period)) in correlated_keys
        for rep_id, period in ground_truth[["rep_id", "period"]].itertuples(index=False, name=None)
    ]
    return tables, ground_truth
