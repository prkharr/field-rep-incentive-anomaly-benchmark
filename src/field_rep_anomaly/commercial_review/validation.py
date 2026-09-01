"""Automated relational, chronology, reconciliation, and leakage validation."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .anomalies import (
    SCENARIOS,
    audit_injected_record_changes,
    ground_truth_trace_pairs,
)
from .policy import incentive_reconciliation_error


PRIMARY_KEYS = {
    "rep_master": ["rep_id"],
    "manager_master": ["manager_id"],
    "team_master": ["team_id"],
    "customer_master": ["customer_id"],
    "product_master": ["product_id"],
    "territory_master": ["territory_id"],
    "rep_targets_quotas": ["rep_id", "period"],
    "incentive_policy_rules": ["policy_id", "policy_version", "lower_attainment_pct"],
    "orders": ["order_line_id"],
    "discount_detail": ["discount_id"],
    "returns_cancellations": ["return_id"],
    "field_visits": ["visit_id"],
    "crm_interactions": ["interaction_id"],
    "travel_expenses": ["expense_id"],
    "incentive_calculations": ["incentive_record_id"],
    "capacity_calendar": ["capacity_record_id"],
    "capacity_customer_drilldown": ["rep_id", "period", "customer_id"],
    "capacity_territory_allocation": ["capacity_territory_allocation_id"],
    "capacity_territory_summary": ["capacity_territory_record_id"],
}


def _check(
    rows: list[dict[str, Any]], name: str, passed: bool, value: Any, detail: str
) -> None:
    rows.append(
        {
            "check_name": name,
            "status": "pass" if passed else "fail",
            "value": value,
            "detail": detail,
        }
    )
    if not passed:
        raise ValueError(f"Validation failed: {name}: {detail} (value={value})")


def _fk(
    rows: list[dict[str, Any]], child: pd.DataFrame, column: str, parent: pd.DataFrame, parent_column: str, label: str
) -> None:
    values = child[column].dropna().astype(str)
    if column in {"visit_id"}:
        values = values[values.str.len().gt(0)]
    missing = set(values) - set(parent[parent_column].dropna().astype(str))
    _check(rows, f"fk__{label}", not missing, len(missing), f"{column} -> {parent_column}")


def _fk_composite(
    rows: list[dict[str, Any]],
    child: pd.DataFrame,
    child_columns: list[str],
    parent: pd.DataFrame,
    parent_columns: list[str],
    label: str,
) -> None:
    child_keys = set(
        child[child_columns].dropna().astype(str).itertuples(index=False, name=None)
    )
    parent_keys = set(
        parent[parent_columns].dropna().astype(str).itertuples(index=False, name=None)
    )
    missing = child_keys - parent_keys
    _check(
        rows,
        f"fk__{label}",
        not missing,
        len(missing),
        f"{'+'.join(child_columns)} -> {'+'.join(parent_columns)}",
    )


def validate_relational_benchmark(
    clean: dict[str, pd.DataFrame],
    injected: dict[str, pd.DataFrame],
    ground_truth: pd.DataFrame,
    feature_store: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Validate all required invariants, raising immediately on a failed check."""
    rows: list[dict[str, Any]] = []
    for layer_name, tables in [("clean", clean), ("injected", injected)]:
        for table_name, key in PRIMARY_KEYS.items():
            if table_name not in tables:
                continue
            table = tables[table_name]
            duplicates = int(table.duplicated(key).sum())
            _check(rows, f"{layer_name}__pk__{table_name}", duplicates == 0, duplicates, "+".join(key))

        _fk(rows, tables["orders"], "rep_id", tables["rep_master"], "rep_id", f"{layer_name}_orders_rep")
        _fk(rows, tables["orders"], "customer_id", tables["customer_master"], "customer_id", f"{layer_name}_orders_customer")
        _fk(rows, tables["orders"], "product_id", tables["product_master"], "product_id", f"{layer_name}_orders_product")
        _fk(rows, tables["orders"], "territory_id", tables["territory_master"], "territory_id", f"{layer_name}_orders_territory")
        _fk(rows, tables["rep_master"], "manager_id", tables["manager_master"], "manager_id", f"{layer_name}_rep_manager")
        _fk(rows, tables["rep_master"], "team_id", tables["team_master"], "team_id", f"{layer_name}_rep_team")
        _fk(rows, tables["rep_master"], "territory_id", tables["territory_master"], "territory_id", f"{layer_name}_rep_territory")
        if "team_id" in tables["manager_master"]:
            _fk(rows, tables["manager_master"], "team_id", tables["team_master"], "team_id", f"{layer_name}_manager_team")
        if "manager_id" in tables["team_master"]:
            _fk(rows, tables["team_master"], "manager_id", tables["manager_master"], "manager_id", f"{layer_name}_team_manager")
        if "team_id" in tables["territory_master"]:
            _fk(rows, tables["territory_master"], "team_id", tables["team_master"], "team_id", f"{layer_name}_territory_team")
        _fk(rows, tables["customer_master"], "territory_id", tables["territory_master"], "territory_id", f"{layer_name}_customer_territory")
        _fk(rows, tables["customer_master"], "primary_rep_id", tables["rep_master"], "rep_id", f"{layer_name}_customer_rep")
        _fk(rows, tables["rep_targets_quotas"], "rep_id", tables["rep_master"], "rep_id", f"{layer_name}_target_rep")
        _fk(rows, tables["discount_detail"], "order_line_id", tables["orders"], "order_line_id", f"{layer_name}_discount_order")
        _fk(rows, tables["discount_detail"], "rep_id", tables["rep_master"], "rep_id", f"{layer_name}_discount_rep")
        if "approver_id" in tables["discount_detail"]:
            _fk(rows, tables["discount_detail"], "approver_id", tables["manager_master"], "manager_id", f"{layer_name}_discount_approver")
        _fk(rows, tables["returns_cancellations"], "order_line_id", tables["orders"], "order_line_id", f"{layer_name}_return_order")
        _fk(rows, tables["returns_cancellations"], "order_id", tables["orders"], "order_id", f"{layer_name}_return_order_header")
        _fk(rows, tables["returns_cancellations"], "rep_id", tables["rep_master"], "rep_id", f"{layer_name}_return_rep")
        _fk(rows, tables["returns_cancellations"], "customer_id", tables["customer_master"], "customer_id", f"{layer_name}_return_customer")
        _fk(rows, tables["returns_cancellations"], "product_id", tables["product_master"], "product_id", f"{layer_name}_return_product")
        _fk(rows, tables["field_visits"], "rep_id", tables["rep_master"], "rep_id", f"{layer_name}_visit_rep")
        _fk(rows, tables["field_visits"], "customer_id", tables["customer_master"], "customer_id", f"{layer_name}_visit_customer")
        _fk(rows, tables["field_visits"], "territory_id", tables["territory_master"], "territory_id", f"{layer_name}_visit_territory")
        _fk(rows, tables["crm_interactions"], "rep_id", tables["rep_master"], "rep_id", f"{layer_name}_crm_rep")
        _fk(rows, tables["crm_interactions"], "customer_id", tables["customer_master"], "customer_id", f"{layer_name}_crm_customer")
        if "visit_id" in tables["crm_interactions"]:
            _fk(rows, tables["crm_interactions"], "visit_id", tables["field_visits"], "visit_id", f"{layer_name}_crm_visit")
        _fk(rows, tables["travel_expenses"], "visit_id", tables["field_visits"], "visit_id", f"{layer_name}_expense_visit")
        _fk(rows, tables["travel_expenses"], "rep_id", tables["rep_master"], "rep_id", f"{layer_name}_expense_rep")
        for column, parent_name, parent_column in [
            ("rep_id", "rep_master", "rep_id"),
            ("manager_id", "manager_master", "manager_id"),
            ("team_id", "team_master", "team_id"),
            ("territory_id", "territory_master", "territory_id"),
            ("policy_id", "incentive_policy_rules", "policy_id"),
        ]:
            _fk(rows, tables["incentive_calculations"], column, tables[parent_name], parent_column, f"{layer_name}_incentive_{column}")
        _fk_composite(
            rows,
            tables["incentive_calculations"],
            ["policy_id", "policy_version"],
            tables["incentive_policy_rules"],
            ["policy_id", "policy_version"],
            f"{layer_name}_incentive_policy_version",
        )
        _fk(rows, tables["capacity_calendar"], "rep_id", tables["rep_master"], "rep_id", f"{layer_name}_capacity_rep")
        for column, parent_name in [
            ("manager_id", "manager_master"),
            ("team_id", "team_master"),
            ("territory_id", "territory_master"),
        ]:
            if column in tables["capacity_calendar"]:
                _fk(rows, tables["capacity_calendar"], column, tables[parent_name], column, f"{layer_name}_capacity_{column}")
        if "capacity_territory_allocation" in tables:
            allocation = tables["capacity_territory_allocation"]
            _fk(
                rows,
                allocation,
                "rep_id",
                tables["rep_master"],
                "rep_id",
                f"{layer_name}_capacity_allocation_rep",
            )
            _fk(
                rows,
                allocation,
                "territory_id",
                tables["territory_master"],
                "territory_id",
                f"{layer_name}_capacity_allocation_territory",
            )
            _fk_composite(
                rows,
                allocation,
                ["rep_id", "period"],
                tables["capacity_calendar"],
                ["rep_id", "period"],
                f"{layer_name}_capacity_allocation_calendar",
            )
        if "capacity_territory_summary" in tables:
            _fk(
                rows,
                tables["capacity_territory_summary"],
                "territory_id",
                tables["territory_master"],
                "territory_id",
                f"{layer_name}_capacity_summary_territory",
            )
        if "capacity_customer_drilldown" in tables:
            drill = tables["capacity_customer_drilldown"]
            _fk(rows, drill, "rep_id", tables["rep_master"], "rep_id", f"{layer_name}_coverage_rep")
            _fk(rows, drill, "customer_id", tables["customer_master"], "customer_id", f"{layer_name}_coverage_customer")
            if "territory_id" in drill:
                _fk(rows, drill, "territory_id", tables["territory_master"], "territory_id", f"{layer_name}_coverage_territory")

        orders = tables["orders"]
        expected_discount = orders["gross_sales"].abs() * orders["discount_pct"]
        expected_net = orders["gross_sales"] - np.sign(orders["gross_sales"]) * expected_discount
        discount_error = float((orders["discount_amount"] - expected_discount).abs().max())
        net_error = float((orders["net_sales"] - expected_net).abs().max())
        _check(
            rows,
            f"{layer_name}__order_discount_arithmetic",
            discount_error <= 1e-6 and net_error <= 1e-6,
            {"discount_max_error": discount_error, "net_max_error": net_error},
            "discount_amount=abs(gross_sales)*discount_pct and signed net sales reconcile",
        )
        detail_reconciliation = tables["discount_detail"].merge(
            orders[["order_line_id", "discount_pct", "discount_amount"]],
            on="order_line_id",
            how="left",
            suffixes=("_detail", "_order"),
            validate="one_to_one",
        )
        detail_pct_error = float(
            (detail_reconciliation["discount_pct_detail"] - detail_reconciliation["discount_pct_order"])
            .abs()
            .max()
        )
        detail_amount_error = float(
            (detail_reconciliation["discount_amount_detail"] - detail_reconciliation["discount_amount_order"])
            .abs()
            .max()
        )
        _check(
            rows,
            f"{layer_name}__discount_detail_reconciliation",
            detail_pct_error <= 1e-6 and detail_amount_error <= 1e-6,
            {"pct_max_error": detail_pct_error, "amount_max_error": detail_amount_error},
            "Discount detail agrees with its order line",
        )

        order_dates = pd.to_datetime(orders["order_date"])
        invoice_dates = pd.to_datetime(orders["invoice_date"])
        fulfillment_dates = pd.to_datetime(orders["fulfillment_date"])
        order_chronology = order_dates.le(invoice_dates) & invoice_dates.le(fulfillment_dates)
        _check(
            rows,
            f"{layer_name}__order_invoice_fulfillment_chronology",
            bool(order_chronology.all()),
            int((~order_chronology).sum()),
            "Order date <= invoice date <= fulfillment date",
        )

        returns = tables["returns_cancellations"]
        chronology = pd.to_datetime(returns["return_date"]).ge(pd.to_datetime(returns["original_order_date"]))
        _check(rows, f"{layer_name}__returns_after_orders", bool(chronology.all()), int((~chronology).sum()), "Return date must not precede original order date")
        return_linkage = returns.merge(
            orders[["order_line_id", "order_id", "order_date"]],
            on="order_line_id",
            how="left",
            suffixes=("_return", "_order"),
            validate="many_to_one",
        )
        linked_dates_match = pd.to_datetime(return_linkage["original_order_date"]).eq(
            pd.to_datetime(return_linkage["order_date"])
        )
        _check(
            rows,
            f"{layer_name}__return_original_order_linkage",
            bool(linked_dates_match.all()),
            int((~linked_dates_match).sum()),
            "Return original_order_date agrees with the linked current order line",
        )
        return_header_matches = return_linkage["order_id_return"].astype(str).eq(
            return_linkage["order_id_order"].astype(str)
        )
        _check(
            rows,
            f"{layer_name}__return_order_header_linkage",
            bool(return_header_matches.all()),
            int((~return_header_matches).sum()),
            "Return order_id agrees with the header of its linked order line",
        )
        incentive = tables["incentive_calculations"]
        after_period = pd.to_datetime(incentive["payout_date"]).gt(pd.to_datetime(incentive["period"]) + pd.offsets.MonthEnd(1))
        _check(rows, f"{layer_name}__payout_after_period", bool(after_period.all()), int((~after_period).sum()), "Payout occurs after period close")
        return_timing = returns.merge(
            incentive[["rep_id", "period", "payout_date"]].rename(columns={"period": "payout_period"}),
            on=["rep_id", "payout_period"],
            how="left",
            validate="many_to_one",
        )
        expected_after_payout = pd.to_datetime(return_timing["return_date"]).gt(
            pd.to_datetime(return_timing["payout_date"])
        )
        flag_matches = expected_after_payout.eq(
            return_timing["after_incentive_payout_flag"].astype(bool)
        )
        _check(
            rows,
            f"{layer_name}__post_payout_return_flag",
            bool(flag_matches.all()),
            int((~flag_matches).sum()),
            "Post-payout return flag uses the effective policy payout date",
        )
        finite_capacity = np.isfinite(tables["capacity_calendar"][["available_field_hours", "required_total_hours", "utilization_pct", "required_fte", "available_fte", "fte_gap"]].to_numpy(float)).all()
        _check(rows, f"{layer_name}__finite_capacity", bool(finite_capacity), bool(finite_capacity), "Capacity arithmetic is finite")
        capacity = tables["capacity_calendar"]
        capacity_residual = (
            capacity["required_total_hours"]
            - capacity[
                [
                    "planned_visit_hours",
                    "planned_travel_hours",
                    "required_customer_coverage_hours",
                    "workload_buffer_hours",
                ]
            ].sum(axis=1)
        ).abs()
        _check(
            rows,
            f"{layer_name}__capacity_required_hours_reconciliation",
            bool(capacity_residual.le(1e-8).all()),
            float(capacity_residual.max()),
            "Required total hours equal all documented workload components",
        )

        visits = tables["field_visits"]
        actual_start = pd.to_datetime(visits["actual_start_time"])
        actual_end = pd.to_datetime(visits["actual_end_time"])
        elapsed_minutes = (actual_end - actual_start).dt.total_seconds() / 60.0
        visit_duration = pd.to_numeric(visits["visit_duration_minutes"], errors="coerce")
        visit_chronology = actual_end.ge(actual_start)
        duration_matches = (elapsed_minutes - visit_duration).abs().le(1e-6)
        _check(rows, f"{layer_name}__visit_chronology", bool(visit_chronology.all()), int((~visit_chronology).sum()), "Actual visit end is not before start")
        _check(rows, f"{layer_name}__visit_duration_reconciliation", bool(duration_matches.all()), float((elapsed_minutes - visit_duration).abs().max()), "Visit duration agrees with timestamps")
        if layer_name == "clean":
            scheduled = visits[["rep_id", "period", "actual_start_time", "actual_end_time"]].copy()
            scheduled["actual_start_time"] = pd.to_datetime(scheduled["actual_start_time"])
            scheduled["actual_end_time"] = pd.to_datetime(scheduled["actual_end_time"])
            scheduled = scheduled.sort_values(
                ["rep_id", "period", "actual_start_time", "actual_end_time"],
                kind="mergesort",
            )
            scheduled["prior_max_end"] = scheduled.groupby(
                ["rep_id", "period"], observed=True
            )["actual_end_time"].transform(lambda values: values.shift().cummax())
            clean_overlaps = scheduled["actual_start_time"].lt(scheduled["prior_max_end"])
            duplicate_starts = scheduled.duplicated(
                ["rep_id", "actual_start_time"], keep=False
            )
            _check(
                rows,
                "clean__visit_schedule_no_overlap",
                not bool(clean_overlaps.any()) and not bool(duplicate_starts.any()),
                {
                    "overlaps": int(clean_overlaps.sum()),
                    "duplicate_starts": int(duplicate_starts.sum()),
                },
                "Normal visit schedule is non-overlapping with unique rep start times",
            )
            max_visits = int(
                config["synthetic"].get("max_normal_visits_per_rep_period", 84)
            )
            observed_max = int(
                visits.groupby(["rep_id", "period"], observed=True).size().max()
            )
            _check(
                rows,
                "clean__visit_schedule_capacity",
                observed_max <= max_visits,
                observed_max,
                "Normal rep-period visits do not exceed the configured schedule ceiling",
            )

        expenses = tables["travel_expenses"]
        expected_deviation = 100.0 * (
            expenses["claimed_amount"] - expenses["expected_amount"]
        ) / expenses["expected_amount"].clip(lower=1e-12)
        expense_error = float((expenses["deviation_pct"] - expected_deviation).abs().max())
        _check(rows, f"{layer_name}__expense_deviation_reconciliation", expense_error <= 1e-8, expense_error, "Expense deviation agrees with claimed and expected amounts")

    reconciliation = incentive_reconciliation_error(clean["incentive_calculations"])
    tolerance = clean["incentive_calculations"]["reconciliation_tolerance"]
    _check(rows, "clean__incentive_policy_reconciliation", bool(reconciliation.le(tolerance).all()), float(reconciliation.max()), "Clean final payout reconciles to centralized policy within tolerance")

    prohibited = [column for column in feature_columns if any(token in column.casefold() for token in ["ground_truth", "anomaly_type", "severity", "injection_id", "correlated_case"])]
    _check(rows, "feature_store__no_ground_truth_leakage", not prohibited, prohibited, "Controlled labels are excluded from feature allowlist")
    missing = int(feature_store[feature_columns].isna().sum().sum())
    finite = bool(np.isfinite(feature_store[feature_columns].to_numpy(float)).all())
    _check(rows, "feature_store__no_nan", missing == 0, missing, "Model features contain no NaN")
    _check(rows, "feature_store__finite", finite, finite, "Model features contain no infinities")
    unique_observations = int(feature_store.duplicated(["rep_id", "period"]).sum())
    _check(rows, "feature_store__unique_grain", unique_observations == 0, unique_observations, "Rep-period grain")

    present_types = set(ground_truth["anomaly_type"])
    required_types = {scenario[0] for scenario in SCENARIOS}
    _check(rows, "ground_truth__all_required_types", required_types.issubset(present_types), sorted(required_types - present_types), "All 22 controlled scenarios are present")
    severities = set(ground_truth["severity"])
    _check(rows, "ground_truth__severity_coverage", {"low", "medium", "high"}.issubset(severities), sorted(severities), "Low, medium, and high severities")
    configured_mix = pd.Series(
        config["anomalies"]["severity_mix"], dtype=float
    )
    configured_mix = configured_mix / configured_mix.sum()
    actual_mix = ground_truth["severity"].value_counts(normalize=True).reindex(
        configured_mix.index, fill_value=0.0
    )
    severity_mix_error = (actual_mix - configured_mix).abs()
    severity_tolerance = max(2.0 / max(len(ground_truth), 1), 0.03)
    _check(
        rows,
        "ground_truth__configured_severity_distribution",
        bool(severity_mix_error.le(severity_tolerance).all()),
        severity_mix_error.to_dict(),
        f"Configured severity mix tolerance={severity_tolerance:.4f}",
    )
    _check(rows, "ground_truth__nonempty_record_ids", bool(ground_truth["affected_record_ids"].astype(str).str.len().gt(2).all()), int(len(ground_truth)), "Every injection identifies affected records")
    changed_payload = ground_truth["original_value"].astype(str).ne(
        ground_truth["injected_value"].astype(str)
    )
    _check(
        rows,
        "ground_truth__non_noop_payloads",
        bool(changed_payload.all()),
        int((~changed_payload).sum()),
        "Every controlled truth row records a changed original/injected payload",
    )

    injected_change_audit = audit_injected_record_changes(clean, injected)
    wrong_lineage = injected_change_audit.loc[
        ~injected_change_audit["data_lineage"].eq("synthetic_injected"),
        ["dataset", "record_id", "data_lineage"],
    ]
    _check(
        rows,
        "injected_diff__changed_rows_have_injected_lineage",
        wrong_lineage.empty,
        wrong_lineage.head(20).to_dict("records"),
        "Every added or value-modified injection-stage fact has synthetic_injected lineage",
    )
    truth_trace_pairs = ground_truth_trace_pairs(ground_truth, injected)
    untraced = [
        (str(row.dataset), str(row.record_id))
        for row in injected_change_audit.itertuples(index=False)
        if (str(row.dataset), str(row.record_id)) not in truth_trace_pairs
    ]
    _check(
        rows,
        "injected_diff__changed_rows_are_ground_truth_traceable",
        not untraced,
        untraced[:20],
        "Every added or value-modified injection-stage fact is identified in ground truth",
    )
    changed_pairs = {
        (str(row.dataset), str(row.record_id))
        for row in injected_change_audit.itertuples(index=False)
    }
    unbacked_truth: list[str] = []
    for truth_index, truth_row in ground_truth.iterrows():
        row_pairs = ground_truth_trace_pairs(
            pd.DataFrame([truth_row.to_dict()]), injected
        )
        if not row_pairs.intersection(changed_pairs):
            unbacked_truth.append(
                str(truth_row.get("injection_id", truth_index))
            )
    _check(
        rows,
        "ground_truth__each_injection_has_changed_fact",
        not unbacked_truth,
        unbacked_truth[:20],
        "Every ground-truth injection references at least one added or value-modified fact",
    )

    commercial_truth = ground_truth.loc[~ground_truth["anomaly_category"].eq("capacity")].copy()
    commercial_cases = commercial_truth[["rep_id", "period"]].drop_duplicates()
    commercial_rate = len(commercial_cases) / max(len(injected["capacity_calendar"]), 1)
    configured_commercial = float(config["anomalies"]["rep_period_prevalence"])
    commercial_tolerance = max(1 / max(len(injected["capacity_calendar"]), 1), 0.01)
    _check(
        rows,
        "ground_truth__configured_rep_period_prevalence",
        abs(commercial_rate - configured_commercial) <= commercial_tolerance,
        commercial_rate,
        f"Configured={configured_commercial:.3f}, tolerance={commercial_tolerance:.3f}",
    )
    correlated_cases = (
        commercial_truth.groupby(["rep_id", "period"], observed=True)["correlated_case_flag"]
        .max()
        .astype(bool)
    )
    correlated_share = float(correlated_cases.mean()) if len(correlated_cases) else 0.0
    configured_correlated = float(config["anomalies"].get("correlated_case_share", 0.20))
    correlated_tolerance = max(1 / max(len(correlated_cases), 1), 0.05)
    _check(
        rows,
        "ground_truth__configured_correlated_case_share",
        abs(correlated_share - configured_correlated) <= correlated_tolerance,
        correlated_share,
        f"Configured={configured_correlated:.3f}, tolerance={correlated_tolerance:.3f}",
    )
    order_rate = float(injected["orders"]["data_lineage"].eq("synthetic_injected").mean())
    configured_order = float(config["anomalies"]["order_level_prevalence"])
    order_tolerance = max(1 / max(len(injected["orders"]), 1), 0.005)
    _check(
        rows,
        "orders__configured_injected_line_prevalence",
        abs(order_rate - configured_order) <= order_tolerance,
        order_rate,
        f"Configured={configured_order:.3f}, tolerance={order_tolerance:.3f}",
    )

    capacity = injected["capacity_calendar"]
    injected_capacity_ids = set()
    valid_capacity_ids = set(capacity["capacity_record_id"].astype(str))
    for value in ground_truth.loc[
        ground_truth["anomaly_type"].eq("territory_workload_exceeds_capacity"),
        "affected_record_ids",
    ].astype(str):
        try:
            injected_capacity_ids.update(
                str(item)
                for item in json.loads(value)
                if str(item) in valid_capacity_ids
            )
        except (TypeError, ValueError):
            continue
    overload_rate = len(injected_capacity_ids) / max(len(capacity), 1)
    configured = float(config["anomalies"]["capacity_overload_prevalence"])
    tolerance_rate = max(1 / max(len(capacity), 1), 0.015)
    _check(rows, "capacity__configured_overload_prevalence", abs(overload_rate - configured) <= tolerance_rate, overload_rate, f"Configured={configured:.3f}, tolerance={tolerance_rate:.3f}")
    return pd.DataFrame(rows)


def validate_dashboard_files(directory: str | Any, filenames: list[str]) -> pd.DataFrame:
    """Validate that dashboard CSV contracts exist and contain at least one row."""
    from pathlib import Path

    root = Path(directory)
    rows = []
    for filename in filenames:
        path = root / filename
        exists = path.exists()
        count = len(pd.read_csv(path)) if exists else 0
        rows.append({"file": filename, "exists": exists, "rows": count, "status": "pass" if exists and count > 0 else "fail"})
        if not exists or count == 0:
            raise ValueError(f"Dashboard output missing or empty: {path}")
    return pd.DataFrame(rows)
