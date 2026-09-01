"""Build the additive twelve-file dashboard semantic layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .capacity import (
    build_capacity_territory_allocation,
    build_capacity_territory_summary,
)
from .io import write_dashboard_csv


REQUIRED_DASHBOARD_FILES = [
    "dashboard_kpi_summary.csv",
    "dashboard_manager_review_queue.csv",
    "dashboard_rep_period_summary.csv",
    "dashboard_anomaly_evidence.csv",
    "dashboard_feature_contributions.csv",
    "dashboard_peer_comparison.csv",
    "dashboard_capacity_summary.csv",
    "dashboard_capacity_customer_drilldown.csv",
    "dashboard_model_metrics.csv",
    "dashboard_anomaly_type_metrics.csv",
    "dashboard_data_quality.csv",
    "dashboard_run_manifest.csv",
]

OPTIONAL_SEMANTIC_DASHBOARD_FILES = [
    "dashboard_capacity_territory_allocation.csv",
    "dashboard_capacity_territory_summary.csv",
]

BENCHMARK_ONLY_FILES = {
    "dashboard_anomaly_evidence.csv",
    "dashboard_anomaly_type_metrics.csv",
    "dashboard_model_metrics.csv",
}

TRUTH_COLUMNS = {
    "ground_truth_label",
    "anomaly_type",
    "anomaly_category",
    "severity",
    "correlated_case_flag",
    "injection_id",
    "affected_dataset",
    "affected_record_ids",
}


def _drop_truth(frame: pd.DataFrame) -> pd.DataFrame:
    suspicious = [
        column for column in frame.columns
        if column in TRUTH_COLUMNS
        or "ground_truth" in column.casefold()
        or "injection_id" in column.casefold()
        or "correlated_case" in column.casefold()
    ]
    return frame.drop(columns=suspicious, errors="ignore")


def _metric(metrics: pd.DataFrame, candidates: list[str], default: float = np.nan) -> float:
    if metrics.empty:
        return default
    for candidate in candidates:
        if candidate in metrics:
            value = pd.to_numeric(metrics[candidate], errors="coerce").dropna()
            if len(value):
                return float(value.iloc[0])
    return default


def _capacity_names(capacity: pd.DataFrame, clean_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    result = capacity.copy()
    for name, key, columns in [
        ("rep_master", "rep_id", ["rep_id", "rep_name", "manager_id", "team_id", "territory_id"]),
        ("manager_master", "manager_id", ["manager_id", "manager_name"]),
        ("team_master", "team_id", ["team_id", "team_name"]),
        ("territory_master", "territory_id", ["territory_id", "territory_name"]),
    ]:
        lookup = clean_tables[name][columns].drop_duplicates(key)
        collisions = [column for column in lookup if column in result and column != key]
        lookup = lookup.drop(columns=collisions)
        result = result.merge(lookup, on=key, how="left", validate="many_to_one")
    return result


def _territory_capacity_names(
    capacity: pd.DataFrame, clean_tables: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    result = capacity.copy()
    territory = clean_tables["territory_master"][["territory_id", "territory_name"]].drop_duplicates(
        "territory_id"
    )
    territory = territory.rename(columns={"territory_name": "_master_territory_name"})
    result = result.merge(territory, on="territory_id", how="left", validate="many_to_one")
    if "territory_name" in result:
        result["territory_name"] = result["_master_territory_name"].where(
            result["_master_territory_name"].notna(), result["territory_name"]
        )
    else:
        result["territory_name"] = result["_master_territory_name"]
    return result.drop(columns="_master_territory_name")


def build_dashboard_datasets(
    clean_tables: dict[str, pd.DataFrame],
    injected_tables: dict[str, pd.DataFrame],
    clean_features: pd.DataFrame,
    injected_features: pd.DataFrame,
    model_results: dict[str, Any],
    ground_truth: pd.DataFrame,
    peer_comparison: pd.DataFrame,
    capacity_customer_drilldown: pd.DataFrame,
    capacity_metrics: pd.DataFrame,
    data_quality: pd.DataFrame,
    run_manifest: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Create compact manager and benchmark datasets without model fitting."""
    scored = model_results["injected_scores"].copy()
    scored["period"] = pd.to_datetime(scored["period"])
    scored["risk_band"] = pd.cut(
        scored["anomaly_score"], [-np.inf, 0.95, 0.99, np.inf], labels=["low", "medium", "high"]
    ).astype(str)
    scored["review_priority"] = np.select(
        [scored["manager_review_flag"].astype(bool) & scored["anomaly_score"].ge(0.99), scored["manager_review_flag"].astype(bool)],
        ["High", "Medium"],
        default="Context only",
    )
    scored = scored.sort_values(["raw_score", "observation_id"], ascending=[False, True], kind="mergesort")
    scored["review_rank"] = np.arange(1, len(scored) + 1)
    scored["incentive_residual"] = scored.get("incentive_calculation_residual", 0)
    queue = scored.loc[scored["manager_review_flag"].astype(bool)].copy()
    manager_columns = [
        "review_rank", "observation_id", "rep_id", "rep_name", "manager_id", "manager_name",
        "team_id", "team_name", "territory_id", "territory_name", "period", "split",
        "anomaly_score", "anomaly_percentile", "raw_score", "raw_threshold", "threshold_flag",
        "manager_review_flag", "review_priority", "risk_band", "primary_reason_code", "primary_reason",
        "secondary_reason", "driver_1_name", "driver_1_value", "driver_1_peer_value", "driver_1_percentile",
        "driver_2_name", "driver_2_value", "driver_2_peer_value", "driver_3_name", "driver_3_value",
        "recommended_review_action", "gross_sales", "net_sales", "target_sales", "attainment_pct",
        "final_incentive_paid", "expected_incentive", "incentive_residual", "payout_to_peer_median_ratio",
        "average_discount_pct", "post_incentive_return_rate", "impossible_travel_count",
        "capacity_utilization_pct", "fte_gap", "currency_code", "data_lineage",
    ]
    queue = _drop_truth(queue[[column for column in manager_columns if column in queue]])

    summary_columns = [
        "observation_id", "rep_id", "rep_name", "manager_id", "manager_name", "team_id", "team_name",
        "territory_id", "territory_name", "period", "payout_date", "split", "anomaly_score", "anomaly_percentile",
        "raw_score", "threshold_flag", "manager_review_flag", "review_priority", "risk_band", "primary_reason",
        "gross_sales", "net_sales", "sales_growth", "rolling_sales_mean", "target_sales", "attainment_pct",
        "final_incentive_paid", "expected_incentive", "incentive_calculation_residual", "average_discount_pct",
        "return_rate", "post_incentive_return_rate", "end_of_period_sales_share", "completed_visit_count",
        "average_visit_duration", "crm_interaction_count", "claimed_expense_amount", "distance_claim_ratio",
        "available_field_hours", "required_total_hours", "capacity_utilization_pct", "required_fte", "available_fte",
        "fte_gap", "priority_customer_coverage_gap", "capacity_risk_band", "currency_code", "data_lineage",
    ]
    rep_period = _drop_truth(scored[[column for column in summary_columns if column in scored]])

    truth_keys = ground_truth.copy()
    truth_keys["period"] = pd.to_datetime(truth_keys["period"])
    truth_rollup = (
        truth_keys.groupby(["rep_id", "period"], observed=True)
        .agg(
            ground_truth_label=("ground_truth_label", "max"),
            anomaly_type=("anomaly_type", lambda values: "|".join(sorted(set(values.astype(str))))),
            anomaly_category=("anomaly_category", lambda values: "|".join(sorted(set(values.astype(str))))),
            severity=("severity", lambda values: "|".join(sorted(set(values.astype(str))))),
            injection_count=("injection_id", "nunique"),
        )
        .reset_index()
    )
    evidence = scored.merge(
        truth_rollup,
        on=["rep_id", "period"],
        how="left",
        suffixes=("", "_truth"),
    )
    for column in ["ground_truth_label", "injection_count"]:
        evidence[column] = evidence[column].fillna(0).astype(int)
    for column in ["anomaly_type", "anomaly_category", "severity"]:
        evidence[column] = evidence[column].fillna("none")
    evidence["benchmark_mode_disclosure"] = "Controlled synthetic labels for evaluation only; not evidence of fraud or misconduct."
    evidence_columns = [
        "observation_id", "rep_id", "rep_name", "manager_id", "manager_name", "team_id", "team_name",
        "territory_id", "territory_name", "period", "split", "anomaly_score", "raw_score", "threshold_flag",
        "manager_review_flag", "primary_reason", "recommended_review_action", "ground_truth_label", "anomaly_type",
        "anomaly_category", "severity", "injection_count", "benchmark_mode_disclosure",
    ]
    evidence = evidence[[column for column in evidence_columns if column in evidence]]

    contributions = model_results["feature_contributions"].copy()
    contributions = contributions.loc[contributions["population"].eq("injected")].reset_index(drop=True)
    peer = peer_comparison.merge(
        injected_features[["observation_id", "rep_id", "period", "rep_name", "manager_name", "team_name", "territory_name"]],
        on="observation_id", how="left", validate="many_to_one",
    )

    # Manager-facing capacity views use the clean deterministic calendar.  The
    # injected capacity layer remains benchmark-only in model metrics/evidence;
    # exposing it here would make ``synthetic_injected`` a label proxy.
    capacity = _capacity_names(clean_tables["capacity_calendar"], clean_tables)
    capacity = _drop_truth(capacity)
    territory_capacity_source = clean_tables.get("capacity_territory_summary")
    if territory_capacity_source is None:
        territory_capacity_source = build_capacity_territory_summary(
            build_capacity_territory_allocation(clean_tables["capacity_calendar"])
        )
    territory_capacity = _territory_capacity_names(
        territory_capacity_source, clean_tables
    )
    territory_capacity = _drop_truth(territory_capacity)
    territory_allocation_source = clean_tables.get("capacity_territory_allocation")
    if territory_allocation_source is None:
        territory_allocation_source = build_capacity_territory_allocation(
            clean_tables["capacity_calendar"]
        )
    territory_allocation = _territory_capacity_names(
        territory_allocation_source, clean_tables
    )
    territory_allocation = _drop_truth(territory_allocation)
    coverage = capacity_customer_drilldown.copy()
    coverage = _drop_truth(coverage)
    customer_names = clean_tables["customer_master"][["customer_id", "customer_name", "customer_priority", "potential_score"]]
    collisions = [column for column in customer_names if column in coverage and column != "customer_id"]
    coverage = coverage.merge(customer_names.drop(columns=collisions), on="customer_id", how="left", validate="many_to_one")
    if "rep_name" not in coverage:
        coverage = coverage.merge(clean_tables["rep_master"][["rep_id", "rep_name"]], on="rep_id", how="left", validate="many_to_one")

    metrics = model_results["metrics_summary"].copy()
    metrics["metric_scope"] = "finalized PCA test at frozen validation threshold"
    top = model_results["top_k_metrics"].copy()
    top_long = top.melt(
        id_vars=[column for column in ["model", "split", "review_fraction", "review_count", "positive_count", "captured_count"] if column in top],
        value_vars=[column for column in ["precision", "recall", "lift"] if column in top],
        var_name="metric_name", value_name="metric_value",
    )
    metric_long = metrics.melt(
        id_vars=[column for column in ["model", "split", "threshold", "manager_review_fraction"] if column in metrics],
        value_vars=[column for column in metrics.select_dtypes(include=[np.number]).columns if column not in {"threshold", "manager_review_fraction"}],
        var_name="metric_name", value_name="metric_value",
    )
    metric_long["review_fraction"] = metrics["manager_review_fraction"].iloc[0]
    dashboard_metrics = pd.concat([metric_long, top_long], ignore_index=True, sort=False)
    if not capacity_metrics.empty:
        cap_long = capacity_metrics.copy()
        cap_long["model"] = "Deterministic capacity calendar"
        cap_long["split"] = "injected evaluation"
        dashboard_metrics = pd.concat([dashboard_metrics, cap_long], ignore_index=True, sort=False)

    type_metrics = model_results["group_metrics"].copy()
    type_metrics = type_metrics.loc[type_metrics["grouping"].isin(["anomaly_type", "anomaly_category", "severity", "manager_id", "team_id", "territory_id"])]
    quality = data_quality.copy()
    table_counts = []
    for layer, tables in [("clean", clean_tables), ("injected", injected_tables)]:
        for name, frame in tables.items():
            table_counts.append({"check_name": f"row_count__{layer}__{name}", "status": "pass", "value": len(frame), "detail": f"{layer} generated layer"})
    quality = pd.concat([quality, pd.DataFrame(table_counts)], ignore_index=True, sort=False)

    review_count = len(queue)
    high_count = int(queue["review_priority"].eq("High").sum()) if "review_priority" in queue else 0
    overload = (
        territory_capacity.loc[
            territory_capacity["capacity_risk_band"].isin(["high", "critical"])
        ]
        if "capacity_risk_band" in territory_capacity
        else territory_capacity.iloc[0:0]
    )
    metrics_summary = model_results["metrics_summary"]
    total_incentive = float(injected_tables["incentive_calculations"]["final_incentive_paid"].sum())
    kpi = pd.DataFrame(
        [
            {
                "total_gross_sales": float(injected_tables["orders"]["gross_sales"].sum()),
                "total_net_sales": float(injected_tables["orders"]["net_sales"].sum()),
                "total_incentive_paid": total_incentive,
                "review_candidate_count": review_count,
                "high_priority_review_candidate_count": high_count,
                "review_rate": review_count / max(len(scored), 1),
                "overloaded_representative_count": int(
                    capacity.loc[
                        capacity["capacity_risk_band"].isin(["high", "critical"]), "rep_id"
                    ].nunique()
                ) if "capacity_risk_band" in capacity else 0,
                "overloaded_territory_count": int(overload["territory_id"].nunique(dropna=False)) if len(overload) else 0,
                "total_positive_fte_gap": float(capacity["fte_gap"].clip(lower=0).sum()),
                "selected_threshold": _metric(metrics_summary, ["threshold"]),
                "test_precision_at_selected_threshold": _metric(metrics_summary, ["precision"]),
                "test_recall_at_selected_threshold": _metric(metrics_summary, ["recall"]),
                "benchmark_mode_label_disclosure": "Injected labels are controlled synthetic labels and never prove fraud.",
                "currency_code": clean_tables["orders"]["currency_code"].iloc[0],
            }
        ]
    )

    manifest = pd.DataFrame(
        [
            {
                **{
                    key: (
                        json.dumps(value, sort_keys=True, default=str)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in run_manifest.items()
                },
                "output_file_names": json.dumps(
                    REQUIRED_DASHBOARD_FILES + OPTIONAL_SEMANTIC_DASHBOARD_FILES
                ),
                "output_row_counts": "pending_write",
            }
        ]
    )
    datasets = {
        "dashboard_kpi_summary.csv": kpi,
        "dashboard_manager_review_queue.csv": queue,
        "dashboard_rep_period_summary.csv": rep_period,
        "dashboard_anomaly_evidence.csv": evidence,
        "dashboard_feature_contributions.csv": contributions,
        "dashboard_peer_comparison.csv": peer,
        "dashboard_capacity_summary.csv": capacity,
        "dashboard_capacity_customer_drilldown.csv": coverage,
        "dashboard_capacity_territory_allocation.csv": territory_allocation,
        "dashboard_capacity_territory_summary.csv": territory_capacity,
        "dashboard_model_metrics.csv": dashboard_metrics,
        "dashboard_anomaly_type_metrics.csv": type_metrics,
        "dashboard_data_quality.csv": quality,
        "dashboard_run_manifest.csv": manifest,
    }
    return datasets


def write_dashboard_datasets(
    datasets: dict[str, pd.DataFrame], directory: str | Path
) -> list[Path]:
    """Write the twelve required contracts and any available semantic add-ons."""
    missing = set(REQUIRED_DASHBOARD_FILES) - set(datasets)
    if missing:
        raise ValueError(f"Missing dashboard datasets: {sorted(missing)}")
    paths = []
    output_files = REQUIRED_DASHBOARD_FILES + [
        filename
        for filename in OPTIONAL_SEMANTIC_DASHBOARD_FILES
        if filename in datasets
    ]
    for filename in output_files:
        frame = datasets[filename]
        if frame.empty:
            raise ValueError(f"Dashboard dataset is empty: {filename}")
        if filename not in BENCHMARK_ONLY_FILES:
            prohibited = [column for column in frame if column in TRUTH_COLUMNS or "ground_truth" in column.casefold()]
            if prohibited:
                raise ValueError(f"Production-style dashboard dataset leaks benchmark labels: {filename}: {prohibited}")
        paths.append(write_dashboard_csv(frame, directory, filename))
    counts = {filename: len(datasets[filename]) for filename in output_files}
    manifest_path = Path(directory) / "dashboard_run_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    manifest["output_row_counts"] = json.dumps(counts, sort_keys=True)
    manifest.to_csv(manifest_path, index=False)
    return paths


def build_optional_model_dashboard_tables(model_results: dict[str, Any]) -> dict[str, pd.DataFrame]:
    curves = []
    for name, key in [("roc", "roc_curve"), ("precision_recall", "pr_curve"), ("lift", "lift_curve")]:
        frame = model_results[key].copy()
        frame["curve_type"] = name
        curves.append(frame)
    metadata = model_results["pca_metadata"]
    variance = pd.DataFrame(
        {
            "component": np.arange(1, len(metadata["explained_variance_ratio"]) + 1),
            "explained_variance_ratio": metadata["explained_variance_ratio"],
            "cumulative_explained_variance": metadata["cumulative_explained_variance"],
        }
    )
    summary = model_results["metrics_summary"].iloc[0]
    confusion = pd.DataFrame(
        [
            {"actual": "positive", "predicted": "positive", "count": summary.get("true_positives", 0)},
            {"actual": "positive", "predicted": "negative", "count": summary.get("false_negatives", 0)},
            {"actual": "negative", "predicted": "positive", "count": summary.get("false_positives", 0)},
            {"actual": "negative", "predicted": "negative", "count": summary.get("true_negatives", 0)},
        ]
    )
    return {
        "dashboard_model_curve.csv": pd.concat(curves, ignore_index=True, sort=False),
        "dashboard_pca_variance.csv": variance,
        "dashboard_confusion_matrix.csv": confusion,
        "dashboard_score_distribution.csv": model_results["score_distributions"],
        "dashboard_period_stability.csv": model_results["period_stability"],
    }
