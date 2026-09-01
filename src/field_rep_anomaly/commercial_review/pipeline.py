"""One-command controlled commercial-review benchmark pipeline."""

from __future__ import annotations

import argparse
import json
import platform
import time
import xml.etree.ElementTree as ET
from importlib.metadata import PackageNotFoundError, version
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .anomalies import inject_controlled_anomalies
from .dashboard_data import (
    OPTIONAL_SEMANTIC_DASHBOARD_FILES,
    REQUIRED_DASHBOARD_FILES,
    build_dashboard_datasets,
    build_optional_model_dashboard_tables,
    write_dashboard_datasets,
)
from .features import build_feature_store
from .foundation import file_sha256, profile_and_normalize_source, profile_json
from .io import (
    build_output_manifest_rows,
    config_sha256,
    generate_data_dictionary,
    git_commit,
    git_worktree_dirty,
    implementation_sha256,
    write_dashboard_csv,
    write_full_table,
    write_json,
)
from .modeling import run_finalized_pca
from .synthetic import generate_clean_datasets
from .validation import validate_dashboard_files, validate_relational_benchmark


def _software_versions() -> dict[str, str]:
    packages = {
        "pandas": "pandas",
        "numpy": "numpy",
        "scikit_learn": "scikit-learn",
        "pyarrow": "pyarrow",
        "streamlit": "streamlit",
        "matplotlib": "matplotlib",
        "pyyaml": "PyYAML",
    }
    result = {"python": platform.python_version()}
    for label, package in packages.items():
        try:
            result[label] = version(package)
        except PackageNotFoundError:
            result[label] = "not-installed"
    return result


def _pytest_result_note(repository_root: Path) -> str:
    junit = repository_root / "artifacts" / "reports" / "commercial_review_tests.xml"
    if not junit.exists():
        return (
            "No JUnit result existed when this pipeline run generated the report; "
            "run the documented pytest command and retain its XML artifact."
        )
    try:
        root = ET.parse(junit).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.findall("./testsuite"))
        totals = {
            name: sum(int(float(suite.attrib.get(name, 0))) for suite in suites)
            for name in ["tests", "failures", "errors", "skipped"]
        }
        elapsed = sum(float(suite.attrib.get("time", 0.0)) for suite in suites)
    except (ET.ParseError, OSError, ValueError):
        return f"JUnit artifact exists but could not be parsed: `{junit}`."
    status = "passed" if totals["failures"] == 0 and totals["errors"] == 0 else "failed"
    return (
        f"Latest retained pytest JUnit result: **{status}**; {totals['tests']} tests, "
        f"{totals['failures']} failures, {totals['errors']} errors, "
        f"{totals['skipped']} skipped, {elapsed:.2f}s. Artifact: "
        "`artifacts/reports/commercial_review_tests.xml`."
    )


def _capacity_type_metric_rows(
    clean_capacity: pd.DataFrame,
    injected_capacity: pd.DataFrame,
    truth: pd.DataFrame,
    capacity_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Expose the two capacity scenarios beside commercial type metrics."""
    overload_truth = truth.loc[
        truth["anomaly_type"].eq("territory_workload_exceeds_capacity")
    ]
    overload_support = int(len(overload_truth))
    metric = capacity_metrics.iloc[0] if not capacity_metrics.empty else pd.Series(dtype=float)
    rows: list[dict[str, Any]] = [
        {
            "group_kind": "ground_truth",
            "grouping": "anomaly_type",
            "value": "territory_workload_exceeds_capacity",
            "observations": len(injected_capacity),
            "positive_support": overload_support,
            "overall_truth_support": overload_support,
            "selected_at_threshold": int(metric.get("predicted_overload_count", 0)),
            "captured_at_threshold": int(metric.get("true_positive", 0)),
            "false_positives_at_threshold": int(metric.get("false_positive", 0)),
            "precision_at_threshold": float(metric.get("precision", np.nan)),
            "recall_at_threshold": float(metric.get("recall", np.nan)),
            "detection_rate_at_threshold": float(metric.get("recall", np.nan)),
            "captured_at_top5pct": np.nan,
            "recall_at_top5pct": np.nan,
            "detection_rate_at_top5pct": np.nan,
            "support_status": "evaluated_by_capacity_rule",
            "evaluation_scope": "deterministic capacity overload rule",
        }
    ]

    under_truth = truth.loc[
        truth["anomaly_type"].eq("persistent_priority_undercoverage")
    ]
    under_ids: set[str] = set()
    for payload in under_truth["affected_record_ids"].astype(str):
        try:
            under_ids.update(
                str(value)
                for value in json.loads(payload)
                if str(value).startswith("CAPCAL_")
            )
        except (TypeError, ValueError):
            continue
    clean_threshold = float(
        clean_capacity["priority_customer_coverage_gap"].quantile(0.95)
    )
    predicted = injected_capacity["priority_customer_coverage_gap"].gt(clean_threshold)
    labels = injected_capacity["capacity_record_id"].astype(str).isin(under_ids)
    tp = int((predicted & labels).sum())
    fp = int((predicted & ~labels).sum())
    support = int(labels.sum())
    rows.append(
        {
            "group_kind": "ground_truth",
            "grouping": "anomaly_type",
            "value": "persistent_priority_undercoverage",
            "observations": len(injected_capacity),
            "positive_support": support,
            "overall_truth_support": len(under_truth),
            "selected_at_threshold": int(predicted.sum()),
            "captured_at_threshold": tp,
            "false_positives_at_threshold": fp,
            "precision_at_threshold": tp / max(tp + fp, 1),
            "recall_at_threshold": tp / support if support else np.nan,
            "detection_rate_at_threshold": tp / support if support else np.nan,
            "captured_at_top5pct": np.nan,
            "recall_at_top5pct": np.nan,
            "detection_rate_at_top5pct": np.nan,
            "support_status": "evaluated_by_capacity_rule" if support else "no_capacity_truth_support",
            "evaluation_scope": (
                "priority coverage gap above frozen clean 95th percentile "
                f"({clean_threshold:.6g})"
            ),
        }
    )
    return pd.DataFrame(rows)


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _output_paths(output_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    paths = config["paths"]
    return {
        "generated": _resolve(output_root, paths["generated_dir"]),
        "dashboard": _resolve(output_root, paths["dashboard_dir"]),
        "artifacts": _resolve(output_root, paths["artifact_dir"]),
        "reports": _resolve(output_root, paths["report_dir"]),
        "docs": output_root / "docs",
    }


def _write_generated_layers(
    normalized: pd.DataFrame,
    mappings: dict[str, pd.DataFrame],
    clean: dict[str, pd.DataFrame],
    injected: dict[str, pd.DataFrame],
    clean_features: pd.DataFrame,
    injected_features: pd.DataFrame,
    truth: pd.DataFrame,
    paths: dict[str, Path],
    config: dict[str, Any],
) -> list[Path]:
    settings = config["output"]
    files: list[Path] = []
    clean_dir = paths["generated"] / "clean"
    injected_dir = paths["generated"] / "injected"
    benchmark_dir = paths["generated"] / "benchmark"
    files += write_full_table(normalized, clean_dir, "normalized_source_transactions", settings["write_parquet"], settings["write_csv_gz"])
    for name, frame in mappings.items():
        files += write_full_table(frame, clean_dir, name, settings["write_parquet"], settings["write_csv_gz"])
    for name, frame in clean.items():
        files += write_full_table(frame, clean_dir, name, settings["write_parquet"], settings["write_csv_gz"])
    files += write_full_table(clean_features, clean_dir, "feature_store_rep_period", settings["write_parquet"], settings["write_csv_gz"])
    for name, frame in injected.items():
        if name.endswith("_master") or name == "incentive_policy_rules":
            continue
        files += write_full_table(frame, injected_dir, name, settings["write_parquet"], settings["write_csv_gz"])
    files += write_full_table(injected_features, injected_dir, "feature_store_rep_period", settings["write_parquet"], settings["write_csv_gz"])
    files += write_full_table(truth, benchmark_dir, "anomaly_ground_truth", settings["write_parquet"], settings["write_csv_gz"])
    if settings.get("write_small_samples", True):
        sample_dir = paths["generated"].parent / "samples" / "commercial_review"
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_rows = int(settings.get("sample_rows", 25))
        for name, frame in {**clean, "anomaly_ground_truth": truth, "feature_store_rep_period": injected_features}.items():
            sample = sample_dir / f"{name}_sample.csv"
            frame.head(sample_rows).to_csv(sample, index=False)
            files.append(sample)
    return files


def _make_figures(
    model_results: dict[str, Any], capacity: pd.DataFrame, directory: Path
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []

    distribution = model_results["score_distributions"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for population, group in distribution.groupby("population", observed=True):
        x = group.get("bin_midpoint", group.get("bin_center", pd.Series(np.arange(len(group)))))
        y = group.get("count", pd.Series(np.zeros(len(group))))
        ax.plot(x, y, marker="o", label=str(population))
    ax.set_title("Clean versus injected PCA score distribution")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Observations")
    ax.legend()
    fig.tight_layout()
    path = directory / "clean_vs_injected_score_distribution.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    files.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    roc = model_results["roc_curve"]
    axes[0].plot(roc.get("false_positive_rate", roc.get("fpr")), roc.get("true_positive_rate", roc.get("tpr")))
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_title("ROC curve — controlled test")
    axes[0].set_xlabel("False-positive rate")
    axes[0].set_ylabel("True-positive rate")
    pr = model_results["pr_curve"]
    axes[1].plot(pr.get("recall"), pr.get("precision"))
    axes[1].set_title("Precision-recall curve — controlled test")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    fig.tight_layout()
    path = directory / "roc_precision_recall_curves.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    files.append(path)

    metadata = model_results["pca_metadata"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(1, metadata["retained_components"] + 1), metadata["cumulative_explained_variance"], marker="o", markersize=3)
    ax.axhline(0.95, color="gray", linestyle="--")
    ax.set_ylim(0, 1.01)
    ax.set_title("PCA cumulative explained variance")
    ax.set_xlabel("Retained components")
    ax.set_ylabel("Cumulative explained variance")
    fig.tight_layout()
    path = directory / "pca_cumulative_explained_variance.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    files.append(path)

    fig, ax = plt.subplots(figsize=(9, 5))
    top = capacity.sort_values("utilization_pct", ascending=False).head(20)
    labels = top["rep_id"].astype(str) + " / " + pd.to_datetime(top["period"]).dt.strftime("%Y-%m")
    ax.barh(labels.iloc[::-1], top["utilization_pct"].iloc[::-1])
    ax.axvline(100, color="red", linestyle="--", label="100% utilization")
    ax.set_title("Highest controlled rep-period capacity utilization")
    ax.set_xlabel("Utilization (%)")
    ax.legend()
    fig.tight_layout()
    path = directory / "capacity_utilization_review.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    files.append(path)
    return files


def _write_reports(
    output_root: Path,
    profile: dict[str, Any],
    clean: dict[str, pd.DataFrame],
    truth: pd.DataFrame,
    clean_features: pd.DataFrame,
    model_results: dict[str, Any],
    capacity_metrics: pd.DataFrame,
    dashboard_tables: dict[str, pd.DataFrame],
    tests_note: str = "Run pytest for the separately recorded automated-test result.",
) -> Path:
    metrics = model_results["metrics_summary"].iloc[0]
    top = model_results["top_k_metrics"].set_index("review_fraction")
    anomaly_counts = truth.groupby("anomaly_type").size().sort_values(ascending=False)
    dataset_counts = {name: len(frame) for name, frame in clean.items()}
    capacity = clean["capacity_calendar"]
    grouped = model_results["group_metrics"].copy()
    type_detection = grouped.loc[
        grouped["grouping"].isin(["anomaly_type", "anomaly_category", "severity"]),
        [
            "grouping",
            "value",
            "positive_support",
            "overall_truth_support",
            "detection_rate_at_threshold",
            "detection_rate_at_top5pct",
            "support_status",
            "evaluation_scope",
        ],
    ]
    organization_detection = grouped.loc[
        grouped["grouping"].isin(["manager_id", "team_id", "territory_id"]),
        [
            "grouping",
            "value",
            "observations",
            "positive_support",
            "precision_at_threshold",
            "recall_at_threshold",
        ],
    ]
    stability = model_results["period_stability"].tail(20)
    false_positive_review = model_results["false_positive_review"]
    dashboard_inventory = pd.DataFrame(
        [
            {"dashboard_dataset": name, "rows": len(frame)}
            for name, frame in sorted(dashboard_tables.items())
        ]
    )
    text = f"""# Final pharma commercial-review controlled benchmark report

Executed source: `{profile['source_file']}` ({profile['input_rows']:,} × {profile['input_columns']}); modeling rows {profile['modeling_rows']:,}.
Coverage: {profile['date_min']} to {profile['date_max']}. The brief's approximate 254,082 × 18 shape matched: **{profile['matches_expected_dimensions']}**.
Currency: source currency unavailable; generated monetary tables use `UNK` and perform no conversion.

## Finalized architecture

The frozen primary review model is **PCA Reconstruction** at rep × month. It retained {model_results['pca_metadata']['retained_components']} components with cumulative explained variance {model_results['pca_metadata']['total_explained_variance']:.4f}. Raw score is mean squared reconstruction error after clean-training median imputation, signed-log compression and robust scaling. Contributions are non-causal reconstruction evidence, not SHAP values.

Selected raw threshold: {metrics['threshold']:.8g}, chosen from the unlabeled validation review budget before test evaluation. Manager review capacity: {metrics['manager_review_fraction']:.1%}.

## Test metrics (controlled synthetic labels)

- ROC-AUC: {metrics.get('roc_auc', float('nan')):.4f}
- PR-AUC / average precision: {metrics.get('pr_auc', float('nan')):.4f}
- Precision / recall / F1 at frozen threshold: {metrics.get('precision', float('nan')):.4f} / {metrics.get('recall', float('nan')):.4f} / {metrics.get('f1', float('nan')):.4f}
- Precision@1/5/10%: {top.loc[0.01, 'precision']:.4f} / {top.loc[0.05, 'precision']:.4f} / {top.loc[0.10, 'precision']:.4f}
- Recall@1/5/10%: {top.loc[0.01, 'recall']:.4f} / {top.loc[0.05, 'recall']:.4f} / {top.loc[0.10, 'recall']:.4f}
- Lift@1/5/10%: {top.loc[0.01, 'lift']:.4f} / {top.loc[0.05, 'lift']:.4f} / {top.loc[0.10, 'lift']:.4f}

Confusion matrix at the frozen threshold (final test):

```
                 Predicted normal  Predicted review
Actual normal    {int(metrics.get('true_negatives', 0)):>16}  {int(metrics.get('false_positives', 0)):>16}
Actual injected  {int(metrics.get('false_negatives', 0)):>16}  {int(metrics.get('true_positives', 0)):>16}
```

Detection by controlled type/category/severity (`no_final_test_support` is reported as N/A, never zero):

```
{type_detection.to_string(index=False)}
```

Detection by manager, team and territory (final holdout):

```
{organization_detection.to_string(index=False)}
```

Score stability across the most recent periods:

```
{stability.to_string(index=False)}
```

False-positive review by leading PCA reconstruction driver:

```
{false_positive_review.to_string(index=False) if not false_positive_review.empty else 'No false positives occurred at the frozen threshold.'}
```

The complete executed tables are persisted under `artifacts/commercial_review/model/`.

## Generated clean datasets

{json.dumps(dataset_counts, indent=2)}

Feature-store shape: {clean_features.shape[0]:,} × {clean_features.shape[1]:,} including identifiers/context; model feature count {model_results['pca_metadata']['feature_count']}.

## Controlled injections

Commercial rep-period injections and capacity overload truth remain in a separate benchmark table. Counts by scenario:

{anomaly_counts.to_string()}

## Capacity

The new deterministic hours calendar preserves the existing normalized workload index as an audit feature while adding working/leave/training/admin/meeting availability, visit/travel/required-coverage hours, utilization, required/available FTE, FTE gap and coverage gaps. Clean high/critical capacity rows: {int(capacity['capacity_risk_band'].isin(['high','critical']).sum())}. Capacity output supports workload review, territory redesign, sharing or further hiring analysis; it is not an automated hiring decision.

Capacity evaluation:

{capacity_metrics.to_string(index=False) if not capacity_metrics.empty else 'No capacity metric table was returned.'}

## Dashboard datasets and visualizations

Compact dashboard inventory:

```
{dashboard_inventory.to_string(index=False)}
```

The existing Streamlit application was extended with seven commercial-review pages: Executive Overview, Manager Review Queue, Rep Anomaly Drill-down, Team and Manager View, Capacity Overview, Model Benchmark View, and Data and Model Health. Generated figures are `clean_vs_injected_score_distribution.png`, `roc_precision_recall_curves.png`, `pca_cumulative_explained_variance.png`, and `capacity_utilization_review.png` under `reports/figures/`.

## Reproduction commands

```powershell
.\.venv\Scripts\python.exe -m field_rep_anomaly.commercial_review.pipeline --config configs/synthetic_data.yaml --input data/raw/pharma-data.csv
.\.venv\Scripts\python.exe -m pytest -q --junitxml=artifacts/reports/commercial_review_tests.xml
.\.venv\Scripts\streamlit.exe run app.py
```

## Automated-test record

{tests_note}

## Implementation and artifact inventory

The extension adds `configs/synthetic_data.yaml`, the `src/field_rep_anomaly/commercial_review/` package, five focused `tests/test_commercial_review_*.py` suites, the synthetic methodology/model card/dashboard-layer documentation, and the compact dashboard/model/report artifacts listed above. It updates `app.py`, `README.md`, packaging/dependency metadata, and ignore rules while preserving the legacy benchmark. The exhaustive generated-file names, hashes and row counts are recorded in `artifacts/commercial_review/output_manifest.csv` and `run_manifest.json`.

## Responsible use and limitations

- The benchmark does **not** prove fraud. A high score is a review candidate or unusual observation.
- All targets, policies, incentives, discounts, visits, CRM, expenses, capacity inputs and injected labels are controlled synthetic or derived data.
- Results require human validation against governed source systems and must not support punitive action without further investigation.
- Capacity outputs support, but never automate, hiring or employment decisions.
- Synthetic relationships simplify real compensation plans, approvals, territories, travel and customer engagement.
- Actual source coverage changes between Poland and Germany remain material context.

"""
    path = output_root / "reports" / "final_benchmark_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def run_commercial_review_pipeline(
    config_path: str | Path,
    input_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Execute all data, model, capacity, evaluation, dashboard, and report phases."""
    started = time.perf_counter()
    repository_root = Path(__file__).resolve().parents[3]
    config_file = Path(config_path).resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    destination_root = Path(output_root).resolve() if output_root else repository_root
    paths = _output_paths(destination_root, config)
    source_path = Path(input_path).resolve() if input_path else _resolve(repository_root, config["project"]["source_input"])
    source_hash_before = file_sha256(source_path)
    seed = int(config["project"]["seed"])
    normalized, profile, mappings, source_quality = profile_and_normalize_source(
        source_path,
        seed=seed,
        currency_code=str(config["project"].get("currency_code", "UNK")),
        generated_day_min=int(config["source"]["generated_transaction_day_min"]),
        generated_day_max=int(config["source"]["generated_transaction_day_max"]),
    )
    source_profile_file = write_json(paths["artifacts"] / "source_profile.json", profile)

    clean = generate_clean_datasets(normalized, config)
    from .capacity import (
        build_capacity_calendar,
        build_capacity_territory_allocation,
        build_capacity_territory_summary,
        evaluate_capacity,
    )
    clean_capacity, coverage = build_capacity_calendar(
        normalized,
        clean["rep_master"],
        clean["customer_master"],
        clean["field_visits"],
        config,
    )
    clean["capacity_calendar"] = clean_capacity
    clean["capacity_customer_drilldown"] = coverage
    clean_territory_allocation = build_capacity_territory_allocation(
        clean_capacity,
        normalized,
        field_visits=clean["field_visits"],
        capacity_customer_drilldown=coverage,
        config=config,
    )
    clean_territory_summary = build_capacity_territory_summary(
        clean_territory_allocation
    )
    clean["capacity_territory_allocation"] = clean_territory_allocation
    clean["capacity_territory_summary"] = clean_territory_summary
    injected, ground_truth = inject_controlled_anomalies(clean, config)
    injected_territory_allocation = build_capacity_territory_allocation(
        injected["capacity_calendar"],
        normalized,
        field_visits=injected["field_visits"],
        capacity_customer_drilldown=injected["capacity_customer_drilldown"],
        config=config,
    )
    injected_territory_summary = build_capacity_territory_summary(
        injected_territory_allocation
    )
    injected["capacity_territory_allocation"] = injected_territory_allocation
    injected["capacity_territory_summary"] = injected_territory_summary

    clean_features, feature_columns, _ = build_feature_store(clean, config)
    injected_features, injected_feature_columns, peer_comparison = build_feature_store(injected, config)
    if feature_columns != injected_feature_columns:
        raise ValueError("Clean and injected feature allowlists differ")

    validation_report = validate_relational_benchmark(
        clean, injected, ground_truth, injected_features, feature_columns, config
    )
    commercial_truth = ground_truth.loc[~ground_truth["anomaly_category"].eq("capacity")].copy()
    # Capacity overloads are evaluated by their own deterministic module.  They
    # must not become negative examples or feature perturbations in the PCA
    # commercial benchmark.
    commercial_injected = {
        name: frame.copy(deep=True) for name, frame in injected.items()
    }
    commercial_injected["capacity_calendar"] = clean_capacity.copy(deep=True)
    commercial_injected["capacity_customer_drilldown"] = coverage.copy(deep=True)
    commercial_injected["capacity_territory_allocation"] = clean_territory_allocation.copy(
        deep=True
    )
    commercial_injected["capacity_territory_summary"] = clean_territory_summary.copy(
        deep=True
    )
    commercial_injected_features, commercial_feature_columns, peer_comparison = build_feature_store(
        commercial_injected, config
    )
    if commercial_feature_columns != feature_columns:
        raise ValueError("Commercial PCA feature allowlist differs from clean feature allowlist")
    model_config = json.loads(json.dumps(config, default=str))
    model_config["model"]["feature_columns"] = feature_columns
    model_config["model"]["group_columns"] = [
        "manager_id", "team_id", "territory_id"
    ]
    model_results = run_finalized_pca(
        clean_features,
        commercial_injected_features,
        commercial_truth,
        model_config,
        output_dir=paths["artifacts"] / "model",
    )
    capacity_metrics = evaluate_capacity(
        clean_capacity,
        injected["capacity_calendar"],
        ground_truth,
        clean_territory_allocation=clean_territory_allocation,
        injected_territory_allocation=injected_territory_allocation,
    )
    if isinstance(capacity_metrics, dict):
        capacity_metrics = pd.DataFrame([capacity_metrics])
    capacity_type_metrics = _capacity_type_metric_rows(
        clean_capacity, injected["capacity_calendar"], ground_truth, capacity_metrics
    )
    model_results["group_metrics"] = pd.concat(
        [model_results["group_metrics"], capacity_type_metrics],
        ignore_index=True,
        sort=False,
    )
    model_results["group_metrics"].to_csv(
        paths["artifacts"] / "model" / "pca_group_metrics.csv", index=False
    )

    source_unchanged = file_sha256(source_path) == source_hash_before
    if not source_unchanged:
        raise RuntimeError("Source CSV fingerprint changed during pipeline execution")

    quality = pd.concat([source_quality, validation_report], ignore_index=True, sort=False)
    metrics_row = model_results["metrics_summary"].iloc[0].to_dict()
    run_manifest: dict[str, Any] = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "input_file_name": source_path.name,
        "input_file_hash": profile["source_sha256"],
        "input_rows": profile["input_rows"],
        "input_columns": profile["input_columns"],
        "modeling_row_count": profile["modeling_rows"],
        "source_date_min": profile["date_min"],
        "source_date_max": profile["date_max"],
        "primary_analytical_grain": "representative x month",
        "model_feature_count": model_results["pca_metadata"]["feature_count"],
        "capacity_methodology": (
            "deterministic hours plus legacy normalized workload with "
            "fractional territory allocation"
        ),
        "key_software_versions": _software_versions(),
        "configuration_hash": config_sha256(config),
        "configuration_file": config_file.name,
        "finalized_model_name": "PCA Reconstruction",
        "model_parameters": {
            "pca_retained_variance": config["model"]["pca_retained_variance"],
            "retained_components": model_results["pca_metadata"]["retained_components"],
            "preprocessing": model_results["pca_metadata"]["preprocessing"],
        },
        "scoring_threshold": model_results["pca_metadata"]["threshold"],
        "manager_review_fraction": config["model"]["manager_review_fraction"],
        "evaluation_metrics": {
            "commercial_pca": metrics_row,
            "capacity": capacity_metrics.iloc[0].to_dict() if not capacity_metrics.empty else {},
        },
        "git_commit_hash": git_commit(repository_root),
        "git_worktree_dirty": git_worktree_dirty(repository_root),
        "implementation_sha256": implementation_sha256(
            sorted((repository_root / "src" / "field_rep_anomaly" / "commercial_review").glob("*.py"))
            + [config_file]
        ),
        "source_unchanged": source_unchanged,
        "manifest_scope": (
            "All generated data, dashboard, model, report, figure, dictionary, and source-profile outputs; "
            "self-referential output_manifest/run_manifest/executed_summary files are excluded"
        ),
    }
    dashboard = build_dashboard_datasets(
        clean,
        injected,
        clean_features,
        commercial_injected_features,
        model_results,
        commercial_truth,
        peer_comparison,
        coverage,
        capacity_metrics,
        quality,
        run_manifest,
    )
    dashboard_files = write_dashboard_datasets(dashboard, paths["dashboard"])
    optional = build_optional_model_dashboard_tables(model_results)
    for filename, frame in optional.items():
        dashboard_files.append(write_dashboard_csv(frame, paths["dashboard"], filename))
    dashboard_validation = validate_dashboard_files(paths["dashboard"], REQUIRED_DASHBOARD_FILES)
    # Preserve the generated-layer row-count checks added by the dashboard
    # builder, then append the file-contract checks.  The data-quality CSV is
    # itself one of those checked files, so record its final (post-append) size.
    final_quality_rows = len(dashboard["dashboard_data_quality.csv"]) + len(dashboard_validation)
    dashboard_validation.loc[
        dashboard_validation["file"].eq("dashboard_data_quality.csv"), "rows"
    ] = final_quality_rows
    quality = pd.concat(
        [
            dashboard["dashboard_data_quality.csv"],
            dashboard_validation.rename(columns={"file": "check_name", "rows": "value"}),
        ],
        ignore_index=True,
        sort=False,
    )
    dashboard["dashboard_data_quality.csv"] = quality
    write_dashboard_csv(quality, paths["dashboard"], "dashboard_data_quality.csv")
    dashboard_counts = {
        filename: len(dashboard[filename])
        for filename in REQUIRED_DASHBOARD_FILES + OPTIONAL_SEMANTIC_DASHBOARD_FILES
        if filename in dashboard
    }
    dashboard_manifest = dashboard["dashboard_run_manifest.csv"].copy()
    dashboard_manifest["output_row_counts"] = json.dumps(dashboard_counts, sort_keys=True)
    dashboard["dashboard_run_manifest.csv"] = dashboard_manifest
    write_dashboard_csv(
        dashboard_manifest, paths["dashboard"], "dashboard_run_manifest.csv"
    )

    generated_files = _write_generated_layers(
        normalized, mappings, clean, injected, clean_features, injected_features,
        ground_truth, paths, config,
    )
    figure_files = _make_figures(model_results, injected["capacity_calendar"], paths["reports"] / "figures")
    report_file = _write_reports(
        destination_root,
        profile,
        clean,
        ground_truth,
        clean_features,
        model_results,
        capacity_metrics,
        {**dashboard, **optional},
        tests_note=_pytest_result_note(repository_root),
    )
    dictionary_datasets = {
        "normalized_source_transactions": normalized,
        **clean,
        "anomaly_ground_truth": ground_truth,
        "feature_store_rep_period": injected_features,
        **{name.removesuffix(".csv"): frame for name, frame in dashboard.items()},
        **{name.removesuffix(".csv"): frame for name, frame in optional.items()},
        **{
            f"model_{name}": frame
            for name, frame in model_results.items()
            if isinstance(frame, pd.DataFrame)
        },
    }
    dictionary_file = generate_data_dictionary(
        paths["docs"] / "commercial_review_data_dictionary.md", dictionary_datasets
    )
    model_artifact_files = sorted(
        path for path in (paths["artifacts"] / "model").glob("*") if path.is_file()
    )
    all_files = (
        generated_files
        + dashboard_files
        + model_artifact_files
        + figure_files
        + [source_profile_file, report_file, dictionary_file]
    )
    output_manifest = build_output_manifest_rows(all_files)
    output_manifest["output_file"] = [
        Path(value).resolve().relative_to(destination_root).as_posix()
        for value in output_manifest["output_file"]
    ]
    output_manifest_path = paths["artifacts"] / "output_manifest.csv"
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.to_csv(output_manifest_path, index=False)
    run_manifest["output_file_names"] = output_manifest["output_file"].tolist()
    run_manifest["output_row_counts"] = {
        str(output_file): (int(row_count) if pd.notna(row_count) else None)
        for output_file, row_count in zip(
            output_manifest["output_file"], output_manifest["output_rows"]
        )
    }
    run_manifest["runtime_seconds"] = time.perf_counter() - started
    run_manifest["source_unchanged"] = file_sha256(source_path) == source_hash_before
    write_json(paths["artifacts"] / "run_manifest.json", run_manifest)
    if not run_manifest["source_unchanged"]:
        raise RuntimeError("Source CSV fingerprint changed during pipeline execution")
    summary = {
        "source_profile": profile,
        "dataset_row_counts": {name: len(frame) for name, frame in clean.items()},
        "injection_type_counts": ground_truth["anomaly_type"].value_counts().to_dict(),
        "feature_store_shape": list(injected_features.shape),
        "model_metrics": metrics_row,
        "pca_metadata": model_results["pca_metadata"],
        "capacity_metrics": capacity_metrics.to_dict("records"),
        "dashboard_row_counts": {name: len(frame) for name, frame in dashboard.items()},
        "run_manifest": run_manifest,
    }
    write_json(paths["artifacts"] / "executed_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/synthetic_data.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    summary = run_commercial_review_pipeline(args.config, args.input, args.output_root)
    print(
        json.dumps(
            {
                "input_dimensions": [
                    summary["source_profile"]["input_rows"],
                    summary["source_profile"]["input_columns"],
                ],
                "feature_store_shape": summary["feature_store_shape"],
                "threshold": summary["pca_metadata"]["threshold"],
                "runtime_seconds": summary["run_manifest"]["runtime_seconds"],
            },
            default=str,
        )
    )


if __name__ == "__main__":
    main()
