"""Benchmark tables, weighted model selection, and narrative reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _minmax(series: pd.Series, higher: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty:
        scaled = pd.Series(0.0, index=series.index)
    elif float(finite.max() - finite.min()) < 1e-12:
        scaled = pd.Series(0.5, index=series.index)
    else:
        scaled = ((values - finite.min()) / (finite.max() - finite.min())).fillna(0.0).clip(0, 1)
    return scaled if higher else 1.0 - scaled


def weighted_model_selection(
    final_metrics: pd.DataFrame,
    segmentation_weights: Mapping[str, float],
    anomaly_weights: Mapping[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score segmentation and anomaly usefulness separately using configured weights."""
    frame = final_metrics.copy().reset_index(drop=True)
    directions = {
        "davies_bouldin_score": False,
        "runtime_seconds": False,
    }
    contributions = []
    for score_name, weights in (("segmentation_score", segmentation_weights), ("anomaly_score", anomaly_weights)):
        total_weight = float(sum(weights.values()))
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            raise ValueError(f"{score_name} weights must sum to 1.0; got {total_weight:.6f}")
        frame[score_name] = 0.0
        for metric, weight in weights.items():
            if metric not in frame:
                raise ValueError(f"Selection metric is missing: {metric}")
            if metric == "runtime_seconds":
                # Use a fixed bounded transform so sub-second timing jitter cannot flip a winner.
                runtime = pd.to_numeric(frame[metric], errors="coerce").clip(lower=0)
                normalised = (1.0 / (1.0 + runtime)).fillna(0.0)
            else:
                normalised = _minmax(frame[metric], higher=directions.get(metric, True))
            contribution = normalised * float(weight)
            frame[score_name] += contribution
            for index, model in enumerate(frame["model"]):
                contributions.append(
                    {
                        "selection_type": score_name,
                        "model": model,
                        "metric": metric,
                        "raw_value": frame.loc[index, metric],
                        "normalised_value": float(normalised.iloc[index]),
                        "weight": float(weight),
                        "weighted_contribution": float(contribution.iloc[index]),
                    }
                )
    best_segmentation = str(frame.loc[frame["segmentation_score"].idxmax(), "model"])
    best_anomaly = str(frame.loc[frame["anomaly_score"].idxmax(), "model"])
    frame["best_segmentation_model"] = frame["model"].eq(best_segmentation)
    frame["best_anomaly_detection_model"] = frame["model"].eq(best_anomaly)
    return frame, pd.DataFrame(contributions)


def build_clustering_benchmark(final_metrics: pd.DataFrame) -> pd.DataFrame:
    """Build the requested Metric × model presentation table."""
    indexed = final_metrics.set_index("model")
    models = ["K-Means", "DBSCAN"]
    descriptors = {
        "K-Means": {
            "Interpretability": "High: compact segments, centroid distance, feature contributions",
            "Strength": "Stable segmentation and continuous distance ranking",
            "Limitation": "Assumes roughly compact clusters and requires k",
        },
        "DBSCAN": {
            "Interpretability": "Medium-high: noise label, density distance, unusual features",
            "Strength": "Finds irregular dense regions and explicit noise without k",
            "Limitation": "Scale/eps sensitive; no native out-of-sample predict",
        },
    }
    metric_map = [
        ("Best Parameters", "best_parameters"),
        ("Number of Clusters", "number_of_clusters"),
        ("Noise %", "noise_percentage"),
        ("Silhouette", "silhouette_score"),
        ("Davies-Bouldin", "davies_bouldin_score"),
        ("Calinski-Harabasz", "calinski_harabasz_score"),
        ("Stability", "stability_score"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("F2", "f2"),
        ("PR-AUC", "pr_auc"),
        ("ROC-AUC", "roc_auc"),
        ("Precision@5%", "precision_at_5pct"),
        ("Recall@5%", "recall_at_5pct"),
        ("Lift@5%", "lift_at_5pct"),
        ("Top-Decile Capture", "top_decile_capture"),
        ("Runtime", "runtime_seconds"),
    ]
    rows = []
    for label, column in metric_map:
        row: dict[str, Any] = {"Metric": label}
        for model in models:
            row[model] = indexed.loc[model, column] if model in indexed.index and column in indexed else np.nan
        rows.append(row)
    for label in ("Interpretability", "Strength", "Limitation"):
        rows.append({"Metric": label, **{model: descriptors[model][label] for model in models}})
    return pd.DataFrame(rows)


def write_execution_reports(
    reports_dir: Path,
    source_metadata: Mapping[str, Any],
    analytical: pd.DataFrame,
    injection_audit: pd.DataFrame,
    selection: pd.DataFrame,
    final_metrics: pd.DataFrame,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    best_seg = str(selection.loc[selection["best_segmentation_model"], "model"].iloc[0])
    best_anom = str(selection.loc[selection["best_anomaly_detection_model"], "model"].iloc[0])
    metric_index = final_metrics.set_index("model")
    best = metric_index.loc[best_anom]
    fallback = bool(source_metadata.get("fallback_used"))
    summary = f"""# Executed benchmark summary

## Data provenance

- Source type: **{source_metadata.get('source_type')}**
- Fallback used: **{'Yes' if fallback else 'No'}**
- Source path: `{source_metadata.get('source_path')}`
- Analytical grain: **Field rep × product × territory × month**
- Analytical observations: **{len(analytical):,}**
- Synthetic reps: **{analytical['rep_id'].nunique():,}**
- Injected anomalies: **{int(analytical['injected_anomaly_flag'].sum()):,} ({analytical['injected_anomaly_flag'].mean():.2%})**

## Weighted conclusions

- Best segmentation model: **{best_seg}**
- Best anomaly-detection model: **{best_anom}**
- {best_anom} precision: **{float(best['precision']):.3f}**
- {best_anom} recall: **{float(best['recall']):.3f}**
- {best_anom} PR-AUC: **{float(best['pr_auc']):.3f}**
- {best_anom} Lift@5%: **{float(best['lift_at_5pct']):.2f}×**

Selection uses the configured weighted framework; synthetic anomaly labels are used only for benchmark evaluation and anomaly-model comparison, never as clustering features or unsupervised tuning inputs.

## Guardrail

Flags identify observations for business review. They do not establish fraud, misconduct, or incorrect payment.
"""
    (reports_dir / "executed_benchmark_summary.md").write_text(summary, encoding="utf-8")
    type_counts = injection_audit["anomaly_type"].value_counts().sort_index()
    injection_report = "# Controlled anomaly injection\n\n" + (
        f"Injected {len(injection_audit):,} labeled anomalies across {len(type_counts)} types. Severity varies continuously and the before/after audit is stored in `data/processed/anomaly_injection_audit.csv`.\n\n"
        + "\n".join(f"- `{name}`: {count}" for name, count in type_counts.items())
        + "\n\nEvaluation labels (`injected_anomaly_flag`, `anomaly_type`, `anomaly_severity`) are excluded from preprocessing and model features.\n"
    )
    (reports_dir / "anomaly_injection_report.md").write_text(injection_report, encoding="utf-8")
    selection_payload = {
        "best_segmentation_model": best_seg,
        "best_anomaly_detection_model": best_anom,
        "selection_rows": selection.replace({np.nan: None}).to_dict(orient="records"),
        "caution": "Synthetic-label benchmark performance is demo evidence, not proof of production generalisation.",
    }
    with (reports_dir / "model_selection.json").open("w", encoding="utf-8") as handle:
        json.dump(selection_payload, handle, indent=2, default=str)
