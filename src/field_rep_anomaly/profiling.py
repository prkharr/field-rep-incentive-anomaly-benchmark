"""Business-readable cluster profiling derived from actual cluster statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd


PROFILE_METRICS = {
    "total_sales": "sales",
    "total_quantity": "quantity",
    "unique_customers": "customers",
    "total_calls": "activity",
    "target_attainment_pct": "target attainment",
    "actual_incentive_paid": "incentive",
    "territory_market_potential": "territory potential",
    "customer_coverage_pct": "customer coverage",
}


def _mode(series: pd.Series) -> str:
    values = series.dropna().astype(str)
    if values.empty:
        return "Unknown"
    counts = values.value_counts()
    return str(sorted(counts[counts == counts.max()].index.tolist())[0])


def _interpret(row: pd.Series, profile_means: pd.DataFrame, overall_anomaly_rate: float) -> str:
    descriptors: list[str] = []
    for metric, label in PROFILE_METRICS.items():
        means = profile_means[metric]
        spread = float(means.std(ddof=0))
        z = (float(row[f"mean_{metric}"]) - float(means.mean())) / spread if spread > 1e-12 else 0.0
        if z >= 0.65:
            descriptors.append(f"higher {label}")
        elif z <= -0.65:
            descriptors.append(f"lower {label}")
    if float(row["anomaly_rate"]) >= overall_anomaly_rate + 0.03:
        descriptors.append("elevated review-label rate")
    if not descriptors:
        descriptors.append("near-population-average operating pattern")
    prefix = "Density-isolated; " if int(row["cluster"]) == -1 else ""
    return prefix + "; ".join(descriptors[:5]).capitalize() + "."


def create_cluster_profiles(data: pd.DataFrame, labels: np.ndarray, model_name: str) -> pd.DataFrame:
    frame = data.copy()
    frame["cluster"] = np.asarray(labels, dtype=int)
    overall_mean = frame[list(PROFILE_METRICS)].mean(numeric_only=True)
    overall_median = frame[list(PROFILE_METRICS)].median(numeric_only=True)
    overall_anomaly_rate = float(frame["injected_anomaly_flag"].mean())
    rows = []
    for cluster, group in frame.groupby("cluster", observed=True):
        row: dict[str, object] = {
            "model": model_name,
            "cluster": int(cluster),
            "population": int(len(group)),
            "population_pct": float(len(group) / len(frame) * 100.0),
            "total_sales_sum": float(group["total_sales"].sum()),
            "anomaly_rate": float(group["injected_anomaly_flag"].mean()),
            "dominant_product": _mode(group["product_name"]),
            "dominant_geography": _mode(group["city"]),
        }
        for metric in PROFILE_METRICS:
            mean_value = float(group[metric].mean())
            median_value = float(group[metric].median())
            row[f"mean_{metric}"] = mean_value
            row[f"median_{metric}"] = median_value
            row[f"{metric}_vs_population_mean_pct"] = (
                (mean_value - float(overall_mean[metric])) / max(abs(float(overall_mean[metric])), 1e-9) * 100.0
            )
            row[f"{metric}_vs_population_median_pct"] = (
                (median_value - float(overall_median[metric])) / max(abs(float(overall_median[metric])), 1e-9) * 100.0
            )
        rows.append(row)
    profiles = pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
    means_for_interpretation = profiles[[f"mean_{metric}" for metric in PROFILE_METRICS]].rename(
        columns={f"mean_{metric}": metric for metric in PROFILE_METRICS}
    )
    profiles["business_interpretation"] = [
        _interpret(row, means_for_interpretation, overall_anomaly_rate) for _, row in profiles.iterrows()
    ]
    return profiles
