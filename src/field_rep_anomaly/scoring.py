"""Human-readable anomaly explanations and representative-level rollups."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .models.dbscan import DBSCANClusteringModel
from .models.kmeans import KMeansClusteringModel


FRIENDLY_NAMES = {
    "total_sales": "Sales",
    "sales_growth": "Sales growth",
    "rolling_sales_growth": "Rolling sales growth",
    "total_quantity": "Quantity",
    "quantity_growth": "Quantity growth",
    "average_price": "Average selling price",
    "sales_per_customer": "Sales per customer",
    "sales_per_call": "Sales per call",
    "customer_coverage_pct": "Customer coverage",
    "total_calls": "Calls",
    "calls_per_working_day": "Calls per working day",
    "activity_efficiency": "Activity efficiency",
    "travel_per_customer": "Travel per customer",
    "workload_index": "Workload",
    "incentive_to_sales_ratio": "Incentive-to-sales ratio",
    "incentive_per_customer": "Incentive per customer",
    "incentive_per_call": "Incentive per call",
    "incentive_variance": "Incentive variance",
    "incentive_variance_pct": "Incentive variance percentage",
    "incentive_to_target_ratio": "Incentive-to-target ratio",
    "sales_vs_peer_median": "Sales versus peer median",
    "incentive_vs_peer_median": "Incentive versus peer median",
    "activity_vs_peer_median": "Activity versus peer median",
    "sales_zscore_within_peer": "Sales within peer group",
    "incentive_zscore_within_peer": "Incentive within peer group",
    "activity_zscore_within_peer": "Activity within peer group",
    "territory_market_potential": "Territory market potential",
    "territory_customer_density": "Territory customer density",
    "opportunity_index": "Territory opportunity",
    "market_potential_adjusted_sales": "Opportunity-adjusted sales",
    "target_attainment_pct": "Target attainment",
    "call_plan_adherence_pct": "Call-plan adherence",
    "product_concentration": "Product concentration",
    "dominant_product_share": "Dominant product share",
}


def _name(feature: str) -> str:
    return FRIENDLY_NAMES.get(feature, feature.replace("_", " ").title())


def kmeans_explanations(
    frame: pd.DataFrame,
    X: np.ndarray,
    model: KMeansClusteringModel,
    feature_names: list[str],
    top_n: int = 5,
) -> list[str]:
    contributions = model.feature_contributions(X)
    labels = model.predict(X)
    residual = np.asarray(X) - model.cluster_centers_[labels]
    cluster_medians = frame.assign(_cluster=labels).groupby("_cluster", observed=True)[feature_names].median(numeric_only=True)
    explanations: list[str] = []
    for row_index in range(len(frame)):
        top = np.argsort(-contributions[row_index])[:top_n]
        reasons = []
        for feature_index in top:
            feature = feature_names[feature_index]
            direction = "above" if residual[row_index, feature_index] >= 0 else "below"
            share = contributions[row_index, feature_index] * 100.0
            original_value = float(pd.to_numeric(frame.iloc[row_index][feature], errors="coerce"))
            median_value = float(cluster_medians.loc[labels[row_index], feature])
            median_delta = original_value - median_value
            reasons.append(
                f"{_name(feature)} {direction} cluster center ({share:.0f}% of centroid distance; cluster-median difference {median_delta:,.2f})"
            )
        explanations.append(" | ".join(reasons))
    return explanations


def dbscan_explanations(
    frame: pd.DataFrame,
    X: np.ndarray,
    model: DBSCANClusteringModel,
    feature_names: list[str],
    top_n: int = 5,
) -> list[str]:
    matrix = np.asarray(X, dtype=float)
    medians = np.median(matrix, axis=0)
    q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
    scale = np.where((q75 - q25) > 1e-9, q75 - q25, 1.0)
    deviations = (matrix - medians) / scale
    distances = model.neighbor_distances(matrix)
    distance_pct = np.searchsorted(np.sort(distances), distances, side="right") / len(distances)
    explanations: list[str] = []
    for row_index in range(len(frame)):
        density = (
            f"Low local density: {model.min_samples}-neighbor distance {distances[row_index]:.2f} "
            f"({distance_pct[row_index]:.0%} percentile)"
        )
        top = np.argsort(-np.abs(deviations[row_index]))[: max(1, top_n - 1)]
        reasons = [density]
        for feature_index in top:
            feature = feature_names[feature_index]
            direction = "above" if deviations[row_index, feature_index] >= 0 else "below"
            reasons.append(f"{_name(feature)} {abs(deviations[row_index, feature_index]):.1f} robust units {direction} population median")
        explanations.append(" | ".join(reasons[:top_n]))
    return explanations


def build_rep_risk_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Roll row-level model scores to one investigation record per representative."""
    context = [
        "rep_id", "territory_id", "country", "city", "sales_manager", "sales_team",
        "product_name", "product_class", "date", "total_sales", "target_attainment_pct",
        "actual_incentive_paid", "anomaly_score", "anomaly_flag", "model", "top_anomaly_drivers",
        "injected_anomaly_flag", "anomaly_type",
    ]
    ordered = scored.sort_values(["rep_id", "anomaly_score"], ascending=[True, False], kind="stable")
    top = ordered.drop_duplicates("rep_id", keep="first")[[column for column in context if column in ordered]].copy()
    aggregates = scored.groupby("rep_id", observed=True).agg(
        total_sales_all_rows=("total_sales", "sum"),
        total_incentive_all_rows=("actual_incentive_paid", "sum"),
        flagged_observations=("anomaly_flag", "sum"),
        observations=("anomaly_flag", "size"),
    ).reset_index()
    result = top.merge(aggregates, on="rep_id", how="left")
    result["flagged_observation_rate"] = result["flagged_observations"] / result["observations"].clip(lower=1)
    return result.sort_values("anomaly_score", ascending=False, kind="stable").reset_index(drop=True)
