"""Unsupervised parameter tuning and clustering-quality metrics."""

from __future__ import annotations

import itertools
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors


def _metric_sample(X: np.ndarray, labels: np.ndarray, sample_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(X) <= sample_size:
        return X, labels
    rng = np.random.default_rng(seed)
    index = rng.choice(len(X), size=sample_size, replace=False)
    return X[index], labels[index]


def clustering_metrics(X: np.ndarray, labels: np.ndarray, sample_size: int = 2000, seed: int = 42) -> dict[str, float | int]:
    """Calculate quality metrics with noise excluded from internal indices."""
    matrix = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)
    non_noise = labels != -1
    clusters = sorted(set(labels[non_noise].tolist()))
    sizes = pd.Series(labels[non_noise]).value_counts() if non_noise.any() else pd.Series(dtype=int)
    denominator = max(len(labels), 1)
    result: dict[str, float | int] = {
        "number_of_clusters": int(len(clusters)),
        "noise_percentage": float((~non_noise).mean() * 100.0),
        "smallest_cluster_pct": float(sizes.min() / denominator * 100.0) if len(sizes) else 0.0,
        "largest_cluster_pct": float(sizes.max() / denominator * 100.0) if len(sizes) else 0.0,
        "cluster_balance": float(sizes.min() / sizes.max()) if len(sizes) and sizes.max() else 0.0,
        "silhouette_score": np.nan,
        "davies_bouldin_score": np.nan,
        "calinski_harabasz_score": np.nan,
    }
    if len(clusters) >= 2 and non_noise.sum() > len(clusters):
        clean_X = matrix[non_noise]
        clean_labels = labels[non_noise]
        sampled_X, sampled_labels = _metric_sample(clean_X, clean_labels, sample_size, seed)
        if len(set(sampled_labels)) >= 2 and len(sampled_X) > len(set(sampled_labels)):
            result["silhouette_score"] = float(silhouette_score(sampled_X, sampled_labels))
            result["davies_bouldin_score"] = float(davies_bouldin_score(sampled_X, sampled_labels))
            result["calinski_harabasz_score"] = float(calinski_harabasz_score(sampled_X, sampled_labels))
    return result


def _normalise(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return pd.Series(0.0, index=series.index)
    low, high = float(finite.min()), float(finite.max())
    if abs(high - low) < 1e-12:
        scaled = pd.Series(0.5, index=series.index)
    else:
        scaled = (numeric - low) / (high - low)
    scaled = scaled.fillna(0.0).clip(0, 1)
    return scaled if higher_is_better else 1.0 - scaled


def _add_selection_score(results: pd.DataFrame, valid_mask: pd.Series) -> pd.DataFrame:
    frame = results.copy()
    interpretability = frame["interpretability_score"] if "interpretability_score" in frame else pd.Series(0.5, index=frame.index)
    frame["selection_score"] = (
        0.30 * _normalise(frame["silhouette_score"])
        + 0.16 * _normalise(frame["davies_bouldin_score"], higher_is_better=False)
        + 0.12 * _normalise(np.log1p(frame["calinski_harabasz_score"].clip(lower=0)))
        + 0.20 * _normalise(frame["cluster_balance"])
        + 0.10 * _normalise(frame["stability_score"])
        + 0.04 * _normalise(frame["runtime_seconds"], higher_is_better=False)
        + 0.08 * _normalise(interpretability)
    )
    frame.loc[~valid_mask, "selection_score"] = -1.0
    return frame


def _kmeans_stability(X: np.ndarray, k: int, repeats: int, seed: int, n_init: int, max_iter: int) -> float:
    labels = [
        KMeans(n_clusters=k, random_state=seed + run, n_init=n_init, max_iter=max_iter).fit_predict(X)
        for run in range(repeats)
    ]
    pairs = [adjusted_rand_score(labels[a], labels[b]) for a, b in itertools.combinations(range(repeats), 2)]
    return float(np.mean(pairs)) if pairs else 1.0


def tune_kmeans(X: np.ndarray, settings: Mapping[str, Any], seed: int = 42) -> tuple[pd.DataFrame, dict[str, Any]]:
    matrix = np.asarray(X, dtype=float)
    k_min = max(2, int(settings.get("k_min", 2)))
    k_max = min(int(settings.get("k_max", 12)), len(matrix) - 1, max(2, len(np.unique(matrix, axis=0)) - 1))
    n_init = int(settings.get("n_init", 20))
    max_iter = int(settings.get("max_iter", 500))
    repeats = int(settings.get("stability_repeats", 4))
    sample_size = int(settings.get("metric_sample_size", 2000))
    min_cluster_pct = float(settings.get("min_cluster_pct", 1.0))
    max_cluster_pct = float(settings.get("max_cluster_pct", 90.0))
    rows: list[dict[str, Any]] = []
    for k in range(k_min, k_max + 1):
        start = time.perf_counter()
        model = KMeans(n_clusters=k, random_state=seed, n_init=n_init, max_iter=max_iter)
        labels = model.fit_predict(matrix)
        runtime = time.perf_counter() - start
        metrics = clustering_metrics(matrix, labels, sample_size=sample_size, seed=seed)
        stability = _kmeans_stability(matrix, k, repeats, seed, n_init, max_iter)
        reasons = []
        if metrics["smallest_cluster_pct"] < min_cluster_pct:
            reasons.append(f"smallest cluster below {min_cluster_pct:.1f}%")
        if metrics["largest_cluster_pct"] > max_cluster_pct:
            reasons.append(f"largest cluster exceeds {max_cluster_pct:.1f}%")
        rows.append(
            {
                "model": "K-Means",
                "configuration_id": f"kmeans_k_{k}",
                "k": k,
                "eps": np.nan,
                "min_samples": np.nan,
                "inertia": float(model.inertia_),
                "runtime_seconds": float(runtime),
                "stability_score": stability,
                "interpretability_score": float(1.0 / (1.0 + 0.08 * max(k - 2, 0))),
                "valid_configuration": len(reasons) == 0,
                "rejection_reason": "; ".join(reasons),
                **metrics,
            }
        )
    raw = pd.DataFrame(rows)
    results = _add_selection_score(raw, raw["valid_configuration"].astype(bool))
    valid = results[results["valid_configuration"]]
    if valid.empty:
        best = results.sort_values(["cluster_balance", "selection_score"], ascending=False).iloc[0]
        results.loc[best.name, "selected_despite_rejection"] = True
    else:
        best = valid.sort_values(["selection_score", "silhouette_score"], ascending=False).iloc[0]
    return results, {"n_clusters": int(best["k"]), "configuration_id": str(best["configuration_id"])}


def calculate_k_distance_curve(X: np.ndarray, min_samples: int) -> np.ndarray:
    count = min(max(2, int(min_samples)), len(X))
    distances, _ = NearestNeighbors(n_neighbors=count).fit(X).kneighbors(X)
    return np.sort(distances[:, -1])


def _dbscan_stability(X: np.ndarray, eps: float, min_samples: int, repeats: int, perturbation_std: float, seed: int) -> float:
    base = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    rng = np.random.default_rng(seed + 509)
    scores = []
    for _ in range(max(1, repeats - 1)):
        perturbed = X + rng.normal(0.0, perturbation_std, size=X.shape)
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(perturbed)
        scores.append(adjusted_rand_score(base, labels))
    return float(np.mean(scores)) if scores else 1.0


def tune_dbscan(X: np.ndarray, settings: Mapping[str, Any], seed: int = 42) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    matrix = np.asarray(X, dtype=float)
    min_samples_values = [int(value) for value in settings.get("min_samples", [5, 10, 15])]
    quantiles = [float(value) for value in settings.get("eps_quantiles", [0.75, 0.82, 0.88, 0.94])]
    max_noise = float(settings.get("max_noise_pct", 55.0))
    min_cluster = float(settings.get("min_cluster_pct", 0.5))
    repeats = int(settings.get("stability_repeats", 4))
    perturbation_std = float(settings.get("perturbation_std", 0.02))
    sample_size = int(settings.get("metric_sample_size", 2000))
    rows: list[dict[str, Any]] = []
    curves: list[pd.DataFrame] = []
    for min_samples in min_samples_values:
        curve = calculate_k_distance_curve(matrix, min_samples)
        curves.append(pd.DataFrame({"rank": np.arange(1, len(curve) + 1), "k_distance": curve, "min_samples": min_samples}))
        eps_values = sorted({max(float(np.quantile(curve, q)), 1e-6) for q in quantiles})
        for eps in eps_values:
            start = time.perf_counter()
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(matrix)
            runtime = time.perf_counter() - start
            metrics = clustering_metrics(matrix, labels, sample_size=sample_size, seed=seed)
            reasons = []
            if metrics["number_of_clusters"] < 2:
                reasons.append("fewer than two non-noise clusters")
            if metrics["noise_percentage"] >= 99.9:
                reasons.append("all observations classified as noise")
            elif metrics["noise_percentage"] > max_noise:
                reasons.append(f"noise exceeds {max_noise:.1f}%")
            if metrics["smallest_cluster_pct"] < min_cluster:
                reasons.append(f"smallest cluster below {min_cluster:.1f}%")
            stability = _dbscan_stability(matrix, eps, min_samples, repeats, perturbation_std, seed)
            rows.append(
                {
                    "model": "DBSCAN",
                    "configuration_id": f"dbscan_eps_{eps:.5f}_min_{min_samples}",
                    "k": np.nan,
                    "eps": eps,
                    "min_samples": min_samples,
                    "inertia": np.nan,
                    "runtime_seconds": float(runtime),
                    "stability_score": stability,
                    "interpretability_score": float(1.0 / (1.0 + 0.08 * max(int(metrics["number_of_clusters"]) - 2, 0) + float(metrics["noise_percentage"]) / 100.0)),
                    "deterministic_fixed_input": True,
                    "valid_configuration": len(reasons) == 0,
                    "rejection_reason": "; ".join(reasons),
                    **metrics,
                }
            )
    raw = pd.DataFrame(rows)
    results = _add_selection_score(raw, raw["valid_configuration"].astype(bool))
    valid = results[results["valid_configuration"]]
    if valid.empty:
        # Preserve an honest fallback: choose the least-degenerate configuration and mark it.
        candidates = results[results["number_of_clusters"] >= 2]
        if candidates.empty:
            candidates = results
        best = candidates.sort_values(["noise_percentage", "silhouette_score"], ascending=[True, False]).iloc[0]
        results.loc[best.name, "selected_despite_rejection"] = True
    else:
        best = valid.sort_values(["selection_score", "silhouette_score"], ascending=False).iloc[0]
    params = {"eps": float(best["eps"]), "min_samples": int(best["min_samples"]), "configuration_id": str(best["configuration_id"])}
    return results, params, pd.concat(curves, ignore_index=True)
