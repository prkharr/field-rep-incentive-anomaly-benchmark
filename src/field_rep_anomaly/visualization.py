"""Static visualizations for the clustering and anomaly benchmark.

The public :func:`generate_all_plots` function deliberately treats plots as
independent reporting artifacts.  A missing column or a degenerate metric
therefore produces an informative placeholder for that plot instead of
preventing the remaining artifacts from being written.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

# The pipeline is commonly run on CI and on headless hackathon environments.
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


_PALETTE = sns.color_palette("colorblind", 10)
_MISSING_MESSAGE = "No valid data were available for this visualization."

_PLOT_NAMES = (
    "data_distributions",
    "correlation_matrix",
    "kmeans_elbow",
    "kmeans_silhouette",
    "dbscan_k_distance",
    "dbscan_parameter_comparison",
    "pca_cluster_visualization",
    "cluster_size_chart",
    "cluster_profile_heatmap",
    "anomaly_score_distributions",
    "roc_curves",
    "precision_recall_curves",
    "lift_curve",
    "top_decile_capture",
    "sales_vs_incentive",
    "sales_vs_activity",
    "region_anomaly_comparison",
    "product_anomaly_comparison",
    "team_anomaly_comparison",
    "final_model_benchmark",
)


def _frame(value: Any) -> pd.DataFrame:
    """Return a defensive DataFrame copy, or an empty frame for bad input."""
    if isinstance(value, pd.DataFrame):
        return value.copy()
    try:
        return pd.DataFrame(value).copy() if value is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _normalise_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _column(frame: pd.DataFrame, candidates: Sequence[str]) -> Any | None:
    """Find the first candidate column, allowing case/punctuation differences."""
    if frame.empty and not len(frame.columns):
        return None
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    lookup: dict[str, Any] = {}
    for col in frame.columns:
        lookup.setdefault(_normalise_name(col), col)
    for candidate in candidates:
        match = lookup.get(_normalise_name(candidate))
        if match is not None:
            return match
    return None


def _numeric(values: Any) -> pd.Series:
    """Coerce one-dimensional values to finite floats while preserving length."""
    try:
        if isinstance(values, pd.Series):
            series = values
        else:
            array = np.asarray(values)
            if array.ndim > 1:
                array = array.reshape(-1)
            series = pd.Series(array)
    except Exception:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _finite(values: Any) -> np.ndarray:
    series = _numeric(values).dropna()
    return series.to_numpy(dtype=float)


def _display_name(value: Any) -> str:
    text = str(value).strip().replace("_", " ")
    return re.sub(r"\s+", " ", text).title() or "Model"


def _message_figure(title: str, message: str = _MISSING_MESSAGE) -> Figure:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=14, weight="semibold")
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        color="0.38",
        fontsize=11,
        wrap=True,
        transform=ax.transAxes,
    )
    return fig


def _combined_view(analytical: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    """Combine row-aligned analytical and scoring columns without changing grain."""
    left = _frame(analytical).reset_index(drop=True)
    right = _frame(scored).reset_index(drop=True)
    if left.empty and not len(left.columns):
        return right
    if right.empty and not len(right.columns):
        return left

    aligned_right = right
    if len(left) != len(right):
        # Some pipelines keep one long scored block per model.  Select a complete
        # K-Means block (or another complete block) when it is row-aligned.
        model_col = _column(right, ("model", "model_name", "algorithm"))
        if model_col is not None:
            groups = list(right.groupby(model_col, sort=False, dropna=False))
            groups.sort(key=lambda item: 0 if "kmeans" in _normalise_name(item[0]).replace("_", "") else 1)
            complete = [group for _, group in groups if len(group) == len(left)]
            if complete:
                aligned_right = complete[0].reset_index(drop=True)

    result = left.copy()
    if len(result) == len(aligned_right):
        for col in aligned_right.columns:
            if col not in result.columns:
                result[col] = aligned_right[col].to_numpy()
    return result


def _kmeans_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Prefer K-Means rows when a long, multi-model scored table is supplied."""
    model_col = _column(frame, ("model", "model_name", "algorithm"))
    if model_col is None:
        return frame
    mask = (
        frame[model_col]
        .astype("string")
        .map(lambda value: _normalise_name(value).replace("_", ""))
        .str.contains("kmeans", na=False)
    )
    selected = frame.loc[mask]
    return selected.reset_index(drop=True) if not selected.empty else frame


def _business_numeric_columns(frame: pd.DataFrame, limit: int = 12) -> list[Any]:
    preferred = (
        "total_sales",
        "actual_incentive_paid",
        "calculated_incentive",
        "total_incentive",
        "total_quantity",
        "total_calls",
        "target_attainment_pct",
        "customer_coverage_pct",
        "unique_customers",
        "sales_growth",
        "incentive_to_sales_ratio",
        "manual_override_amount",
        "territory_market_potential",
        "workload_index",
        "opportunity_index",
    )
    selected: list[Any] = []
    for name in preferred:
        col = _column(frame, (name,))
        if col is not None and col not in selected and _numeric(frame[col]).notna().any():
            selected.append(col)
        if len(selected) >= limit:
            return selected

    candidates: list[tuple[float, Any]] = []
    for col in frame.columns:
        if col in selected:
            continue
        values = _numeric(frame[col])
        valid = values.dropna()
        if valid.empty or valid.nunique() < 2:
            continue
        variance = float(valid.var())
        candidates.append((variance if np.isfinite(variance) else 0.0, col))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected.extend(col for _, col in candidates[: max(0, limit - len(selected))])
    return selected


def _trimmed(values: np.ndarray) -> np.ndarray:
    """Trim only extreme plotting tails; all calculations remain unchanged."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 30 or np.unique(values).size < 3:
        return values
    low, high = np.nanquantile(values, [0.01, 0.99])
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        return values
    trimmed = values[(values >= low) & (values <= high)]
    return trimmed if len(trimmed) >= max(10, int(0.5 * len(values))) else values


def _plot_data_distributions(frame: pd.DataFrame) -> Figure:
    columns = _business_numeric_columns(frame, limit=6)
    if not columns:
        return _message_figure("Data distributions")
    ncols = 3 if len(columns) > 2 else len(columns)
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)
    for ax, col, color in zip(axes.flat, columns, _PALETTE):
        values = _trimmed(_finite(frame[col]))
        if values.size:
            bins = int(np.clip(np.sqrt(values.size), 12, 45))
            ax.hist(values, bins=bins, color=color, alpha=0.82, edgecolor="white", linewidth=0.4)
            if np.unique(values).size > 1:
                ax.axvline(np.nanmedian(values), color="0.2", linestyle="--", linewidth=1.2, label="Median")
                ax.legend(frameon=False, fontsize=8)
        else:
            ax.text(0.5, 0.5, "No valid values", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(_display_name(col))
        ax.set_xlabel("")
        ax.set_ylabel("Observations")
    for ax in axes.flat[len(columns) :]:
        ax.axis("off")
    fig.suptitle("Analytical feature distributions", fontsize=15, weight="semibold", y=1.01)
    fig.tight_layout()
    return fig


def _plot_correlation_matrix(frame: pd.DataFrame) -> Figure:
    columns = _business_numeric_columns(frame, limit=12)
    usable: list[Any] = []
    numeric = pd.DataFrame(index=frame.index)
    for col in columns:
        values = _numeric(frame[col])
        if values.notna().sum() >= 2 and values.nunique(dropna=True) > 1:
            numeric[str(col)] = values
            usable.append(col)
    if len(usable) < 2:
        return _message_figure("Correlation matrix", "At least two varying numeric features are required.")
    corr = numeric.corr(min_periods=2)
    corr = corr.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if corr.shape[0] < 2:
        return _message_figure("Correlation matrix", "At least two correlated numeric features are required.")
    labels = [_display_name(col) for col in corr.columns]
    size = max(8.5, 0.72 * len(labels) + 3.5)
    fig, ax = plt.subplots(figsize=(size, size * 0.82))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.35,
        linecolor="white",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Pearson correlation", "shrink": 0.78},
    )
    ax.set_title("Feature correlation matrix", fontsize=15, pad=14, weight="semibold")
    ax.tick_params(axis="x", rotation=48, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    return fig


def _tuning_xy(
    frame: pd.DataFrame,
    x_candidates: Sequence[str],
    y_candidates: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    x_col = _column(frame, x_candidates)
    y_col = _column(frame, y_candidates)
    if x_col is None or y_col is None:
        return np.array([]), np.array([])
    data = pd.DataFrame({"x": _numeric(frame[x_col]), "y": _numeric(frame[y_col])}).dropna()
    if data.empty:
        return np.array([]), np.array([])
    data = data.groupby("x", as_index=False, sort=True)["y"].mean()
    return data["x"].to_numpy(dtype=float), data["y"].to_numpy(dtype=float)


def _line_metric_figure(
    frame: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    x_candidates: Sequence[str],
    y_candidates: Sequence[str],
    best: str,
) -> Figure:
    x, y = _tuning_xy(frame, x_candidates, y_candidates)
    if not len(x):
        return _message_figure(title)
    fig, ax = plt.subplots(figsize=(9, 5.7))
    ax.plot(x, y, marker="o", linewidth=2.0, color=_PALETTE[0])
    finite = np.isfinite(y)
    if finite.any():
        positions = np.flatnonzero(finite)
        chosen = positions[np.argmin(y[finite])] if best == "min" else positions[np.argmax(y[finite])]
        ax.scatter([x[chosen]], [y[chosen]], s=95, marker="*", color=_PALETTE[3], zorder=4, label="Best observed")
        ax.legend(frameon=False)
    if np.allclose(x, np.round(x), equal_nan=True):
        ax.set_xticks(x)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=15, pad=12, weight="semibold")
    fig.tight_layout()
    return fig


def _plot_dbscan_k_distance(scored: pd.DataFrame, tuning: pd.DataFrame) -> Figure:
    distance_col = _column(
        scored,
        (
            "dbscan_neighbor_distance",
            "dbscan_neighbour_distance",
            "neighbor_distance",
            "nearest_neighbor_distance",
            "k_distance",
        ),
    )
    values = _finite(scored[distance_col]) if distance_col is not None else np.array([])
    if not len(values):
        fallback = _column(tuning, ("k_distance", "neighbor_distance", "dbscan_neighbor_distance"))
        values = _finite(tuning[fallback]) if fallback is not None else np.array([])
    if not len(values):
        return _message_figure("DBSCAN k-distance curve")
    values = np.sort(values)
    fig, ax = plt.subplots(figsize=(9, 5.7))
    percentile = np.linspace(0, 100, len(values), endpoint=True)
    ax.plot(percentile, values, color=_PALETTE[1], linewidth=2)
    for quantile, linestyle in ((90, "--"), (95, ":")):
        value = float(np.nanpercentile(values, quantile))
        ax.axhline(value, color="0.35", linestyle=linestyle, linewidth=1, label=f"{quantile}th pct: {value:.3g}")
    ax.set_xlabel("Observations ordered by neighbor distance (percentile)")
    ax.set_ylabel("Neighbor distance")
    ax.set_title("DBSCAN k-nearest-neighbor distance", fontsize=15, pad=12, weight="semibold")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _plot_dbscan_parameter_comparison(frame: pd.DataFrame) -> Figure:
    eps_col = _column(frame, ("eps", "epsilon"))
    min_col = _column(frame, ("min_samples", "minimum_samples"))
    silhouette_col = _column(frame, ("silhouette_score", "silhouette"))
    noise_col = _column(frame, ("noise_percentage", "noise_pct", "noise_percent", "percentage_noise"))
    if eps_col is None or (silhouette_col is None and noise_col is None):
        return _message_figure("DBSCAN parameter comparison")

    data = pd.DataFrame({"eps": _numeric(frame[eps_col])})
    data["min_samples"] = _numeric(frame[min_col]) if min_col is not None else 0
    if silhouette_col is not None:
        data["silhouette"] = _numeric(frame[silhouette_col])
    if noise_col is not None:
        data["noise"] = _numeric(frame[noise_col])
    data = data.dropna(subset=["eps"])
    if data.empty:
        return _message_figure("DBSCAN parameter comparison")

    available = [("silhouette", "Silhouette score"), ("noise", "Noise (%)")]
    available = [(key, label) for key, label in available if key in data and data[key].notna().any()]
    fig, axes = plt.subplots(1, len(available), figsize=(7.2 * len(available), 5.5), squeeze=False)
    groups = list(data.groupby("min_samples", dropna=False, sort=True))
    for ax, (metric, ylabel) in zip(axes.flat, available):
        for idx, (minimum, group) in enumerate(groups):
            clean = group[["eps", metric]].dropna().sort_values("eps")
            if clean.empty:
                continue
            label = f"min_samples={int(minimum) if float(minimum).is_integer() else minimum:g}"
            ax.plot(
                clean["eps"],
                clean[metric],
                marker="o",
                linewidth=1.8,
                color=_PALETTE[idx % len(_PALETTE)],
                label=label,
            )
        ax.set_xlabel("eps")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        if groups:
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("DBSCAN tuning comparison", fontsize=15, weight="semibold", y=1.01)
    fig.tight_layout()
    return fig


def _plot_pca_clusters(scored: pd.DataFrame) -> Figure:
    scored = _kmeans_rows(scored)
    x_col = _column(scored, ("pca_1", "pca1", "principal_component_1", "pc1"))
    y_col = _column(scored, ("pca_2", "pca2", "principal_component_2", "pc2"))
    cluster_col = _column(scored, ("kmeans_cluster", "k_means_cluster", "cluster", "cluster_label"))
    if x_col is None or y_col is None or cluster_col is None:
        return _message_figure("PCA cluster visualization")
    data = pd.DataFrame(
        {"x": _numeric(scored[x_col]), "y": _numeric(scored[y_col]), "cluster": scored[cluster_col].astype("string")}
    ).dropna(subset=["x", "y", "cluster"])
    if data.empty:
        return _message_figure("PCA cluster visualization")
    if len(data) > 5000:
        data = data.sample(5000, random_state=42)
    labels = sorted(data["cluster"].unique(), key=lambda value: str(value))
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for idx, label in enumerate(labels):
        group = data[data["cluster"] == label]
        ax.scatter(
            group["x"],
            group["y"],
            s=24,
            alpha=0.66,
            color=_PALETTE[idx % len(_PALETTE)],
            edgecolors="none",
            label=f"Cluster {label}",
            rasterized=True,
        )
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.set_title("K-Means clusters in PCA space", fontsize=15, pad=12, weight="semibold")
    if len(labels) <= 15:
        ax.legend(frameon=False, ncol=max(1, math.ceil(len(labels) / 7)), fontsize=8)
    fig.tight_layout()
    return fig


def _plot_cluster_sizes(scored: pd.DataFrame, profiles: pd.DataFrame) -> Figure:
    scored = _kmeans_rows(scored)
    profiles = _kmeans_rows(profiles)
    cluster_col = _column(scored, ("kmeans_cluster", "k_means_cluster", "cluster", "cluster_label"))
    if cluster_col is not None:
        labels = scored[cluster_col].dropna().astype("string")
        counts = labels.value_counts().sort_index()
    else:
        profile_cluster = _column(profiles, ("kmeans_cluster", "cluster", "cluster_id", "cluster_label"))
        size_col = _column(profiles, ("population", "cluster_size", "count", "n_observations", "size"))
        if profile_cluster is None or size_col is None:
            return _message_figure("Cluster size chart")
        temp = pd.DataFrame(
            {"cluster": profiles[profile_cluster].astype("string"), "count": _numeric(profiles[size_col])}
        ).dropna()
        counts = temp.groupby("cluster")["count"].sum().sort_index()
    if counts.empty:
        return _message_figure("Cluster size chart")
    fig, ax = plt.subplots(figsize=(9, 5.8))
    bars = ax.bar([str(value) for value in counts.index], counts.to_numpy(), color=_PALETTE[0], alpha=0.85)
    total = float(counts.sum())
    for bar, count in zip(bars, counts.to_numpy()):
        pct = 100 * float(count) / total if total else 0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(count):,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("K-Means cluster")
    ax.set_ylabel("Observations")
    ax.set_title("K-Means cluster sizes", fontsize=15, pad=12, weight="semibold")
    ax.margins(y=0.16)
    fig.tight_layout()
    return fig


def _plot_cluster_profiles(profiles: pd.DataFrame) -> Figure:
    if profiles.empty:
        return _message_figure("Cluster profile heatmap")
    profiles = _kmeans_rows(profiles)
    cluster_col = _column(profiles, ("kmeans_cluster", "cluster", "cluster_id", "cluster_label"))
    excluded_names = {
        _normalise_name(value)
        for value in (
            cluster_col,
            "population",
            "cluster_size",
            "count",
            "anomaly_count",
            "business_interpretation",
        )
        if value is not None
    }
    profile_metric_names = (
        "mean_total_sales",
        "mean_total_quantity",
        "mean_unique_customers",
        "mean_total_calls",
        "mean_target_attainment_pct",
        "mean_actual_incentive_paid",
        "mean_territory_market_potential",
        "mean_customer_coverage_pct",
        "anomaly_rate",
    )
    preferred = [
        col
        for name in profile_metric_names
        if (col := _column(profiles, (name,))) is not None and _numeric(profiles[col]).notna().any()
    ]
    for col in _business_numeric_columns(profiles, limit=14):
        if col not in preferred:
            preferred.append(col)
        if len(preferred) >= 14:
            break
    columns = [col for col in preferred if _normalise_name(col) not in excluded_names]
    if not columns:
        return _message_figure("Cluster profile heatmap")
    numeric = pd.DataFrame({str(col): _numeric(profiles[col]) for col in columns})
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.empty:
        return _message_figure("Cluster profile heatmap")
    means = numeric.mean(axis=0)
    std = numeric.std(axis=0, ddof=0).replace(0, np.nan)
    standardised = ((numeric - means) / std).fillna(0.0).clip(-3, 3)
    if cluster_col is not None:
        standardised.index = [f"Cluster {value}" for value in profiles[cluster_col].astype("string").fillna("Unknown")]
    else:
        standardised.index = [f"Cluster {idx}" for idx in range(len(standardised))]
    standardised.columns = [_display_name(col) for col in standardised.columns]
    width = max(10, 0.78 * standardised.shape[1] + 3)
    height = max(4.8, 0.55 * standardised.shape[0] + 2.6)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        standardised,
        ax=ax,
        cmap="vlag",
        center=0,
        vmin=-3,
        vmax=3,
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "Standard deviations from profile mean", "shrink": 0.8},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Standardized K-Means cluster profiles", fontsize=15, pad=13, weight="semibold")
    ax.tick_params(axis="x", rotation=48, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    fig.tight_layout()
    return fig


def _score_mapping(scores_by_model: Mapping[str, Any] | None, scored: pd.DataFrame) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    if isinstance(scores_by_model, Mapping):
        for name, values in scores_by_model.items():
            array = _numeric(values).to_numpy(dtype=float)
            if array.size and np.isfinite(array).any():
                result[str(name)] = array

    # Make standalone plotting useful even when the caller omitted the mapping.
    model_col = _column(scored, ("model", "model_name", "algorithm"))
    generic_score_col = _column(scored, ("anomaly_score",))
    if model_col is not None and generic_score_col is not None:
        for model_name, group in scored.groupby(model_col, sort=False, dropna=False):
            key = str(model_name)
            if any(_normalise_name(existing) == _normalise_name(key) for existing in result):
                continue
            array = _numeric(group[generic_score_col]).to_numpy(dtype=float)
            if array.size and np.isfinite(array).any():
                result[key] = array
    for col in scored.columns:
        normalised = _normalise_name(col)
        if not (normalised.endswith("anomaly_score") or normalised in {"anomaly_score", "score"}):
            continue
        if col == generic_score_col and model_col is not None:
            continue
        model_name = re.sub(r"_?anomaly_score$", "", normalised) or "Anomaly model"
        if any(_normalise_name(existing) == model_name for existing in result):
            continue
        array = _numeric(scored[col]).to_numpy(dtype=float)
        if array.size and np.isfinite(array).any():
            result[_display_name(model_name)] = array
    return result


def _aligned_scores(y_true: Any, score_map: Mapping[str, np.ndarray]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    y = _numeric(y_true).to_numpy(dtype=float)
    aligned: list[tuple[str, np.ndarray, np.ndarray]] = []
    for name, scores in score_map.items():
        score = np.asarray(scores, dtype=float).reshape(-1)
        n = min(len(y), len(score))
        if n == 0:
            continue
        mask = np.isfinite(y[:n]) & np.isfinite(score[:n])
        if not mask.any():
            continue
        labels = (y[:n][mask] > 0).astype(int)
        aligned.append((str(name), labels, score[:n][mask]))
    return aligned


def _plot_score_distributions(y_true: Any, score_map: Mapping[str, np.ndarray]) -> Figure:
    aligned = _aligned_scores(y_true, score_map)
    if not aligned:
        # Labels are optional for a distribution; retain raw scores when possible.
        raw = [(name, _finite(values)) for name, values in score_map.items()]
        raw = [(name, values) for name, values in raw if len(values)]
        if not raw:
            return _message_figure("Anomaly score distributions")
        ncols = min(2, len(raw))
        nrows = math.ceil(len(raw) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.3 * nrows), squeeze=False)
        for ax, (name, scores), color in zip(axes.flat, raw, _PALETTE):
            ax.hist(scores, bins=int(np.clip(np.sqrt(len(scores)), 12, 45)), color=color, alpha=0.82)
            ax.set_title(_display_name(name))
            ax.set_xlabel("Anomaly score")
            ax.set_ylabel("Observations")
        for ax in axes.flat[len(raw) :]:
            ax.axis("off")
        fig.suptitle("Anomaly score distributions", fontsize=15, weight="semibold", y=1.01)
        fig.tight_layout()
        return fig

    ncols = min(2, len(aligned))
    nrows = math.ceil(len(aligned) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.4 * nrows), squeeze=False)
    for ax, (name, labels, scores) in zip(axes.flat, aligned):
        bins = int(np.clip(np.sqrt(len(scores)), 12, 45))
        normal = scores[labels == 0]
        anomalous = scores[labels == 1]
        if len(normal):
            ax.hist(normal, bins=bins, density=True, alpha=0.55, color=_PALETTE[0], label="Non-injected")
        if len(anomalous):
            ax.hist(anomalous, bins=bins, density=True, alpha=0.58, color=_PALETTE[3], label="Injected anomaly")
        ax.set_title(_display_name(name))
        ax.set_xlabel("Anomaly score")
        ax.set_ylabel("Density")
        ax.legend(frameon=False, fontsize=8)
    for ax in axes.flat[len(aligned) :]:
        ax.axis("off")
    fig.suptitle("Anomaly score distributions by ground truth", fontsize=15, weight="semibold", y=1.01)
    fig.tight_layout()
    return fig


def _plot_roc(y_true: Any, score_map: Mapping[str, np.ndarray]) -> Figure:
    aligned = _aligned_scores(y_true, score_map)
    valid = [(name, labels, scores) for name, labels, scores in aligned if np.unique(labels).size == 2]
    if not valid:
        return _message_figure("ROC curves", "ROC curves require both positive and negative ground-truth labels.")
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for idx, (name, labels, scores) in enumerate(valid):
        fpr, tpr, _ = roc_curve(labels, scores)
        auc_value = roc_auc_score(labels, scores)
        ax.plot(fpr, tpr, linewidth=2, color=_PALETTE[idx % len(_PALETTE)], label=f"{_display_name(name)} (AUC={auc_value:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.45", linewidth=1.2, label="Random")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False positive rate", ylabel="True positive rate")
    ax.set_title("Anomaly-detection ROC curves", fontsize=15, pad=12, weight="semibold")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def _plot_precision_recall(y_true: Any, score_map: Mapping[str, np.ndarray]) -> Figure:
    aligned = _aligned_scores(y_true, score_map)
    valid = [(name, labels, scores) for name, labels, scores in aligned if np.unique(labels).size == 2]
    if not valid:
        return _message_figure(
            "Precision-recall curves",
            "Precision-recall curves require both positive and negative ground-truth labels.",
        )
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    prevalence = float(np.mean(valid[0][1]))
    for idx, (name, labels, scores) in enumerate(valid):
        precision, recall, _ = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)
        ax.plot(recall, precision, linewidth=2, color=_PALETTE[idx % len(_PALETTE)], label=f"{_display_name(name)} (AP={ap:.3f})")
    ax.axhline(prevalence, linestyle="--", color="0.45", linewidth=1.2, label=f"Prevalence ({prevalence:.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="Recall", ylabel="Precision")
    ax.set_title("Anomaly-detection precision-recall curves", fontsize=15, pad=12, weight="semibold")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _plot_lift(y_true: Any, score_map: Mapping[str, np.ndarray]) -> Figure:
    aligned = _aligned_scores(y_true, score_map)
    valid = [(name, labels, scores) for name, labels, scores in aligned if labels.sum() > 0 and len(labels) > 1]
    if not valid:
        return _message_figure("Lift curve", "Lift requires at least one positive ground-truth anomaly.")
    fig, ax = plt.subplots(figsize=(8.8, 6.1))
    for idx, (name, labels, scores) in enumerate(valid):
        order = np.argsort(-scores, kind="stable")
        ranked = labels[order]
        ranks = np.arange(1, len(ranked) + 1, dtype=float)
        population_fraction = ranks / len(ranked)
        base_rate = ranked.mean()
        lift = (np.cumsum(ranked) / ranks) / base_rate
        ax.plot(population_fraction * 100, lift, linewidth=2, color=_PALETTE[idx % len(_PALETTE)], label=_display_name(name))
    ax.axhline(1, color="0.45", linestyle="--", linewidth=1.2, label="Random baseline")
    ax.axvline(5, color="0.65", linestyle=":", linewidth=1)
    ax.axvline(10, color="0.65", linestyle=":", linewidth=1)
    ax.set_xlabel("Population reviewed (%)")
    ax.set_ylabel("Cumulative lift")
    ax.set_title("Cumulative anomaly lift", fontsize=15, pad=12, weight="semibold")
    ax.set_xlim(0, 100)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _plot_top_decile_capture(y_true: Any, score_map: Mapping[str, np.ndarray]) -> Figure:
    aligned = _aligned_scores(y_true, score_map)
    rows: list[tuple[str, float]] = []
    for name, labels, scores in aligned:
        positives = int(labels.sum())
        if positives == 0:
            continue
        reviewed = max(1, int(math.ceil(0.10 * len(labels))))
        order = np.argsort(-scores, kind="stable")[:reviewed]
        rows.append((_display_name(name), 100 * float(labels[order].sum()) / positives))
    if not rows:
        return _message_figure("Top-decile capture", "Top-decile capture requires positive ground-truth anomalies.")
    fig, ax = plt.subplots(figsize=(8.6, 5.7))
    names, values = zip(*rows)
    bars = ax.bar(names, values, color=[_PALETTE[idx % len(_PALETTE)] for idx in range(len(rows))], alpha=0.86)
    ax.axhline(10, linestyle="--", linewidth=1.2, color="0.45", label="Random expectation (10%)")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Injected anomalies captured (%)")
    ax.set_ylim(0, max(105, max(values) * 1.15))
    ax.set_title("Anomaly capture in the top-ranked decile", fontsize=15, pad=12, weight="semibold")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _anomaly_signal(frame: pd.DataFrame) -> tuple[pd.Series, str, bool] | None:
    """Return a row-level anomaly signal, label, and whether it is binary."""
    flag_candidates = (
        "selected_anomaly_flag",
        "final_anomaly_flag",
        "ensemble_anomaly_flag",
        "consensus_anomaly_flag",
        "predicted_anomaly_flag",
        "is_anomaly",
        "anomaly_flag",
        "kmeans_anomaly_flag",
        "kmeans_is_anomaly",
        "dbscan_anomaly_flag",
        "dbscan_is_anomaly",
    )
    flag = _column(frame, flag_candidates)
    if flag is not None:
        values = _numeric(frame[flag])
        if values.notna().any():
            return (values.fillna(0).gt(0).astype(float), "Review candidate", True)

    score_cols = [
        col
        for col in frame.columns
        if _normalise_name(col).endswith("anomaly_score") or _normalise_name(col) == "anomaly_score"
    ]
    numeric_scores = [_numeric(frame[col]) for col in score_cols if _numeric(frame[col]).notna().any()]
    if numeric_scores:
        signal = pd.concat(numeric_scores, axis=1).max(axis=1, skipna=True)
        if signal.notna().any():
            return (signal, "Anomaly score", False)

    injected = _column(frame, ("injected_anomaly_flag", "ground_truth_anomaly", "y_true"))
    if injected is not None:
        values = _numeric(frame[injected])
        if values.notna().any():
            return (values.fillna(0).gt(0).astype(float), "Injected anomaly", True)
    return None


def _plot_scatter(
    frame: pd.DataFrame,
    *,
    x_candidates: Sequence[str],
    y_candidates: Sequence[str],
    title: str,
    xlabel: str,
    ylabel: str,
) -> Figure:
    x_col = _column(frame, x_candidates)
    y_col = _column(frame, y_candidates)
    if x_col is None or y_col is None:
        return _message_figure(title)
    data = pd.DataFrame({"x": _numeric(frame[x_col]), "y": _numeric(frame[y_col])})
    signal = _anomaly_signal(frame)
    if signal is not None:
        data["signal"] = signal[0]
    data = data.dropna(subset=["x", "y"])
    if data.empty:
        return _message_figure(title)
    if len(data) > 6000:
        data = data.sample(6000, random_state=42)

    fig, ax = plt.subplots(figsize=(9, 6.2))
    if signal is not None and "signal" in data and data["signal"].notna().any():
        _, signal_label, binary = signal
        if binary:
            normal = data[data["signal"].fillna(0) <= 0]
            flagged = data[data["signal"].fillna(0) > 0]
            if len(normal):
                ax.scatter(normal["x"], normal["y"], s=25, alpha=0.48, color=_PALETTE[0], edgecolors="none", label="Other")
            if len(flagged):
                ax.scatter(flagged["x"], flagged["y"], s=38, alpha=0.8, color=_PALETTE[3], edgecolors="white", linewidths=0.25, label=signal_label)
            ax.legend(frameon=False)
        else:
            plotted = ax.scatter(
                data["x"],
                data["y"],
                c=data["signal"],
                cmap="viridis",
                s=27,
                alpha=0.68,
                edgecolors="none",
                rasterized=True,
            )
            fig.colorbar(plotted, ax=ax, label=signal_label, shrink=0.82)
    else:
        ax.scatter(data["x"], data["y"], s=27, alpha=0.58, color=_PALETTE[0], edgecolors="none", rasterized=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=15, pad=12, weight="semibold")
    fig.tight_layout()
    return fig


def _plot_category_comparison(
    frame: pd.DataFrame,
    *,
    candidates: Sequence[str],
    title: str,
    axis_label: str,
) -> Figure:
    category_col = _column(frame, candidates)
    signal = _anomaly_signal(frame)
    if category_col is None or signal is None:
        return _message_figure(title)
    indicator, signal_label, binary = signal
    if not binary:
        valid = indicator.dropna()
        if valid.empty:
            return _message_figure(title)
        threshold = float(valid.quantile(0.90))
        indicator = indicator.ge(threshold).astype(float) if valid.nunique() > 1 else pd.Series(0.0, index=indicator.index)
        signal_label = "Top-decile anomaly score"
    category = frame[category_col].astype("string").fillna("Unknown")
    data = pd.DataFrame({"category": category, "flag": indicator}).dropna(subset=["flag"])
    if data.empty:
        return _message_figure(title)
    grouped = data.groupby("category", observed=True)["flag"].agg(["mean", "size"])
    if len(grouped) > 15:
        grouped = grouped.nlargest(15, "size")
    grouped = grouped.sort_values("mean", ascending=True)
    if grouped.empty:
        return _message_figure(title)

    fig, ax = plt.subplots(figsize=(9.5, max(5, 0.38 * len(grouped) + 2.3)))
    values = grouped["mean"].to_numpy(dtype=float) * 100
    labels = [str(label)[:42] for label in grouped.index]
    bars = ax.barh(labels, values, color=_PALETTE[0], alpha=0.85)
    overall = float(data["flag"].mean() * 100)
    ax.axvline(overall, color=_PALETTE[3], linestyle="--", linewidth=1.3, label=f"Overall: {overall:.1f}%")
    for bar, value, count in zip(bars, values, grouped["size"].to_numpy()):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {value:.1f}% (n={int(count):,})", va="center", fontsize=8)
    ax.set_xlabel(f"{signal_label} rate (%)")
    ax.set_ylabel(axis_label)
    ax.set_title(title, fontsize=15, pad=12, weight="semibold")
    ax.legend(frameon=False)
    ax.margins(x=0.20)
    fig.tight_layout()
    return fig


def _benchmark_wide(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce either model-row or Metric/K-Means/DBSCAN tables to model rows."""
    if frame.empty:
        return pd.DataFrame()
    model_col = _column(frame, ("model", "model_name", "algorithm", "method"))
    if model_col is not None:
        result = frame.copy()
        result = result.rename(columns={model_col: "model"})
        return result
    metric_col = _column(frame, ("metric", "measure", "statistic"))
    if metric_col is None or len(frame.columns) < 2:
        return pd.DataFrame()
    value_cols = [col for col in frame.columns if col != metric_col]
    melted = frame.melt(id_vars=[metric_col], value_vars=value_cols, var_name="model", value_name="value")
    melted["metric_key"] = melted[metric_col].map(_normalise_name)
    melted["value"] = _numeric(melted["value"])
    wide = melted.pivot_table(index="model", columns="metric_key", values="value", aggfunc="first").reset_index()
    wide.columns.name = None
    return wide


def _metric_column(frame: pd.DataFrame, aliases: Sequence[str]) -> Any | None:
    return _column(frame, aliases)


def _plot_final_benchmark(frame: pd.DataFrame) -> Figure:
    wide = _benchmark_wide(frame)
    if wide.empty:
        return _message_figure("Final model benchmark")
    model_col = _column(wide, ("model", "model_name", "algorithm", "method"))
    if model_col is None:
        return _message_figure("Final model benchmark")

    metric_specs = (
        ("Precision", ("precision",)),
        ("Recall", ("recall",)),
        ("F1", ("f1", "f1_score")),
        ("F2", ("f2", "f2_score")),
        ("PR-AUC", ("pr_auc", "average_precision")),
        ("ROC-AUC", ("roc_auc", "roc_auc_score")),
        ("Lift@5%", ("lift_at_5pct", "lift_5pct", "lift_at_5", "lift_5")),
        ("Top-decile capture", ("top_decile_capture", "anomaly_capture_in_top_decile")),
        ("Stability", ("stability_score", "stability")),
        ("Silhouette", ("silhouette_score", "silhouette")),
    )
    chosen: list[tuple[str, Any]] = []
    for label, aliases in metric_specs:
        col = _metric_column(wide, aliases)
        if col is not None and _numeric(wide[col]).notna().any():
            chosen.append((label, col))
    if not chosen:
        # Last-resort support for custom numeric benchmark metrics.
        for col in wide.columns:
            if col == model_col:
                continue
            if _numeric(wide[col]).notna().any():
                chosen.append((_display_name(col), col))
            if len(chosen) == 8:
                break
    if not chosen:
        return _message_figure("Final model benchmark")

    records: list[dict[str, Any]] = []
    for _, row in wide.iterrows():
        for label, col in chosen:
            value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if pd.notna(value) and np.isfinite(value):
                records.append({"Model": _display_name(row[model_col]), "Metric": label, "Value": float(value)})
    tidy = pd.DataFrame(records)
    if tidy.empty:
        return _message_figure("Final model benchmark")

    lift_mask = tidy["Metric"].str.contains("Lift", case=False, na=False)
    bounded = tidy.loc[~lift_mask]
    lift = tidy.loc[lift_mask]
    if not lift.empty and not bounded.empty:
        fig, axes = plt.subplots(
            1, 2, figsize=(max(12, 1.02 * len(chosen) + 6), 6.3),
            gridspec_kw={"width_ratios": [max(len(bounded["Metric"].unique()), 3), 1.5]},
        )
        ax, lift_ax = axes
        sns.barplot(data=lift, x="Metric", y="Value", hue="Model", ax=lift_ax, palette="colorblind", errorbar=None)
        lift_ax.set_ylabel("Lift multiplier (×)")
        lift_ax.set_xlabel("")
        lift_ax.tick_params(axis="x", rotation=25)
        if lift_ax.legend_ is not None:
            lift_ax.legend_.remove()
        ax.set_ylim(0, max(1.05, float(bounded["Value"].max()) * 1.08))
    else:
        fig, ax = plt.subplots(figsize=(max(10, 1.08 * len(chosen) + 5), 6.3))
    main_data = bounded if not bounded.empty else tidy
    sns.barplot(data=main_data, x="Metric", y="Value", hue="Model", ax=ax, palette="colorblind", errorbar=None)
    ax.set_xlabel("")
    ax.set_ylabel("Score (0–1)")
    fig.suptitle("Final K-Means and DBSCAN benchmark", fontsize=15, weight="semibold", y=1.01)
    ax.tick_params(axis="x", rotation=38)
    ax.legend(frameon=False, title="Model", ncol=2)
    fig.tight_layout()
    return fig


def _save_one(
    name: str,
    factory: Callable[[], Figure],
    plots_dir: Path,
    dpi: int,
) -> Path:
    path = plots_dir / f"{name}.png"
    fig: Figure | None = None
    try:
        fig = factory()
        if not isinstance(fig, Figure):
            raise TypeError("plot factory did not return a matplotlib Figure")
    except Exception as exc:  # A reporting artifact must not abort the pipeline.
        if fig is not None:
            plt.close(fig)
        reason = re.sub(r"\s+", " ", str(exc)).strip()
        if len(reason) > 180:
            reason = reason[:177] + "..."
        message = "This plot could not be rendered from the supplied data."
        if reason:
            message += f"\n\nReason: {reason}"
        fig = _message_figure(_display_name(name), message)
    try:
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)
    return path


def generate_all_plots(
    analytical: pd.DataFrame,
    scored: pd.DataFrame,
    kmeans_tuning: pd.DataFrame,
    dbscan_tuning: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
    benchmark_model_rows: pd.DataFrame,
    y_true: np.ndarray,
    scores_by_model: dict[str, np.ndarray],
    plots_dir: Path,
    dpi: int = 140,
) -> list[Path]:
    """Generate the complete static benchmark plot suite.

    Parameters are intentionally tables/arrays rather than fitted estimators so
    report generation remains decoupled from model training.  The returned
    paths follow a stable order and every requested filename is attempted.  If
    an input required by one chart is absent, that PNG contains an explanatory
    placeholder and the rest of the suite continues.
    """
    analytical_df = _frame(analytical)
    scored_df = _frame(scored)
    kmeans_df = _frame(kmeans_tuning)
    dbscan_df = _frame(dbscan_tuning)
    profiles_df = _frame(cluster_profiles)
    benchmark_df = _frame(benchmark_model_rows)
    view = _combined_view(analytical_df, scored_df)
    score_map = _score_mapping(scores_by_model, scored_df)

    destination = Path(plots_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    try:
        plot_dpi = int(dpi)
    except (TypeError, ValueError):
        plot_dpi = 140
    plot_dpi = int(np.clip(plot_dpi, 72, 600))

    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    factories: dict[str, Callable[[], Figure]] = {
        "data_distributions": lambda: _plot_data_distributions(analytical_df),
        "correlation_matrix": lambda: _plot_correlation_matrix(analytical_df),
        "kmeans_elbow": lambda: _line_metric_figure(
            kmeans_df,
            title="K-Means elbow curve",
            ylabel="Inertia",
            x_candidates=("k", "n_clusters", "number_of_clusters"),
            y_candidates=("inertia", "within_cluster_sum_of_squares", "wcss"),
            best="min",
        ),
        "kmeans_silhouette": lambda: _line_metric_figure(
            kmeans_df,
            title="K-Means silhouette comparison",
            ylabel="Silhouette score",
            x_candidates=("k", "n_clusters", "number_of_clusters"),
            y_candidates=("silhouette_score", "silhouette"),
            best="max",
        ),
        "dbscan_k_distance": lambda: _plot_dbscan_k_distance(scored_df, dbscan_df),
        "dbscan_parameter_comparison": lambda: _plot_dbscan_parameter_comparison(dbscan_df),
        "pca_cluster_visualization": lambda: _plot_pca_clusters(scored_df),
        "cluster_size_chart": lambda: _plot_cluster_sizes(scored_df, profiles_df),
        "cluster_profile_heatmap": lambda: _plot_cluster_profiles(profiles_df),
        "anomaly_score_distributions": lambda: _plot_score_distributions(y_true, score_map),
        "roc_curves": lambda: _plot_roc(y_true, score_map),
        "precision_recall_curves": lambda: _plot_precision_recall(y_true, score_map),
        "lift_curve": lambda: _plot_lift(y_true, score_map),
        "top_decile_capture": lambda: _plot_top_decile_capture(y_true, score_map),
        "sales_vs_incentive": lambda: _plot_scatter(
            view,
            x_candidates=("total_sales", "sales", "net_sales"),
            y_candidates=("actual_incentive_paid", "total_incentive", "incentive_paid", "calculated_incentive"),
            title="Sales versus incentive payout",
            xlabel="Total sales",
            ylabel="Actual incentive paid",
        ),
        "sales_vs_activity": lambda: _plot_scatter(
            view,
            x_candidates=("total_sales", "sales", "net_sales"),
            y_candidates=("total_calls", "activity_count", "field_calls", "calls"),
            title="Sales versus field activity",
            xlabel="Total sales",
            ylabel="Total calls",
        ),
        "region_anomaly_comparison": lambda: _plot_category_comparison(
            view,
            candidates=("region", "country", "territory_id", "territory", "city"),
            title="Anomaly comparison by region",
            axis_label="Region",
        ),
        "product_anomaly_comparison": lambda: _plot_category_comparison(
            view,
            candidates=("product_name", "dominant_product", "product", "product_class"),
            title="Anomaly comparison by product",
            axis_label="Product",
        ),
        "team_anomaly_comparison": lambda: _plot_category_comparison(
            view,
            candidates=("sales_team", "team", "team_name", "manager_id", "sales_manager"),
            title="Anomaly comparison by sales team",
            axis_label="Sales team",
        ),
        "final_model_benchmark": lambda: _plot_final_benchmark(benchmark_df),
    }

    paths = [_save_one(name, factories[name], destination, plot_dpi) for name in _PLOT_NAMES]
    return paths


__all__ = ["generate_all_plots"]
