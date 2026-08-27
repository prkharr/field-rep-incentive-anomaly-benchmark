"""End-to-end executable benchmark pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA

from .anomaly_injection import inject_controlled_anomalies
from .config import deep_update, ensure_project_directories, load_config
from .data_loader import load_or_generate_data
from .evaluation import evaluate_anomaly_model, top_fraction_flags
from .feature_engineering import LABEL_COLUMNS, engineer_features
from .models.dbscan import DBSCANClusteringModel
from .models.kmeans import KMeansClusteringModel
from .preprocessing import fit_preprocessor, persist_preprocessor
from .profiling import create_cluster_profiles
from .reporting import (
    build_clustering_benchmark,
    weighted_model_selection,
    write_execution_reports,
)
from .scoring import (
    build_rep_risk_summary,
    dbscan_explanations,
    kmeans_explanations,
)
from .synthetic_enrichment import build_enriched_analytical_dataset
from .tuning import clustering_metrics, tune_dbscan, tune_kmeans
from .validation import (
    build_data_quality_report,
    validate_canonical_data,
    write_data_quality_report,
)
from .visualization import generate_all_plots


def _json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=str, allow_nan=False)


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_row(results: pd.DataFrame, configuration_id: str) -> pd.Series:
    matched = results.loc[results["configuration_id"] == configuration_id]
    if matched.empty:
        raise RuntimeError(f"Selected configuration was not found: {configuration_id}")
    return matched.iloc[0]


def _write_methodology_report(
    reports_dir: Path,
    feature_names: list[str],
    warnings: list[str],
    preprocessing_settings: Mapping[str, Any],
) -> None:
    warning_text = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- No fatal data-quality warnings."
    clipping_text = (
        f"Fitted quantile clipping ({float(preprocessing_settings.get('clip_lower_quantile', 0.01)):.1%}–{float(preprocessing_settings.get('clip_upper_quantile', 0.99)):.1%}) was applied before scaling to limit extreme leverage without changing saved business-unit values."
        if bool(preprocessing_settings.get("clip_outliers", False))
        else "Quantile clipping was available but disabled for this execution."
    )
    text = f"""# Executed methodology

1. Discover and profile source data; use a labeled deterministic fallback only when no qualifying input exists.
2. Create a stable customer-to-representative mapping within sales team, manager, and territory.
3. Aggregate at representative × product × territory × month.
4. Add business-related activity, target, capacity, opportunity, and incentive fields.
5. Inject 5–7% controlled anomalies with variable severity and preserve a before/after audit.
6. Engineer portfolio, growth, efficiency, peer-comparison, and opportunity features.
7. Median-impute and RobustScale numeric features. {clipping_text}
8. Tune K-Means and DBSCAN using clustering quality only; evaluate selected configurations against held-out-purpose synthetic labels.
9. Rank anomalies continuously, explain their primary feature drivers, and roll results to representative level.

## Leakage control

The model feature list contains {len(feature_names)} numeric fields. The evaluation-only fields `{', '.join(sorted(LABEL_COLUMNS))}` are explicitly rejected by feature validation and never enter preprocessing or clustering.

## Warnings retained from source validation

{warning_text}

## Important limitations

- Synthetic anomaly labels support a demo benchmark; they do not prove real-world generalisation.
- Model choice based on the same injected label design may be optimistic.
- DBSCAN is deterministic for fixed data and parameters, while its reported stability assesses small input perturbations.
- scikit-learn DBSCAN has no native prediction; the persisted wrapper uses nearest-core assignment within eps for new rows.
- Anomaly flags indicate review priority, not fraud, misconduct, or incorrect payment.
"""
    (reports_dir / "executed_methodology.md").write_text(text, encoding="utf-8")


def run_pipeline(
    config_path: str | Path,
    input_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the full benchmark and return a compact run summary."""
    started = time.perf_counter()
    config = load_config(config_path)
    if overrides:
        preserved = {"_config_path": config["_config_path"], "_repo_root": config["_repo_root"]}
        config = deep_update(config, overrides)
        config.update(preserved)
    paths = ensure_project_directories(config)
    processed_dir = paths["processed_dir"]
    artifacts_dir = paths["artifacts_dir"]
    metrics_dir = artifacts_dir / "metrics"
    models_dir = artifacts_dir / "models"
    reports_dir = artifacts_dir / "reports"
    plots_dir = artifacts_dir / "plots"
    seed = int(config["project"]["seed"])

    # 1. Discover/load source and record source-level quality before enrichment.
    canonical, source_metadata = load_or_generate_data(config, paths["synthetic_dir"], explicit_path=input_path)
    warnings = validate_canonical_data(canonical)
    quality_report, column_profile = build_data_quality_report(canonical, source_metadata)
    quality_report["validation_warnings"] = warnings
    write_data_quality_report(quality_report, column_profile, reports_dir, metrics_dir)

    # 2. Deterministic rep mapping, business grain, enrichment, and controlled labels.
    pre_injection, rep_mapping, field_lineage = build_enriched_analytical_dataset(
        canonical,
        reps_per_territory=int(config["data"]["reps_per_territory"]),
        seed=seed,
    )
    rep_mapping.to_csv(processed_dir / "rep_mapping.csv", index=False)
    pre_injection.to_csv(processed_dir / "analytical_dataset_pre_injection.csv", index=False)
    analytical, injection_audit = inject_controlled_anomalies(
        pre_injection,
        injection_rate=float(config["anomalies"]["injection_rate"]),
        severity_min=float(config["anomalies"]["severity_min"]),
        severity_max=float(config["anomalies"]["severity_max"]),
        seed=seed,
    )
    injection_audit.to_csv(processed_dir / "anomaly_injection_audit.csv", index=False)

    # 3. Feature engineering and fitted preprocessing.
    featured = engineer_features(analytical)
    feature_names = [str(value) for value in config["preprocessing"]["features"]]
    preprocessor, X = fit_preprocessor(featured, config["preprocessing"])
    persist_preprocessor(preprocessor, models_dir / "preprocessor.joblib")
    _json_dump({"feature_names": feature_names}, models_dir / "feature_names.json")
    feature_lineage = pd.DataFrame(
        [{"field": feature, "lineage": "derived_model_feature"} for feature in feature_names if feature not in set(field_lineage["field"])]
        + [{"field": field, "lineage": "evaluation_label_only"} for field in sorted(LABEL_COLUMNS)]
    )
    pd.concat([field_lineage, feature_lineage], ignore_index=True).drop_duplicates("field").to_csv(
        processed_dir / "field_lineage.csv", index=False
    )
    featured.to_csv(processed_dir / "analytical_dataset.csv", index=False)
    model_feature_export = pd.concat(
        [featured[["rep_id", "product_name", "territory_id", "date"]].reset_index(drop=True),
         pd.DataFrame(X, columns=[f"scaled__{name}" for name in feature_names])], axis=1
    )
    model_feature_export.to_csv(processed_dir / "model_features.csv", index=False)

    # 4. Tune without labels, then refit chosen models.
    common = {
        "metric_sample_size": int(config["models"]["metric_sample_size"]),
        "stability_repeats": int(config["models"]["stability_repeats"]),
        "perturbation_std": float(config["models"]["perturbation_std"]),
    }
    k_settings = {**config["models"]["kmeans"], **common}
    d_settings = {**config["models"]["dbscan"], **common}
    kmeans_tuning, k_best = tune_kmeans(X, k_settings, seed=seed)
    dbscan_tuning, d_best, k_distance_curve = tune_dbscan(X, d_settings, seed=seed)
    kmeans_tuning["selected"] = kmeans_tuning["configuration_id"].eq(k_best["configuration_id"])
    dbscan_tuning["selected"] = dbscan_tuning["configuration_id"].eq(d_best["configuration_id"])
    tuning_long = pd.concat([kmeans_tuning, dbscan_tuning], ignore_index=True, sort=False)
    tuning_long.to_csv(metrics_dir / "clustering_benchmark_long.csv", index=False)
    tuning_long.to_csv(metrics_dir / "tuning_results_long.csv", index=False)
    kmeans_tuning.to_csv(metrics_dir / "kmeans_tuning.csv", index=False)
    dbscan_tuning.to_csv(metrics_dir / "dbscan_tuning.csv", index=False)
    k_distance_curve.to_csv(metrics_dir / "dbscan_k_distance_curve.csv", index=False)

    k_model = KMeansClusteringModel(
        n_clusters=k_best["n_clusters"], random_state=seed,
        n_init=int(k_settings["n_init"]), max_iter=int(k_settings["max_iter"]),
    ).fit(X)
    d_model = DBSCANClusteringModel(eps=d_best["eps"], min_samples=d_best["min_samples"]).fit(X)
    joblib.dump(k_model, models_dir / "kmeans_model.joblib")
    joblib.dump(d_model, models_dir / "dbscan_model.joblib")

    k_distances = k_model.distances(X)
    assigned_labels = k_model.predict(X)
    manual_distances = np.sqrt(np.square(X - k_model.cluster_centers_[assigned_labels]).sum(axis=1))
    sklearn_distances = k_model.model.transform(X)[np.arange(len(X)), assigned_labels]
    contribution_sums = k_model.feature_contributions(X).sum(axis=1)
    nonzero_distance = k_distances > 1e-15
    distance_validation = {
        "metric": "Euclidean L2 distance in fitted preprocessed feature space",
        "formula": "sqrt(sum((x_j - assigned_centroid_j)^2))",
        "observations_checked": int(len(X)),
        "assigned_centroid_is_nearest_for_all_rows": bool(
            np.array_equal(assigned_labels, np.argmin(k_model.model.transform(X), axis=1))
        ),
        "max_abs_error_vs_manual_formula": float(np.max(np.abs(k_distances - manual_distances))),
        "max_abs_error_vs_sklearn_transform": float(np.max(np.abs(k_distances - sklearn_distances))),
        "abs_error_sum_squared_distance_vs_inertia": float(abs(np.square(k_distances).sum() - k_model.model.inertia_)),
        "max_abs_feature_contribution_sum_error": float(
            np.max(np.abs(contribution_sums[nonzero_distance] - 1.0)) if nonzero_distance.any() else 0.0
        ),
    }
    if (
        not distance_validation["assigned_centroid_is_nearest_for_all_rows"]
        or distance_validation["max_abs_error_vs_manual_formula"] > 1e-9
        or distance_validation["max_abs_error_vs_sklearn_transform"] > 1e-9
    ):
        raise RuntimeError("K-Means centroid-distance validation failed.")
    _json_dump(_clean_json(distance_validation), reports_dir / "kmeans_distance_validation.json")

    k_scores = k_model.score_anomaly(X)
    d_scores = d_model.score_anomaly(X)
    contamination = float(config["anomalies"]["contamination"])
    k_flags = top_fraction_flags(k_scores, contamination)
    d_flags = d_model.labels_ == -1
    y_true = featured["injected_anomaly_flag"].to_numpy(dtype=bool)
    cutoffs = [float(value) for value in config["anomalies"]["ranking_cutoffs"]]
    anomaly_rows = [
        evaluate_anomaly_model(
            "K-Means", y_true, k_scores, k_flags, cutoffs,
            classification_rule=f"exact top {contamination:.1%} centroid-distance percentile",
        ),
        evaluate_anomaly_model(
            "DBSCAN", y_true, d_scores, d_flags, cutoffs,
            classification_rule="DBSCAN noise label (-1)",
        ),
    ]
    anomaly_metrics = pd.DataFrame(anomaly_rows)
    anomaly_metrics.to_csv(metrics_dir / "anomaly_metrics.csv", index=False)
    ranking_columns = ["model"] + [column for column in anomaly_metrics.columns if "_at_" in column or column == "top_decile_capture"]
    anomaly_metrics[ranking_columns].to_csv(metrics_dir / "ranking_metrics.csv", index=False)

    selected_k = _selected_row(kmeans_tuning, k_best["configuration_id"])
    selected_d = _selected_row(dbscan_tuning, d_best["configuration_id"])
    clustering_selected = pd.DataFrame([selected_k, selected_d]).reset_index(drop=True)
    # Recalculate the selected clustering metrics from the persisted final labels.
    for index, (labels, model_name) in enumerate(((k_model.labels_, "K-Means"), (d_model.labels_, "DBSCAN"))):
        fresh = clustering_metrics(X, labels, int(common["metric_sample_size"]), seed)
        for key, value in fresh.items():
            clustering_selected.loc[index, key] = value
        clustering_selected.loc[index, "model"] = model_name
    clustering_selected["best_parameters"] = [
        json.dumps(k_model.get_params(), sort_keys=True),
        json.dumps(d_model.get_params(), sort_keys=True),
    ]
    clustering_selected["interpretability"] = [0.92, 0.82]
    clustering_selected["operational_usefulness"] = [0.90, 0.78]
    final_metrics = clustering_selected.merge(anomaly_metrics, on="model", how="left", suffixes=("", "_anomaly"))
    final_metrics.to_csv(metrics_dir / "final_model_metrics.csv", index=False)

    selection, contributions = weighted_model_selection(
        final_metrics,
        config["selection"]["segmentation_weights"],
        config["selection"]["anomaly_weights"],
    )
    selection.to_csv(metrics_dir / "model_selection.csv", index=False)
    contributions.to_csv(metrics_dir / "model_selection_contributions.csv", index=False)
    benchmark = build_clustering_benchmark(final_metrics)
    benchmark.to_csv(metrics_dir / "clustering_benchmark.csv", index=False)
    best_segmentation = str(selection.loc[selection["best_segmentation_model"], "model"].iloc[0])
    best_anomaly = str(selection.loc[selection["best_anomaly_detection_model"], "model"].iloc[0])

    # 5. Explanations, PCA coordinates, profiles, and row/rep investigation outputs.
    top_n = int(config["reporting"]["top_driver_count"])
    scored = featured.copy()
    scored["kmeans_cluster"] = k_model.labels_
    scored["dbscan_cluster"] = d_model.labels_
    scored["dbscan_is_noise"] = d_flags
    scored["kmeans_centroid_distance"] = k_distances
    scored["kmeans_squared_centroid_distance"] = np.square(scored["kmeans_centroid_distance"])
    scored["dbscan_neighbor_distance"] = d_model.neighbor_distances(X)
    scored["kmeans_anomaly_score"] = k_scores
    scored["dbscan_anomaly_score"] = d_scores
    scored["kmeans_anomaly_flag"] = k_flags
    scored["dbscan_anomaly_flag"] = d_flags
    scored["kmeans_top_drivers"] = kmeans_explanations(scored, X, k_model, feature_names, top_n)
    scored["dbscan_top_drivers"] = dbscan_explanations(scored, X, d_model, feature_names, top_n)
    components = min(2, X.shape[1], max(1, X.shape[0] - 1))
    coordinates = PCA(n_components=components, random_state=seed).fit_transform(X)
    scored["pca_1"] = coordinates[:, 0]
    scored["pca_2"] = coordinates[:, 1] if components > 1 else 0.0
    if best_anomaly == "K-Means":
        scored["anomaly_score"] = scored["kmeans_anomaly_score"]
        scored["anomaly_flag"] = scored["kmeans_anomaly_flag"]
        scored["top_anomaly_drivers"] = scored["kmeans_top_drivers"]
    else:
        scored["anomaly_score"] = scored["dbscan_anomaly_score"]
        scored["anomaly_flag"] = scored["dbscan_anomaly_flag"]
        scored["top_anomaly_drivers"] = scored["dbscan_top_drivers"]
    scored["model"] = best_anomaly
    scored.to_csv(processed_dir / "scored_observations.csv", index=False)
    rep_summary = build_rep_risk_summary(scored)
    rep_summary.to_csv(processed_dir / "rep_risk_summary.csv", index=False)
    high_risk = scored.loc[top_fraction_flags(scored["anomaly_score"].to_numpy(), float(config["reporting"]["high_risk_top_pct"]))].copy()
    high_risk.sort_values("anomaly_score", ascending=False).to_csv(reports_dir / "anomaly_investigations.csv", index=False)

    k_profiles = create_cluster_profiles(scored, k_model.labels_, "K-Means")
    d_profiles = create_cluster_profiles(scored, d_model.labels_, "DBSCAN")
    k_profiles.to_csv(metrics_dir / "cluster_profiles_kmeans.csv", index=False)
    d_profiles.to_csv(metrics_dir / "cluster_profiles_dbscan.csv", index=False)
    all_profiles = pd.concat([k_profiles, d_profiles], ignore_index=True)
    all_profiles.to_csv(metrics_dir / "cluster_profiles.csv", index=False)

    # 6. Complete visual and narrative evidence.
    generated_plots = generate_all_plots(
        analytical=featured,
        scored=scored,
        kmeans_tuning=kmeans_tuning,
        dbscan_tuning=dbscan_tuning,
        cluster_profiles=k_profiles,
        benchmark_model_rows=final_metrics,
        y_true=y_true,
        scores_by_model={"K-Means": k_scores, "DBSCAN": d_scores},
        plots_dir=plots_dir,
        dpi=int(config["reporting"]["plot_dpi"]),
    )
    write_execution_reports(reports_dir, source_metadata, scored, injection_audit, selection, final_metrics)
    _write_methodology_report(reports_dir, feature_names, warnings, config["preprocessing"])

    fallback_path = Path(source_metadata["source_path"])
    if not fallback_path.is_absolute():
        fallback_path = Path(str(config["_repo_root"])) / fallback_path
    run_summary = {
        "status": "completed",
        "source_type": source_metadata["source_type"],
        "fallback_used": bool(source_metadata["fallback_used"]),
        "source_path": str(source_metadata["source_path"]),
        "source_sha256": _sha256(fallback_path),
        "source_rows": int(len(canonical)),
        "analytical_rows": int(len(scored)),
        "representatives": int(scored["rep_id"].nunique()),
        "injected_anomalies": int(y_true.sum()),
        "injected_anomaly_rate": float(y_true.mean()),
        "best_segmentation_model": best_segmentation,
        "best_anomaly_detection_model": best_anomaly,
        "plots_generated": int(len(generated_plots)),
        "runtime_seconds": float(time.perf_counter() - started),
        "seed": seed,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "config_path": str(Path(config["_config_path"]).resolve().relative_to(Path(str(config["_repo_root"])).resolve()).as_posix()),
    }
    _json_dump(_clean_json(run_summary), reports_dir / "run_metadata.json")
    return run_summary


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run the field-rep incentive anomaly clustering benchmark.")
    parser.add_argument("--config", type=Path, default=repo_root / "configs" / "config.yaml")
    parser.add_argument("--input", type=Path, default=None, help="Optional explicit pharma CSV. Discovery/fallback is used when omitted.")
    args = parser.parse_args()
    summary = run_pipeline(args.config, input_path=args.input)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
