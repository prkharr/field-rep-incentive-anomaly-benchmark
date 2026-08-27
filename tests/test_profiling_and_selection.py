"""Population reconciliation and configured model-selection tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from field_rep_anomaly.profiling import create_cluster_profiles
from field_rep_anomaly.reporting import weighted_model_selection


def test_cluster_profile_populations_sales_and_labels_reconcile(injected_data):
    row_number = np.arange(len(injected_data))
    labels = np.where(row_number % 7 == 0, -1, row_number % 3)

    profiles = create_cluster_profiles(injected_data, labels, model_name="fixture-model")

    assert set(profiles["cluster"]) == set(np.unique(labels))
    assert int(profiles["population"].sum()) == len(injected_data)
    assert np.isclose(profiles["population_pct"].sum(), 100.0)
    assert np.isclose(profiles["total_sales_sum"].sum(), injected_data["total_sales"].sum())
    assert np.isclose(
        (profiles["anomaly_rate"] * profiles["population"]).sum(),
        injected_data["injected_anomaly_flag"].sum(),
    )
    assert profiles["business_interpretation"].str.len().gt(0).all()
    assert profiles.loc[profiles["cluster"].eq(-1), "business_interpretation"].str.startswith(
        "Density-isolated;"
    ).all()


def test_config_weights_sum_to_one_and_contributions_reconcile(project_config):
    segmentation_weights = project_config["selection"]["segmentation_weights"]
    anomaly_weights = project_config["selection"]["anomaly_weights"]
    assert np.isclose(sum(segmentation_weights.values()), 1.0)
    assert np.isclose(sum(anomaly_weights.values()), 1.0)

    metric_names = sorted(set(segmentation_weights) | set(anomaly_weights))
    final_metrics = pd.DataFrame({"model": ["K-Means", "DBSCAN"]})
    for position, metric in enumerate(metric_names, start=1):
        final_metrics[metric] = [float(position), float(position + 1)]

    selection, contributions = weighted_model_selection(
        final_metrics,
        segmentation_weights=segmentation_weights,
        anomaly_weights=anomaly_weights,
    )

    assert selection["segmentation_score"].between(0.0, 1.0).all()
    assert selection["anomaly_score"].between(0.0, 1.0).all()
    assert selection["best_segmentation_model"].sum() == 1
    assert selection["best_anomaly_detection_model"].sum() == 1
    for score_name in ("segmentation_score", "anomaly_score"):
        summed = (
            contributions.loc[contributions["selection_type"].eq(score_name)]
            .groupby("model", observed=True)["weighted_contribution"]
            .sum()
        )
        expected = selection.set_index("model")[score_name]
        pd.testing.assert_series_equal(
            summed.sort_index(), expected.sort_index(), check_names=False, atol=1e-12, rtol=1e-12
        )


def test_model_selection_rejects_misconfigured_weights(project_config):
    segmentation_weights = dict(project_config["selection"]["segmentation_weights"])
    anomaly_weights = project_config["selection"]["anomaly_weights"]
    segmentation_weights["silhouette_score"] += 0.10
    metrics = sorted(set(segmentation_weights) | set(anomaly_weights))
    frame = pd.DataFrame({"model": ["one", "two"], **{name: [0.0, 1.0] for name in metrics}})

    with pytest.raises(ValueError, match="weights must sum to 1.0"):
        weighted_model_selection(frame, segmentation_weights, anomaly_weights)
