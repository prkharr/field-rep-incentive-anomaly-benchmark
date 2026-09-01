"""Focused contracts for the finalized commercial-review PCA runner."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from field_rep_anomaly.commercial_review.modeling import run_finalized_pca


FEATURES = [
    "gross_sales",
    "final_incentive_paid",
    "average_discount_pct",
    "return_rate",
    "gross_sales_peer_median",
    "final_incentive_paid_peer_median",
]


def _benchmark_frames():
    periods = pd.date_range("2020-01-01", periods=18, freq="MS")
    rows = []
    for month_number, period in enumerate(periods):
        seasonal = 12 * np.sin(month_number / 2.5)
        for rep_number in range(6):
            sales = 100 + seasonal + 7 * rep_number + 1.5 * month_number
            incentive = 8 + 0.055 * sales + 0.25 * rep_number
            rows.append(
                {
                    "observation_id": f"obs_{month_number:02d}_{rep_number:02d}",
                    "rep_id": f"REP_{rep_number:02d}",
                    "manager_id": f"MANAGER_{rep_number // 3}",
                    "team_id": f"TEAM_{rep_number // 2}",
                    "territory_id": f"TERRITORY_{rep_number}",
                    "period": period,
                    "gross_sales": sales,
                    "final_incentive_paid": incentive,
                    "average_discount_pct": 0.03 + 0.002 * rep_number + 0.001 * (month_number % 3),
                    "return_rate": 0.01 + 0.001 * ((rep_number + month_number) % 4),
                    "data_lineage": "synthetic_normal",
                }
            )
    clean = pd.DataFrame(rows)
    clean["gross_sales_peer_median"] = clean.groupby("period").gross_sales.transform("median")
    clean["final_incentive_paid_peer_median"] = clean.groupby("period").final_incentive_paid.transform("median")
    injected = clean.copy(deep=True)
    validation_or_test = injected.period.gt("2020-08-01")
    sales_anomaly = validation_or_test & injected.rep_id.eq("REP_00")
    payout_anomaly = validation_or_test & injected.rep_id.eq("REP_01") & injected.period.dt.month.mod(2).eq(0)
    injected.loc[sales_anomaly, "gross_sales"] *= 2.8
    injected.loc[payout_anomaly, "final_incentive_paid"] *= 3.2
    truth = injected[
        ["rep_id", "period", "manager_id", "team_id", "territory_id"]
    ].copy()
    truth["ground_truth_label"] = sales_anomaly | payout_anomaly
    truth["anomaly_type"] = np.select(
        [sales_anomaly, payout_anomaly], ["sales_spike", "payout_mismatch"], default="none"
    )
    truth["severity"] = np.where(truth.ground_truth_label, "high", "none")
    config = {
        "model": {
            "finalized_model": "PCA Reconstruction",
            "train_end": "2020-08-01",
            "validation_end": "2021-01-01",
            "manager_review_fraction": 0.20,
            "pca_retained_variance": 0.90,
            "signed_log1p": True,
            "scaler": "robust",
            "feature_columns": FEATURES,
            "id_columns": ["rep_id", "period"],
            "top_contribution_count": 3,
            "distribution_bins": 8,
        }
    }
    return clean, injected, truth, config


def test_finalized_pca_returns_complete_deterministic_contract():
    clean, injected, truth, config = _benchmark_frames()
    clean_before = clean.copy(deep=True)
    injected_before = injected.copy(deep=True)
    truth_before = truth.copy(deep=True)

    result = run_finalized_pca(clean, injected, truth, config)

    expected = {
        "clean_scores",
        "injected_scores",
        "metrics_summary",
        "top_k_metrics",
        "group_metrics",
        "period_stability",
        "score_distributions",
        "roc_curve",
        "pr_curve",
        "lift_curve",
        "feature_contributions",
        "false_positive_review",
        "pca_metadata",
    }
    assert set(result) == expected
    pd.testing.assert_frame_equal(clean, clean_before)
    pd.testing.assert_frame_equal(injected, injected_before)
    pd.testing.assert_frame_equal(truth, truth_before)

    clean_scores = result["clean_scores"]
    benchmark_scores = result["injected_scores"]
    assert len(clean_scores) == len(clean) == len(benchmark_scores)
    assert clean_scores.observation_id.is_unique and benchmark_scores.observation_id.is_unique
    assert clean_scores.anomaly_score.between(0, 1).all()
    assert benchmark_scores.anomaly_score.between(0, 1).all()
    np.testing.assert_array_equal(
        benchmark_scores.manager_review_flag, benchmark_scores.threshold_flag
    )
    np.testing.assert_array_equal(
        benchmark_scores.review_budget_flag, benchmark_scores.threshold_flag
    )
    assert "ground_truth_label" not in clean_scores
    assert {"primary_reason_code", "primary_reason", "secondary_reason", "recommended_review_action"} <= set(clean_scores)
    assert clean_scores.primary_reason.str.contains("PCA reconstruction deviation").all()
    assert clean_scores.recommended_review_action.str.len().gt(20).all()

    metadata = result["pca_metadata"]
    assert metadata["fit_population"] == "clean train only"
    assert metadata["feature_columns"] == FEATURES
    assert not set(FEATURES) & set(metadata["ground_truth_columns_excluded"])
    cumulative = np.asarray(metadata["cumulative_explained_variance"])
    assert len(cumulative) == metadata["retained_components"]
    assert np.all(np.diff(cumulative) >= 0)
    assert cumulative[-1] == pytest.approx(metadata["total_explained_variance"])
    assert cumulative[-1] >= 0.90

    assert set(result["top_k_metrics"].review_fraction) == {0.01, 0.05, 0.10}
    assert {"anomaly_type", "severity", "manager_id", "team_id", "territory_id", "manager_team_territory"} <= set(
        result["group_metrics"].grouping
    )
    contributions = result["feature_contributions"]
    assert len(contributions) == 3 * (len(clean) + len(injected))
    assert contributions.contribution.ge(0).all()
    assert contributions.peer_value.notna().any()


def test_threshold_is_validation_budget_based_and_label_independent():
    clean, injected, truth, config = _benchmark_frames()
    first = run_finalized_pca(clean, injected, truth, config)
    changed_truth = truth.copy()
    validation = changed_truth.period.gt("2020-08-01") & changed_truth.period.le("2021-01-01")
    changed_truth.loc[validation, "ground_truth_label"] = ~changed_truth.loc[validation, "ground_truth_label"]
    changed_truth.loc[validation, "anomaly_type"] = "changed_label_only"
    second = run_finalized_pca(clean, injected, changed_truth, config)

    first_meta, second_meta = first["pca_metadata"], second["pca_metadata"]
    assert first_meta["threshold"] == pytest.approx(second_meta["threshold"])
    pd.testing.assert_series_equal(first["injected_scores"].raw_score, second["injected_scores"].raw_score)
    pd.testing.assert_series_equal(first["injected_scores"].threshold_flag, second["injected_scores"].threshold_flag)

    validation_scores = first["injected_scores"].query("split == 'validation'").raw_score.to_numpy()
    expected_count = math.ceil(len(validation_scores) * 0.20)
    expected_threshold = np.sort(validation_scores)[::-1][expected_count - 1]
    assert first_meta["threshold"] == pytest.approx(expected_threshold)
    test = first["injected_scores"].query("split == 'test'")
    np.testing.assert_array_equal(test.threshold_flag, test.raw_score.ge(expected_threshold))


def test_ground_truth_leakage_is_rejected():
    clean, injected, truth, config = _benchmark_frames()
    clean["ground_truth_label"] = 0
    injected["ground_truth_label"] = truth.ground_truth_label.astype(int)
    config["model"]["feature_columns"] = FEATURES + ["ground_truth_label"]
    with pytest.raises(ValueError, match="Ground-truth leakage"):
        run_finalized_pca(clean, injected, truth, config)


def test_clean_future_rows_cannot_change_fitted_model_or_threshold():
    clean, injected, truth, config = _benchmark_frames()
    baseline = run_finalized_pca(clean, injected, truth, config)
    changed = clean.copy(deep=True)
    future = changed.period.gt("2020-08-01")
    changed.loc[future, FEATURES] = changed.loc[future, FEATURES] * 1000 + 500
    rerun = run_finalized_pca(changed, injected, truth, config)

    assert rerun["pca_metadata"]["threshold"] == pytest.approx(baseline["pca_metadata"]["threshold"])
    np.testing.assert_allclose(
        rerun["pca_metadata"]["explained_variance_ratio"],
        baseline["pca_metadata"]["explained_variance_ratio"],
    )
    pd.testing.assert_series_equal(rerun["injected_scores"].raw_score, baseline["injected_scores"].raw_score)


def test_optional_persistence_is_confined_to_output_directory(tmp_path):
    clean, injected, truth, config = _benchmark_frames()
    untouched = tmp_path / "no_output"
    run_finalized_pca(clean, injected, truth, config)
    assert not untouched.exists()

    destination = tmp_path / "model_outputs"
    result = run_finalized_pca(clean, injected, truth, config, output_dir=destination)
    expected_files = {
        "pca_clean_scores.csv",
        "pca_injected_scores.csv",
        "pca_metrics_summary.csv",
        "pca_top_k_metrics.csv",
        "pca_group_metrics.csv",
        "pca_period_stability.csv",
        "pca_score_distributions.csv",
        "pca_roc_curve.csv",
        "pca_pr_curve.csv",
        "pca_lift_curve.csv",
        "pca_feature_contributions.csv",
        "pca_false_positive_review.csv",
        "pca_metadata.json",
    }
    assert {path.name for path in destination.iterdir()} == expected_files
    assert len(pd.read_csv(destination / "pca_clean_scores.csv")) == len(result["clean_scores"])
    assert not [path for path in tmp_path.iterdir() if path != destination]
