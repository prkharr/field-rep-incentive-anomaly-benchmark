"""Semantic dashboard contracts using executed outputs, without fitting models.

The artifact integration fixture writes only to pytest's temporary directory.
Small rep-month fixtures separately exercise missing calendar periods and the
as-of boundary, which are easy to conceal in a complete production panel.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from field_rep_anomaly.dashboard_data import (
    build_all_dashboard_datasets,
    build_anomaly_dashboard_dataset,
    build_model_summary_dataset,
    build_rep_summary_dataset,
)
from field_rep_anomaly.dashboard_capacity_data import (
    build_capacity_base_dataset,
    capacity_priority,
)


ROOT = Path(__file__).resolve().parents[1]
ANOMALY = "dashboard_anomaly_review.csv"
REP = "dashboard_rep_summary.csv"
BASE = "dashboard_capacity_base.csv"
SCENARIOS = "dashboard_capacity_scenarios.csv"
MODELS = "dashboard_model_summary.csv"
METADATA = "dashboard_metadata.json"
GRAINS = {
    ANOMALY: ["representative", "product_class", "month"],
    REP: ["representative"],
    BASE: ["team", "country", "product_class"],
    SCENARIOS: ["team", "country", "product_class", "scenario_name"],
    MODELS: ["model"],
}
REQUIRED_COLUMNS = {
    ANOMALY: """
        observation_id representative manager team country product_class month
        sales quantity transaction_value unique_customers new_customers
        repeat_customers lost_customers product_breadth pca_raw_score
        pca_score_percentile pca_review_flag pca_threshold_exceedance pca_rank
        review_priority review_rank top_driver_1 top_driver_1_contribution
        top_driver_2 top_driver_2_contribution top_driver_3
        top_driver_3_contribution strongest_peer_deviation_metric
        strongest_peer_deviation_value strongest_history_deviation_metric
        strongest_history_deviation_value temporal_review_flag
        temporal_history_length business_rule_flag robust_peer_flag
        kmeans_cluster kmeans_distance number_of_supporting_signals
        model_agreement_summary simulated_target simulated_expected_incentive
        simulated_actual_payout simulated_adjustment simulated_payout_delta
    """.split(),
    REP: """
        representative manager team latest_month_available total_observations
        high_priority_review_count medium_priority_review_count
        top_5_percent_review_count maximum_pca_percentile mean_pca_percentile
        latest_pca_percentile latest_review_priority strongest_recent_driver
        temporal_flag_count business_rule_flag_count peer_flag_count
        model_agreement_high_count total_sales recent_3m_sales prior_3m_sales
        sales_growth_3m unique_customers_latest customer_change_3m
    """.split(),
    BASE: """
        team country product_class forecast_horizon
        eligible_for_capacity_recommendation selected_forecast_method
        forecast_workload forecast_error_metric_used_for_selection
        validation_wape test_wape forecast_lower_scenario forecast_upper_scenario
        sustainable_workload_per_rep allocated_fte required_fte fte_gap
        required_fte_lower required_fte_upper fte_gap_lower fte_gap_upper
        capacity_priority customer_load transaction_load geography_load
        product_load distributor_load workload_score_raw
        workload_score_winsorized latest_observed_workload recent_workload_growth
    """.split(),
    SCENARIOS: """
        team country product_class scenario_name scenario_description
        forecast_workload allocated_fte sustainable_capacity_per_rep required_fte
        fte_gap capacity_priority eligible_for_capacity_recommendation
        source_unit target_unit fte_reallocated
    """.split(),
    MODELS: """
        model role recall_at_5pct lift_at_5pct precision_at_5pct pr_auc f1 f2
        stability runtime_seconds selected_for_primary_use business_interpretation
    """.split(),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected_hashes() -> dict[str, str]:
    """Include every original technical output, not just the model selection."""
    paths = [
        path
        for folder in (ROOT / "data/processed", ROOT / "artifacts")
        for path in folder.rglob("*")
        if path.is_file()
    ]
    raw = ROOT / "data/raw/pharma-data.csv"
    if raw.exists():
        paths.append(raw)
    return {path.relative_to(ROOT).as_posix(): _sha256(path) for path in paths}


@pytest.fixture(scope="module")
def dashboard_bundle(tmp_path_factory):
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.neural_network import MLPRegressor

    output = tmp_path_factory.mktemp("dashboard-semantic-data")
    before = _protected_hashes()

    def forbidden_fit(*args, **kwargs):
        pytest.fail("Dashboard consolidation must never fit or retrain a model")

    with pytest.MonkeyPatch.context() as patch:
        for model in (DBSCAN, KMeans, PCA, IsolationForest, MLPRegressor):
            patch.setattr(model, "fit", forbidden_fit)
        returned = build_all_dashboard_datasets(ROOT, output_dir=output)
    after = _protected_hashes()
    frames = {name: pd.read_csv(output / name) for name in GRAINS}
    return output, frames, before, after, returned


@pytest.fixture(scope="module")
def anomaly_inputs():
    paths = {
        "analytical": "data/processed/analytical_dataset.csv",
        "scores": "data/processed/clean_scores_long.csv",
        "pca_contributions": "artifacts/reports/clean_pca_reconstruction_contributions.csv",
        "peer_explanations": "artifacts/reports/clean_peer_explanations.csv",
        "temporal_scores": "data/processed/clean_time_series_scores.csv",
        "rule_signals": "artifacts/reports/clean_rule_signals.csv",
        "queue": "artifacts/reports/clean_investigation_queue.csv",
    }
    return {key: pd.read_csv(ROOT / path) for key, path in paths.items()}


@pytest.fixture(scope="module")
def capacity_inputs():
    paths = {
        "planning": "artifacts/planning/hiring_need_by_business_unit.csv",
        "forecast_metrics": "artifacts/planning/forecast_metrics.csv",
        "capacity_assumptions": "artifacts/planning/capacity_assumptions.csv",
        "forecast_backtest": "artifacts/planning/forecast_backtest.csv",
        "planning_view": "data/processed/planning_view.csv",
        "allocation": "artifacts/planning/fte_allocation.csv",
        "cleaning_sensitivity": "artifacts/planning/anomaly_cleaning_sensitivity.csv",
    }
    return {key: pd.read_csv(ROOT / path) for key, path in paths.items()}


def test_all_dashboard_files_are_created_and_populated(dashboard_bundle):
    output, frames, _, _, returned = dashboard_bundle
    assert set(returned) == set(GRAINS)
    assert (output / METADATA).is_file()
    for name, frame in frames.items():
        assert (output / name).stat().st_size > 0
        assert not frame.empty, name
        assert isinstance(returned[name], pd.DataFrame)
        assert len(returned[name]) == len(frame)


@pytest.mark.parametrize("filename", GRAINS)
def test_dashboard_schemas_and_declared_grains(dashboard_bundle, filename):
    frame = dashboard_bundle[1][filename]
    assert set(REQUIRED_COLUMNS[filename]) <= set(frame)
    assert not frame[GRAINS[filename]].isna().any().any()
    assert not frame.duplicated(GRAINS[filename]).any()


@pytest.mark.parametrize("filename", GRAINS)
def test_manager_data_excludes_labels_and_unfounded_certainty(dashboard_bundle, filename):
    frame = dashboard_bundle[1][filename]
    forbidden = {
        "injected_label", "anomaly_label", "injected_type", "injected_severity",
        "injected_anomaly_flag", "anomaly_type", "severity", "fraud",
        "fraud_probability", "fraudulent", "risk_probability",
    }
    assert not forbidden.intersection(frame.columns)
    assert not any("injected" in c.lower() or "fraud" in c.lower() for c in frame)


def test_clean_population_and_all_incentive_columns_have_simulated_prefix(dashboard_bundle, anomaly_inputs):
    frame = dashboard_bundle[1][ANOMALY]
    analytical = anomaly_inputs["analytical"].set_index("observation_id")
    assert set(frame.observation_id) == set(analytical.index)
    assert frame.observation_id.is_unique
    for column in frame:
        if any(token in column.lower() for token in ("incentive", "payout", "attainment", "adjustment")):
            assert column.startswith("simulated_"), column
    mappings = {
        "sales": "total_sales", "quantity": "total_quantity",
        "transaction_value": "average_transaction_value",
        "unique_customers": "distinct_customers", "product_breadth": "distinct_products",
        "simulated_target": "simulated_target_sales",
        "simulated_expected_incentive": "simulated_expected_incentive",
        "simulated_actual_payout": "simulated_actual_payout",
        "simulated_adjustment": "simulated_adjustment",
        "simulated_payout_delta": "simulated_payout_delta",
    }
    aligned = analytical.reindex(frame.observation_id)
    for target, source in mappings.items():
        np.testing.assert_allclose(frame[target], aligned[source], equal_nan=True)


def test_pca_calibration_flags_and_raw_scores_are_preserved(dashboard_bundle, anomaly_inputs):
    frame = dashboard_bundle[1][ANOMALY].set_index("observation_id")
    scores = anomaly_inputs["scores"]
    pca = scores.loc[scores.model_name.eq("PCA Reconstruction")].set_index("observation_id").reindex(frame.index)
    assert frame.pca_score_percentile.between(0, 1).all()
    np.testing.assert_allclose(frame.pca_raw_score, pca.raw_score)
    np.testing.assert_allclose(frame.pca_score_percentile, pca.anomaly_score)
    np.testing.assert_array_equal(frame.pca_review_flag, pca.anomaly_flag)
    np.testing.assert_array_equal(frame.pca_threshold_exceedance, pca.threshold_flag)
    np.testing.assert_allclose(frame.pca_raw_threshold, pca.threshold)
    np.testing.assert_array_equal(frame.source_partition, pca.split)
    for partition, group in frame.groupby("source_partition"):
        original = pca.loc[pca.split.eq(partition)]
        assert group.pca_review_flag.sum() == original.anomaly_flag.sum()
        assert group.pca_review_flag.sum() == int(np.ceil(len(group) * 0.05))


def test_review_ranks_follow_score_and_stable_observation_ties(dashboard_bundle, anomaly_inputs):
    frame = dashboard_bundle[1][ANOMALY]
    expected = frame.sort_values(["pca_raw_score", "observation_id"], ascending=[False, True])
    np.testing.assert_array_equal(expected.review_rank, np.arange(1, len(frame) + 1))
    np.testing.assert_array_equal(expected.pca_rank, expected.review_rank)
    assert frame.sort_values("review_rank").pca_raw_score.is_monotonic_decreasing

    # Ties must not depend on the order in which CSV rows happen to be stored.
    tied = {key: value.copy(deep=True) for key, value in anomaly_inputs.items()}
    tied["scores"].loc[tied["scores"].model_name.eq("PCA Reconstruction"), "raw_score"] = 7.0
    first = build_anomaly_dashboard_dataset(**tied).sort_values("review_rank")
    shuffled = {key: value.sample(frac=1, random_state=17).reset_index(drop=True) for key, value in tied.items()}
    second = build_anomaly_dashboard_dataset(**shuffled).sort_values("review_rank")
    assert first.observation_id.tolist() == sorted(first.observation_id)
    assert first.observation_id.tolist() == second.observation_id.tolist()


def test_review_priority_is_deterministic_and_support_is_not_correlated_model_count(dashboard_bundle, anomaly_inputs):
    frame = dashboard_bundle[1][ANOMALY]
    counts = sum(frame[column].astype(int) for column in ("temporal_review_flag", "robust_peer_flag", "business_rule_flag"))
    np.testing.assert_array_equal(frame.number_of_supporting_signals, counts)
    high = frame.pca_review_flag | (frame.pca_score_percentile.ge(0.99) & counts.ge(1))
    medium = frame.pca_score_percentile.ge(0.95) | counts.ge(2)
    expected = np.select([high, medium], ["High", "Medium"], default="Low")
    np.testing.assert_array_equal(frame.review_priority, expected)
    assert set(frame.review_priority) <= {"High", "Medium", "Low"}
    scores = anomaly_inputs["scores"]
    for model, column in (("EWMA Residual", "temporal_review_flag"), ("Robust Peer Baseline", "robust_peer_flag")):
        source = scores.loc[scores.model_name.eq(model)].set_index("observation_id").reindex(frame.observation_id)
        np.testing.assert_array_equal(frame[column], source.anomaly_flag)
    rules = anomaly_inputs["rule_signals"].groupby("observation_id").flag.any().reindex(frame.observation_id)
    np.testing.assert_array_equal(frame.business_rule_flag, rules)


def test_pca_explanations_are_existing_contributions_not_recomputed(dashboard_bundle, anomaly_inputs):
    frame = dashboard_bundle[1][ANOMALY].set_index("observation_id")
    source = anomaly_inputs["pca_contributions"]
    sums = frame[[f"top_driver_{i}_contribution" for i in (1, 2, 3)]].sum(axis=1)
    # These saved values are squared reconstruction errors, not probabilities
    # or normalized shares, so a legitimate contribution may exceed one.
    assert sums.ge(0).all() and np.isfinite(sums).all()
    top = source.sort_values(["observation_id", "contribution", "feature"], ascending=[True, False, True]).groupby("observation_id").head(3)
    expected = top.groupby("observation_id").contribution.sum().reindex(frame.index)
    np.testing.assert_allclose(sums, expected)


def test_anomaly_transform_does_not_mutate_input_frames(anomaly_inputs):
    inputs = {key: frame.copy(deep=True) for key, frame in anomaly_inputs.items()}
    build_anomaly_dashboard_dataset(**inputs)
    for key, frame in inputs.items():
        pd.testing.assert_frame_equal(frame, anomaly_inputs[key])


def test_historical_fields_do_not_change_when_future_rows_change(anomaly_inputs):
    original = build_anomaly_dashboard_dataset(**anomaly_inputs)
    changed = {key: value.copy(deep=True) for key, value in anomaly_inputs.items()}
    cutoff = pd.Timestamp("2018-12-01")
    future_ids = set(changed["analytical"].loc[pd.to_datetime(changed["analytical"].date).gt(cutoff), "observation_id"])
    changed_columns = {
        "analytical": ["total_sales", "total_quantity", "distinct_customers", "new_customers", "simulated_target_sales", "simulated_expected_incentive"],
        "peer_explanations": ["observed", "expected", "robust_z"],
        "temporal_scores": ["observed", "expected", "residual", "normalized_residual", "score"],
    }
    changed_columns["analytical"] += [column for column in changed["analytical"] if column.endswith(("_history_deviation", "_peer_z"))]
    for key, columns in changed_columns.items():
        table = changed[key]
        future = table.observation_id.isin(future_ids)
        for column in columns:
            table.loc[future, column] *= 3.0
    updated = build_anomaly_dashboard_dataset(**changed)
    historical = [
        "sales", "quantity", "unique_customers", "new_customers", "repeat_customers",
        "lost_customers", "strongest_history_deviation_metric", "strongest_history_deviation_value",
        "temporal_history_length", "simulated_target", "simulated_expected_incentive",
    ]
    def past(frame):
        return frame.loc[pd.to_datetime(frame.month).le(cutoff)].set_index("observation_id")[historical].sort_index()
    pd.testing.assert_frame_equal(past(original), past(updated))


def test_unavailable_and_nonintegral_customer_counts_remain_null(anomaly_inputs):
    changed = {key: value.copy(deep=True) for key, value in anomaly_inputs.items()}
    data = changed["analytical"]
    rep, product_class = data.loc[0, ["representative", "product_class"]]
    indices = data.loc[data.representative.eq(rep) & data.product_class.eq(product_class)].sort_values("date").index
    first, second = indices[:2]
    data.loc[[first, second], "distinct_customers"] = 10
    data.loc[first, "new_customers"] = 10
    data.loc[second, "new_customers"] = 5
    data.loc[second, "distinct_customers_lag_1"] = 10
    data.loc[first, "repeat_customer_ratio"] = 0.75
    data.loc[second, "repeat_customer_ratio"] = 0.5
    data.loc[[first, second], "customer_loss_rate"] = 0.5
    frame = build_anomaly_dashboard_dataset(**changed).set_index("observation_id")
    a, b = frame.loc[data.loc[first, "observation_id"]], frame.loc[data.loc[second, "observation_id"]]
    assert pd.isna(a.repeat_customers), "A nonintegral estimate is not an observed customer count"
    assert b.repeat_customers == 5
    assert pd.isna(a.lost_customers), "No prior calendar month means the lost-customer count is unknown"
    assert b.lost_customers == 5


def _rep_summary_inputs():
    """Class customers intentionally overlap; only the rep rollup is distinct."""
    rows, rollup = [], []
    for index, month in enumerate(pd.date_range("2019-01-01", periods=6, freq="MS"), 1):
        for product, percentile in (("A", index / 10), ("B", index / 10 + 0.35)):
            rows.append({
                "observation_id": f"R-{index}-{product}", "representative": "R",
                "manager": "M", "team": "T", "country": "Germany", "product_class": product,
                "month": month.strftime("%Y-%m-%d"), "sales": index * 50.0,
                "unique_customers": 100, "pca_raw_score": percentile,
                "pca_score_percentile": percentile, "pca_review_flag": percentile >= 0.9,
                "review_priority": "High" if percentile >= 0.9 else "Low",
                "review_rank": 20 - len(rows), "top_driver_1": f"Driver {product}",
                "temporal_review_flag": False, "business_rule_flag": False,
                "robust_peer_flag": False, "peer_flag": False, "number_of_supporting_signals": 0,
                "model_agreement_summary": "PCA only",
            })
        rollup.append({"representative": "R", "date": month.strftime("%Y-%m-%d"),
                       "total_sales": index * 100.0, "distinct_customers": 90 + index * 10})
    return pd.DataFrame(rows), pd.DataFrame(rollup)


def test_rep_summary_uses_exact_calendar_windows_and_distinct_customer_rollup():
    anomaly, rollup = _rep_summary_inputs()
    row = build_rep_summary_dataset(anomaly, rollup).iloc[0]
    assert row.total_observations == 12
    assert row.total_sales == pytest.approx(2100)
    assert row.recent_3m_sales == pytest.approx(1500)
    assert row.prior_3m_sales == pytest.approx(600)
    assert row.sales_growth_3m == pytest.approx(1.5)
    assert row.unique_customers_latest == 150
    assert row.customer_change_3m == 30
    assert row.latest_pca_percentile == pytest.approx(0.95)
    assert row.latest_review_priority == "High"
    assert row.strongest_recent_driver == "Driver B"


@pytest.mark.parametrize("missing_month,missing_metric", [("2019-04-01", "recent_3m_sales"), ("2019-01-01", "prior_3m_sales"), ("2019-03-01", "customer_change_3m")])
def test_rep_summary_never_bridges_unknown_calendar_months(missing_month, missing_metric):
    anomaly, rollup = _rep_summary_inputs()
    anomaly = anomaly.loc[anomaly.month.ne(missing_month)]
    rollup = rollup.loc[rollup.date.ne(missing_month)]
    row = build_rep_summary_dataset(anomaly, rollup).iloc[0]
    assert pd.isna(row[missing_metric])
    if missing_metric != "customer_change_3m":
        assert pd.isna(row.sales_growth_3m)


def test_rep_summary_ignores_rollup_observations_after_its_asof_month():
    anomaly, rollup = _rep_summary_inputs()
    anomaly = anomaly.loc[anomaly.month.le("2019-03-01")]
    before = build_rep_summary_dataset(anomaly, rollup)
    altered = rollup.copy()
    altered.loc[altered.date.gt("2019-03-01"), ["total_sales", "distinct_customers"]] = 999999
    after = build_rep_summary_dataset(anomaly, altered)
    pd.testing.assert_frame_equal(before, after)
    assert before.iloc[0].recent_3m_sales == 600
    assert before.iloc[0].unique_customers_latest == 120


def test_rep_summary_aggregations_reconcile(dashboard_bundle):
    anomaly, reps = dashboard_bundle[1][ANOMALY], dashboard_bundle[1][REP]
    reps = reps.set_index("representative")
    for representative, observations in anomaly.groupby("representative"):
        row = reps.loc[representative]
        assert row.total_observations == len(observations)
        assert row.high_priority_review_count == observations.review_priority.eq("High").sum()
        assert row.medium_priority_review_count == observations.review_priority.eq("Medium").sum()
        assert row.top_5_percent_review_count == observations.pca_review_flag.sum()
        assert row.total_sales == pytest.approx(observations.sales.sum())
        assert row.maximum_pca_percentile == pytest.approx(observations.pca_score_percentile.max())
        assert row.mean_pca_percentile == pytest.approx(observations.pca_score_percentile.mean())
        latest = observations.loc[observations.month.eq(observations.month.max())]
        assert row.latest_pca_percentile == pytest.approx(latest.pca_score_percentile.max())


def test_capacity_arithmetic_allocation_and_existing_results_reconcile(dashboard_bundle):
    base = dashboard_bundle[1][BASE]
    eligible = base.loc[base.eligible_for_capacity_recommendation]
    assert set(base.forecast_horizon) == {"2019-05-01"}
    np.testing.assert_allclose(eligible.required_fte, eligible.forecast_workload / eligible.sustainable_workload_per_rep)
    np.testing.assert_allclose(eligible.fte_gap, eligible.required_fte - eligible.allocated_fte)
    for bound in ("lower", "upper"):
        np.testing.assert_allclose(eligible[f"fte_gap_{bound}"], eligible[f"required_fte_{bound}"] - eligible.allocated_fte)
    assert eligible.allocated_fte.sum() == pytest.approx(13)
    allocation = pd.read_csv(ROOT / "artifacts/planning/fte_allocation.csv")
    assert eligible.allocated_fte.sum() == pytest.approx(allocation.allocated_fte.sum())
    assert allocation.representative.nunique() == 13
    original = pd.read_csv(ROOT / "artifacts/planning/hiring_need_by_business_unit.csv").set_index(GRAINS[BASE]).reindex(eligible.set_index(GRAINS[BASE]).index)
    for target, source in (("required_fte", "required_fte"), ("fte_gap", "fte_gap"), ("allocated_fte", "allocated_current_fte"), ("forecast_workload", "forecast_workload")):
        np.testing.assert_allclose(eligible[target], original[source])


def test_poland_stale_coverage_is_never_zero_demand_or_zero_staffing(dashboard_bundle):
    base = dashboard_bundle[1][BASE]
    stale = base.loc[base.country.eq("Poland")]
    assert len(stale) == 24
    assert not stale.eligible_for_capacity_recommendation.any()
    assert stale.capacity_priority.eq("Ineligible / Stale Coverage").all()
    assert stale[["forecast_workload", "allocated_fte", "required_fte", "fte_gap", "fte_gap_lower", "fte_gap_upper"]].isna().all().all()
    assert set(base.capacity_priority) <= {"Potential Capacity Gap", "Balanced", "Potential Spare Capacity", "Ineligible / Stale Coverage"}


def test_forecast_selection_and_wape_come_from_validation_artifacts(dashboard_bundle):
    eligible = dashboard_bundle[1][BASE].query("eligible_for_capacity_recommendation")
    metrics = pd.read_csv(ROOT / "artifacts/planning/forecast_metrics.csv")
    candidates = metrics.loc[metrics.metric.eq("workload") & metrics.split.eq("validation")]
    best = candidates.sort_values("WAPE").iloc[0]
    assert set(eligible.selected_forecast_method) == {best.method}
    assert eligible.forecast_error_metric_used_for_selection.str.contains("WAPE", case=False).all()
    for split in ("validation", "test"):
        expected = metrics.loc[metrics.metric.eq("workload") & metrics.method.eq(best.method) & metrics.split.eq(split), "WAPE"].iloc[0]
        np.testing.assert_allclose(eligible[f"{split}_wape"], expected)
    assert (eligible.forecast_lower_scenario <= eligible.forecast_workload).all()
    assert (eligible.forecast_upper_scenario >= eligible.forecast_workload).all()


def test_capacity_history_does_not_read_past_saved_planning_horizon(capacity_inputs):
    original = build_capacity_base_dataset(**capacity_inputs)
    altered = {key: frame.copy(deep=True) for key, frame in capacity_inputs.items()}
    for key in ("forecast_backtest", "planning_view"):
        source = altered[key]
        extra = source.loc[source.date.eq("2019-04-01")].copy()
        extra["date"] = "2019-05-01"
        for column in ("observed", "total_sales", "distinct_customers"):
            if column in extra:
                extra[column] = 999999
        altered[key] = pd.concat([source, extra], ignore_index=True)
    pd.testing.assert_frame_equal(original, build_capacity_base_dataset(**altered))


def test_capacity_growth_requires_six_actual_calendar_months(capacity_inputs):
    altered = {key: frame.copy(deep=True) for key, frame in capacity_inputs.items()}
    backtest = altered["forecast_backtest"]
    altered["forecast_backtest"] = backtest.loc[~(backtest.date.eq("2018-11-01") & backtest.metric.eq("workload"))].copy()
    result = build_capacity_base_dataset(**altered)
    assert result.loc[result.country.eq("Germany"), "recent_workload_growth"].isna().all()


@pytest.mark.parametrize("gap,eligible,expected", [(1.0, True, "Potential Capacity Gap"), (-1.0, True, "Potential Spare Capacity"), (0.0, True, "Balanced"), (1e-10, True, "Balanced"), (np.nan, False, "Ineligible / Stale Coverage")])
def test_capacity_priority_is_a_deterministic_gap_label(gap, eligible, expected):
    assert capacity_priority(gap, eligible) == expected


def test_all_scenarios_preserve_original_rows_and_reconcile_base_assumptions(dashboard_bundle):
    base = dashboard_bundle[1][BASE].set_index(GRAINS[BASE])
    scenarios = dashboard_bundle[1][SCENARIOS]
    assert set(scenarios.scenario_name) == {"Base", "Demand +10%", "Demand +20%", "Add 1 FTE", "Add 2 FTE", "Capacity -10%", "Product Launch", "Net-zero Reallocation"}
    original = pd.read_csv(ROOT / "artifacts/planning/hiring_scenarios.csv")
    assert len(scenarios) == len(original)
    assert scenarios.eligible_for_capacity_recommendation.all()
    assert "Poland" not in set(scenarios.country)
    np.testing.assert_allclose(scenarios.required_fte, scenarios.forecast_workload / scenarios.sustainable_capacity_per_rep)
    np.testing.assert_allclose(scenarios.fte_gap, scenarios.required_fte - scenarios.allocated_fte)
    specs = {"Base": (1.0, 0, 1.0), "Demand +10%": (1.1, 0, 1.0), "Demand +20%": (1.2, 0, 1.0), "Add 1 FTE": (1.0, 1, 1.0), "Add 2 FTE": (1.0, 2, 1.0), "Capacity -10%": (1.0, 0, 0.9), "Product Launch": (1.3, 0, 1.0)}
    for name, (demand, additional, capacity) in specs.items():
        rows = scenarios.loc[scenarios.scenario_name.eq(name)].set_index(GRAINS[BASE])
        reference = base.reindex(rows.index)
        np.testing.assert_allclose(rows.forecast_workload, reference.forecast_workload * demand)
        np.testing.assert_allclose(rows.allocated_fte, reference.allocated_fte + additional)
        np.testing.assert_allclose(rows.sustainable_capacity_per_rep, reference.sustainable_workload_per_rep * capacity)
    reallocation = scenarios.loc[scenarios.scenario_name.eq("Net-zero Reallocation")].set_index(GRAINS[BASE])
    reference = base.reindex(reallocation.index)
    assert len(reallocation) == 2
    assert reallocation.fte_reallocated.sum() == pytest.approx(0)
    assert reallocation.allocated_fte.sum() == pytest.approx(reference.allocated_fte.sum())
    np.testing.assert_allclose(reallocation.allocated_fte - reference.allocated_fte, reallocation.fte_reallocated)
    assert reallocation.source_unit.notna().all() and reallocation.target_unit.notna().all()
    assert reallocation.source_unit.nunique() == reallocation.target_unit.nunique() == 1
    assert reallocation.allocated_fte.ge(-1e-9).all()


def test_model_summary_loads_all_metrics_and_validation_selection(dashboard_bundle):
    frame = dashboard_bundle[1][MODELS].set_index("model")
    benchmark = pd.read_csv(ROOT / "artifacts/metrics/final_anomaly_model_benchmark.csv").set_index("model")
    selection = json.loads((ROOT / "artifacts/reports/extended_model_selection.json").read_text())
    assert set(frame.index) == set(benchmark.index)
    mappings = {"recall_at_5pct": "Recall@5%", "lift_at_5pct": "Lift@5%", "precision_at_5pct": "Precision@5%", "pr_auc": "PR_AUC", "f1": "F1", "f2": "F2", "stability": "stability", "runtime_seconds": "runtime_seconds"}
    for target, source in mappings.items():
        np.testing.assert_allclose(frame[target], benchmark.reindex(frame.index)[source], equal_nan=True)
    assert frame.loc[selection["recommended_model"], "selected_for_primary_use"]
    assert frame.loc["PCA Reconstruction", "role"] == "Primary anomaly ranking"
    assert frame.loc["K-Means", "role"] == "Segmentation"
    assert frame.loc["EWMA Residual", "role"] == "Temporal specialist"
    assert not frame.loc["Best Ensemble", "selected_for_primary_use"]

    # A changed persisted metric must flow through; display code has no constants.
    modified = benchmark.reset_index()
    modified.loc[modified.model.eq("PCA Reconstruction"), "Recall@5%"] = 0.123456
    rebuilt = build_model_summary_dataset(modified, selection).set_index("model")
    assert rebuilt.loc["PCA Reconstruction", "recall_at_5pct"] == pytest.approx(0.123456)


def test_metadata_records_sources_models_grains_and_provenance(dashboard_bundle):
    output, frames, _, _, _ = dashboard_bundle
    metadata = json.loads((output / METADATA).read_text())
    original = json.loads((ROOT / "artifacts/reports/extended_run_metadata.json").read_text())
    required = {"generated_timestamp", "source_csv_sha256", "source_row_count", "modeling_row_count", "source_date_range", "primary_analytical_grain", "dashboard_files_generated", "datasets", "selected_primary_anomaly_model", "selected_temporal_model", "selected_capacity_forecast_methodology", "planning_horizon", "known_poland_coverage_limitation", "feature_count", "seed", "pipeline_version", "git_commit", "git_worktree_dirty"}
    assert required <= set(metadata)
    assert not pd.isna(pd.Timestamp(metadata["generated_timestamp"]))
    assert metadata["source_csv_sha256"] == original["sha256"]
    assert metadata["source_row_count"] == original["raw_rows"]
    assert metadata["modeling_row_count"] == original["analytical_rows"] == len(frames[ANOMALY])
    assert metadata["source_date_range"] == {"start": original["date_min"], "end": original["date_max"]}
    assert set(GRAINS) <= set(metadata["dashboard_files_generated"])
    assert metadata["selected_primary_anomaly_model"] == "PCA Reconstruction"
    assert metadata["selected_temporal_model"] == "EWMA Residual"
    horizon = metadata["planning_horizon"]
    assert (horizon if isinstance(horizon, list) else [horizon]) == ["2019-05-01"]
    assert metadata["feature_count"] == original["feature_count"]
    assert metadata["seed"] == original["seed"]
    assert "Poland" in str(metadata["known_poland_coverage_limitation"])
    for filename, frame in frames.items():
        assert metadata["datasets"][filename]["row_count"] == len(frame)


def test_dashboard_generation_does_not_mutate_source_or_technical_artifacts(dashboard_bundle):
    _, _, before, after, _ = dashboard_bundle
    assert before == after
    raw = ROOT / "data/raw/pharma-data.csv"
    if raw.exists():
        metadata = json.loads((ROOT / "artifacts/reports/extended_run_metadata.json").read_text())
        assert _sha256(raw) == metadata["sha256"]
