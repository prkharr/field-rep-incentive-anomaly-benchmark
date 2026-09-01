"""Focused startup and governance tests for the commercial-review dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from field_rep_anomaly.commercial_review import dashboard


EXPECTED_REQUIRED = {
    "dashboard_kpi_summary.csv",
    "dashboard_manager_review_queue.csv",
    "dashboard_rep_period_summary.csv",
    "dashboard_anomaly_evidence.csv",
    "dashboard_feature_contributions.csv",
    "dashboard_peer_comparison.csv",
    "dashboard_capacity_summary.csv",
    "dashboard_capacity_customer_drilldown.csv",
    "dashboard_model_metrics.csv",
    "dashboard_anomaly_type_metrics.csv",
    "dashboard_data_quality.csv",
    "dashboard_run_manifest.csv",
}


def _write_csv(directory: Path, filename: str, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(directory / filename, index=False)


@pytest.fixture
def dashboard_root(tmp_path: Path) -> Path:
    directory = tmp_path / "data" / "dashboard"
    directory.mkdir(parents=True)
    _write_csv(
        directory,
        "dashboard_kpi_summary.csv",
        [
            {
                "total_gross_sales": 1_000_000,
                "total_net_sales": 940_000,
                "total_incentive_paid": 25_000,
                "review_candidate_count": 2,
                "high_priority_review_candidate_count": 1,
                "review_rate": 0.10,
                "overloaded_territory_count": 1,
                "total_positive_fte_gap": 0.6,
                "selected_threshold": 1.23,
                "test_precision_at_selected_threshold": 0.50,
                "test_recall_at_selected_threshold": 0.40,
                "benchmark_mode_label_disclosure": "Controlled synthetic labels are evaluation-only.",
                "currency_code": "USD",
            }
        ],
    )
    periods = ["2025-01-01", "2025-02-01", "2025-03-01"]
    summary_rows = []
    for rep_index, (rep, manager, team, territory) in enumerate(
        [("Rep One", "Manager A", "Team East", "Territory 1"), ("Rep Two", "Manager B", "Team West", "Territory 2")],
        start=1,
    ):
        for period_index, period in enumerate(periods, start=1):
            observation = f"OBS_{rep_index}_{period_index}"
            flagged = (rep_index, period_index) in {(1, 3), (2, 2)}
            summary_rows.append(
                {
                    "observation_id": observation,
                    "rep_id": f"REP_{rep_index}",
                    "rep_name": rep,
                    "manager_id": f"MGR_{rep_index}",
                    "manager_name": manager,
                    "team_id": f"TEAM_{rep_index}",
                    "team_name": team,
                    "territory_id": f"TERR_{rep_index}",
                    "territory_name": territory,
                    "period": period,
                    "anomaly_score": 0.99 if flagged else 0.30 + period_index / 10,
                    "manager_review_flag": flagged,
                    "threshold_flag": flagged,
                    "review_priority": "High" if flagged else "Context only",
                    "risk_band": "high" if flagged else "low",
                    "primary_reason": "Paid incentive differed from policy expectation" if flagged else "Context only",
                    "gross_sales": 100_000 + rep_index * 1_000 + period_index * 100,
                    "net_sales": 95_000 + rep_index * 1_000 + period_index * 100,
                    "target_sales": 90_000,
                    "attainment_pct": 105 + period_index,
                    "final_incentive_paid": 2_000 + period_index * 100,
                    "expected_incentive": 1_900,
                    "incentive_calculation_residual": 100 + period_index * 100,
                    "average_discount_pct": 0.04 + period_index / 100,
                    "return_rate": 0.01,
                    "post_incentive_return_rate": period_index / 100,
                    "end_of_period_sales_share": 0.20 + period_index / 100,
                    "completed_visit_count": 12 + period_index,
                    "average_visit_duration": 35 - period_index,
                    "crm_interaction_count": 16 + period_index,
                    "claimed_expense_amount": 400 + period_index * 10,
                    "distance_claim_ratio": 1 + period_index / 10,
                    "impossible_travel_count": int(flagged),
                    "available_field_hours": 120,
                    "required_total_hours": 130 if flagged else 105,
                    "capacity_utilization_pct": 108 if flagged else 88,
                    "required_fte": 1.1 if flagged else 0.9,
                    "available_fte": 1.0,
                    "fte_gap": 0.1 if flagged else -0.1,
                    "priority_customer_coverage_gap": int(flagged),
                    "capacity_risk_band": "high" if flagged else "medium",
                    "data_lineage": "synthetic_derived",
                }
            )
    _write_csv(directory, "dashboard_rep_period_summary.csv", summary_rows)

    queue_rows = []
    for rank, observation in enumerate(("OBS_1_3", "OBS_2_2"), start=1):
        row = next(item for item in summary_rows if item["observation_id"] == observation).copy()
        row.update(
            review_rank=rank,
            anomaly_percentile=99.0 - rank,
            incentive_residual=row["incentive_calculation_residual"],
            payout_to_peer_median_ratio=1.4,
            driver_1_name="Incentive residual",
            driver_2_name="End-period sales share",
            driver_3_name="Distance claim ratio",
            recommended_review_action="Validate policy tier, source sales, and supporting activity.",
        )
        # Deliberately no benchmark label, anomaly type, or controlled severity.
        queue_rows.append(row)
    _write_csv(directory, "dashboard_manager_review_queue.csv", queue_rows)

    evidence = []
    contributions = []
    peers = []
    for row in summary_rows:
        evidence.append(
            {
                "observation_id": row["observation_id"],
                "rep_id": row["rep_id"],
                "period": row["period"],
                "ground_truth_label": int(row["manager_review_flag"]),
                "anomaly_type": "incorrect_accelerator" if row["manager_review_flag"] else "none",
                "severity": "high" if row["manager_review_flag"] else "none",
                "benchmark_mode_disclosure": "Controlled labels only.",
            }
        )
        for rank, (name, contribution) in enumerate(
            [("Incentive residual", 3.0), ("End period sales share", 2.0), ("Distance claim ratio", 1.0)], start=1
        ):
            contributions.append(
                {
                    "observation_id": row["observation_id"],
                    "population": "injected",
                    "driver_rank": rank,
                    "name": name,
                    "contribution": contribution,
                    "value": contribution,
                    "peer_value": contribution / 2,
                }
            )
        for metric, actual, median in [("net_sales", row["net_sales"], 90_000), ("final_incentive_paid", row["final_incentive_paid"], 1_900)]:
            peers.append(
                {
                    "observation_id": row["observation_id"],
                    "rep_id": row["rep_id"],
                    "period": row["period"],
                    "metric_name": metric,
                    "actual_value": actual,
                    "peer_median_value": median,
                    "peer_z_score": (actual - median) / max(abs(median), 1),
                    "peer_percentile": 0.90,
                    "peer_group_basis": "team / potential / tenure",
                }
            )
    _write_csv(directory, "dashboard_anomaly_evidence.csv", evidence)
    _write_csv(directory, "dashboard_feature_contributions.csv", contributions)
    _write_csv(directory, "dashboard_peer_comparison.csv", peers)

    capacity_rows = []
    customer_rows = []
    for index, (territory, team, gap, risk) in enumerate(
        [("Territory 1", "Team East", 0.6, "high"), ("Territory 3", "Team East", -0.4, "low"), ("Territory 2", "Team West", -0.2, "medium")],
        start=1,
    ):
        capacity_rows.append(
            {
                "rep_id": f"CAP_REP_{index}",
                "rep_name": f"Capacity Rep {index}",
                "manager_name": "Manager A" if team == "Team East" else "Manager B",
                "team_id": "TEAM_1" if team == "Team East" else "TEAM_2",
                "team_name": team,
                "territory_id": f"CAP_TERR_{index}",
                "territory_name": territory,
                "period": "2025-03-01",
                "required_total_hours": 150 if gap > 0 else 90,
                "available_field_hours": 120,
                "capacity_utilization_pct": 125 if gap > 0 else 75,
                "required_fte": 1 + gap,
                "available_fte": 1.0,
                "fte_gap": gap,
                "priority_customer_coverage_gap": 3 if gap > 0 else 0,
                "workload_per_active_customer": 4.5,
                "capacity_risk_band": risk,
                "capacity_overload_flag": gap > 0,
                "data_lineage": "synthetic_derived",
            }
        )
        customer_rows.append(
            {
                "rep_id": f"CAP_REP_{index}",
                "rep_name": f"Capacity Rep {index}",
                "customer_id": f"CUSTOMER_{index}",
                "customer_name": f"Customer {index}",
                "territory_id": f"CAP_TERR_{index}",
                "period": "2025-03-01",
                "customer_priority": "high",
                "required_visit_count": 3,
                "completed_visit_count": 2 if gap > 0 else 3,
                "priority_customer_coverage_gap": 1 if gap > 0 else 0,
                "coverage_status": "Coverage gap" if gap > 0 else "Covered",
                "data_lineage": "synthetic_derived",
            }
        )
    _write_csv(directory, "dashboard_capacity_summary.csv", capacity_rows)
    _write_csv(directory, "dashboard_capacity_customer_drilldown.csv", customer_rows)
    allocation_rows = []
    for row in capacity_rows:
        shares = (
            [(row["territory_id"], row["territory_name"], 0.75), ("CAP_TERR_ALT", "Territory Alternate", 0.25)]
            if row["rep_id"] == "CAP_REP_1"
            else [(row["territory_id"], row["territory_name"], 1.0)]
        )
        for territory_id, territory_name, share in shares:
            allocation_rows.append(
                {
                    "capacity_territory_allocation_id": f"ALLOC_{row['rep_id']}_{territory_id}",
                    "rep_id": row["rep_id"],
                    "rep_name": row["rep_name"],
                    "manager_name": row["manager_name"],
                    "team_id": row["team_id"],
                    "team_name": row["team_name"],
                    "territory_id": territory_id,
                    "territory_name": territory_name,
                    "period": row["period"],
                    "territory_allocation_share": share,
                    "required_total_hours": row["required_total_hours"] * share,
                    "available_field_hours": row["available_field_hours"] * share,
                    "required_fte": row["required_fte"] * share,
                    "available_fte": row["available_fte"] * share,
                    "fte_gap": row["fte_gap"] * share,
                    "priority_customer_coverage_gap": row["priority_customer_coverage_gap"] * share,
                    "capacity_risk_band": row["capacity_risk_band"],
                    "capacity_overload_flag": row["capacity_overload_flag"],
                    "risk_medium_threshold_pct": 85.0,
                    "risk_high_threshold_pct": 100.0,
                    "risk_critical_threshold_pct": 120.0,
                    "overload_threshold_pct": 100.0,
                    "data_lineage": "synthetic_derived",
                }
            )
    _write_csv(
        directory,
        "dashboard_capacity_territory_allocation.csv",
        allocation_rows,
    )
    _write_csv(
        directory,
        "dashboard_capacity_territory_summary.csv",
        [
            {
                "capacity_territory_record_id": f"CAPTERR_{index}",
                "territory_id": row["territory_id"],
                "territory_name": row["territory_name"],
                "period": row["period"],
                "rep_count": 1,
                "fractional_rep_equivalent": 1.0,
                "required_total_hours": row["required_total_hours"],
                "available_field_hours": row["available_field_hours"],
                "capacity_utilization_pct": row["capacity_utilization_pct"],
                "required_fte": row["required_fte"],
                "available_fte": row["available_fte"],
                "fte_gap": row["fte_gap"],
                "priority_customer_coverage_gap": row["priority_customer_coverage_gap"],
                "capacity_risk_band": row["capacity_risk_band"],
                "capacity_overload_flag": row["capacity_overload_flag"],
                "data_lineage": "synthetic_derived",
            }
            for index, row in enumerate(capacity_rows, start=1)
        ],
    )

    _write_csv(
        directory,
        "dashboard_model_metrics.csv",
        [
            {"model": "PCA Reconstruction", "split": "test", "threshold": 1.23, "metric_name": "precision", "metric_value": 0.50},
            {"model": "PCA Reconstruction", "split": "test", "threshold": 1.23, "metric_name": "recall", "metric_value": 0.40},
            {"model": "PCA Reconstruction", "split": "test", "review_fraction": 0.05, "metric_name": "lift", "metric_value": 4.0},
        ],
    )
    _write_csv(
        directory,
        "dashboard_anomaly_type_metrics.csv",
        [
            {"group_kind": "ground_truth", "grouping": "anomaly_type", "value": "incorrect_accelerator", "observations": 3, "recall_at_threshold": 0.67},
            {"group_kind": "ground_truth", "grouping": "severity", "value": "high", "observations": 3, "recall_at_threshold": 0.67},
            {"group_kind": "organization", "grouping": "manager_id", "value": "MGR_1", "observations": 3, "recall_at_threshold": 0.50},
        ],
    )
    _write_csv(
        directory,
        "dashboard_data_quality.csv",
        [
            {"check_name": "missing_rate__sales", "status": "pass", "value": 0, "detail": "No missing sales"},
            {"check_name": "fk__orders_rep", "status": "pass", "value": 0, "detail": "orders.rep_id -> rep_master.rep_id"},
            {"check_name": "row_count__clean__orders", "status": "pass", "value": 100, "detail": "clean generated layer"},
        ],
    )
    _write_csv(
        directory,
        "dashboard_run_manifest.csv",
        [
            {
                "execution_timestamp": "2025-04-01T00:00:00+00:00",
                "random_seed": 42,
                "input_file_name": "pharma-data.csv",
                "input_file_hash": "example-sha256",
                "configuration_hash": "example-config-hash",
                "configuration_file": "synthetic_data.yaml",
                "finalized_model_name": "PCA Reconstruction",
                "scoring_threshold": 1.23,
                "manager_review_fraction": 0.05,
            }
        ],
    )

    curve_rows = []
    for value in (0.0, 0.5, 1.0):
        curve_rows.extend(
            [
                {"curve_type": "roc", "false_positive_rate": value, "true_positive_rate": min(value + 0.2, 1.0)},
                {"curve_type": "precision_recall", "recall": value, "precision": 1.0 - value / 2},
                {"curve_type": "lift", "review_fraction": max(value, 0.01), "lift": 5.0 - value * 3},
            ]
        )
    _write_csv(directory, "dashboard_model_curve.csv", curve_rows)
    _write_csv(
        directory,
        "dashboard_pca_variance.csv",
        [
            {"component": 1, "explained_variance_ratio": 0.60, "cumulative_explained_variance": 0.60},
            {"component": 2, "explained_variance_ratio": 0.35, "cumulative_explained_variance": 0.95},
        ],
    )
    _write_csv(
        directory,
        "dashboard_confusion_matrix.csv",
        [
            {"actual": "positive", "predicted": "positive", "count": 4},
            {"actual": "positive", "predicted": "negative", "count": 6},
            {"actual": "negative", "predicted": "positive", "count": 4},
            {"actual": "negative", "predicted": "negative", "count": 86},
        ],
    )
    _write_csv(
        directory,
        "dashboard_score_distribution.csv",
        [
            {"population": population, "split": "test", "score_lower": lower, "score_upper": lower + 0.5, "count": count}
            for population, lower, count in [("clean", 0.0, 20), ("clean", 0.5, 5), ("injected_anomaly", 0.0, 2), ("injected_anomaly", 0.5, 8)]
        ],
    )
    _write_csv(
        directory,
        "dashboard_period_stability.csv",
        [
            {"population": population, "period": period, "mean_anomaly_score": score, "review_rate": 0.05}
            for population in ("clean", "injected")
            for period, score in [("2025-01-01", 0.40), ("2025-02-01", 0.42), ("2025-03-01", 0.41)]
        ],
    )
    return tmp_path


def _app(root: Path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    source = (
        "from pathlib import Path\n"
        "import streamlit as st\n"
        "from field_rep_anomaly.commercial_review.dashboard import render_commercial_review\n"
        "def table(frame, height=430):\n"
        "    st.dataframe(frame, height=height)\n"
        "def section(title, caption):\n"
        "    st.subheader(title)\n"
        "    st.caption(caption)\n"
        f"render_commercial_review(Path({str(root)!r}), table, section)\n"
    )
    return AppTest.from_string(source, default_timeout=60).run()


def test_contract_reads_only_required_and_declared_optional_csvs(dashboard_root, monkeypatch):
    assert set(dashboard.REQUIRED_DASHBOARD_FILES) == EXPECTED_REQUIRED
    expected = EXPECTED_REQUIRED | set(dashboard.OPTIONAL_DASHBOARD_FILES)
    reads = []
    original = dashboard._read_dashboard_csv

    def recording_read(path, modified_ns):
        reads.append(Path(path).name)
        return original(path, modified_ns)

    monkeypatch.setattr(dashboard, "_read_dashboard_csv", recording_read)
    tables, missing, errors = dashboard._load_tables(dashboard_root)
    assert not missing and not errors
    assert set(tables) == expected
    assert set(reads) == expected
    assert all(Path(name).suffix == ".csv" for name in reads)


def test_capacity_territory_view_uses_allocated_summary_and_active_filters(dashboard_root):
    tables, missing, errors = dashboard._load_tables(dashboard_root)
    assert not missing and not errors
    rep_capacity = tables["dashboard_capacity_summary.csv"]
    rep_capacity = rep_capacity.loc[rep_capacity["territory_id"].eq("CAP_TERR_1")]
    territory_summary = tables["dashboard_capacity_territory_summary.csv"].copy()
    territory_summary.loc[
        territory_summary["territory_id"].eq("CAP_TERR_1"), "required_total_hours"
    ] = 777.0

    view = dashboard._territory_capacity_view(rep_capacity, territory_summary)
    assert view["territory_id"].tolist() == ["CAP_TERR_1"]
    assert view["required_total_hours"].iloc[0] == pytest.approx(777.0)
    assert "capacity_territory_record_id" in view


def test_capacity_filters_use_non_dominant_allocation_territory(dashboard_root):
    tables, missing, errors = dashboard._load_tables(dashboard_root)
    assert not missing and not errors
    reps, allocation = dashboard._filter_capacity_allocation(
        tables["dashboard_capacity_summary.csv"],
        tables["dashboard_capacity_territory_allocation.csv"],
        {
            "manager_name": ["Manager A"],
            "team_name": ["Team East"],
            "territory_name": ["Territory Alternate"],
            "period": ["2025-03-01"],
            "capacity_risk_band": ["high"],
        },
    )
    assert allocation["territory_id"].tolist() == ["CAP_TERR_ALT"]
    assert reps["rep_id"].tolist() == ["CAP_REP_1"]
    territory = dashboard.build_capacity_territory_summary(allocation)
    assert territory["territory_id"].tolist() == ["CAP_TERR_ALT"]
    assert territory["required_total_hours"].iloc[0] == pytest.approx(37.5)


def test_manager_safe_view_drops_controlled_labels_without_requiring_them():
    label_free = pd.DataFrame([{"observation_id": "OBS", "anomaly_score": 0.9}])
    pd.testing.assert_frame_equal(dashboard._manager_safe(label_free), label_free)
    contaminated = label_free.assign(
        ground_truth_label=1,
        anomaly_type="synthetic_test",
        severity="high",
        injection_id="INJ_1",
    )
    safe = dashboard._manager_safe(contaminated)
    assert list(safe.columns) == ["observation_id", "anomaly_score"]


def test_all_seven_pages_start_and_manager_queue_does_not_need_labels(dashboard_root):
    app = _app(dashboard_root)
    assert not app.exception, [item.message for item in app.exception]
    navigation = next(item for item in app.sidebar.radio if item.label == "Commercial review workspace")
    assert tuple(navigation.options) == dashboard.PAGES
    for page in dashboard.PAGES:
        navigation = next(item for item in app.sidebar.radio if item.label == "Commercial review workspace")
        navigation.set_value(page).run()
        assert not app.exception, (page, [item.message for item in app.exception])
        if page == "Manager Review Queue":
            labels = {item.label for item in app.multiselect}
            assert {"Manager", "Team", "Territory", "Representative", "Period", "Score severity", "Reason"} <= labels
            for table in app.dataframe:
                assert not dashboard._LABEL_TOKENS.intersection(table.value.columns)
        if page == "Model Benchmark View":
            assert any("Benchmark mode" in item.value for item in app.warning)


def test_missing_semantic_files_show_readiness_message_without_fallback(tmp_path):
    app = _app(tmp_path)
    assert not app.exception, [item.message for item in app.exception]
    assert any("dashboard-ready datasets are not yet complete" in item.value for item in app.info)
