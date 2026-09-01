"""End-to-end contract test for the relational commercial-review pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from field_rep_anomaly.commercial_review.anomalies import SCENARIOS, inject_controlled_anomalies
from field_rep_anomaly.commercial_review.capacity import build_capacity_calendar
from field_rep_anomaly.commercial_review.dashboard_data import REQUIRED_DASHBOARD_FILES
from field_rep_anomaly.commercial_review.features import build_feature_store, validate_no_future_leakage
from field_rep_anomaly.commercial_review.foundation import profile_and_normalize_source
from field_rep_anomaly.commercial_review.pipeline import run_commercial_review_pipeline
from field_rep_anomaly.commercial_review.policy import calculate_incentives
from field_rep_anomaly.commercial_review.synthetic import build_masters, build_targets, generate_clean_datasets


def _source_fixture(path: Path) -> None:
    rows = []
    months = pd.date_range("2017-01-01", "2019-04-01", freq="MS")
    for rep_number in range(8):
        manager_number = rep_number // 4
        team_number = manager_number
        country = "Germany" if team_number == 0 else "Poland"
        for period_number, period in enumerate(months):
            for customer_number in range(12):
                product_number = (rep_number + customer_number + period_number) % 12
                quantity = 4 + ((rep_number * 3 + customer_number + period_number) % 18)
                price = 45 + product_number * 8
                rows.append(
                    {
                        "Distributor": f"Distributor {customer_number % 4}",
                        "Customer Name": f"Customer {rep_number:02d}-{customer_number:02d}",
                        "City": f"City {team_number}-{customer_number:02d}",
                        "Country": country,
                        "Latitude": 50.0 + team_number + customer_number / 100,
                        "Longitude": 10.0 + team_number + customer_number / 100,
                        "Channel": "Hospital" if customer_number % 2 else "Pharmacy",
                        "Sub-channel": ["Private", "Retail", "Institution"][customer_number % 3],
                        "Product Name": f"Product {product_number:02d}",
                        "Product Class": f"Class {product_number % 6}",
                        "Quantity": quantity,
                        "Price": price,
                        "Sales": quantity * price,
                        "Month": period.strftime("%B"),
                        "Year": period.year,
                        "Name of Sales Rep": f"Rep {rep_number:02d}",
                        "Manager": f"Manager {manager_number}",
                        "Sales Team": f"Team {team_number}",
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_full_pipeline_from_clean_output_directory(tmp_path: Path) -> None:
    source = tmp_path / "pharma-data.csv"
    _source_fixture(source)
    repository = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((repository / "configs" / "synthetic_data.yaml").read_text(encoding="utf-8"))
    config["anomalies"]["rep_period_prevalence"] = 0.12
    config["synthetic"]["visit_sampling_rate"] = {"low": 0.25, "medium": 0.35, "high": 0.48}
    config["output"]["write_small_samples"] = False
    config_path = tmp_path / "synthetic_data.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    output_root = tmp_path / "fresh-output"
    assert not output_root.exists()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    # Reproducibility of normalization with the same seed.
    first, _, _, _ = profile_and_normalize_source(source, seed=42, currency_code="UNK")
    second, _, _, _ = profile_and_normalize_source(source, seed=42, currency_code="UNK")
    pd.testing.assert_series_equal(first["source_row_id"], second["source_row_id"])
    pd.testing.assert_series_equal(first["transaction_date"], second["transaction_date"])

    changed_future_source = first.copy(deep=True)
    last_source_period = pd.to_datetime(changed_future_source["period"]).max()
    changed_future_source.loc[
        pd.to_datetime(changed_future_source["period"]).eq(last_source_period), "sales"
    ] *= 9.0
    original_masters = build_masters(first, config)
    future_changed_masters = build_masters(changed_future_source, config)
    for dataset_name in original_masters:
        pd.testing.assert_frame_equal(
            original_masters[dataset_name], future_changed_masters[dataset_name]
        )

    clean_first = generate_clean_datasets(first, config)
    clean_second = generate_clean_datasets(second, config)
    for dataset_name in clean_first:
        pd.testing.assert_frame_equal(clean_first[dataset_name], clean_second[dataset_name])
    assert clean_first["rep_master"]["currency_code"].eq("UNK").all()
    assert clean_first["customer_master"]["currency_code"].eq("UNK").all()

    # Feature construction consumes the finalized capacity module. Build that
    # derived layer explicitly for both otherwise-identical clean generations.
    for clean_tables, normalized in [(clean_first, first), (clean_second, second)]:
        capacity_calendar, coverage_drilldown = build_capacity_calendar(
            normalized,
            clean_tables["rep_master"],
            clean_tables["customer_master"],
            clean_tables["field_visits"],
            config,
        )
        clean_tables["capacity_calendar"] = capacity_calendar
        clean_tables["capacity_customer_drilldown"] = coverage_drilldown
    pd.testing.assert_frame_equal(
        clean_first["capacity_calendar"], clean_second["capacity_calendar"]
    )
    pd.testing.assert_frame_equal(
        clean_first["capacity_customer_drilldown"],
        clean_second["capacity_customer_drilldown"],
    )

    # Exogenous policy weights cannot look ahead to later training-period sales.
    later_training_source = first.copy(deep=True)
    train_end = pd.Timestamp(config["model"]["train_end"])
    later_training_period = pd.to_datetime(
        later_training_source.loc[
            pd.to_datetime(later_training_source["period"]).le(train_end), "period"
        ]
    ).max()
    later_training_source.loc[
        pd.to_datetime(later_training_source["period"]).eq(later_training_period),
        "sales",
    ] *= 11.0
    later_training_product = build_masters(later_training_source, config)[
        "product_master"
    ]
    pd.testing.assert_frame_equal(
        clean_first["product_master"][
            ["product_id", "incentive_eligible_flag", "incentive_weight"]
        ],
        later_training_product[
            ["product_id", "incentive_eligible_flag", "incentive_weight"]
        ],
    )
    counterfactual_tables = {
        name: frame.copy(deep=True) for name, frame in clean_first.items()
    }
    counterfactual_tables["product_master"] = later_training_product
    counterfactual_incentives = calculate_incentives(
        counterfactual_tables["orders"],
        counterfactual_tables["discount_detail"],
        counterfactual_tables["returns_cancellations"],
        counterfactual_tables["rep_targets_quotas"],
        counterfactual_tables["product_master"],
        counterfactual_tables["rep_master"],
        counterfactual_tables["incentive_policy_rules"],
        str(config["project"]["currency_code"]),
    )
    pd.testing.assert_frame_equal(
        clean_first["incentive_calculations"], counterfactual_incentives
    )
    counterfactual_features, counterfactual_columns, _ = build_feature_store(
        counterfactual_tables, config
    )
    baseline_master_features, baseline_master_columns, _ = build_feature_store(
        clean_first, config
    )
    assert counterfactual_columns == baseline_master_columns
    pd.testing.assert_frame_equal(
        baseline_master_features, counterfactual_features
    )

    future_orders = clean_first["orders"].copy(deep=True)
    final_order_period = pd.to_datetime(future_orders["period"]).max()
    final_order_mask = pd.to_datetime(future_orders["period"]).eq(final_order_period)
    future_orders.loc[final_order_mask, ["gross_sales", "net_sales", "quantity"]] *= 4.0
    rebuilt_targets = build_targets(
        future_orders,
        clean_first["rep_master"],
        clean_first["customer_master"],
        clean_first["product_master"],
        clean_first["territory_master"],
        config,
    )
    pd.testing.assert_frame_equal(clean_first["rep_targets_quotas"], rebuilt_targets)

    late_return_indices = clean_first["returns_cancellations"].index[
        clean_first["returns_cancellations"]["after_incentive_payout_flag"].astype(bool)
    ]
    assert len(late_return_indices) > 0
    late_return_index = late_return_indices[0]
    changed_returns = clean_first["returns_cancellations"].copy(deep=True)
    changed_returns.loc[late_return_index, "return_amount"] *= 25.0
    recalculated = calculate_incentives(
        clean_first["orders"],
        clean_first["discount_detail"],
        changed_returns,
        clean_first["rep_targets_quotas"],
        clean_first["product_master"],
        clean_first["rep_master"],
        clean_first["incentive_policy_rules"],
        str(config["project"]["currency_code"]),
    )
    pd.testing.assert_frame_equal(clean_first["incentive_calculations"], recalculated)
    for clean_layer, normalized in ((clean_first, first), (clean_second, second)):
        calendar, drill = build_capacity_calendar(
            normalized,
            clean_layer["rep_master"],
            clean_layer["customer_master"],
            clean_layer["field_visits"],
            config,
        )
        clean_layer["capacity_calendar"] = calendar
        clean_layer["capacity_customer_drilldown"] = drill
    injected_first, truth_first = inject_controlled_anomalies(clean_first, config)
    injected_second, truth_second = inject_controlled_anomalies(clean_second, config)
    pd.testing.assert_frame_equal(truth_first, truth_second)
    for dataset_name in injected_first:
        pd.testing.assert_frame_equal(injected_first[dataset_name], injected_second[dataset_name])

    baseline_features, baseline_columns, _ = build_feature_store(clean_first, config)
    future_changed = {name: frame.copy(deep=True) for name, frame in clean_first.items()}
    last_period = pd.to_datetime(future_changed["orders"]["period"]).max()
    future_mask = pd.to_datetime(future_changed["orders"]["period"]).eq(last_period)
    future_changed["orders"].loc[future_mask, ["gross_sales", "net_sales"]] *= 3.0
    changed_features, changed_columns, _ = build_feature_store(future_changed, config)
    assert baseline_columns == changed_columns
    validate_no_future_leakage(
        baseline_features,
        changed_features,
        last_period - pd.offsets.MonthBegin(1),
        baseline_columns,
    )

    summary = run_commercial_review_pipeline(config_path, source, output_root)
    assert output_root.exists()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    assert summary["run_manifest"]["source_unchanged"] is True

    clean_dir = output_root / "data" / "generated" / "clean"
    injected_dir = output_root / "data" / "generated" / "injected"
    truth = pd.read_parquet(output_root / "data" / "generated" / "benchmark" / "anomaly_ground_truth.parquet")
    orders = pd.read_parquet(clean_dir / "orders.parquet")
    discounts = pd.read_parquet(clean_dir / "discount_detail.parquet")
    returns = pd.read_parquet(clean_dir / "returns_cancellations.parquet")
    visits = pd.read_parquet(clean_dir / "field_visits.parquet")
    expenses = pd.read_parquet(clean_dir / "travel_expenses.parquet")
    incentives = pd.read_parquet(clean_dir / "incentive_calculations.parquet")
    injected_capacity = pd.read_parquet(injected_dir / "capacity_calendar.parquet")
    injected_orders = pd.read_parquet(injected_dir / "orders.parquet")
    injected_discounts = pd.read_parquet(injected_dir / "discount_detail.parquet")
    injected_returns = pd.read_parquet(injected_dir / "returns_cancellations.parquet")
    injected_visits = pd.read_parquet(injected_dir / "field_visits.parquet")
    injected_expenses = pd.read_parquet(injected_dir / "travel_expenses.parquet")
    injected_incentives = pd.read_parquet(injected_dir / "incentive_calculations.parquet")
    features = pd.read_parquet(injected_dir / "feature_store_rep_period.parquet")

    # Primary keys and foreign keys.
    assert orders["order_line_id"].is_unique
    assert discounts["discount_id"].is_unique
    assert visits["visit_id"].is_unique
    assert expenses["expense_id"].is_unique
    assert set(discounts["order_line_id"]).issubset(set(orders["order_line_id"]))
    assert set(returns["order_line_id"]).issubset(set(orders["order_line_id"]))
    assert set(expenses["visit_id"]).issubset(set(visits["visit_id"]))

    # Chronology, centralized policy reconciliation, and no leakage/nonfinite inputs.
    assert pd.to_datetime(returns["return_date"]).ge(pd.to_datetime(returns["original_order_date"])).all()
    assert pd.to_datetime(incentives["payout_date"]).gt(pd.to_datetime(incentives["period"]) + pd.offsets.MonthEnd(1)).all()
    assert pd.to_datetime(injected_orders["order_date"]).le(pd.to_datetime(injected_orders["invoice_date"])).all()
    assert pd.to_datetime(injected_orders["invoice_date"]).le(pd.to_datetime(injected_orders["fulfillment_date"])).all()
    linked_order_date = injected_returns["order_line_id"].map(
        injected_orders.set_index("order_line_id")["order_date"]
    )
    assert pd.to_datetime(injected_returns["original_order_date"]).eq(pd.to_datetime(linked_order_date)).all()
    elapsed = (
        pd.to_datetime(injected_visits["actual_end_time"])
        - pd.to_datetime(injected_visits["actual_start_time"])
    ).dt.total_seconds() / 60.0
    np.testing.assert_allclose(elapsed, injected_visits["visit_duration_minutes"])
    clean_schedule = visits.sort_values(
        ["rep_id", "period", "actual_start_time"], kind="mergesort"
    ).copy()
    clean_schedule["prior_end"] = clean_schedule.groupby(
        ["rep_id", "period"], observed=True
    )["actual_end_time"].transform(
        lambda values: pd.to_datetime(values).shift().cummax()
    )
    assert (
        clean_schedule["prior_end"].isna()
        | pd.to_datetime(clean_schedule["actual_start_time"]).ge(
            clean_schedule["prior_end"]
        )
    ).all()
    assert not visits.duplicated(["rep_id", "actual_start_time"]).any()
    assert visits.groupby(["rep_id", "period"], observed=True).size().max() <= int(
        config["synthetic"]["max_normal_visits_per_rep_period"]
    )
    expected_deviation = 100 * (
        injected_expenses["claimed_amount"] - injected_expenses["expected_amount"]
    ) / injected_expenses["expected_amount"]
    np.testing.assert_allclose(expected_deviation, injected_expenses["deviation_pct"])
    reconciliation = (incentives["final_incentive_paid"] - incentives["calculated_incentive"] - incentives["manual_adjustment"]).abs()
    assert reconciliation.le(incentives["reconciliation_tolerance"]).all()
    below_minimum = incentives["attainment_pct"].lt(50.0)
    assert below_minimum.any()
    assert incentives.loc[below_minimum, "final_incentive_paid"].eq(0.0).all()
    policy = pd.read_parquet(clean_dir / "incentive_policy_rules.parquet")
    decelerated = incentives.loc[
        incentives["attainment_pct"].between(50.0, 99.999, inclusive="both")
    ].iloc[0]
    applicable_rule = policy.loc[
        policy["policy_version"].eq(decelerated["policy_version"])
        & policy["lower_attainment_pct"].le(decelerated["attainment_pct"])
        & policy["upper_attainment_pct"].gt(decelerated["attainment_pct"])
    ].iloc[0]
    expected_base = round(
        decelerated["eligible_net_sales"]
        * applicable_rule["payout_rate"]
        * (decelerated["attainment_pct"] / 100.0)
        * applicable_rule["decelerator_multiplier"],
        2,
    )
    assert decelerated["base_incentive"] == pytest.approx(expected_base, abs=0.01)
    assert not any(token in column.casefold() for column in features for token in ["ground_truth", "anomaly_type", "severity", "injection_id"])
    numeric = features.select_dtypes(include=[np.number])
    assert numeric.notna().all().all()
    assert np.isfinite(numeric.to_numpy(float)).all()

    # Every controlled type/severity, configured capacity truth, and independent clean layer.
    assert {item[0] for item in SCENARIOS}.issubset(set(truth["anomaly_type"]))
    assert {"low", "medium", "high"}.issubset(set(truth["severity"]))
    expected_mix = pd.Series(config["anomalies"]["severity_mix"], dtype=float)
    expected_mix /= expected_mix.sum()
    observed_mix = truth["severity"].value_counts(normalize=True).reindex(
        expected_mix.index, fill_value=0.0
    )
    assert (observed_mix - expected_mix).abs().le(
        max(2.0 / len(truth), 0.03)
    ).all()
    capacity_truth = truth.loc[truth["anomaly_type"].eq("territory_workload_exceeds_capacity")]
    assert len(capacity_truth) > 0
    assert injected_capacity["data_lineage"].eq("synthetic_injected").sum() >= round(len(injected_capacity) * 0.10)
    np.testing.assert_allclose(
        injected_capacity["required_total_hours"],
        injected_capacity[
            [
                "planned_visit_hours",
                "planned_travel_hours",
                "required_customer_coverage_hours",
                "workload_buffer_hours",
            ]
        ].sum(axis=1),
    )
    persistent = truth.loc[
        truth["anomaly_type"].eq("persistent_priority_undercoverage")
    ].iloc[0]
    persistent_ids = [
        value for value in json.loads(persistent["affected_record_ids"])
        if str(value).startswith("CAPCAL_")
    ]
    assert len(persistent_ids) >= 2
    persistent_periods = injected_capacity.loc[
        injected_capacity["capacity_record_id"].isin(persistent_ids), "period"
    ].sort_values()
    assert len(persistent_periods) == len(persistent_ids)
    assert all(
        later == earlier + pd.offsets.MonthBegin(1)
        for earlier, later in zip(persistent_periods.iloc[:-1], persistent_periods.iloc[1:])
    )
    commercial_truth = truth.loc[~truth["anomaly_category"].eq("capacity")]
    correlated_share = (
        commercial_truth.groupby(["rep_id", "period"])["correlated_case_flag"].max().mean()
    )
    assert correlated_share == pytest.approx(config["anomalies"]["correlated_case_share"], abs=0.06)
    assert injected_orders["data_lineage"].eq("synthetic_injected").mean() == pytest.approx(
        config["anomalies"]["order_level_prevalence"], abs=0.005
    )
    multi = truth.loc[truth["anomaly_type"].eq("multi_signal_sales_discount_returns")].iloc[0]
    affected_ids = set(json.loads(multi["affected_record_ids"]))
    affected_discounts = injected_discounts.loc[injected_discounts["discount_id"].isin(affected_ids)]
    assert len(affected_discounts) > 0
    assert affected_discounts["data_lineage"].eq("synthetic_injected").all()
    activity_case = truth.loc[
        truth["anomaly_type"].eq("high_activity_low_engagement")
    ].iloc[0]
    clean_activity = baseline_features.loc[
        baseline_features["rep_id"].eq(activity_case["rep_id"])
        & pd.to_datetime(baseline_features["period"]).eq(
            pd.Timestamp(activity_case["period"])
        )
    ].iloc[0]
    injected_activity = features.loc[
        features["rep_id"].eq(activity_case["rep_id"])
        & pd.to_datetime(features["period"]).eq(pd.Timestamp(activity_case["period"]))
    ].iloc[0]
    assert (
        injected_activity["completed_visit_count"]
        > clean_activity["completed_visit_count"]
    )
    assert (
        injected_activity["visit_to_sales_conversion"]
        < clean_activity["visit_to_sales_conversion"]
    )
    territory_case = truth.loc[
        truth["anomaly_type"].eq("territory_potential_explained_performance")
    ].iloc[0]
    clean_territory_case = baseline_features.loc[
        baseline_features["rep_id"].eq(territory_case["rep_id"])
        & pd.to_datetime(baseline_features["period"]).eq(
            pd.Timestamp(territory_case["period"])
        )
    ].iloc[0]
    injected_territory_case = features.loc[
        features["rep_id"].eq(territory_case["rep_id"])
        & pd.to_datetime(features["period"]).eq(pd.Timestamp(territory_case["period"]))
    ].iloc[0]
    assert abs(injected_territory_case["territory_adjusted_sales_residual"]) < abs(
        clean_territory_case["territory_adjusted_sales_residual"]
    )
    threshold_case = truth.loc[
        truth["anomaly_type"].eq("threshold_crossing_discount")
    ].iloc[0]
    threshold_incentive = injected_incentives.loc[
        injected_incentives["rep_id"].eq(threshold_case["rep_id"])
        & pd.to_datetime(injected_incentives["period"]).eq(pd.Timestamp(threshold_case["period"]))
    ].iloc[0]
    assert 100.0 <= threshold_incentive["attainment_pct"] <= 106.0

    # Dashboard and executed model/capacity outputs are all non-empty.
    dashboard_dir = output_root / "data" / "dashboard"
    for filename in REQUIRED_DASHBOARD_FILES:
        frame = pd.read_csv(dashboard_dir / filename)
        assert len(frame) > 0, filename
    dashboard_manifest = pd.read_csv(dashboard_dir / "dashboard_run_manifest.csv").iloc[0]
    assert bool(dashboard_manifest["source_unchanged"])
    embedded_counts = json.loads(dashboard_manifest["output_row_counts"])
    for filename in REQUIRED_DASHBOARD_FILES:
        assert embedded_counts[filename] == len(pd.read_csv(dashboard_dir / filename)), filename
    dashboard_quality = pd.read_csv(dashboard_dir / "dashboard_data_quality.csv")
    assert dashboard_quality["check_name"].str.startswith("row_count__clean__").any()
    for filename in ["dashboard_capacity_summary.csv", "dashboard_capacity_customer_drilldown.csv"]:
        production_capacity = pd.read_csv(dashboard_dir / filename)
        assert not production_capacity["data_lineage"].eq("synthetic_injected").any()
    queue = pd.read_csv(dashboard_dir / "dashboard_manager_review_queue.csv")
    assert not any("ground_truth" in column.casefold() for column in queue)
    assert queue["threshold_flag"].astype(bool).all()
    injected_scores = pd.read_csv(
        output_root
        / "artifacts"
        / "commercial_review"
        / "model"
        / "pca_injected_scores.csv"
    )
    assert set(queue["observation_id"]) == set(
        injected_scores.loc[
            injected_scores["threshold_flag"].astype(bool), "observation_id"
        ]
    )
    assert "payout_date" in pd.read_csv(
        dashboard_dir / "dashboard_rep_period_summary.csv", nrows=1
    ).columns
    type_metrics = pd.read_csv(dashboard_dir / "dashboard_anomaly_type_metrics.csv")
    reported_types = set(
        type_metrics.loc[type_metrics["grouping"].eq("anomaly_type"), "value"]
    )
    assert {item[0] for item in SCENARIOS} <= reported_types
    scenario_metrics = type_metrics.loc[
        type_metrics["grouping"].eq("anomaly_type")
        & type_metrics["value"].isin({item[0] for item in SCENARIOS})
    ]
    assert scenario_metrics["positive_support"].gt(0).all()
    capacity_types = {
        "territory_workload_exceeds_capacity",
        "persistent_priority_undercoverage",
    }
    assert scenario_metrics.loc[
        ~scenario_metrics["value"].isin(capacity_types), "support_status"
    ].eq("evaluated_on_final_test").all()
    assert scenario_metrics.loc[
        scenario_metrics["value"].isin(capacity_types), "support_status"
    ].eq("evaluated_by_capacity_rule").all()
    severity_metrics = type_metrics.loc[
        type_metrics["grouping"].eq("severity")
        & type_metrics["value"].isin({"low", "medium", "high"})
    ]
    assert set(severity_metrics["value"]) == {"low", "medium", "high"}
    assert severity_metrics["positive_support"].gt(0).all()
    confusion = pd.read_csv(dashboard_dir / "dashboard_confusion_matrix.csv")
    assert confusion["count"].sum() == summary["model_metrics"]["test_rows"]
    assert confusion["count"].gt(0).any()
    assert summary["feature_store_shape"][0] == len(features)
    assert np.isfinite(float(summary["pca_metadata"]["threshold"]))
    assert summary["capacity_metrics"][0]["ground_truth_overload_count"] > 0
    assert summary["run_manifest"]["output_row_counts"]["data/generated/clean/orders.parquet"] == len(orders)
    assert summary["run_manifest"]["output_row_counts"]["reports/final_benchmark_report.md"] is None
    assert "artifacts/commercial_review/source_profile.json" in summary["run_manifest"]["output_file_names"]
    assert "artifacts/commercial_review/model/pca_metadata.json" in summary["run_manifest"]["output_file_names"]
    assert {"commercial_pca", "capacity"} == set(
        summary["run_manifest"]["evaluation_metrics"]
    )
    assert len(summary["run_manifest"]["implementation_sha256"]) == 64
    assert summary["run_manifest"]["modeling_row_count"] == summary[
        "source_profile"
    ]["modeling_rows"]
    assert summary["run_manifest"]["primary_analytical_grain"] == (
        "representative x month"
    )
    assert summary["run_manifest"]["model_feature_count"] == summary[
        "pca_metadata"
    ]["feature_count"]
    assert {"python", "pandas", "numpy", "scikit_learn", "streamlit"} <= set(
        summary["run_manifest"]["key_software_versions"]
    )
    dictionary_text = (output_root / "docs" / "commercial_review_data_dictionary.md").read_text(encoding="utf-8")
    assert "dashboard_confusion_matrix" in dictionary_text
    assert "model_false_positive_review" in dictionary_text
