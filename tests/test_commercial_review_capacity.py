"""Focused contracts for the additive deterministic capacity layer."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from field_rep_anomaly.commercial_review.capacity import (
    CAPACITY_ALLOCATION_ADDITIVE_COLUMNS,
    LEGACY_WORKLOAD_METRICS,
    build_capacity_calendar,
    build_capacity_territory_allocation,
    build_capacity_territory_summary,
    evaluate_capacity,
)


@pytest.fixture
def capacity_inputs():
    transactions = pd.DataFrame(
        [
            ("R1", "C1", "2024-01-01", "Alpha", "P1", "D1"),
            ("R1", "C2", "2024-01-01", "Beta", "P2", "D1"),
            ("R2", "C3", "2024-01-01", "Gamma", "P1", "D2"),
            ("R1", "C1", "2024-02-01", "Alpha", "P1", "D1"),
            ("R1", "C1", "2024-02-01", "Alpha", "P2", "D1"),
            ("R2", "C3", "2024-02-01", "Gamma", "P1", "D2"),
            ("R2", "C4", "2024-02-01", "Delta", "P3", "D3"),
        ],
        columns=["rep_id", "customer_id", "period", "city", "product_id", "distributor"],
    )
    rep_master = pd.DataFrame(
        [
            {
                "rep_id": "R1",
                "rep_name": "Rep One",
                "manager_id": "M1",
                "manager_name": "Manager One",
                "team_id": "TEAM1",
                "team_name": "Team One",
                "territory_id": "T1",
                "territory_name": "North",
                "standard_field_hours_per_day": 8.0,
                "standard_working_days_per_month": 20.0,
                "leave_days": 1.0,
                "holiday_days": 1.0,
                "training_hours": 4.0,
                "administrative_hours": 8.0,
                "meeting_hours": 4.0,
            },
            {
                "rep_id": "R2",
                "rep_name": "Rep Two",
                "manager_id": "M2",
                "manager_name": "Manager Two",
                "team_id": "TEAM2",
                "team_name": "Team Two",
                "territory_id": "T2",
                "territory_name": "South",
                "standard_field_hours_per_day": 8.0,
                "standard_working_days_per_month": 20.0,
                "leave_days": 1.0,
                "holiday_days": 1.0,
                "training_hours": 4.0,
                "administrative_hours": 8.0,
                "meeting_hours": 4.0,
            },
        ]
    )
    customer_master = pd.DataFrame(
        [
            ("C1", "Customer One", "R1", "T1", "High", 2.0, 0.90),
            ("C2", "Customer Two", "R1", "T1", "Medium", 1.0, 0.60),
            ("C3", "Customer Three", "R2", "T2", "High", 2.0, 0.85),
            ("C4", "Customer Four", "R2", "T2", "Low", 0.5, 0.25),
        ],
        columns=[
            "customer_id",
            "customer_name",
            "primary_rep_id",
            "territory_id",
            "customer_priority",
            "required_visit_frequency",
            "potential_score",
        ],
    )
    field_visits = pd.DataFrame(
        [
            ("V1", "R1", "C1", "2024-01-12", 60.0, 42.0, True),
            ("V2", "R1", "C2", "2024-01-18", 30.0, 21.0, False),
            ("V3", "R2", "C3", "2024-01-15", 45.0, 21.0, True),
            ("V4", "R1", "C1", "2024-02-10", 60.0, 42.0, True),
            ("V5", "R2", "C3", "2024-02-11", 45.0, 21.0, True),
            ("V6", "R2", "C4", "2024-02-21", 30.0, 21.0, True),
        ],
        columns=[
            "visit_id",
            "rep_id",
            "customer_id",
            "visit_date",
            "visit_duration_minutes",
            "estimated_travel_km",
            "visit_completed_flag",
        ],
    )
    config = {
        "project": {"seed": 42},
        "model": {"train_end": "2024-01-01"},
        "synthetic": {"average_visit_minutes": 42, "average_speed_kmh": 42.0},
        "capacity": {
            "numeric_visit_frequency_period_divisor": 1.0,
            "workload_weights": {
                "distinct_customers": 0.40,
                "transaction_count": 0.25,
                "distinct_cities": 0.15,
                "distinct_products": 0.10,
                "distributor_count": 0.10,
            },
            "visit_hours_per_required_call": 0.70,
            "default_travel_hours_per_visit": 0.50,
            "administrative_buffer_pct": 0.08,
            "risk_thresholds": {"medium": 0.85, "high": 1.0, "critical": 1.2},
        },
    }
    return transactions, rep_master, customer_master, field_visits, config


def test_capacity_calendar_is_reproducible_finite_and_reconciled(capacity_inputs):
    snapshots = [frame.copy(deep=True) if isinstance(frame, pd.DataFrame) else frame for frame in capacity_inputs]
    first_calendar, first_drill = build_capacity_calendar(*capacity_inputs)
    second_calendar, second_drill = build_capacity_calendar(*capacity_inputs)

    pd.testing.assert_frame_equal(first_calendar, second_calendar)
    pd.testing.assert_frame_equal(first_drill, second_drill)
    for original, snapshot in zip(capacity_inputs[:4], snapshots[:4]):
        pd.testing.assert_frame_equal(original, snapshot)

    assert not first_calendar.duplicated(["rep_id", "period"]).any()
    assert first_calendar["capacity_record_id"].is_unique
    assert not first_drill.duplicated(["rep_id", "period", "customer_id"]).any()
    assert first_calendar["data_lineage"].eq("synthetic_derived").all()
    assert first_drill["data_lineage"].eq("synthetic_derived").all()
    assert first_calendar["synthetic_seed"].eq(42).all()
    assert np.isfinite(first_calendar.select_dtypes(include=[np.number])).all().all()
    assert np.isfinite(first_drill.select_dtypes(include=[np.number])).all().all()
    assert first_calendar["observed_visit_count"].sum() == pytest.approx(len(capacity_inputs[3]))
    assert first_drill["observed_visit_count"].sum() == pytest.approx(len(capacity_inputs[3]))
    np.testing.assert_allclose(
        first_drill["planned_visit_hours"] + first_drill["excess_service_visit_hours"],
        first_drill["observed_visit_hours"],
    )
    np.testing.assert_allclose(
        first_drill["planned_travel_hours"] + first_drill["excess_service_travel_hours"],
        first_drill["observed_travel_hours"],
    )
    np.testing.assert_allclose(
        first_calendar["planned_visit_hours"]
        + first_calendar["excess_service_visit_hours"],
        first_calendar["observed_visit_hours"],
    )
    np.testing.assert_allclose(
        first_calendar["planned_travel_hours"]
        + first_calendar["excess_service_travel_hours"],
        first_calendar["observed_travel_hours"],
    )

    np.testing.assert_allclose(
        first_calendar["available_field_hours"],
        first_calendar["gross_rostered_field_hours"]
        - first_calendar["training_hours"]
        - first_calendar["administrative_hours"]
        - first_calendar["meeting_hours"],
    )
    np.testing.assert_allclose(
        first_calendar["required_total_hours"],
        first_calendar["planned_visit_hours"]
        + first_calendar["planned_travel_hours"]
        + first_calendar["required_customer_coverage_hours"]
        + first_calendar["workload_buffer_hours"],
    )
    np.testing.assert_allclose(
        first_calendar["utilization_pct"],
        first_calendar["required_total_hours"] / first_calendar["available_field_hours"] * 100.0,
    )
    np.testing.assert_allclose(
        first_calendar["required_fte"],
        first_calendar["required_total_hours"] / first_calendar["nominal_full_time_hours"],
    )
    np.testing.assert_allclose(
        first_calendar["available_fte"],
        first_calendar["available_field_hours"] / first_calendar["nominal_full_time_hours"],
    )
    np.testing.assert_allclose(
        first_calendar["fte_gap"], first_calendar["required_fte"] - first_calendar["available_fte"]
    )
    assert set(first_calendar["workload_risk_band"]) <= {"low", "medium", "high", "critical"}

    r1_january = first_drill.loc[
        first_drill.rep_id.eq("R1") & first_drill.period.eq(pd.Timestamp("2024-01-01"))
    ].set_index("customer_id")
    assert r1_january.loc["C1", "planned_coverage_gap_count"] == pytest.approx(1.0)
    assert r1_january.loc["C1", "priority_customer_coverage_gap"] == pytest.approx(1.0)
    assert r1_january.loc["C2", "planned_coverage_gap_count"] == pytest.approx(0.0)
    assert r1_january.loc["C2", "customer_coverage_gap"] == pytest.approx(1.0)


def test_legacy_workload_index_uses_configurable_weights_and_training_medians(capacity_inputs):
    transactions, reps, customers, visits, config = capacity_inputs
    changed = {
        **config,
        "capacity": {
            **config["capacity"],
            "workload_weights": {
                "distinct_customers": 1.0,
                "transaction_count": 0.0,
                "distinct_cities": 0.0,
                "distinct_products": 0.0,
                "distributor_count": 0.0,
            },
            "training_medians": {metric: (4.0 if metric == "distinct_customers" else 1.0) for metric in LEGACY_WORKLOAD_METRICS},
        },
    }
    calendar, _ = build_capacity_calendar(transactions, reps, customers, visits, changed)
    row = calendar.loc[
        calendar.rep_id.eq("R1") & calendar.period.eq(pd.Timestamp("2024-01-01"))
    ].iloc[0]
    assert row.legacy_distinct_customers_training_median == pytest.approx(4.0)
    assert row.legacy_normalized_workload_index == pytest.approx(2.0 / 4.0)


def test_bare_capacity_config_and_missing_month_calendar_are_supported(capacity_inputs):
    transactions, reps, customers, visits, _ = capacity_inputs
    transactions = transactions.copy()
    transactions.loc[transactions["period"].eq("2024-02-01"), "period"] = "2024-03-01"
    visits = visits.copy()
    visits["visit_date"] = pd.to_datetime(visits["visit_date"])
    visits.loc[visits["visit_date"].dt.month.eq(2), "visit_date"] += pd.offsets.MonthBegin(1)

    calendar, drill = build_capacity_calendar(
        transactions,
        reps,
        customers,
        visits,
        {"numeric_visit_frequency_period_divisor": 1.0},
    )

    assert calendar["period"].drop_duplicates().tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-02-01"),
        pd.Timestamp("2024-03-01"),
    ]
    assert calendar.loc[calendar["period"].eq("2024-02-01"), "transaction_count"].eq(0).all()
    assert drill.loc[
        drill["customer_id"].eq("C1") & drill["period"].eq("2024-02-01"),
        "required_visit_count",
    ].iloc[0] == pytest.approx(2.0)


def test_period_specific_territory_owner_and_cross_servicing_attribution():
    transactions = pd.DataFrame(
        [
            ("R1", "C1", "2024-01-01", "T_B", "Beta"),
            ("R1", "C1", "2024-01-01", "T_A", "Alpha"),
            ("R1", "C1", "2024-01-01", "T_B", "Beta"),
            ("R1", "C1", "2024-01-01", "T_A", "Alpha"),
            ("R2", "C2", "2024-02-01", "T_R2", "South"),
            ("R1", "C1", "2024-03-01", "T_C", "Central"),
            ("R2", "C1", "2024-03-01", "T_R2", "South"),
            ("R2", "C1", "2024-03-01", "T_R2", "South"),
            ("R2", "C1", "2024-03-01", "T_R2", "South"),
        ],
        columns=["rep_id", "customer_id", "period", "territory_id", "territory_name"],
    )
    reps = pd.DataFrame(
        [
            {
                "rep_id": "R1",
                "territory_id": "T0",
                "territory_name": "Roster fallback",
                "standard_field_hours_per_day": 8.0,
                "standard_working_days_per_month": 20.0,
            },
            {
                "rep_id": "R2",
                "territory_id": "T_R2",
                "territory_name": "South",
                "standard_field_hours_per_day": 8.0,
                "standard_working_days_per_month": 20.0,
            },
        ]
    )
    customers = pd.DataFrame(
        [
            ("C1", "R1", "T0", "high", 2.0),
            ("C2", "R2", "T_R2", "low", 1.0),
        ],
        columns=[
            "customer_id",
            "primary_rep_id",
            "territory_id",
            "customer_priority",
            "required_visit_frequency",
        ],
    )
    visits = pd.DataFrame(
        [
            ("V1", "R1", "C1", "2024-01-05", 60.0, 42.0, True),
            ("V2", "R2", "C1", "2024-01-06", 60.0, 42.0, True),
            ("V3", "R2", "C1", "2024-01-07", 60.0, 42.0, True),
            ("V4", "R2", "C1", "2024-01-08", 60.0, 42.0, True),
        ],
        columns=[
            "visit_id",
            "rep_id",
            "customer_id",
            "visit_date",
            "visit_duration_minutes",
            "estimated_travel_km",
            "visit_completed_flag",
        ],
    )
    config = {
        "project": {"seed": 7},
        "synthetic": {"average_visit_minutes": 60.0, "average_speed_kmh": 42.0},
        "capacity": {
            "numeric_visit_frequency_period_divisor": 1.0,
            "administrative_buffer_pct": 0.0,
        },
    }

    calendar, drill = build_capacity_calendar(transactions, reps, customers, visits, config)

    r1_territory = calendar.loc[calendar["rep_id"].eq("R1")].set_index("period")
    assert r1_territory.loc[pd.Timestamp("2024-01-01"), "territory_id"] == "T_A"
    assert r1_territory.loc[pd.Timestamp("2024-01-01"), "territory_name"] == "Alpha"
    assert r1_territory.loc[pd.Timestamp("2024-02-01"), "territory_id"] == "T0"
    assert r1_territory.loc[pd.Timestamp("2024-03-01"), "territory_id"] == "T_C"

    c1 = drill.loc[drill["customer_id"].eq("C1")].set_index("period")
    assert c1.loc[pd.Timestamp("2024-01-01"), "rep_id"] == "R1"
    assert c1.loc[pd.Timestamp("2024-02-01"), "rep_id"] == "R1"
    assert c1.loc[pd.Timestamp("2024-03-01"), "rep_id"] == "R2"
    assert not drill.duplicated(["rep_id", "period", "customer_id"]).any()

    march = calendar.loc[calendar["period"].eq(pd.Timestamp("2024-03-01"))].set_index("rep_id")
    assert march.loc["R1", "required_visit_count"] == pytest.approx(0.0)
    assert march.loc["R2", "required_visit_count"] == pytest.approx(3.0)

    january = calendar.loc[calendar["period"].eq(pd.Timestamp("2024-01-01"))].set_index("rep_id")
    assert january.loc["R1", "observed_visit_count"] == pytest.approx(1.0)
    assert january.loc["R2", "observed_visit_count"] == pytest.approx(3.0)
    assert january.loc["R1", "planned_visit_count"] == pytest.approx(0.5)
    assert january.loc["R2", "planned_visit_count"] == pytest.approx(1.5)
    assert january.loc["R1", "planned_visit_hours"] == pytest.approx(0.5)
    assert january.loc["R2", "planned_visit_hours"] == pytest.approx(1.5)
    assert january.loc["R1", "planned_travel_hours"] == pytest.approx(0.5)
    assert january.loc["R2", "planned_travel_hours"] == pytest.approx(1.5)
    assert january.loc["R1", "excess_service_hours"] == pytest.approx(1.0)
    assert january.loc["R2", "excess_service_hours"] == pytest.approx(3.0)
    assert january["planned_visit_count"].sum() == pytest.approx(
        c1.loc[pd.Timestamp("2024-01-01"), "planned_visit_count"]
    )
    assert january["required_total_hours"].sum() == pytest.approx(4.0)


def test_capacity_evaluation_uses_separate_truth_and_numeric_truth_when_available(capacity_inputs):
    clean, _ = build_capacity_calendar(*capacity_inputs)
    injected = clean.copy(deep=True)
    selected = injected.index[
        injected.rep_id.eq("R1") & injected.period.eq(pd.Timestamp("2024-01-01"))
    ][0]
    available = float(injected.at[selected, "available_field_hours"])
    nominal = float(injected.at[selected, "nominal_full_time_hours"])
    required = 200.0
    utilization = required / available * 100.0
    injected.at[selected, "required_total_hours"] = required
    injected.at[selected, "required_hours"] = required
    injected.at[selected, "utilization_pct"] = utilization
    injected.at[selected, "capacity_utilization_pct"] = utilization
    injected.at[selected, "required_fte"] = required / nominal
    injected.at[selected, "fte_gap"] = required / nominal - injected.at[selected, "available_fte"]
    # Evaluation must recompute the deterministic rule, not consume injected flags.
    injected.at[selected, "overload_flag"] = False
    injected.at[selected, "capacity_overload_flag"] = False
    injected.at[selected, "workload_risk_band"] = "critical"
    injected.at[selected, "capacity_risk_band"] = "critical"

    truth = pd.DataFrame(
        [
            {
                "injection_id": "CAP_001",
                "entity_type": "capacity_record",
                "entity_id": clean.at[selected, "capacity_record_id"],
                "rep_id": "R1",
                "territory_id": "T1",
                "period": "2024-01-01",
                "anomaly_type": "territory_workload_exceeds_available_capacity",
                "ground_truth_label": 1,
                "affected_record_ids": json.dumps([clean.at[selected, "capacity_record_id"]]),
                "injected_value": json.dumps(
                    {
                        "required_total_hours": required,
                        "available_field_hours": available,
                    }
                ),
            },
            {
                "injection_id": "CAP_002",
                "entity_type": "rep_period",
                "entity_id": "R2|2024-01-01",
                "rep_id": "R2",
                "territory_id": "T2",
                "period": "2024-01-01",
                "anomaly_type": "persistent_priority_undercoverage",
                "ground_truth_label": 1,
                "affected_record_ids": "[]",
                "injected_value": json.dumps({"priority_customer_coverage_gap": 5}),
            }
        ]
    )
    result = evaluate_capacity(clean, injected, truth).iloc[0]
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.true_positive == 1
    assert result.false_positive == 0
    assert result.false_negative == 0
    assert result.ground_truth_overload_count == 1
    assert result.capacity_ground_truth_row_count == 2
    assert result.undercoverage_ground_truth_row_count == 1
    assert result.predicted_overload_count == 1
    assert result.mae_required_total_hours == pytest.approx(0.0)
    assert result.mae_utilization_pct == pytest.approx(0.0)
    assert result.workload_mae_observations == 1
    assert np.isnan(result.territory_ranking_agreement)
    assert np.isfinite(result.territory_allocation_sensitivity_spearman)
    assert result.territory_allocation_sensitivity_spearman > 0
    assert not result.territory_truth_independent_flag
    assert result.overloaded_territory_count == 1
    assert result.total_fte_gap == pytest.approx(result.total_required_fte - result.total_available_fte)


def test_multi_territory_allocation_normalizes_and_conserves_additive_capacity():
    period = pd.Timestamp("2024-01-01")
    calendar = pd.DataFrame(
        [
            {
                "capacity_record_id": "CAP_R1",
                "rep_id": "R1",
                "rep_name": "Rep One",
                "manager_id": "M1",
                "team_id": "TEAM1",
                "territory_id": "T_B",
                "territory_name": "Beta",
                "period": period,
                "transaction_count": 4.0,
                "available_field_hours": 80.0,
                "required_total_hours": 100.0,
                "required_hours": 100.0,
                "nominal_full_time_hours": 100.0,
                "required_fte": 1.0,
                "available_fte": 0.8,
                "fte_gap": 0.2,
                "risk_medium_threshold_pct": 85.0,
                "risk_high_threshold_pct": 100.0,
                "risk_critical_threshold_pct": 120.0,
                "overload_threshold_pct": 100.0,
                "capacity_risk_band": "critical",
                "data_lineage": "synthetic_derived",
            },
            {
                "capacity_record_id": "CAP_R2",
                "rep_id": "R2",
                "rep_name": "Rep Two",
                "manager_id": "M1",
                "team_id": "TEAM1",
                "territory_id": "T_B",
                "territory_name": "Beta",
                "period": period,
                "transaction_count": 2.0,
                "available_field_hours": 100.0,
                "required_total_hours": 50.0,
                "required_hours": 50.0,
                "nominal_full_time_hours": 100.0,
                "required_fte": 0.5,
                "available_fte": 1.0,
                "fte_gap": -0.5,
                "risk_medium_threshold_pct": 85.0,
                "risk_high_threshold_pct": 100.0,
                "risk_critical_threshold_pct": 120.0,
                "overload_threshold_pct": 100.0,
                "capacity_risk_band": "low",
                "data_lineage": "synthetic_derived",
            },
        ]
    )
    transactions = pd.DataFrame(
        [
            ("R1", "C1", period, "T_A", "Alpha"),
            ("R1", "C2", period, "T_B", "Beta"),
            ("R1", "C3", period, "T_B", "Beta"),
            ("R1", "C4", period, "T_B", "Beta"),
            ("R2", "C5", period, "T_B", "Beta"),
            ("R2", "C6", period, "T_B", "Beta"),
        ],
        columns=["rep_id", "customer_id", "period", "territory_id", "territory_name"],
    )

    first = build_capacity_territory_allocation(calendar, transactions)
    second = build_capacity_territory_allocation(calendar, transactions)
    pd.testing.assert_frame_equal(first, second)
    assert not first.duplicated(["rep_id", "territory_id", "period"]).any()
    assert {
        "rep_id",
        "manager_id",
        "team_id",
        "territory_id",
        "period",
        "capacity_risk_band",
        "geographic_workload_attribution_flag",
        "allocation_scope",
    } <= set(first.columns)
    assert not first["geographic_workload_attribution_flag"].any()
    shares = first.groupby(["rep_id", "period"])["territory_allocation_share"].sum()
    np.testing.assert_allclose(shares, 1.0)
    r1 = first.loc[first["rep_id"].eq("R1")].set_index("territory_id")
    assert r1.loc["T_A", "territory_allocation_share"] == pytest.approx(0.25)
    assert r1.loc["T_B", "territory_allocation_share"] == pytest.approx(0.75)

    additive = [
        column for column in CAPACITY_ALLOCATION_ADDITIVE_COLUMNS if column in calendar
    ]
    allocated = first.groupby(["rep_id", "period"])[additive].sum().sort_index()
    source = calendar.set_index(["rep_id", "period"])[additive].sort_index()
    pd.testing.assert_frame_equal(allocated, source, check_dtype=False)

    territory = build_capacity_territory_summary(first).set_index("territory_id")
    assert territory.loc["T_A", "required_total_hours"] == pytest.approx(25.0)
    assert territory.loc["T_A", "available_field_hours"] == pytest.approx(20.0)
    assert territory.loc["T_A", "capacity_utilization_pct"] == pytest.approx(125.0)
    assert bool(territory.loc["T_A", "capacity_overload_flag"])
    assert territory.loc["T_B", "required_total_hours"] == pytest.approx(125.0)
    assert territory.loc["T_B", "available_field_hours"] == pytest.approx(160.0)
    assert territory.loc["T_B", "capacity_utilization_pct"] == pytest.approx(78.125)
    assert not bool(territory.loc["T_B", "capacity_overload_flag"])


def test_exact_territory_workload_uses_visits_not_transaction_share():
    period = pd.Timestamp("2024-01-01")
    transactions = pd.DataFrame(
        [("R1", "C_A", period, "T_A", "Alpha", "P1", "D1")]
        + [
            ("R1", "C_B", period, "T_B", "Beta", "P1", "D1")
            for _ in range(9)
        ],
        columns=[
            "rep_id",
            "customer_id",
            "period",
            "territory_id",
            "city",
            "product_id",
            "distributor",
        ],
    )
    reps = pd.DataFrame(
        [
            {
                "rep_id": "R1",
                "territory_id": "T_B",
                "territory_name": "Beta",
                "standard_field_hours_per_day": 8.0,
                "standard_working_days_per_month": 20.0,
            }
        ]
    )
    customers = pd.DataFrame(
        [
            ("C_A", "R1", "T_A", "Alpha", "high", 1.0),
            ("C_B", "R1", "T_B", "Beta", "low", 0.0),
        ],
        columns=[
            "customer_id",
            "primary_rep_id",
            "territory_id",
            "territory_name",
            "customer_priority",
            "required_visit_frequency",
        ],
    )
    visits = pd.DataFrame(
        [
            {
                "visit_id": "V1",
                "rep_id": "R1",
                "customer_id": "C_A",
                "territory_id": "T_A",
                "territory_name": "Alpha",
                "visit_date": "2024-01-12",
                "visit_duration_minutes": 60.0,
                "estimated_travel_km": 42.0,
                "visit_completed_flag": True,
            }
        ]
    )
    config = {
        "project": {"seed": 11},
        "synthetic": {"average_visit_minutes": 60.0, "average_speed_kmh": 42.0},
        "capacity": {
            "numeric_visit_frequency_period_divisor": 1.0,
            "administrative_buffer_pct": 0.0,
        },
    }
    calendar, drill = build_capacity_calendar(
        transactions, reps, customers, visits, config
    )
    allocation = build_capacity_territory_allocation(
        calendar,
        transactions,
        field_visits=visits,
        capacity_customer_drilldown=drill,
        config=config,
    ).set_index("territory_id")

    assert allocation.loc["T_A", "territory_allocation_share"] == pytest.approx(0.10)
    assert allocation.loc["T_B", "territory_allocation_share"] == pytest.approx(0.90)
    assert allocation.loc["T_A", "planned_visit_hours"] == pytest.approx(1.0)
    assert allocation.loc["T_A", "planned_travel_hours"] == pytest.approx(1.0)
    assert allocation.loc["T_B", "planned_visit_hours"] == pytest.approx(0.0)
    assert allocation.loc["T_B", "planned_travel_hours"] == pytest.approx(0.0)
    assert allocation.loc["T_A", "required_total_hours"] == pytest.approx(2.0)
    assert allocation.loc["T_B", "required_total_hours"] == pytest.approx(0.0)
    assert allocation["geographic_workload_attribution_flag"].all()
    assert not allocation["availability_geographic_attribution_flag"].any()
    np.testing.assert_allclose(
        allocation[list(CAPACITY_ALLOCATION_ADDITIVE_COLUMNS)].sum().to_numpy(float),
        calendar.iloc[0][list(CAPACITY_ALLOCATION_ADDITIVE_COLUMNS)].to_numpy(float),
    )

    injected = calendar.copy(deep=True)
    injected.loc[0, "planned_visit_hours"] += 10.0
    injected.loc[0, "core_required_hours"] += 10.0
    injected.loc[0, "required_total_hours"] += 10.0
    injected.loc[0, "required_hours"] += 10.0
    injected.loc[0, "required_fte"] = (
        injected.loc[0, "required_total_hours"]
        / injected.loc[0, "nominal_full_time_hours"]
    )
    injected.loc[0, "fte_gap"] = (
        injected.loc[0, "required_fte"] - injected.loc[0, "available_fte"]
    )
    injected.loc[0, "data_lineage"] = "synthetic_injected"
    injected_allocation = build_capacity_territory_allocation(
        injected,
        transactions,
        field_visits=visits,
        capacity_customer_drilldown=drill,
        config=config,
    ).set_index("territory_id")
    assert injected_allocation.loc["T_B", "planned_visit_hours"] == pytest.approx(10.0)
    assert bool(injected_allocation.loc["T_B", "residual_allocation_flag"])
    assert bool(injected_allocation.loc["T_B", "injected_residual_allocation_flag"])
    assert injected_allocation.loc[
        "T_B", "unrepresented_workload_residual_hours"
    ] == pytest.approx(10.0)
    assert "planned_visit_hours" in injected_allocation.loc[
        "T_B", "residual_allocation_columns"
    ]


def test_capacity_evaluation_counts_all_allocated_overloaded_territories():
    period = pd.Timestamp("2024-01-01")
    clean = pd.DataFrame(
        [
            {
                "capacity_record_id": "CAP_R1",
                "rep_id": "R1",
                "territory_id": "T_B",
                "territory_name": "Beta",
                "period": period,
                "fractional_territory_allocation": "T_A:0.25|T_B:0.75",
                "required_total_hours": 40.0,
                "required_hours": 40.0,
                "available_field_hours": 80.0,
                "nominal_full_time_hours": 100.0,
                "required_fte": 0.4,
                "available_fte": 0.8,
                "fte_gap": -0.4,
                "utilization_pct": 50.0,
                "risk_medium_threshold_pct": 85.0,
                "risk_high_threshold_pct": 100.0,
                "risk_critical_threshold_pct": 120.0,
                "overload_threshold_pct": 100.0,
                "capacity_risk_band": "low",
                "data_lineage": "synthetic_derived",
            }
        ]
    )
    injected = clean.copy(deep=True)
    injected.loc[0, ["required_total_hours", "required_hours", "required_fte", "utilization_pct"]] = [
        100.0,
        100.0,
        1.0,
        125.0,
    ]
    injected.loc[0, "fte_gap"] = 0.2
    injected.loc[0, "capacity_risk_band"] = "critical"
    truth = pd.DataFrame(
        [
            {
                "injection_id": "CAP_001",
                "entity_type": "capacity_record",
                "entity_id": "CAP_R1",
                "rep_id": "R1",
                "territory_id": "T_B",
                "period": period,
                "anomaly_type": "territory_workload_exceeds_available_capacity",
                "ground_truth_label": 1,
                "affected_record_ids": json.dumps(["CAP_R1"]),
                "injected_value": json.dumps(
                    {"required_total_hours": 100.0, "utilization_pct": 125.0}
                ),
            }
        ]
    )
    clean_allocation = build_capacity_territory_allocation(clean)
    injected_allocation = build_capacity_territory_allocation(injected)

    result = evaluate_capacity(
        clean,
        injected,
        truth,
        clean_territory_allocation=clean_allocation,
        injected_territory_allocation=injected_allocation,
    ).iloc[0]
    assert result.overloaded_territory_count == 2
    assert result.overloaded_territory_period_count == 2
    assert result.territories_above_critical_threshold == 2
    assert result.territory_count == 2
    assert result.territory_capacity_basis.startswith("conserved rep-territory-period")
