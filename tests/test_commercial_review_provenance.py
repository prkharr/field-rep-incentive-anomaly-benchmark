"""Focused tests for clean-versus-injected row provenance."""

from __future__ import annotations

import json

import pandas as pd

from field_rep_anomaly.commercial_review.anomalies import (
    _repair_injected_provenance,
    audit_injected_record_changes,
    ground_truth_trace_pairs,
)


def test_provenance_repair_marks_and_traces_cascaded_rows() -> None:
    period = pd.Timestamp("2019-03-01")
    clean = {
        "orders": pd.DataFrame(
            [
                {
                    "order_line_id": "LINE_1",
                    "rep_id": "REP_1",
                    "period": period,
                    "net_sales": 100.0,
                    "data_lineage": "source_observed",
                }
            ]
        ),
        "discount_detail": pd.DataFrame(
            [
                {
                    "discount_id": "DISC_1",
                    "order_line_id": "LINE_1",
                    "rep_id": "REP_1",
                    "discount_amount": 10.0,
                    "data_lineage": "synthetic_normal",
                }
            ]
        ),
        "incentive_calculations": pd.DataFrame(
            [
                {
                    "incentive_record_id": "INC_1",
                    "rep_id": "REP_1",
                    "period": period,
                    "final_incentive_paid": 5.0,
                    "data_lineage": "synthetic_normal",
                }
            ]
        ),
        "crm_interactions": pd.DataFrame(
            columns=[
                "interaction_id",
                "rep_id",
                "period",
                "sentiment_or_interest_score",
                "data_lineage",
            ]
        ),
    }
    injected = {name: frame.copy(deep=True) for name, frame in clean.items()}
    injected["orders"].loc[0, "net_sales"] = 125.0
    injected["discount_detail"].loc[0, "discount_amount"] = 15.0
    injected["incentive_calculations"].loc[0, "final_incentive_paid"] = 7.0
    injected["crm_interactions"] = pd.DataFrame(
        [
            {
                "interaction_id": "CRM_ADDED_1",
                "rep_id": "REP_1",
                "period": period,
                "sentiment_or_interest_score": 0.05,
                "data_lineage": "synthetic_normal",
            }
        ]
    )
    original_value = json.dumps({"net_sales": 100.0})
    injected_value = json.dumps({"net_sales": 125.0})
    truth = [
        {
            "rep_id": "REP_1",
            "period": period,
            "affected_dataset": "orders",
            "affected_record_ids": json.dumps(["LINE_1"]),
            "original_value": original_value,
            "injected_value": injected_value,
        }
    ]

    _repair_injected_provenance(clean, injected, truth)
    _repair_injected_provenance(clean, injected, truth)

    audit = audit_injected_record_changes(clean, injected)
    assert set(audit["dataset"]) == {
        "orders",
        "discount_detail",
        "incentive_calculations",
        "crm_interactions",
    }
    assert audit["data_lineage"].eq("synthetic_injected").all()
    assert truth[0]["original_value"] == original_value
    assert truth[0]["injected_value"] == injected_value
    assert set(truth[0]["affected_dataset"].split("|")) == set(audit["dataset"])
    assert set(json.loads(truth[0]["affected_record_ids"])) == set(audit["record_id"])
    assert ground_truth_trace_pairs(truth, injected) == set(
        zip(audit["dataset"], audit["record_id"])
    )
