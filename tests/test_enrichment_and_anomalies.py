"""Contract tests for the analytical grain and controlled anomaly labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from field_rep_anomaly.anomaly_injection import ANOMALY_TYPES, AUDIT_FIELDS, inject_controlled_anomalies
from field_rep_anomaly.feature_engineering import LABEL_COLUMNS
from field_rep_anomaly.synthetic_enrichment import GRAIN


def test_analytical_grain_reconciles_and_enrichment_identities_hold(
    canonical_transactions, enrichment_bundle
):
    analytical, mapping, lineage = enrichment_bundle

    assert len(analytical) >= 200  # enough rows for 5-7% injection to cover all 12 patterns
    assert not analytical.duplicated(GRAIN).any()
    assert np.isclose(analytical["total_sales"].sum(), canonical_transactions["sales"].sum())
    assert set(lineage["field"]) == set(analytical.columns)
    assert lineage["field"].is_unique
    assert mapping["rep_id"].notna().all()

    np.testing.assert_allclose(
        analytical["target_attainment_pct"],
        analytical["total_sales"] / analytical["sales_target"] * 100.0,
    )
    np.testing.assert_allclose(
        analytical["incentive_attainment_pct"],
        analytical["actual_incentive_paid"] / analytical["target_incentive"] * 100.0,
    )
    np.testing.assert_allclose(
        analytical["payout_adjustment"],
        analytical["actual_incentive_paid"] - analytical["calculated_incentive"],
    )
    np.testing.assert_allclose(
        analytical["customer_coverage_pct"],
        np.minimum(
            100.0,
            analytical["unique_customers_contacted"]
            / analytical["assigned_customer_portfolio"]
            * 100.0,
        ),
    )
    np.testing.assert_allclose(
        analytical["workload_index_raw"],
        0.55 * analytical["total_calls"] / analytical["rep_capacity"]
        + 0.45 * analytical["assigned_customer_portfolio"] / analytical["rep_capacity"],
    )

    numeric = analytical.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy()).all()
    for field in ("customer_concentration", "channel_mix", "subchannel_mix"):
        assert analytical[field].between(0.0, 1.0 + 1e-12).all()
    assert analytical["customer_coverage_pct"].between(0.0, 100.0).all()
    assert analytical["call_plan_adherence_pct"].between(25.0, 100.0).all()
    assert analytical["rep_tenure_months"].between(6, 120).all()
    assert analytical["rep_capacity"].between(80, 120).all()
    assert (analytical[["sales_target", "quantity_target", "target_incentive"]] > 0).all().all()
    assert (analytical[["total_calls", "unique_customers_contacted"]] >= 1).all().all()
    assert (analytical["actual_incentive_paid"] >= 0).all()


def test_injection_is_exact_auditable_deterministic_and_label_separated(
    analytical_data, injected_bundle, project_config
):
    injected, audit = injected_bundle
    repeated, repeated_audit = inject_controlled_anomalies(
        analytical_data,
        injection_rate=0.06,
        severity_min=0.35,
        severity_max=1.0,
        seed=17,
    )
    pd.testing.assert_frame_equal(injected, repeated)
    pd.testing.assert_frame_equal(audit, repeated_audit)

    expected = min(max(len(ANOMALY_TYPES), int(round(len(analytical_data) * 0.06))), len(analytical_data))
    flagged = injected["injected_anomaly_flag"]
    assert int(flagged.sum()) == expected
    assert 0.05 <= float(flagged.mean()) <= 0.07
    assert set(injected.loc[flagged, "anomaly_type"]) == set(ANOMALY_TYPES)
    assert injected.loc[~flagged, "anomaly_type"].eq("none").all()
    assert injected.loc[~flagged, "anomaly_severity"].eq(0.0).all()
    assert injected.loc[flagged, "anomaly_severity"].between(0.35, 1.0).all()

    assert len(audit) == expected
    assert not audit.duplicated(GRAIN).any()
    changes = np.column_stack(
        [
            ~np.isclose(
                audit[f"before_{field}"].to_numpy(dtype=float),
                audit[f"after_{field}"].to_numpy(dtype=float),
                equal_nan=True,
            )
            for field in AUDIT_FIELDS
        ]
    )
    assert changes.any(axis=1).all()

    configured_features = set(project_config["preprocessing"]["features"])
    assert configured_features.isdisjoint(LABEL_COLUMNS)
    assert LABEL_COLUMNS <= set(injected.columns)
    assert not LABEL_COLUMNS.intersection(analytical_data.columns)

    np.testing.assert_allclose(
        injected["average_price"], injected["total_sales"] / injected["total_quantity"]
    )
    np.testing.assert_allclose(
        injected["target_attainment_pct"], injected["total_sales"] / injected["sales_target"] * 100.0
    )
    np.testing.assert_allclose(
        injected["payout_adjustment"],
        injected["actual_incentive_paid"] - injected["calculated_incentive"],
    )
