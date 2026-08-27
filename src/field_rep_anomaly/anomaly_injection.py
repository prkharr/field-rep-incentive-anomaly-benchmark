"""Controlled, auditable anomaly injection for objective benchmark evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


ANOMALY_TYPES = [
    "high_incentive_weak_sales",
    "high_incentive_low_target_attainment",
    "very_high_sales_low_activity",
    "extreme_sales_spike",
    "extreme_quantity_spike",
    "abnormally_high_calls",
    "low_coverage_high_incentive",
    "large_manual_incentive_override",
    "sales_inconsistent_with_territory_opportunity",
    "sales_inconsistent_with_peer_group",
    "unusual_product_mix",
    "duplicate_suspicious_activity_pattern",
]


AUDIT_FIELDS = [
    "total_sales", "total_quantity", "average_price", "total_calls", "customer_coverage_pct",
    "actual_incentive_paid", "manual_override_amount", "payout_adjustment", "target_attainment_pct",
]


def inject_controlled_anomalies(
    data: pd.DataFrame,
    injection_rate: float = 0.06,
    severity_min: float = 0.35,
    severity_max: float = 1.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject 12 anomaly patterns into approximately 5–7% of rows.

    Exactly one primary anomaly type is assigned to each selected row. The returned
    audit table contains before/after values and is never used as a model input.
    """
    if not 0.05 <= injection_rate <= 0.07:
        raise ValueError("Controlled anomaly injection rate must remain between 5% and 7%.")
    if not 0 < severity_min <= severity_max <= 1.5:
        raise ValueError("Invalid severity bounds.")
    frame = data.copy().reset_index(drop=True)
    frame["total_quantity"] = pd.to_numeric(frame["total_quantity"], errors="coerce").astype(float)
    frame["injected_anomaly_flag"] = False
    frame["anomaly_type"] = "none"
    frame["anomaly_severity"] = 0.0
    rng = np.random.default_rng(seed + 211)
    n_anomalies = max(len(ANOMALY_TYPES), int(round(len(frame) * injection_rate)))
    n_anomalies = min(n_anomalies, len(frame))
    selected = rng.choice(frame.index.to_numpy(), size=n_anomalies, replace=False)
    selected = selected[rng.permutation(len(selected))]
    severity = rng.uniform(severity_min, severity_max, size=n_anomalies)
    type_cycle = np.resize(np.array(ANOMALY_TYPES, dtype=object), n_anomalies)
    type_cycle = type_cycle[rng.permutation(n_anomalies)]
    before = frame.loc[selected, AUDIT_FIELDS].copy()

    for idx, kind, sev in zip(selected, type_cycle, severity):
        sales = max(float(frame.at[idx, "total_sales"]), 1.0)
        incentive = max(float(frame.at[idx, "actual_incentive_paid"]), 1.0)
        if kind == "high_incentive_weak_sales":
            frame.at[idx, "total_sales"] = sales * (0.55 - 0.22 * sev)
            frame.at[idx, "actual_incentive_paid"] = incentive * (1.55 + 1.00 * sev)
        elif kind == "high_incentive_low_target_attainment":
            frame.at[idx, "total_sales"] = min(sales, float(frame.at[idx, "sales_target"]) * (0.62 - 0.18 * sev))
            frame.at[idx, "actual_incentive_paid"] = max(incentive, float(frame.at[idx, "target_incentive"]) * (1.30 + sev))
        elif kind == "very_high_sales_low_activity":
            frame.at[idx, "total_sales"] = sales * (1.65 + 1.10 * sev)
            frame.at[idx, "total_calls"] = max(1, int(frame.at[idx, "total_calls"] * (0.50 - 0.25 * sev)))
            frame.at[idx, "unique_customers_contacted"] = max(1, int(frame.at[idx, "unique_customers_contacted"] * 0.55))
        elif kind == "extreme_sales_spike":
            frame.at[idx, "total_sales"] = sales * (2.00 + 2.20 * sev)
        elif kind == "extreme_quantity_spike":
            frame.at[idx, "total_quantity"] = max(1.0, float(frame.at[idx, "total_quantity"]) * (2.20 + 2.30 * sev))
        elif kind == "abnormally_high_calls":
            frame.at[idx, "total_calls"] = max(1, int(frame.at[idx, "total_calls"] * (2.50 + 3.00 * sev)))
            frame.at[idx, "digital_engagements"] = max(0, int(frame.at[idx, "digital_engagements"] * (1.5 + sev)))
        elif kind == "low_coverage_high_incentive":
            frame.at[idx, "customer_coverage_pct"] = max(2.0, float(frame.at[idx, "customer_coverage_pct"]) * (0.40 - 0.20 * sev))
            frame.at[idx, "unique_customers_contacted"] = max(1, int(frame.at[idx, "unique_customers_contacted"] * 0.35))
            frame.at[idx, "actual_incentive_paid"] = incentive * (1.45 + 1.10 * sev)
        elif kind == "large_manual_incentive_override":
            override = max(float(frame.at[idx, "target_incentive"]) * (0.65 + 1.25 * sev), incentive * 0.55)
            frame.at[idx, "manual_override_amount"] = override
            frame.at[idx, "actual_incentive_paid"] = float(frame.at[idx, "calculated_incentive"]) + override
        elif kind == "sales_inconsistent_with_territory_opportunity":
            potential = max(float(frame.at[idx, "territory_market_potential"]), 1.0)
            if rng.random() < 0.5:
                frame.at[idx, "total_sales"] = max(sales, potential * (0.70 + 0.55 * sev))
            else:
                frame.at[idx, "total_sales"] = min(sales, potential * (0.025 + 0.025 * (1 - sev)))
        elif kind == "sales_inconsistent_with_peer_group":
            frame.at[idx, "total_sales"] = sales * ((0.25 + 0.18 * (1 - sev)) if rng.random() < 0.5 else (2.10 + 1.70 * sev))
        elif kind == "unusual_product_mix":
            frame.at[idx, "total_sales"] = sales * (2.20 + 1.70 * sev)
            frame.at[idx, "total_quantity"] = max(1.0, float(frame.at[idx, "total_quantity"]) * (1.35 + sev))
        elif kind == "duplicate_suspicious_activity_pattern":
            donors = frame.index.difference(pd.Index(selected))
            donor = int(rng.choice(donors)) if len(donors) else int((idx + 1) % len(frame))
            for field in ("total_calls", "priority_customer_calls", "digital_engagements", "working_days", "travel_distance_km"):
                frame.at[idx, field] = frame.at[donor, field]
            frame.at[idx, "total_sales"] = sales * (0.48 - 0.12 * sev)
            frame.at[idx, "actual_incentive_paid"] = incentive * (1.35 + 0.65 * sev)
        frame.at[idx, "injected_anomaly_flag"] = True
        frame.at[idx, "anomaly_type"] = kind
        frame.at[idx, "anomaly_severity"] = float(sev)

    # Recompute identities that are expected to respond to altered primitive values.
    frame["average_price"] = np.divide(
        frame["total_sales"], frame["total_quantity"],
        out=frame["average_price"].to_numpy(dtype=float, copy=True),
        where=frame["total_quantity"].to_numpy(dtype=float) > 0,
    )
    frame["target_attainment_pct"] = np.divide(
        frame["total_sales"], frame["sales_target"], out=np.zeros(len(frame)), where=frame["sales_target"].to_numpy(dtype=float) > 0
    ) * 100.0
    frame["incentive_attainment_pct"] = np.divide(
        frame["actual_incentive_paid"], frame["target_incentive"], out=np.zeros(len(frame)), where=frame["target_incentive"].to_numpy(dtype=float) > 0
    ) * 100.0
    frame["payout_adjustment"] = frame["actual_incentive_paid"] - frame["calculated_incentive"]
    after = frame.loc[selected, AUDIT_FIELDS].copy()
    audit = frame.loc[selected, ["rep_id", "product_name", "territory_id", "date", "anomaly_type", "anomaly_severity"]].copy()
    for field in AUDIT_FIELDS:
        audit[f"before_{field}"] = before[field].to_numpy()
        audit[f"after_{field}"] = after[field].to_numpy()
    audit = audit.sort_values(["anomaly_type", "anomaly_severity"], ascending=[True, False]).reset_index(drop=True)
    return frame, audit
