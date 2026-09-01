"""Central, versioned synthetic incentive-policy calculation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_policy_rules(config: dict[str, Any]) -> pd.DataFrame:
    """Expand configured policy versions and attainment tiers into one rule table."""
    rows: list[dict[str, Any]] = []
    versions = config["synthetic"]["policy_versions"]
    tiers = config["synthetic"]["attainment_tiers"]
    for version in versions:
        for tier_number, tier in enumerate(tiers, start=1):
            rows.append(
                {
                    "policy_id": version["policy_id"],
                    "policy_version": str(version["policy_version"]),
                    "effective_start_date": pd.Timestamp(version["effective_start_date"]),
                    "effective_end_date": pd.Timestamp(version["effective_end_date"]),
                    "metric_name": "eligible_net_sales_attainment",
                    "lower_attainment_pct": float(tier["lower"]) * 100,
                    "upper_attainment_pct": float(tier["upper"]) * 100,
                    "payout_rate": float(version["base_rate"]),
                    "accelerator_multiplier": float(tier["accelerator"]),
                    "decelerator_multiplier": float(tier["decelerator"]),
                    "maximum_payout": float(version["maximum_payout"]),
                    "minimum_eligibility": 0.50,
                    "product_weight": 1.0,
                    "new_customer_bonus": 50.0,
                    "discount_penalty_threshold_pct": 0.13,
                    "discount_penalty_rule": (
                        "Period-average discount above 13%; penalty scaled by "
                        "the unapproved-discount rate"
                    ),
                    "return_clawback_rule": "Base payout rate applied only to linked returns known by payout date",
                    "payout_delay_days": int(version["payout_delay_days"]),
                    "policy_description": (
                        f"Synthetic controlled benchmark policy tier {tier_number}; "
                        "not an observed compensation plan"
                    ),
                    "currency_code": str(
                        config.get("project", {}).get("currency_code", "UNK")
                    ),
                    "data_lineage": "synthetic_normal",
                }
            )
    return pd.DataFrame(rows)


def _policy_for_period(policy_rules: pd.DataFrame, period: pd.Timestamp) -> pd.DataFrame:
    candidates = policy_rules.loc[
        policy_rules["effective_start_date"].le(period)
        & policy_rules["effective_end_date"].ge(period)
    ]
    if candidates.empty:
        raise ValueError(f"No incentive policy is effective for {period.date()}")
    return candidates.sort_values("lower_attainment_pct")


def calculate_incentives(
    orders: pd.DataFrame,
    discounts: pd.DataFrame,
    returns: pd.DataFrame,
    targets: pd.DataFrame,
    product_master: pd.DataFrame,
    rep_master: pd.DataFrame,
    policy_rules: pd.DataFrame,
    currency_code: str = "UNK",
) -> pd.DataFrame:
    """Calculate reproducible rep-period incentives from the central policy table."""
    order = orders.merge(
        product_master[
            ["product_id", "incentive_eligible_flag", "incentive_weight", "expected_discount_pct"]
        ],
        on="product_id",
        how="left",
        validate="many_to_one",
    )
    order["eligible_weight"] = np.where(
        order["incentive_eligible_flag"].fillna(False),
        order["incentive_weight"].fillna(1.0),
        0.0,
    )
    order["eligible_gross"] = order["gross_sales"] * order["eligible_weight"]
    order["eligible_net"] = order["net_sales"] * order["eligible_weight"]
    order["eligible_priority_sales"] = np.where(
        order["incentive_weight"].fillna(1.0).ge(1.15), order["net_sales"], 0.0
    )
    agg = (
        order.groupby(["rep_id", "period"], observed=True)
        .agg(
            eligible_gross_sales=("eligible_gross", "sum"),
            eligible_net_sales=("eligible_net", "sum"),
            priority_product_sales=("eligible_priority_sales", "sum"),
        )
        .reset_index()
    )
    first_orders = (
        order.loc[order["net_sales"].gt(0)]
        .groupby(["rep_id", "customer_id"], observed=True)["period"]
        .min()
        .rename("first_period")
        .reset_index()
    )
    new_counts = (
        first_orders.groupby(["rep_id", "first_period"], observed=True)
        .size()
        .rename("new_customer_count")
        .reset_index()
        .rename(columns={"first_period": "period"})
    )
    agg = agg.merge(new_counts, on=["rep_id", "period"], how="left")
    agg["new_customer_count"] = agg["new_customer_count"].fillna(0)

    if discounts.empty:
        discount_agg = pd.DataFrame(columns=["rep_id", "period", "average_discount_pct", "unapproved_discount_rate"])
    else:
        detail = discounts.merge(
            orders[["order_line_id", "period"]], on="order_line_id", how="left", validate="many_to_one"
        )
        discount_agg = (
            detail.groupby(["rep_id", "period"], observed=True)
            .agg(
                average_discount_pct=("discount_pct", "mean"),
                unapproved_discount_rate=("approved_flag", lambda values: 1 - values.astype(bool).mean()),
            )
            .reset_index()
        )
    agg = agg.merge(discount_agg, on=["rep_id", "period"], how="left")
    agg[["average_discount_pct", "unapproved_discount_rate"]] = agg[
        ["average_discount_pct", "unapproved_discount_rate"]
    ].fillna(0)

    if returns.empty:
        return_agg = pd.DataFrame(columns=["rep_id", "period", "return_amount"])
    else:
        returned = returns.copy()
        returned["period"] = pd.to_datetime(returned["payout_period"])
        payout_delays = {
            period: int(_policy_for_period(policy_rules, pd.Timestamp(period)).iloc[0]["payout_delay_days"])
            for period in returned["period"].drop_duplicates()
        }
        returned["calculation_cutoff_date"] = (
            returned["period"]
            + pd.offsets.MonthEnd(1)
            + pd.to_timedelta(returned["period"].map(payout_delays), unit="D")
        )
        returned = returned.loc[
            pd.to_datetime(returned["return_date"]).le(returned["calculation_cutoff_date"])
        ]
        return_agg = (
            returned.groupby(["rep_id", "period"], observed=True)["return_amount"]
            .sum()
            .rename("return_amount")
            .reset_index()
        )
    agg = agg.merge(return_agg, on=["rep_id", "period"], how="left")
    agg["return_amount"] = agg["return_amount"].fillna(0)
    target_columns = [
        "rep_id",
        "period",
        "target_sales",
        "target_units",
        "target_priority_product_sales",
        "target_new_customer_sales",
        "target_visit_count",
        "target_version",
    ]
    agg = agg.merge(targets[target_columns], on=["rep_id", "period"], how="left", validate="one_to_one")
    agg = agg.merge(
        rep_master[["rep_id", "manager_id", "team_id", "territory_id"]],
        on="rep_id",
        how="left",
        validate="many_to_one",
    )

    results: list[dict[str, Any]] = []
    for row in agg.itertuples(index=False):
        period = pd.Timestamp(row.period)
        rules = _policy_for_period(policy_rules, period)
        provisional_attainment = float(row.eligible_net_sales) / max(
            float(row.target_sales), 1.0
        )
        chosen = rules.loc[
            rules["lower_attainment_pct"].le(provisional_attainment * 100)
            & rules["upper_attainment_pct"].gt(provisional_attainment * 100)
        ]
        if chosen.empty:
            chosen = rules.tail(1)
        rule = chosen.iloc[0]
        policy_product_weight = float(rule.product_weight)
        eligible_gross_sales = float(row.eligible_gross_sales) * policy_product_weight
        eligible_net_sales = float(row.eligible_net_sales) * policy_product_weight
        priority_product_sales = float(row.priority_product_sales) * policy_product_weight
        attainment = eligible_net_sales / max(float(row.target_sales), 1.0)
        eligible = attainment >= float(rule.minimum_eligibility)
        attainment_multiplier = (
            float(rule.decelerator_multiplier) if attainment < 1.0 else 1.0
        )
        base = (
            max(eligible_net_sales, 0.0)
            * float(rule.payout_rate)
            * min(attainment, 1.0)
            * attainment_multiplier
            if eligible
            else 0.0
        )
        accelerator = (
            base * max(attainment - 1.0, 0.0) * float(rule.accelerator_multiplier)
            if eligible
            else 0.0
        )
        priority_share = max(priority_product_sales, 0.0) / max(
            abs(eligible_net_sales), 1.0
        )
        mix_bonus = base * min(priority_share, 1.0) * 0.10 if eligible else 0.0
        new_bonus = float(row.new_customer_count) * float(rule.new_customer_bonus) if eligible else 0.0
        discount_threshold = float(
            rule.get("discount_penalty_threshold_pct", 0.13)
        )
        discount_excess = max(
            float(row.average_discount_pct) - discount_threshold, 0.0
        )
        discount_penalty = (
            max(eligible_net_sales, 0.0)
            * discount_excess
            * float(rule.payout_rate)
            * (1 + float(row.unapproved_discount_rate))
            if eligible
            else 0.0
        )
        clawback = (
            max(float(row.return_amount), 0.0) * float(rule.payout_rate)
            if eligible
            else 0.0
        )
        manual_adjustment = 0.0
        calculated = float(
            np.clip(
                base + accelerator + mix_bonus + new_bonus - discount_penalty - clawback,
                0,
                float(rule.maximum_payout),
            )
        )
        payout_date = period + pd.offsets.MonthEnd(1) + pd.Timedelta(days=int(rule.payout_delay_days))
        results.append(
            {
                "rep_id": row.rep_id,
                "manager_id": row.manager_id,
                "team_id": row.team_id,
                "territory_id": row.territory_id,
                "period": period,
                "policy_id": rule.policy_id,
                "policy_version": rule.policy_version,
                "eligible_gross_sales": round(eligible_gross_sales, 2),
                "eligible_net_sales": round(eligible_net_sales, 2),
                "target_sales": round(float(row.target_sales), 2),
                "attainment_pct": attainment * 100,
                "base_incentive": round(base, 2),
                "accelerator_amount": round(accelerator, 2),
                "product_mix_bonus": round(mix_bonus, 2),
                "new_customer_bonus": round(new_bonus, 2),
                "discount_penalty": round(discount_penalty, 2),
                "return_clawback": round(clawback, 2),
                "manual_adjustment": manual_adjustment,
                "calculated_incentive": round(calculated, 2),
                "final_incentive_paid": round(calculated, 2),
                "payout_date": payout_date,
                "payout_status": "paid",
                "payout_to_sales_ratio": calculated / max(abs(eligible_net_sales), 1.0),
                "currency_code": currency_code,
                "reconciliation_tolerance": 0.02,
                "data_lineage": "synthetic_normal",
            }
        )
    result = pd.DataFrame(results).sort_values(["period", "rep_id"]).reset_index(drop=True)
    result["incentive_record_id"] = [
        f"INC_{index:08d}" for index in range(1, len(result) + 1)
    ]
    return result


def incentive_reconciliation_error(incentives: pd.DataFrame) -> pd.Series:
    """Absolute final-versus-calculated error after an explicit manual adjustment."""
    expected = incentives["calculated_incentive"] + incentives["manual_adjustment"]
    return (incentives["final_incentive_paid"] - expected).abs()
