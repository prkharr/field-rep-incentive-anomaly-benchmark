"""Leakage-aware rep-period commercial feature store."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd


IDENTITY_COLUMNS = [
    "observation_id",
    "rep_id",
    "rep_name",
    "manager_id",
    "manager_name",
    "team_id",
    "team_name",
    "territory_id",
    "territory_name",
    "period",
    "peer_group_id",
    "data_lineage",
]


def _safe(numerator: Any, denominator: Any) -> np.ndarray:
    a = np.asarray(numerator, dtype=float)
    b = np.asarray(denominator, dtype=float)
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=np.abs(b) > 1e-12)


def _hhi_and_top(frame: pd.DataFrame, category: str, value: str, prefix: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(["rep_id", "period", category], observed=True)[value]
        .sum()
        .abs()
        .rename("weight")
        .reset_index()
    )
    totals = grouped.groupby(["rep_id", "period"])["weight"].transform("sum").clip(lower=1)
    grouped["share"] = grouped["weight"] / totals
    return (
        grouped.groupby(["rep_id", "period"], observed=True)
        .agg(**{f"{prefix}_hhi": ("share", lambda values: float(np.square(values).sum())), f"top_{prefix}_share": ("share", "max")})
        .reset_index()
    )


def _mix_shift(frame: pd.DataFrame, category: str, value: str, output: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(["rep_id", "period", category], observed=True)[value]
        .sum()
        .abs()
        .rename("weight")
        .reset_index()
    )
    grouped["share"] = grouped["weight"] / grouped.groupby(["rep_id", "period"])["weight"].transform("sum").clip(lower=1)
    pivot = grouped.pivot_table(index=["rep_id", "period"], columns=category, values="share", fill_value=0)
    pivot = pivot.sort_index()
    previous = pivot.groupby(level=0).shift().fillna(0)
    shift = 0.5 * (pivot - previous).abs().sum(axis=1)
    return shift.rename(output).reset_index()


def _prior_rolling(frame: pd.DataFrame, column: str, window: int, statistic: str) -> pd.Series:
    def calculate(values: pd.Series) -> pd.Series:
        rolling = values.shift().rolling(window, min_periods=1)
        return getattr(rolling, statistic)()

    return frame.groupby("rep_id", sort=False)[column].transform(calculate)


def _prior_customer_context(orders: pd.DataFrame) -> pd.DataFrame:
    """Return customer potential known before each current scoring period.

    The customer master intentionally describes the generated benchmark as a
    whole.  Model features cannot use its behavior-derived potential score for
    early periods, so this lookup rebuilds the score from cumulative *prior*
    order value.  A 50-point neutral cold-start score is used when no customer
    has history yet.
    """
    history = (
        orders.assign(_abs_sales=orders["net_sales"].abs())
        .groupby(["customer_id", "period"], observed=True)["_abs_sales"]
        .sum()
        .rename("period_customer_sales")
        .reset_index()
        .sort_values(["customer_id", "period"], kind="mergesort")
    )
    history["prior_customer_sales"] = history.groupby("customer_id", sort=False)[
        "period_customer_sales"
    ].transform(lambda values: values.cumsum().shift(fill_value=0.0))
    history["prior_customer_rank"] = history.groupby("period", observed=True)[
        "prior_customer_sales"
    ].rank(pct=True, method="average")
    no_history = history.groupby("period", observed=True)["prior_customer_sales"].transform("max").le(0)
    history["asof_customer_potential_score"] = (
        10.0 + 90.0 * history["prior_customer_rank"]
    ).where(~no_history, 50.0).clip(1.0, 100.0)
    history["asof_customer_priority"] = np.select(
        [
            history["asof_customer_potential_score"].ge(75.0),
            history["asof_customer_potential_score"].ge(40.0),
        ],
        ["high", "medium"],
        default="low",
    )
    return history[
        ["customer_id", "period", "asof_customer_potential_score", "asof_customer_priority"]
    ]


def _prior_territory_context(orders: pd.DataFrame) -> pd.DataFrame:
    """Return prior-only territory potential at territory-month grain."""
    sales = (
        orders.assign(_abs_sales=orders["net_sales"].abs())
        .groupby(["territory_id", "period"], observed=True)["_abs_sales"]
        .sum()
        .rename("period_territory_sales")
        .reset_index()
    )
    first_customer = (
        orders.groupby(["territory_id", "customer_id"], observed=True)["period"]
        .min()
        .rename("first_period")
        .reset_index()
        .groupby(["territory_id", "first_period"], observed=True)
        .size()
        .rename("new_territory_customers")
        .reset_index()
        .rename(columns={"first_period": "period"})
    )
    history = sales.merge(first_customer, on=["territory_id", "period"], how="left")
    history["new_territory_customers"] = history["new_territory_customers"].fillna(0.0)
    history = history.sort_values(["territory_id", "period"], kind="mergesort")
    history["prior_territory_sales"] = history.groupby("territory_id", sort=False)[
        "period_territory_sales"
    ].transform(lambda values: values.cumsum().shift(fill_value=0.0))
    history["prior_territory_customers"] = history.groupby("territory_id", sort=False)[
        "new_territory_customers"
    ].transform(lambda values: values.cumsum().shift(fill_value=0.0))
    sales_rank = history.groupby("period", observed=True)["prior_territory_sales"].rank(
        pct=True, method="average"
    )
    customer_rank = history.groupby("period", observed=True)["prior_territory_customers"].rank(
        pct=True, method="average"
    )
    no_history = history.groupby("period", observed=True)["prior_territory_sales"].transform("max").le(0)
    history["territory_potential"] = (
        50.0 + 25.0 * (sales_rank + customer_rank - 1.0)
    ).where(~no_history, 50.0).clip(1.0, 100.0)
    return history[["territory_id", "period", "territory_potential"]]


def _dominant_rep_period_territory(orders: pd.DataFrame) -> pd.DataFrame:
    counts = (
        orders.groupby(["rep_id", "period", "territory_id"], observed=True)
        .size()
        .rename("territory_order_count")
        .reset_index()
        .sort_values(
            ["rep_id", "period", "territory_order_count", "territory_id"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
    )
    return counts.drop_duplicates(["rep_id", "period"])[
        ["rep_id", "period", "territory_id"]
    ]


def _peer_stats(frame: pd.DataFrame, column: str) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    med = pd.Series(np.nan, index=frame.index, dtype=float)
    mad = med.copy()
    percentile = med.copy()
    group_name = pd.Series("", index=frame.index, dtype=object)
    for keys, label, minimum in [
        (["peer_group_id", "period"], "commercial_peer_group", 3),
        (["team_id", "period"], "team_period", 3),
        (["territory_id", "period"], "territory_period", 3),
        (["period"], "period", 1),
    ]:
        grouping = frame.groupby(keys, observed=True)[column]
        candidate_med = grouping.transform("median")
        candidate_mad = (frame[column] - candidate_med).abs().groupby(
            [frame[key] for key in keys], observed=True
        ).transform("median")
        size = grouping.transform("size")
        use = med.isna() & size.ge(minimum)
        med.loc[use] = candidate_med.loc[use]
        mad.loc[use] = candidate_mad.loc[use]
        percentile.loc[use] = grouping.rank(pct=True).loc[use]
        group_name.loc[use] = label
    scale = np.maximum(1.4826 * mad.fillna(0), np.maximum(med.abs() * 0.10, 1.0))
    z = (frame[column] - med) / scale
    return med.fillna(0), z.fillna(0), percentile.fillna(0.5), group_name.replace("", "period")


def build_feature_store(
    tables: dict[str, pd.DataFrame], config: dict[str, Any]
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Aggregate the relational layer to one rep-period feature matrix.

    Current-period operational facts are intended for post-period manager review.
    Historical rolling features shift before rolling so they never use future rows.
    """
    orders = tables["orders"].copy()
    orders["period"] = pd.to_datetime(orders["period"])
    positive_quantity = orders["quantity"].abs()
    order_agg = (
        orders.groupby(["rep_id", "period"], observed=True)
        .agg(
            gross_sales=("gross_sales", "sum"),
            net_sales=("net_sales", "sum"),
            total_quantity=("quantity", "sum"),
            order_count=("order_line_id", "size"),
            average_order_value=("net_sales", "mean"),
            maximum_order_value=("net_sales", lambda values: float(values.abs().max())),
            end_period_order_count=("end_of_period_flag", "sum"),
            end_period_sales=("net_sales", lambda values: 0.0),
        )
        .reset_index()
    )
    end_sales = (
        orders.loc[orders["end_of_period_flag"]]
        .groupby(["rep_id", "period"], observed=True)["net_sales"]
        .sum()
    )
    order_agg["end_period_sales"] = order_agg.set_index(["rep_id", "period"]).index.map(end_sales).astype(float)
    order_agg["end_period_sales"] = order_agg["end_period_sales"].fillna(0)
    order_agg["end_of_period_sales_share"] = _safe(order_agg["end_period_sales"], order_agg["net_sales"].abs())
    order_agg["average_selling_price"] = _safe(order_agg["net_sales"], order_agg["total_quantity"])
    order_agg["order_frequency"] = order_agg["order_count"] / 21.0
    order_agg["unusual_order_timing_score"] = order_agg["end_period_order_count"] / order_agg["order_count"].clip(lower=1)
    duplicate_mask = orders.duplicated(
        ["rep_id", "period", "customer_id", "product_id", "order_date", "net_sales"], keep=False
    )
    duplicate_order = duplicate_mask.groupby([orders["rep_id"], orders["period"]]).mean().rename("duplicate_order_signal").reset_index()
    order_agg = order_agg.merge(duplicate_order, on=["rep_id", "period"], how="left")

    product_lookup = tables["product_master"].set_index("product_id")
    order_product = orders.join(product_lookup[["incentive_weight"]], on="product_id")
    product_price_history = (
        orders.groupby(["product_id", "period"], observed=True)["unit_list_price"]
        .median()
        .rename("period_product_price")
        .reset_index()
        .sort_values(["product_id", "period"], kind="mergesort")
    )
    product_price_history["prior_list_price"] = product_price_history.groupby(
        "product_id", sort=False
    )["period_product_price"].transform(
        lambda values: values.shift().expanding(min_periods=1).median()
    )
    order_product = order_product.merge(
        product_price_history[["product_id", "period", "prior_list_price"]],
        on=["product_id", "period"],
        how="left",
        validate="many_to_one",
    )
    order_product["prior_list_price"] = order_product["prior_list_price"].fillna(
        order_product["unit_list_price"]
    )
    order_product["price_deviation"] = _safe(
        order_product["unit_list_price"] - order_product["prior_list_price"],
        order_product["prior_list_price"],
    )
    product_agg = (
        order_product.groupby(["rep_id", "period"], observed=True)
        .agg(
            active_product_count=("product_id", "nunique"),
            product_price_deviation=("price_deviation", "mean"),
            highly_incentivized_product_sales=("net_sales", lambda values: 0.0),
        )
        .reset_index()
    )
    incentivized_sales = (
        order_product.loc[order_product["incentive_weight"].ge(1.15)]
        .groupby(["rep_id", "period"], observed=True)["net_sales"]
        .sum()
    )
    product_agg["highly_incentivized_product_sales"] = product_agg.set_index(["rep_id", "period"]).index.map(incentivized_sales).astype(float)
    product_agg["highly_incentivized_product_sales"] = product_agg["highly_incentivized_product_sales"].fillna(0)
    product_agg["highly_incentivized_product_share"] = _safe(
        product_agg["highly_incentivized_product_sales"],
        order_agg.set_index(["rep_id", "period"]).loc[product_agg.set_index(["rep_id", "period"]).index, "net_sales"].abs(),
    )
    product_hhi = _hhi_and_top(orders, "product_id", "net_sales", "product_concentration")
    product_agg = product_agg.merge(product_hhi, on=["rep_id", "period"], how="left")
    product_agg["product_mix_entropy"] = -np.log(product_agg["product_concentration_hhi"].clip(lower=1e-9))
    product_agg = product_agg.merge(
        _mix_shift(orders, "product_id", "net_sales", "product_mix_shift"),
        on=["rep_id", "period"], how="left",
    )
    product_agg["priority_product_share"] = product_agg["highly_incentivized_product_share"]
    product_agg["low_volume_product_spike"] = (
        product_agg["product_mix_shift"] * product_agg["highly_incentivized_product_share"]
    )

    customers = tables["customer_master"]
    customer_context = _prior_customer_context(orders)
    order_customer = orders.merge(
        customers[["customer_id", "channel"]],
        on="customer_id", how="left", validate="many_to_one",
    ).merge(
        customer_context,
        on=["customer_id", "period"],
        how="left",
        validate="many_to_one",
    )
    first_customer = (
        orders.loc[orders["net_sales"].gt(0)]
        .groupby(["rep_id", "customer_id"], observed=True)["period"].min().rename("first_period").reset_index()
    )
    new_customer = first_customer.groupby(["rep_id", "first_period"]).size().rename("new_customer_count").reset_index().rename(columns={"first_period": "period"})
    customer_agg = (
        order_customer.groupby(["rep_id", "period"], observed=True)
        .agg(active_customer_count=("customer_id", "nunique"), total_customer_sales=("net_sales", "sum"))
        .reset_index()
        .merge(new_customer, on=["rep_id", "period"], how="left")
    )
    customer_agg["new_customer_count"] = customer_agg["new_customer_count"].fillna(0)
    high_sales = order_customer.loc[order_customer["asof_customer_priority"].eq("high")].groupby(["rep_id", "period"])["net_sales"].sum()
    low_sales = order_customer.loc[order_customer["asof_customer_potential_score"].le(35)].groupby(["rep_id", "period"])["net_sales"].sum()
    index = customer_agg.set_index(["rep_id", "period"]).index
    customer_agg["high_priority_customer_share"] = _safe(index.map(high_sales).astype(float), customer_agg["total_customer_sales"].abs())
    customer_agg["low_potential_customer_sales_share"] = _safe(index.map(low_sales).astype(float), customer_agg["total_customer_sales"].abs())
    customer_agg[["high_priority_customer_share", "low_potential_customer_sales_share"]] = customer_agg[["high_priority_customer_share", "low_potential_customer_sales_share"]].fillna(0)
    customer_hhi = _hhi_and_top(orders, "customer_id", "net_sales", "customer_concentration")
    customer_agg = customer_agg.merge(customer_hhi, on=["rep_id", "period"], how="left")
    customer_agg["customer_sales_concentration"] = customer_agg["customer_concentration_hhi"]
    customer_agg["customer_concentration_hhi"] = customer_agg["customer_concentration_hhi"]
    customer_agg["top_customer_sales_share"] = customer_agg["top_customer_concentration_share"]
    customer_agg = customer_agg.merge(
        _mix_shift(orders, "customer_id", "net_sales", "customer_mix_shift"),
        on=["rep_id", "period"], how="left",
    )

    discount = tables["discount_detail"].merge(
        orders[["order_line_id", "period"]], on="order_line_id", how="left", validate="many_to_one"
    )
    discount_agg = (
        discount.groupby(["rep_id", "period"], observed=True)
        .agg(
            average_discount_pct=("discount_pct", "mean"),
            maximum_discount_pct=("discount_pct", "max"),
            unapproved_discount_rate=("approved_flag", lambda values: 1 - values.astype(bool).mean()),
        )
        .reset_index()
    )
    discount_agg["discount_spike"] = discount_agg["maximum_discount_pct"] - discount_agg["average_discount_pct"]

    returns = tables["returns_cancellations"].copy()
    # Generic return/cancellation behavior is assigned to the month in which
    # the event became observable.  Only the explicitly named post-incentive
    # review signal is mapped back to the original payout period.
    returns["period"] = pd.to_datetime(returns["return_date"]).dt.to_period("M").dt.to_timestamp()
    return_agg = (
        returns.groupby(["rep_id", "period"], observed=True)
        .agg(
            return_amount=("return_amount", "sum"),
            return_count=("return_id", "size"),
            cancellation_count=("cancellation_flag", "sum"),
        )
        .reset_index()
    )
    after = returns.loc[returns["after_incentive_payout_flag"]].copy()
    after["period"] = pd.to_datetime(after["payout_period"])
    post_payout = (
        after.groupby(["rep_id", "period"], observed=True)["return_amount"]
        .sum()
        .rename("post_payout_return_amount")
        .reset_index()
    )
    return_agg = return_agg.merge(post_payout, on=["rep_id", "period"], how="outer")
    return_agg[
        ["return_amount", "return_count", "cancellation_count", "post_payout_return_amount"]
    ] = return_agg[
        ["return_amount", "return_count", "cancellation_count", "post_payout_return_amount"]
    ].fillna(0)

    visits = tables["field_visits"].copy()
    completed_customer_visits = visits.loc[
        visits["visit_completed_flag"].astype(bool),
        ["rep_id", "period", "customer_id"],
    ].drop_duplicates()
    positive_sales_customers = (
        orders.groupby(["rep_id", "period", "customer_id"], observed=True)[
            "net_sales"
        ]
        .sum()
        .gt(0)
        .rename("positive_sales")
        .reset_index()
    )
    converted_customers = completed_customer_visits.merge(
        positive_sales_customers.loc[positive_sales_customers["positive_sales"]],
        on=["rep_id", "period", "customer_id"],
        how="inner",
        validate="one_to_one",
    )
    converted_customer_agg = (
        converted_customers.groupby(["rep_id", "period"], observed=True)
        .size()
        .rename("visit_converted_customer_count")
        .reset_index()
    )
    visit_agg = (
        visits.groupby(["rep_id", "period"], observed=True)
        .agg(
            completed_visit_count=("visit_completed_flag", "sum"),
            average_visit_duration=("visit_duration_minutes", "mean"),
            extremely_short_visit_rate=("visit_duration_minutes", lambda values: values.lt(10).mean()),
            impossible_travel_count=("impossible_travel_flag", "sum"),
            overlapping_visit_count=("overlapping_visit_flag", "sum"),
            estimated_visit_travel_km=("estimated_travel_km", "sum"),
        )
        .reset_index()
        .merge(
            converted_customer_agg,
            on=["rep_id", "period"],
            how="left",
            validate="one_to_one",
        )
    )
    visit_agg["visit_converted_customer_count"] = visit_agg[
        "visit_converted_customer_count"
    ].fillna(0)
    crm = tables["crm_interactions"]
    crm_agg = (
        crm.groupby(["rep_id", "period"], observed=True)
        .agg(crm_interaction_count=("interaction_id", "size"), average_interest_score=("sentiment_or_interest_score", "mean"))
        .reset_index()
    )
    expense = tables["travel_expenses"].copy()
    expense["duplicate"] = expense.duplicated(["rep_id", "period", "visit_id", "expense_category", "claimed_amount"], keep=False)
    expense_agg = (
        expense.groupby(["rep_id", "period"], observed=True)
        .agg(
            claimed_distance_km=("claimed_distance_km", "sum"),
            estimated_distance_km=("estimated_distance_km", "sum"),
            claimed_expense_amount=("claimed_amount", "sum"),
            expected_expense_amount=("expected_amount", "sum"),
            missing_receipt_rate=("receipt_available_flag", lambda values: 1 - values.astype(bool).mean()),
            duplicate_expense_signal=("duplicate", "mean"),
        )
        .reset_index()
    )

    incentives = tables["incentive_calculations"].copy()
    targets = tables["rep_targets_quotas"].copy()
    base = incentives.merge(
        targets[["rep_id", "period", "target_units", "quota_difficulty_index", "target_revision_flag", "target_effective_date", "target_version", "target_visit_count"]],
        on=["rep_id", "period"], how="left", validate="one_to_one",
    )
    for addition in [order_agg, customer_agg, product_agg, discount_agg, return_agg, visit_agg, crm_agg, expense_agg, tables["capacity_calendar"]]:
        use = addition.copy()
        use["period"] = pd.to_datetime(use["period"])
        duplicate_columns = [column for column in use.columns if column in base.columns and column not in ["rep_id", "period"]]
        use = use.drop(columns=duplicate_columns)
        base = base.merge(use, on=["rep_id", "period"], how="left", validate="one_to_one")

    rep = tables["rep_master"][["rep_id", "rep_name", "manager_id", "team_id", "territory_id", "hire_date"]]
    manager = tables["manager_master"][["manager_id", "manager_name"]]
    team = tables["team_master"][["team_id", "team_name"]]
    territory = tables["territory_master"][["territory_id", "territory_name", "travel_complexity_index"]]
    for column in [
        "rep_name", "manager_id", "manager_name", "team_id", "team_name",
        "territory_id", "territory_name", "hire_date",
    ]:
        if column in base:
            base = base.drop(columns=column)
    base = base.merge(rep, on="rep_id", how="left", validate="many_to_one")
    dominant_territory = _dominant_rep_period_territory(orders)
    base = base.drop(columns=["territory_id"], errors="ignore").merge(
        dominant_territory,
        on=["rep_id", "period"],
        how="left",
        validate="one_to_one",
    )
    rep_territory = rep.set_index("rep_id")["territory_id"]
    base["territory_id"] = base["territory_id"].fillna(base["rep_id"].map(rep_territory))
    base = (
        base.merge(manager, on="manager_id", how="left", validate="many_to_one")
        .merge(team, on="team_id", how="left", validate="many_to_one")
        .merge(territory, on="territory_id", how="left", validate="many_to_one")
    )
    base = base.merge(
        _prior_territory_context(orders),
        on=["territory_id", "period"],
        how="left",
        validate="many_to_one",
    )
    base["territory_potential"] = base["territory_potential"].fillna(50.0)
    base = base.sort_values(["rep_id", "period"]).reset_index(drop=True)
    base["tenure_months"] = (
        (base["period"].dt.year - pd.to_datetime(base["hire_date"]).dt.year) * 12
        + base["period"].dt.month
        - pd.to_datetime(base["hire_date"]).dt.month
    ).clip(lower=1)
    base["observation_id"] = [
        "RPER_" + hashlib.sha256(f"{rep_id}|{pd.Timestamp(period).date()}".encode()).hexdigest()[:16]
        for rep_id, period in base[["rep_id", "period"]].itertuples(index=False, name=None)
    ]
    base["tenure_band"] = pd.cut(base["tenure_months"], [-1, 24, 60, np.inf], labels=["developing", "experienced", "senior"]).astype(str)
    base["potential_band"] = pd.cut(base["territory_potential"], [-np.inf, 40, 70, np.inf], labels=["lower", "medium", "higher"]).astype(str)
    base["travel_band"] = pd.cut(base["travel_complexity_index"], [-np.inf, 2, 3.5, np.inf], labels=["compact", "mixed", "distributed"]).astype(str)
    base["peer_group_id"] = base["team_id"] + "|" + base["potential_band"] + "|" + base["tenure_band"] + "|" + base["travel_band"]

    # Sales and order features.
    base["rolling_sales_mean"] = _prior_rolling(base, "net_sales", 3, "mean")
    base["rolling_sales_std"] = _prior_rolling(base, "net_sales", 6, "std")
    prior_sales = base.groupby("rep_id")["net_sales"].shift()
    prior_quantity = base.groupby("rep_id")["total_quantity"].shift()
    base["sales_growth"] = _safe(base["net_sales"] - prior_sales, prior_sales)
    base["quantity_growth"] = _safe(base["total_quantity"] - prior_quantity, prior_quantity)
    base["sales_volatility"] = _safe(base["rolling_sales_std"], base["rolling_sales_mean"])
    base["price_deviation_from_product_norm"] = base["product_price_deviation"]
    base["sales_vs_territory_potential"] = _safe(base["net_sales"], base["territory_potential"])
    base["price_deviation_from_product_norm"] = base["product_price_deviation"]

    # Incentive, quota, and return identities.
    base["target_attainment_pct"] = base["attainment_pct"]
    base["expected_incentive"] = base["calculated_incentive"]
    base["incentive_calculation_residual"] = base["final_incentive_paid"] - base["calculated_incentive"] - base["manual_adjustment"]
    base["manual_adjustment_ratio"] = _safe(base["manual_adjustment"], base["calculated_incentive"])
    base["accelerator_cliff_distance"] = np.minimum.reduce([
        np.abs(base["attainment_pct"] - 80), np.abs(base["attainment_pct"] - 100), np.abs(base["attainment_pct"] - 120)
    ])
    base["return_rate"] = _safe(base["return_amount"], base["net_sales"].abs())
    base["post_incentive_return_rate"] = _safe(base["post_payout_return_amount"], base["net_sales"].abs())
    base["return_clawback_ratio"] = _safe(base["return_clawback"], base["final_incentive_paid"])
    prior_incentive = base.groupby("rep_id")["final_incentive_paid"].shift()
    base["incentive_growth"] = _safe(base["final_incentive_paid"] - prior_incentive, prior_incentive)
    base["incentive_volatility"] = _safe(_prior_rolling(base, "final_incentive_paid", 6, "std"), _prior_rolling(base, "final_incentive_paid", 3, "mean"))
    base["cancelled_order_rate"] = _safe(base["cancellation_count"], base["order_count"])

    # Activity and expense features.
    base["visits_per_customer"] = _safe(base["completed_visit_count"], base["active_customer_count"])
    base["sales_per_visit"] = _safe(base["net_sales"], base["completed_visit_count"])
    base["visit_to_sales_conversion"] = _safe(
        base["visit_converted_customer_count"], base["completed_visit_count"]
    )
    base["missed_priority_visit_count"] = base.get("priority_customer_coverage_gap", 0)
    base["crm_interactions_per_customer"] = _safe(base["crm_interaction_count"], base["active_customer_count"])
    base["distance_claim_ratio"] = _safe(base["claimed_distance_km"], base["estimated_distance_km"])
    base["expense_per_visit"] = _safe(base["claimed_expense_amount"], base["completed_visit_count"])
    base["average_travel_hours"] = _safe(base["planned_travel_hours"], base["completed_visit_count"])
    base["workload_per_active_customer"] = _safe(base["required_total_hours"], base["active_customer_count"])
    base["capacity_utilization_pct"] = base["utilization_pct"]
    base["priority_customer_coverage_gap"] = base["priority_customer_coverage_gap"].fillna(0)
    base["capacity_risk_code"] = base["capacity_risk_band"].map({"low": 0, "medium": 1, "high": 2, "critical": 3}).fillna(0)

    # Peer and adjusted performance; no global all-time comparator is used.
    peer_rows: list[pd.DataFrame] = []
    for column in ["net_sales", "final_incentive_paid", "average_discount_pct", "claimed_expense_amount"]:
        med, z, pct, cohort = _peer_stats(base, column)
        label = {"net_sales": "sales", "final_incentive_paid": "incentive", "average_discount_pct": "discount", "claimed_expense_amount": "expense"}[column]
        base[f"{label}_peer_median"] = med
        base[f"{label}_peer_z"] = z
        base[f"{label}_peer_percentile"] = pct
        base[f"{label}_peer_cohort"] = cohort
        peer_rows.append(pd.DataFrame({"observation_id": base["observation_id"], "metric_name": column, "actual_value": base[column], "peer_median_value": med, "peer_z_score": z, "peer_percentile": pct, "peer_group_basis": cohort}))
    base["payout_to_peer_median_ratio"] = _safe(base["final_incentive_paid"], base["incentive_peer_median"])
    base["discount_vs_peer"] = base["average_discount_pct"] - base["discount_peer_median"]
    base["expense_vs_peer"] = _safe(base["claimed_expense_amount"], base["expense_peer_median"])
    base["peer_group_z_score"] = base["sales_peer_z"]
    base["peer_percentile"] = base["sales_peer_percentile"]
    base["rep_historical_mean"] = _prior_rolling(base, "net_sales", 6, "mean")
    base["rep_historical_std"] = _prior_rolling(base, "net_sales", 6, "std")
    base["rep_historical_z_score"] = _safe(base["net_sales"] - base["rep_historical_mean"], base["rep_historical_std"].abs().clip(lower=1))
    expected_by_potential = base.groupby("period")["net_sales"].transform("sum") * base["territory_potential"] / base.groupby("period")["territory_potential"].transform("sum").clip(lower=1)
    base["territory_adjusted_sales_residual"] = base["net_sales"] - expected_by_potential
    expected_incentive_by_potential = base.groupby("period")["final_incentive_paid"].transform("sum") * base["territory_potential"] / base.groupby("period")["territory_potential"].transform("sum").clip(lower=1)
    base["territory_adjusted_incentive_residual"] = base["final_incentive_paid"] - expected_incentive_by_potential
    tenure_median = base.groupby(["period", "tenure_band"], observed=True)["net_sales"].transform("median")
    base["tenure_adjusted_performance"] = _safe(base["net_sales"] - tenure_median, tenure_median)
    base["product_mix_adjusted_performance"] = base["sales_peer_z"] - base["product_mix_shift"].fillna(0)
    channel_share = order_customer.assign(abs_sales=order_customer["net_sales"].abs()).groupby(["rep_id", "period", "channel"], observed=True)["abs_sales"].sum().groupby(level=[0, 1]).apply(lambda values: float(np.square(values / values.sum()).sum())).rename("channel_hhi").reset_index()
    base = base.merge(channel_share, on=["rep_id", "period"], how="left")
    base["channel_adjusted_performance"] = base["sales_peer_z"] - base["channel_hhi"].fillna(0)
    base["month_over_month_behavior_change"] = base[["sales_growth", "customer_mix_shift", "product_mix_shift", "discount_spike"]].abs().mean(axis=1)
    base["rolling_anomaly_score"] = base.groupby("rep_id")["month_over_month_behavior_change"].transform(lambda values: values.shift().rolling(3, min_periods=1).mean())
    base["threshold_crossing_discount_signal"] = ((base["attainment_pct"].between(100, 105)) & base["average_discount_pct"].gt(base["discount_peer_median"] + 0.04)).astype(int)

    base["data_lineage"] = "synthetic_derived"
    prohibited = {
        "ground_truth_label",
        "anomaly_type",
        "severity",
        "injection_id",
        "correlated_case_flag",
    }
    feature_candidates = [
        column for column in base.columns
        if pd.api.types.is_numeric_dtype(base[column])
        and column not in prohibited
        and column not in {"target_version"}
    ]
    # Preserve a missingness signal for cold starts, then provide finite model inputs.
    cold_start_columns = ["rolling_sales_mean", "rolling_sales_std", "rep_historical_mean", "rep_historical_std"]
    for column in cold_start_columns:
        missing_name = f"{column}_missing"
        base[missing_name] = base[column].isna().astype(int)
        feature_candidates.append(missing_name)
    feature_candidates = list(dict.fromkeys(feature_candidates))
    base[feature_candidates] = base[feature_candidates].replace([np.inf, -np.inf], np.nan).fillna(0)
    if prohibited.intersection(feature_candidates) or any("ground_truth" in column for column in feature_candidates):
        raise ValueError("Ground-truth leakage detected in feature allowlist")
    if not np.isfinite(base[feature_candidates].to_numpy(float)).all():
        raise ValueError("Feature store contains non-finite model inputs")
    peer_comparison = pd.concat(peer_rows, ignore_index=True)
    ordered = [column for column in IDENTITY_COLUMNS if column in base] + [column for column in base if column not in IDENTITY_COLUMNS]
    return base[ordered], feature_candidates, peer_comparison


def validate_no_future_leakage(
    original: pd.DataFrame, modified_future: pd.DataFrame, cutoff: pd.Timestamp, columns: list[str]
) -> None:
    """Raise if changing future rows changes historical feature values."""
    left = original.loc[pd.to_datetime(original["period"]).le(cutoff), ["observation_id"] + columns].sort_values("observation_id")
    right = modified_future.loc[pd.to_datetime(modified_future["period"]).le(cutoff), ["observation_id"] + columns].sort_values("observation_id")
    pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True), check_dtype=False)
