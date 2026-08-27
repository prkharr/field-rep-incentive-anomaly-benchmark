"""Reusable feature engineering at rep-product-territory-month grain."""

from __future__ import annotations

import numpy as np
import pandas as pd


LABEL_COLUMNS = {"injected_anomaly_flag", "anomaly_type", "anomaly_severity"}


def safe_divide(numerator: pd.Series, denominator: pd.Series, fill: float = 0.0) -> pd.Series:
    left = pd.to_numeric(numerator, errors="coerce").to_numpy(dtype=float)
    right = pd.to_numeric(denominator, errors="coerce").to_numpy(dtype=float)
    values = np.divide(left, right, out=np.full_like(left, fill, dtype=float), where=np.isfinite(right) & (np.abs(right) > 1e-12))
    return pd.Series(values, index=numerator.index)


def _past_rolling_mean(series: pd.Series, window: int = 3) -> pd.Series:
    return series.shift(1).rolling(window=window, min_periods=1).mean()


def _hierarchical_peer_stats(frame: pd.DataFrame, value: str, minimum_group: int = 3) -> tuple[pd.Series, pd.Series, pd.Series]:
    levels = [
        ["product_name", "territory_id", "date"],
        ["product_name", "sales_team", "date"],
        ["product_name", "date"],
        ["date"],
    ]
    chosen_median = pd.Series(np.nan, index=frame.index, dtype=float)
    chosen_mean = pd.Series(np.nan, index=frame.index, dtype=float)
    chosen_std = pd.Series(np.nan, index=frame.index, dtype=float)
    for level in levels:
        grouped = frame.groupby(level, dropna=False, observed=True)[value]
        count = grouped.transform("count")
        eligible = chosen_median.isna() & (count >= minimum_group)
        chosen_median.loc[eligible] = grouped.transform("median").loc[eligible]
        chosen_mean.loc[eligible] = grouped.transform("mean").loc[eligible]
        chosen_std.loc[eligible] = grouped.transform("std").loc[eligible]
    global_median = float(frame[value].median())
    global_mean = float(frame[value].mean())
    global_std = float(frame[value].std(ddof=1)) or 1.0
    return chosen_median.fillna(global_median), chosen_mean.fillna(global_mean), chosen_std.fillna(global_std).replace(0, global_std)


def engineer_features(data: pd.DataFrame, minimum_peer_group: int = 3) -> pd.DataFrame:
    """Create commercial, activity, incentive, peer, and opportunity features."""
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["rep_id", "product_name", "territory_id", "date"], kind="stable").reset_index(drop=True)
    history_group = ["rep_id", "product_name", "territory_id"]
    frame["sales_growth"] = frame.groupby(history_group, observed=True)["total_sales"].pct_change(fill_method=None)
    frame["quantity_growth"] = frame.groupby(history_group, observed=True)["total_quantity"].pct_change(fill_method=None)
    prior_sales = frame.groupby(history_group, observed=True)["total_sales"].transform(_past_rolling_mean)
    frame["rolling_sales_growth"] = safe_divide(frame["total_sales"] - prior_sales, prior_sales)

    portfolio_group = ["rep_id", "territory_id", "date"]
    portfolio_sales = frame.groupby(portfolio_group, observed=True)["total_sales"].transform("sum")
    frame["unique_products"] = frame.groupby(portfolio_group, observed=True)["product_name"].transform("nunique")
    product_share = safe_divide(frame["total_sales"], portfolio_sales)
    frame["product_concentration"] = product_share.pow(2).groupby([frame[col] for col in portfolio_group]).transform("sum")
    frame["dominant_product_share"] = product_share.groupby([frame[col] for col in portfolio_group]).transform("max")
    frame["product_sales_share"] = product_share

    frame["sales_per_customer"] = safe_divide(frame["total_sales"], frame["unique_customers"])
    frame["sales_per_product"] = safe_divide(portfolio_sales, frame["unique_products"])
    frame["sales_per_call"] = safe_divide(frame["total_sales"], frame["total_calls"])
    frame["calls_per_working_day"] = safe_divide(frame["total_calls"], frame["working_days"])
    frame["calls_per_customer"] = safe_divide(frame["total_calls"], frame["unique_customers_contacted"])
    frame["activity_efficiency"] = safe_divide(frame["total_sales"], frame["total_calls"] + frame["digital_engagements"])
    frame["travel_per_customer"] = safe_divide(frame["travel_distance_km"], frame["unique_customers_contacted"])
    frame["workload_index"] = 0.55 * safe_divide(frame["total_calls"], frame["rep_capacity"]) + 0.45 * safe_divide(frame["assigned_customer_portfolio"], frame["rep_capacity"])

    frame["incentive_to_sales_ratio"] = safe_divide(frame["actual_incentive_paid"], frame["total_sales"])
    frame["incentive_per_customer"] = safe_divide(frame["actual_incentive_paid"], frame["unique_customers"])
    frame["incentive_per_call"] = safe_divide(frame["actual_incentive_paid"], frame["total_calls"])
    frame["incentive_variance"] = frame["actual_incentive_paid"] - frame["calculated_incentive"]
    frame["incentive_variance_pct"] = safe_divide(frame["incentive_variance"], frame["calculated_incentive"].abs())
    frame["incentive_to_target_ratio"] = safe_divide(frame["actual_incentive_paid"], frame["target_incentive"])

    for value, prefix in (("total_sales", "sales"), ("actual_incentive_paid", "incentive"), ("total_calls", "activity")):
        peer_median, peer_mean, peer_std = _hierarchical_peer_stats(frame, value, minimum_group=minimum_peer_group)
        frame[f"{prefix}_peer_median"] = peer_median
        frame[f"{prefix}_vs_peer_median"] = safe_divide(frame[value] - peer_median, peer_median.abs())
        frame[f"{prefix}_zscore_within_peer"] = safe_divide(frame[value] - peer_mean, peer_std)

    opportunity_raw = pd.to_numeric(frame["opportunity_index_raw"], errors="coerce").fillna(0.0)
    low, high = opportunity_raw.quantile([0.01, 0.99])
    frame["opportunity_index"] = ((opportunity_raw - low) / max(float(high - low), 1e-9)).clip(0, 1)
    frame["market_potential_adjusted_sales"] = safe_divide(frame["total_sales"], frame["territory_market_potential"])

    numeric = frame.select_dtypes(include=[np.number]).columns
    frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan)
    # Growth is undefined on the first observation; zero means no prior evidence of a change.
    for field in ("sales_growth", "quantity_growth", "rolling_sales_growth"):
        frame[field] = frame[field].fillna(0.0).clip(-10, 10)
    return frame


def validate_model_features(frame: pd.DataFrame, features: list[str]) -> None:
    missing = sorted(set(features).difference(frame.columns))
    if missing:
        raise ValueError(f"Configured model features are missing: {', '.join(missing)}")
    leaked = sorted(LABEL_COLUMNS.intersection(features))
    if leaked:
        raise ValueError(f"Evaluation-label leakage detected: {', '.join(leaked)}")
    non_numeric = [feature for feature in features if not pd.api.types.is_numeric_dtype(frame[feature])]
    if non_numeric:
        raise TypeError(f"Configured model features must be numeric: {', '.join(non_numeric)}")
