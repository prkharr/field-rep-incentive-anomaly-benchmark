"""Pure, manager-facing projections of already executed capacity outputs.

No function in this module fits a forecast, changes a capacity assumption, or
writes an input artifact.  Input frames are copied before transformation.

Provenance conventions
----------------------
* Forecast-selection WAPE is the persisted, pooled workload metric, not a
  separately estimated business-unit metric.  Its validation/test populations
  can differ with source coverage.
* ``*_load`` fields are independently forecast commercial counts (customers,
  transactions, cities, products, distributors), NOT weighted contributions to
  the composite workload index.
* Workload scenario bounds are recovered algebraically from saved FTE bounds
  and saved TRAIN capacity quantiles.  They are scenario bounds, not confidence
  intervals.  No forecast is recomputed.
* Recent workload growth compares the latest three observed calendar months
  with the preceding three, as of each unit's saved last-observed month.  All
  six months must exist in saved backtest observations; otherwise it is null.
* Stale units retain genuine historical observations, but no forecast or FTE
  recommendation is substituted with zero.
* ``fte_reallocated`` is signed at the unit grain: positive for a receiver,
  negative for a donor.  A paired transfer therefore sums to zero.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .planning.capacity import required_fte


BASE_GRAIN = ["team", "country", "product_class"]
SCENARIO_GRAIN = BASE_GRAIN + ["scenario_name"]
CAPACITY_GAP_EPSILON = 1e-9

SCENARIO_NAMES = {
    "Base case": "Base",
    "Base": "Base",
    "Demand +10%": "Demand +10%",
    "Demand +20%": "Demand +20%",
    "Add 1 representative": "Add 1 FTE",
    "Add 1 FTE": "Add 1 FTE",
    "Add 2 representatives": "Add 2 FTE",
    "Add 2 FTE": "Add 2 FTE",
    "Capacity -10%": "Capacity -10%",
    "Product launch +30%": "Product Launch",
    "Product Launch": "Product Launch",
    "Reallocate 1 representative (bounded by donor FTE)": "Net-zero Reallocation",
    "Net-zero Reallocation": "Net-zero Reallocation",
}

SCENARIO_DESCRIPTIONS = {
    "Base": "Persisted base workload forecast, fractional FTE allocation, and sustainable per-rep capacity.",
    "Demand +10%": "Workload increases 10%; allocated FTE and sustainable per-rep capacity are unchanged.",
    "Demand +20%": "Workload increases 20%; allocated FTE and sustainable per-rep capacity are unchanged.",
    "Add 1 FTE": "One FTE is added to this unit independently; workload and per-rep capacity are unchanged.",
    "Add 2 FTE": "Two FTE are added to this unit independently; workload and per-rep capacity are unchanged.",
    "Capacity -10%": "Sustainable per-rep capacity decreases 10%; workload and allocated FTE are unchanged.",
    "Product Launch": "Existing launch assumption: workload increases 30%; FTE and per-rep capacity are unchanged.",
    "Net-zero Reallocation": "Existing paired transfer, bounded by donor FTE; signed FTE is positive for the receiver and negative for the donor.",
}


def capacity_priority(
    fte_gap: float, eligible: bool = True, epsilon: float = CAPACITY_GAP_EPSILON
) -> str:
    """Label a deterministic FTE gap; this is not a staffing recommendation.

    ``abs(fte_gap) <= 1e-9`` is Balanced.  No utilization threshold, model
    selection, or observed test outcome changes this presentation rule.
    """
    if not eligible:
        return "Ineligible / Stale Coverage"
    if epsilon < 0 or not np.isfinite(epsilon):
        raise ValueError("Capacity-priority epsilon must be finite and nonnegative")
    if pd.isna(fte_gap) or not np.isfinite(float(fte_gap)):
        raise ValueError("An eligible capacity unit must have a finite FTE gap")
    if float(fte_gap) > epsilon:
        return "Potential Capacity Gap"
    if float(fte_gap) < -epsilon:
        return "Potential Spare Capacity"
    return "Balanced"


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _require_unique(frame: pd.DataFrame, grain: list[str], label: str) -> None:
    _require_columns(frame, grain, label)
    if frame[grain].isna().any().any():
        raise ValueError(f"{label} contains null grain identifiers: {grain}")
    if frame.duplicated(grain).any():
        raise ValueError(f"{label} contains duplicate rows at grain {grain}")


def _number(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="raise").astype(float)


def _bool_series(values: pd.Series, label: str) -> pd.Series:
    normalized = values.astype("string").str.strip().str.lower()
    valid = normalized.isin(["true", "false", "1", "0", "1.0", "0.0"])
    if not valid.all():
        raise ValueError(f"{label} contains missing or ambiguous boolean values")
    return normalized.isin(["true", "1", "1.0"]).astype(bool)


def _assert_close(actual: Any, expected: Any, label: str) -> None:
    if not np.allclose(
        np.asarray(actual, dtype=float), np.asarray(expected, dtype=float),
        rtol=1e-9, atol=1e-9, equal_nan=True,
    ):
        raise ValueError(f"Capacity source reconciliation failed: {label}")


def _assumptions(frame: pd.DataFrame) -> dict[str, float]:
    _require_columns(frame, ["assumption", "value"], "capacity assumptions")
    _require_unique(frame, ["assumption"], "capacity assumptions")
    return dict(zip(frame.assumption, pd.to_numeric(frame.value, errors="raise")))


def _append_forecast_metrics(
    result: pd.DataFrame, forecast_metrics: pd.DataFrame
) -> pd.DataFrame:
    _require_columns(forecast_metrics, ["metric", "method", "split", "WAPE"], "forecast metrics")
    subset = forecast_metrics.loc[
        forecast_metrics.metric.eq("workload")
        & forecast_metrics.split.isin(["validation", "test"])
    ].copy()
    _require_unique(subset, ["method", "split"], "workload forecast metrics")
    for split in ["validation", "test"]:
        values = subset.loc[subset.split.eq(split), ["method", "WAPE"]].rename(
            columns={"method": "selected_forecast_method", "WAPE": f"{split}_wape"}
        )
        result = result.merge(values, on="selected_forecast_method", how="left", validate="many_to_one")
    result["forecast_metric_scope"] = (
        "Pooled workload WAPE across available business-unit observations in each split; "
        "not unit-specific. Validation and test coverage may differ."
    )
    return result


def _append_workload_history(
    result: pd.DataFrame, forecast_backtest: pd.DataFrame, planning_view: pd.DataFrame
) -> pd.DataFrame:
    _require_columns(forecast_backtest, BASE_GRAIN + ["date", "metric", "observed"], "forecast backtest")
    _require_columns(planning_view, BASE_GRAIN + ["date"], "planning view")
    view = planning_view.copy()
    view["date"] = pd.to_datetime(view.date, errors="raise")
    _require_unique(view, BASE_GRAIN + ["date"], "planning view")
    observations = forecast_backtest.loc[
        forecast_backtest.metric.eq("workload"), BASE_GRAIN + ["date", "observed"]
    ].copy()
    observations["date"] = pd.to_datetime(observations.date, errors="raise")
    observations["observed"] = pd.to_numeric(observations.observed, errors="raise")
    # The same actual appears once for each forecasting method; only deduplicate
    # after checking that all methods agree on the actual value.
    distinct = observations.groupby(BASE_GRAIN + ["date"], dropna=False)["observed"].nunique(dropna=False)
    if distinct.gt(1).any():
        raise ValueError("Forecast backtest methods disagree on observed workload")
    observations = observations.drop_duplicates(BASE_GRAIN + ["date"])
    history_groups = {
        key: group.set_index("date").observed.sort_index()
        for key, group in observations.groupby(BASE_GRAIN, dropna=False)
    }
    coverage_groups = {key: group.date for key, group in view.groupby(BASE_GRAIN, dropna=False)}
    latest_values, growth_values = [], []
    for row in result.itertuples(index=False):
        key = tuple(getattr(row, field) for field in BASE_GRAIN)
        last_month = pd.Timestamp(row.last_observed_month)
        horizon = pd.Timestamp(row.forecast_horizon)
        dates = coverage_groups.get(key, pd.Series([], dtype="datetime64[ns]"))
        known_dates = dates.loc[dates.lt(horizon)]
        if known_dates.empty or known_dates.max() != last_month:
            raise ValueError(f"Planning coverage disagrees with last-observed month for {key}")
        history = history_groups.get(key, pd.Series([], dtype=float))
        history = history.loc[(history.index <= last_month) & (history.index < horizon)]
        latest_values.append(float(history.loc[last_month]) if last_month in history.index else np.nan)
        calendar = pd.date_range(last_month - pd.DateOffset(months=5), last_month, freq="MS")
        window = history.reindex(calendar)
        growth = np.nan
        if len(window) == 6 and window.notna().all() and np.isfinite(window).all():
            prior = float(window.iloc[:3].mean())
            if prior > 0:
                growth = float(window.iloc[3:].mean() / prior - 1)
        growth_values.append(growth)
    result["latest_observed_workload"] = latest_values
    result["recent_workload_growth"] = growth_values
    result["recent_workload_growth_basis"] = (
        "Mean workload in the latest 3 calendar months / mean in the preceding 3 months - 1; "
        "as of last_observed_month, requiring all 6 saved actuals."
    )
    return result


def _validate_allocation(result: pd.DataFrame, allocation: pd.DataFrame) -> None:
    _require_columns(allocation, ["representative"] + BASE_GRAIN + ["date", "allocated_fte"], "FTE allocation")
    source = allocation.copy()
    source["date"] = pd.to_datetime(source.date, errors="raise")
    _require_unique(source, ["representative"] + BASE_GRAIN + ["date"], "FTE allocation")
    source["allocated_fte"] = pd.to_numeric(source.allocated_fte, errors="raise")
    if source.empty or source.date.nunique() != 1:
        raise ValueError("Base FTE allocation must describe exactly one observed month")
    if not np.isfinite(source.allocated_fte).all() or source.allocated_fte.lt(0).any():
        raise ValueError("Observed fractional FTE allocation must be finite and nonnegative")
    eligible = result.loc[result.eligible_for_capacity_recommendation].copy()
    if not pd.to_datetime(eligible.last_observed_month).eq(source.date.iloc[0]).all():
        raise ValueError("FTE allocation month differs from eligible units' last-observed month")
    _assert_close(source.groupby("representative").allocated_fte.sum(), 1.0, "one total FTE per observed representative")
    grouped = source.groupby(BASE_GRAIN, as_index=False).allocated_fte.sum()
    matched = grouped.merge(eligible[BASE_GRAIN], on=BASE_GRAIN, how="left", indicator=True, validate="one_to_one")
    if not matched._merge.eq("both").all():
        raise ValueError("Observed FTE allocation includes a unit outside the eligible base")
    checked = eligible[BASE_GRAIN + ["allocated_fte"]].merge(
        grouped, on=BASE_GRAIN, how="left", suffixes=("", "_source"), validate="one_to_one"
    )
    # Zero is valid for an eligible unit with genuinely no saved allocation;
    # this fill never touches an ineligible/stale unit.
    _assert_close(checked.allocated_fte, checked.allocated_fte_source.fillna(0), "base versus observed unit FTE")
    _assert_close(eligible.allocated_fte.sum(), source.representative.nunique(), "base total versus observed representative count")
    if "workload" in source:
        workloads = source.groupby(BASE_GRAIN, as_index=False).workload.sum().rename(columns={"workload": "allocation_workload"})
        checked_work = eligible[BASE_GRAIN + ["latest_observed_workload"]].merge(
            workloads, on=BASE_GRAIN, how="left", validate="one_to_one"
        ).dropna(subset=["latest_observed_workload", "allocation_workload"])
        _assert_close(checked_work.latest_observed_workload, checked_work.allocation_workload, "latest workload versus allocation source")


def build_capacity_base_dataset(
    planning: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    capacity_assumptions: pd.DataFrame,
    forecast_backtest: pd.DataFrame,
    planning_view: pd.DataFrame,
    allocation: pd.DataFrame,
    cleaning_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    """Project persisted planning outputs to one row per team/country/class.

    The existing ``planning_eligible`` flag is authoritative; contradictory
    coverage or arithmetic raises rather than manufacturing a recommendation.
    Forecast dates are ISO first-of-month strings.  Source frames are unchanged.
    """
    required = BASE_GRAIN + [
        "forecast_date", "planning_eligible", "last_observed_month", "forecast_workload",
        "allocated_current_fte", "capacity_per_rep", "required_fte", "fte_gap", "forecast_method",
    ]
    _require_columns(planning, required, "capacity planning")
    _require_unique(planning, BASE_GRAIN, "capacity planning")
    if planning.empty:
        raise ValueError("Capacity planning output is empty")
    source = planning.copy().reset_index(drop=True)
    horizons = pd.to_datetime(source.forecast_date, errors="raise")
    observed_months = pd.to_datetime(source.last_observed_month, errors="raise")
    if horizons.isna().any() or observed_months.isna().any() or horizons.nunique() != 1:
        raise ValueError("Capacity base requires one nonmissing planning horizon and known coverage dates")
    if not horizons.dt.is_month_start.all() or not observed_months.dt.is_month_start.all():
        raise ValueError("Capacity forecast and observed dates must identify calendar months")
    if observed_months.ge(horizons).any():
        raise ValueError("Capacity history must precede the planning horizon")
    eligible = _bool_series(source.planning_eligible, "planning_eligible")
    expected_latest = horizons - pd.offsets.MonthBegin(1)
    if (eligible & observed_months.ne(expected_latest)).any():
        raise ValueError("A stale-coverage unit cannot be eligible for a capacity recommendation")

    result = source[BASE_GRAIN].copy()
    result["forecast_horizon"] = horizons.dt.strftime("%Y-%m-%d")
    result["eligible_for_capacity_recommendation"] = eligible
    result["selected_forecast_method"] = source.forecast_method
    result["forecast_workload"] = _number(source, "forecast_workload")
    result["forecast_error_metric_used_for_selection"] = "WAPE"
    result["sustainable_workload_per_rep"] = _number(source, "capacity_per_rep")
    result["allocated_fte"] = _number(source, "allocated_current_fte")
    result["required_fte"] = _number(source, "required_fte")
    result["fte_gap"] = _number(source, "fte_gap")
    result["required_fte_lower"] = _number(source, "required_fte_low")
    result["required_fte_upper"] = _number(source, "required_fte_high")
    result["fte_gap_lower"] = result.required_fte_lower - result.allocated_fte
    result["fte_gap_upper"] = result.required_fte_upper - result.allocated_fte
    assumptions = _assumptions(capacity_assumptions)
    capacity_low, capacity_high = assumptions.get("capacity_low", np.nan), assumptions.get("capacity_high", np.nan)
    for name, value in [("capacity_low", capacity_low), ("capacity_high", capacity_high)]:
        if pd.notna(value) and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"Saved {name} must be finite and positive")
    if pd.notna(capacity_low) and pd.notna(capacity_high) and capacity_low > capacity_high:
        raise ValueError("Saved lower capacity quantile exceeds the upper capacity quantile")
    result["forecast_lower_scenario"] = result.required_fte_lower * capacity_high
    result["forecast_upper_scenario"] = result.required_fte_upper * capacity_low
    result["forecast_scenario_basis"] = (
        "Recovered from saved FTE scenario bounds and TRAIN capacity quantiles; "
        "validation absolute-error sensitivity, not a statistical confidence interval."
    )

    _require_columns(cleaning_sensitivity, BASE_GRAIN + ["raw_required_fte", "cleaned_required_fte"], "cleaning sensitivity")
    _require_unique(cleaning_sensitivity, BASE_GRAIN, "cleaning sensitivity")
    cleaning = source[BASE_GRAIN].merge(
        cleaning_sensitivity[BASE_GRAIN + ["raw_required_fte", "cleaned_required_fte"]],
        on=BASE_GRAIN, how="left", indicator=True, validate="one_to_one",
    )
    if not cleaning._merge.eq("both").all() or len(cleaning_sensitivity) != len(source):
        raise ValueError("Cleaning sensitivity must cover exactly the planning units")
    for name in ["raw_required_fte", "cleaned_required_fte"]:
        if name in source:
            _assert_close(_number(source, name), _number(cleaning, name), f"persisted {name} versus cleaning sensitivity")
    result["workload_score_raw"] = _number(cleaning, "raw_required_fte") * result.sustainable_workload_per_rep
    result["workload_score_winsorized"] = _number(cleaning, "cleaned_required_fte") * result.sustainable_workload_per_rep
    for friendly, technical in {
        "customer_load": "forecast_distinct_customers",
        "transaction_load": "forecast_transaction_count",
        "geography_load": "forecast_distinct_cities",
        "product_load": "forecast_distinct_products",
        "distributor_load": "forecast_distributor_count",
    }.items():
        result[friendly] = _number(source, technical)
    result["workload_components_basis"] = (
        "Independently forecast counts of customers, transactions, cities, products, and distributors; "
        "not additive weighted-workload contributions."
    )
    result["last_observed_month"] = observed_months.dt.strftime("%Y-%m-%d")
    result["coverage_note"] = source.get("coverage_note", pd.Series(pd.NA, index=source.index))
    result = _append_forecast_metrics(result, forecast_metrics)
    result = _append_workload_history(result, forecast_backtest, planning_view)

    eligible = result.eligible_for_capacity_recommendation
    quantitative_recommendations = [
        "forecast_workload", "validation_wape", "test_wape", "forecast_lower_scenario", "forecast_upper_scenario",
        "sustainable_workload_per_rep", "allocated_fte", "required_fte", "fte_gap", "required_fte_lower",
        "required_fte_upper", "fte_gap_lower", "fte_gap_upper", "workload_score_raw", "workload_score_winsorized",
        "customer_load", "transaction_load", "geography_load", "product_load", "distributor_load",
    ]
    # Source coverage is preserved explicitly.  Absent Poland months are never
    # interpreted as zero demand, zero current staffing, or spare capacity.
    result.loc[~eligible, quantitative_recommendations] = np.nan
    result.loc[~eligible, ["selected_forecast_method", "forecast_error_metric_used_for_selection"]] = pd.NA
    finite_required = ["forecast_workload", "sustainable_workload_per_rep", "allocated_fte", "required_fte", "fte_gap"]
    if not np.isfinite(result.loc[eligible, finite_required].to_numpy(dtype=float)).all():
        raise ValueError("Eligible base capacity recommendations must be finite")
    if result.loc[eligible, "sustainable_workload_per_rep"].le(0).any():
        raise ValueError("Sustainable per-rep workload must be positive")
    if result.loc[eligible, ["forecast_workload", "allocated_fte", "required_fte"]].lt(0).any().any():
        raise ValueError("Workload and FTE values must be nonnegative")
    if np.isinf(result[quantitative_recommendations].to_numpy(dtype=float)).any():
        raise ValueError("Dashboard capacity quantities cannot contain infinite values")
    _assert_close(result.loc[eligible, "fte_gap"], result.loc[eligible, "required_fte"] - result.loc[eligible, "allocated_fte"], "base FTE gap")
    _assert_close(result.loc[eligible, "required_fte"], result.loc[eligible, "forecast_workload"] / result.loc[eligible, "sustainable_workload_per_rep"], "base required FTE")
    _assert_close(result.loc[eligible, "workload_score_raw"], result.loc[eligible, "forecast_workload"], "raw workload reconstruction")
    if "sustainable_capacity_per_rep" in assumptions:
        _assert_close(result.loc[eligible, "sustainable_workload_per_rep"], assumptions["sustainable_capacity_per_rep"], "saved sustainable capacity assumption")
    fte_bounded = eligible & result.required_fte_lower.notna() & result.required_fte_upper.notna()
    if (
        result.loc[fte_bounded, "required_fte_lower"].lt(-CAPACITY_GAP_EPSILON).any()
        or (result.loc[fte_bounded, "required_fte_lower"] > result.loc[fte_bounded, "required_fte"] + CAPACITY_GAP_EPSILON).any()
        or (result.loc[fte_bounded, "required_fte_upper"] < result.loc[fte_bounded, "required_fte"] - CAPACITY_GAP_EPSILON).any()
    ):
        raise ValueError("Saved FTE scenario bounds do not surround the base required FTE")
    bounded = eligible & result.forecast_lower_scenario.notna() & result.forecast_upper_scenario.notna()
    if (
        result.loc[bounded, "forecast_lower_scenario"].lt(-CAPACITY_GAP_EPSILON).any()
        or (result.loc[bounded, "forecast_lower_scenario"] > result.loc[bounded, "forecast_workload"] + CAPACITY_GAP_EPSILON).any()
        or (result.loc[bounded, "forecast_upper_scenario"] < result.loc[bounded, "forecast_workload"] - CAPACITY_GAP_EPSILON).any()
    ):
        raise ValueError("Recovered forecast scenario bounds do not surround the saved workload forecast")
    result["capacity_priority"] = [capacity_priority(gap, flag) for gap, flag in zip(result.fte_gap, eligible)]
    _validate_allocation(result, allocation)
    _require_unique(result, BASE_GRAIN, "dashboard capacity base")
    return result.sort_values(BASE_GRAIN, kind="mergesort").reset_index(drop=True)


def build_capacity_scenario_dataset(scenarios: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Map saved scenario rows without expanding unobserved or unchanged units.

    Existing output has seven scenarios for every eligible unit and two rows
    for its one real net-zero transfer.  The transfer is not a proposal to add
    one rep to every cell.  No stale unit receives a manufactured scenario.
    """
    _require_columns(scenarios, BASE_GRAIN + [
        "scenario", "scenario_change", "workload_before", "capacity_before", "fte_gap_before",
        "workload_after", "capacity_after", "remaining_gap",
    ], "capacity scenarios")
    _require_unique(base, BASE_GRAIN, "dashboard capacity base")
    _require_columns(base, [
        "forecast_horizon", "eligible_for_capacity_recommendation", "forecast_workload",
        "sustainable_workload_per_rep", "allocated_fte", "required_fte", "fte_gap",
    ], "dashboard capacity base")
    source = scenarios.copy()
    source["scenario_name"] = source.scenario.map(SCENARIO_NAMES)
    if source.scenario_name.isna().any():
        unknown = source.loc[source.scenario_name.isna(), "scenario"].unique().tolist()
        raise ValueError(f"Unrecognized persisted capacity scenarios: {unknown}")
    _require_unique(source, SCENARIO_GRAIN, "capacity scenarios")
    source = source.merge(
        base[BASE_GRAIN + ["forecast_horizon", "eligible_for_capacity_recommendation", "forecast_workload",
                           "sustainable_workload_per_rep", "allocated_fte", "required_fte", "fte_gap"]],
        on=BASE_GRAIN, how="left", indicator=True, validate="many_to_one",
    )
    if not source._merge.eq("both").all():
        raise ValueError("A scenario references a unit missing from the base dataset")
    flags = _bool_series(source.eligible_for_capacity_recommendation, "scenario eligibility")
    if not flags.all():
        raise ValueError("A persisted scenario attempts a recommendation for a stale/ineligible unit")
    _assert_close(source.workload_before, source.forecast_workload, "scenario base workload")
    _assert_close(source.capacity_before, source.allocated_fte * source.sustainable_workload_per_rep, "scenario base capacity")
    _assert_close(source.fte_gap_before, source.fte_gap, "scenario base FTE gap")

    output = []
    for row in source.itertuples(index=False):
        name = row.scenario_name
        after_workload, after_capacity = float(row.workload_after), float(row.capacity_after)
        base_fte, per_rep = float(row.allocated_fte), float(row.sustainable_workload_per_rep)
        if not np.isfinite([after_workload, after_capacity, base_fte, per_rep]).all() or min(after_workload, after_capacity, base_fte) < 0 or per_rep <= 0:
            raise ValueError("Scenario workload, FTE, and capacity must be valid finite quantities")
        if name == "Capacity -10%":
            # This saved scenario keeps FTE fixed.  Recover its effective
            # per-rep capacity from the persisted capacity total whenever
            # identifiable, using its named assumption only at zero staffing.
            if base_fte > 0:
                per_rep = after_capacity / base_fte
            elif abs(float(row.remaining_gap)) > CAPACITY_GAP_EPSILON:
                per_rep = (after_workload - after_capacity) / float(row.remaining_gap)
            else:
                per_rep = float(row.sustainable_workload_per_rep) * .9
            _assert_close(per_rep, float(row.sustainable_workload_per_rep) * .9, "Capacity -10% effective per-rep capacity")
        allocated = after_capacity / per_rep
        computed = required_fte(after_workload, per_rep, allocated)
        _assert_close(computed["fte_gap"], row.remaining_gap, f"{name} saved remaining gap")
        demand_factor = {"Demand +10%": 1.1, "Demand +20%": 1.2, "Product Launch": 1.3}.get(name, 1.0)
        _assert_close(after_workload, float(row.workload_before) * demand_factor, f"{name} saved demand assumption")
        change = allocated - base_fte
        if name != "Net-zero Reallocation":
            _assert_close(change, {"Add 1 FTE": 1.0, "Add 2 FTE": 2.0}.get(name, 0.0), f"{name} saved FTE assumption")
        output.append({
            **{column: getattr(row, column) for column in BASE_GRAIN},
            "forecast_horizon": row.forecast_horizon,
            "scenario_name": name,
            "scenario_description": SCENARIO_DESCRIPTIONS[name],
            "forecast_workload": after_workload,
            "allocated_fte": allocated,
            "sustainable_capacity_per_rep": per_rep,
            "required_fte": computed["required_fte"],
            "fte_gap": float(row.remaining_gap),
            "capacity_priority": capacity_priority(float(row.remaining_gap), True),
            "eligible_for_capacity_recommendation": True,
            "source_unit": pd.NA,
            "target_unit": pd.NA,
            "fte_reallocated": np.nan,
            "reallocation_role": pd.NA,
            "allocated_fte_change": change,
            "scenario_source_name": row.scenario,
            "scenario_source_description": row.scenario_change,
        })
    if not output:
        raise ValueError("Capacity scenario output is empty")
    result = pd.DataFrame(output)
    transfer = result.loc[result.scenario_name.eq("Net-zero Reallocation")]
    if not transfer.empty:
        donor = transfer.loc[transfer.allocated_fte_change.lt(-CAPACITY_GAP_EPSILON)]
        receiver = transfer.loc[transfer.allocated_fte_change.gt(CAPACITY_GAP_EPSILON)]
        if len(transfer) != 2 or len(donor) != 1 or len(receiver) != 1:
            raise ValueError("Saved reallocation must identify one unambiguous donor/receiver pair")
        _assert_close(transfer.allocated_fte_change.sum(), 0.0, "net-zero paired FTE transfer")
        if donor.team.iloc[0] != receiver.team.iloc[0]:
            raise ValueError("Saved reallocation pair crosses the existing within-team scenario scope")
        amount = float(receiver.allocated_fte_change.iloc[0])
        donor_base_fte = float(donor.allocated_fte.iloc[0] - donor.allocated_fte_change.iloc[0])
        if amount > min(1.0, donor_base_fte) + CAPACITY_GAP_EPSILON:
            raise ValueError("Saved reallocation exceeds one FTE or the donor's observed allocation")
        source_unit = " | ".join(str(donor.iloc[0][column]) for column in BASE_GRAIN)
        target_unit = " | ".join(str(receiver.iloc[0][column]) for column in BASE_GRAIN)
        result.loc[transfer.index, "source_unit"] = source_unit
        result.loc[transfer.index, "target_unit"] = target_unit
        result.loc[transfer.index, "fte_reallocated"] = transfer.allocated_fte_change
        result.loc[donor.index, "reallocation_role"] = "Donor"
        result.loc[receiver.index, "reallocation_role"] = "Receiver"
    _require_unique(result, SCENARIO_GRAIN, "dashboard capacity scenarios")
    return result.sort_values(SCENARIO_GRAIN, kind="mergesort").reset_index(drop=True)
