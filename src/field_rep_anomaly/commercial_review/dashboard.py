"""Read-only Streamlit views for the additive commercial-review benchmark.

The dashboard consumes only the compact semantic CSV layer written by
``commercial_review.dashboard_data``.  It deliberately contains no model
training or scoring code.  Manager-facing pages also remove controlled label
fields defensively, even though the production queue contract already excludes
them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .capacity import build_capacity_territory_summary


REQUIRED_DASHBOARD_FILES = (
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
)

OPTIONAL_DASHBOARD_FILES = (
    "dashboard_capacity_territory_allocation.csv",
    "dashboard_capacity_territory_summary.csv",
    "dashboard_model_curve.csv",
    "dashboard_pca_variance.csv",
    "dashboard_confusion_matrix.csv",
    "dashboard_score_distribution.csv",
    "dashboard_period_stability.csv",
)

PAGES = (
    "Executive Overview",
    "Manager Review Queue",
    "Rep Anomaly Drill-down",
    "Team and Manager View",
    "Capacity Overview",
    "Model Benchmark View",
    "Data and Model Health",
)

DEFINITIONS = {
    "Anomaly score": (
        "A percentile-calibrated measure of how unusual a rep-period pattern is relative to the "
        "frozen training reference. It is a prioritization signal, not a probability of misconduct."
    ),
    "Review priority": (
        "A deterministic workflow tier based on the review budget and score band; it does not state "
        "that an observation is incorrect."
    ),
    "Peer group": (
        "Commercially similar observations grouped using territory, customer/channel mix, product mix, "
        "tenure, team, customer count, and travel complexity where available."
    ),
    "Attainment": "Net eligible sales divided by the applicable target for the period, expressed as a percentage.",
    "Incentive residual": (
        "Paid or calculated incentive minus the policy-based expected incentive. Monetary values in this "
        "extension are synthetic."
    ),
    "Capacity utilization": "Required field-work hours divided by available field hours, expressed as a percentage.",
    "FTE gap": (
        "Required FTE minus available FTE. Positive values indicate modeled workload pressure; negative "
        "values indicate modeled spare capacity."
    ),
}

_LABEL_TOKENS = {
    "ground_truth_label",
    "anomaly_type",
    "anomaly_category",
    "severity",
    "injection_id",
    "affected_dataset",
    "affected_record_ids",
    "injection_count",
}

TableRenderer = Callable[..., Any]
SectionRenderer = Callable[[str, str], Any]


@st.cache_data(show_spinner=False)
def _read_dashboard_csv(path: str, modified_ns: int) -> pd.DataFrame:
    """Read one semantic CSV; the mtime invalidates Streamlit's data cache."""
    del modified_ns
    return pd.read_csv(path, low_memory=False)


def _load_tables(root: Path) -> tuple[dict[str, pd.DataFrame], list[str], list[str]]:
    directory = root / "data" / "dashboard"
    missing = [name for name in REQUIRED_DASHBOARD_FILES if not (directory / name).is_file()]
    errors: list[str] = []
    tables: dict[str, pd.DataFrame] = {}
    for filename in (*REQUIRED_DASHBOARD_FILES, *OPTIONAL_DASHBOARD_FILES):
        path = directory / filename
        if not path.is_file():
            continue
        try:
            tables[filename] = _read_dashboard_csv(str(path), path.stat().st_mtime_ns)
        except Exception as exc:  # partial/corrupt outputs must not take down the app
            tables[filename] = pd.DataFrame()
            errors.append(f"{filename}: {exc}")
    return tables, missing, errors


def _frame(tables: Mapping[str, pd.DataFrame], filename: str) -> pd.DataFrame:
    value = tables.get(filename)
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _manager_safe(frame: pd.DataFrame) -> pd.DataFrame:
    prohibited = [
        column
        for column in frame.columns
        if str(column).casefold() in _LABEL_TOKENS
        or "ground_truth" in str(column).casefold()
        or "injection_id" in str(column).casefold()
    ]
    return frame.drop(columns=prohibited, errors="ignore")


def _first_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lookup = {str(column).casefold(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    return None


def _available(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype(str).str.strip().str.casefold()
    return numeric.fillna(0).ne(0) | text.isin({"true", "yes", "y", "t"})


def _number(value: Any, default: float = 0.0) -> float:
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(converted) if pd.notna(converted) else default


def _sum(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())


def _metric_value(row: pd.Series, column: str, style: str = "number") -> str:
    value = _number(row.get(column, np.nan), np.nan)
    if not np.isfinite(value):
        return "—"
    if style == "money":
        return f"{value:,.0f}"
    if style == "percent":
        return f"{value * 100:,.1f}%"
    if style == "percent_already":
        return f"{value:,.1f}%"
    if style == "decimal":
        return f"{value:,.2f}"
    return f"{value:,.0f}"


def _download(label: str, frame: pd.DataFrame, filename: str, key: str) -> None:
    st.download_button(
        label,
        frame.to_csv(index=False),
        filename,
        "text/csv",
        key=key,
        disabled=frame.empty,
    )


def _plot(fig: go.Figure, key: str) -> None:
    fig.update_layout(
        margin=dict(l=18, r=18, t=55, b=20),
        legend_title_text="",
        hovermode="closest",
    )
    st.plotly_chart(fig, width="stretch", key=key)


def _line_chart(frame: pd.DataFrame, x: str | None, y: list[str], title: str, key: str) -> bool:
    available_y = [column for column in y if column in frame]
    if frame.empty or not x or x not in frame or not available_y:
        return False
    chart = frame[[x, *available_y]].copy()
    for column in available_y:
        chart[column] = pd.to_numeric(chart[column], errors="coerce")
    chart = chart.dropna(subset=[x]).sort_values(x)
    if chart[available_y].notna().sum().sum() == 0:
        return False
    _plot(px.line(chart, x=x, y=available_y, markers=True, title=title), key)
    return True


def _categorical_filter(
    frame: pd.DataFrame,
    column: str | None,
    label: str,
    key: str,
    container: Any,
) -> pd.DataFrame:
    if not column or column not in frame:
        return frame
    options = sorted(frame[column].dropna().astype(str).unique(), key=str.casefold)
    if not options:
        return frame
    selected = container.multiselect(label, options, key=key)
    if selected:
        return frame.loc[frame[column].astype(str).isin(selected)]
    return frame


def _definitions() -> None:
    with st.expander("Metric definitions and interpretation"):
        for name, definition in DEFINITIONS.items():
            st.markdown(f"**{name}.** {definition}")


def _responsible_use() -> None:
    st.divider()
    st.caption(
        "Responsible use: review signals identify unusual observations that require human validation; "
        "they are not findings of fraud or misconduct. Synthetic incentive, activity, expense, capacity, "
        "and controlled-label data must not be treated as verified production facts. Capacity outputs are "
        "planning aids, not automated employment decisions."
    )


def _empty_panel(message: str) -> None:
    st.info(message)


def _render_executive(
    tables: Mapping[str, pd.DataFrame], render_table: TableRenderer, section: SectionRenderer
) -> None:
    section(
        "Commercial review and capacity pulse",
        "Two separate questions: which rep-periods should managers validate first, and where might workload exceed modeled capacity?",
    )
    kpi = _frame(tables, "dashboard_kpi_summary.csv")
    if kpi.empty:
        _empty_panel("KPI summary is empty. Refresh after the commercial-review pipeline completes.")
        return
    row = kpi.iloc[0]
    currency = str(row.get("currency_code", "monetary units"))
    cards = [
        ("Gross sales", _metric_value(row, "total_gross_sales", "money"), f"Reported-source anchored; currency: {currency}."),
        ("Net sales", _metric_value(row, "total_net_sales", "money"), f"After synthetic discounts/returns; currency: {currency}."),
        ("Incentive paid", _metric_value(row, "total_incentive_paid", "money"), "Synthetic payout data."),
        ("Review candidates", _metric_value(row, "review_candidate_count"), DEFINITIONS["Review priority"]),
        ("High-priority candidates", _metric_value(row, "high_priority_review_candidate_count"), DEFINITIONS["Review priority"]),
        ("Review rate", _metric_value(row, "review_rate", "percent"), "Share placed in the manager review budget."),
        ("Overloaded territories", _metric_value(row, "overloaded_territory_count"), DEFINITIONS["Capacity utilization"]),
        ("Estimated positive FTE gap", _metric_value(row, "total_positive_fte_gap", "decimal"), DEFINITIONS["FTE gap"]),
        (
            "Precision / recall",
            f"{_metric_value(row, 'test_precision_at_selected_threshold', 'percent')} / "
            f"{_metric_value(row, 'test_recall_at_selected_threshold', 'percent')}",
            "Controlled benchmark test metrics at the frozen manager threshold.",
        ),
    ]
    for start in range(0, len(cards), 3):
        columns = st.columns(3)
        for container, (label, value, help_text) in zip(columns, cards[start : start + 3]):
            container.metric(label, value, help=help_text)

    disclosure = str(
        row.get(
            "benchmark_mode_label_disclosure",
            "Benchmark labels are controlled synthetic labels for evaluation only and do not establish misconduct.",
        )
    )
    st.info(f"Benchmark-label disclosure: {disclosure}")

    queue = _manager_safe(_frame(tables, "dashboard_manager_review_queue.csv"))
    capacity = _manager_safe(_frame(tables, "dashboard_capacity_summary.csv"))
    left, right = st.columns(2)
    priority = _first_column(queue, ("review_priority", "risk_band"))
    if priority and not queue.empty:
        counts = queue[priority].fillna("Unspecified").astype(str).value_counts().rename_axis(priority).reset_index(name="candidates")
        with left:
            _plot(px.bar(counts, x=priority, y="candidates", color=priority, title="Review candidates by workflow priority"), "cr_exec_priority")
    else:
        left.info("Review-priority distribution is unavailable.")
    risk = _first_column(capacity, ("capacity_risk_band", "workload_risk_band"))
    if risk and not capacity.empty:
        counts = capacity[risk].fillna("Unspecified").astype(str).value_counts().rename_axis(risk).reset_index(name="rep_periods")
        with right:
            _plot(px.bar(counts, x=risk, y="rep_periods", color=risk, title="Capacity risk by rep-period"), "cr_exec_capacity")
    else:
        right.info("Capacity-risk distribution is unavailable.")

    st.subheader("Highest-priority review candidates")
    rank = _first_column(queue, ("review_rank", "anomaly_score"))
    if rank:
        queue = queue.sort_values(rank, ascending=rank != "review_rank", kind="stable")
    columns = _available(
        queue,
        (
            "review_rank", "review_priority", "rep_name", "manager_name", "team_name", "territory_name",
            "period", "anomaly_score", "primary_reason", "recommended_review_action",
        ),
    )
    render_table(queue[columns].head(12) if columns else queue.head(12), height=390)
    _download("Download KPI summary", kpi, "dashboard_kpi_summary.csv", "cr_download_kpi")


def _render_manager_queue(
    tables: Mapping[str, pd.DataFrame], render_table: TableRenderer, section: SectionRenderer
) -> None:
    section(
        "Ranked manager review queue",
        "Use filters to focus validation work. Controlled labels are intentionally excluded from this production-style view.",
    )
    queue = _manager_safe(_frame(tables, "dashboard_manager_review_queue.csv"))
    if queue.empty:
        _empty_panel("No manager review candidates are currently available. This may be a valid zero-candidate result.")
        return
    original = queue.copy()
    filter_specs = [
        (("manager_name", "manager_id"), "Manager"),
        (("team_name", "team_id"), "Team"),
        (("territory_name", "territory_id"), "Territory"),
        (("rep_name", "rep_id"), "Representative"),
        (("period",), "Period"),
        (("risk_band", "review_priority"), "Score severity"),
        (("primary_reason", "primary_reason_code"), "Reason"),
    ]
    containers = st.columns(4)
    for index, (candidates, label) in enumerate(filter_specs):
        queue = _categorical_filter(
            queue,
            _first_column(original, candidates),
            label,
            f"cr_queue_filter_{index}",
            containers[index % len(containers)],
        )
    rank = _first_column(queue, ("review_rank", "anomaly_score", "anomaly_percentile"))
    if rank:
        queue = queue.sort_values(rank, ascending=rank == "review_rank", kind="stable")
    columns = _available(
        queue,
        (
            "review_rank", "review_priority", "risk_band", "rep_name", "rep_id", "manager_name",
            "team_name", "territory_name", "period", "anomaly_score", "anomaly_percentile",
            "primary_reason", "driver_1_name", "driver_2_name", "driver_3_name", "gross_sales",
            "net_sales", "target_sales", "attainment_pct", "final_incentive_paid", "expected_incentive",
            "incentive_residual", "payout_to_peer_median_ratio", "average_discount_pct",
            "post_incentive_return_rate", "impossible_travel_count", "capacity_utilization_pct", "fte_gap",
            "recommended_review_action",
        ),
    )
    st.caption(f"Showing {len(queue):,} of {len(original):,} review candidates.")
    render_table(queue[columns] if columns else queue, height=540)

    observation = _first_column(queue, ("observation_id",))
    if observation and not queue.empty:
        labels = {}
        for _, row in queue.iterrows():
            identifier = str(row[observation])
            rep = row.get("rep_name", row.get("rep_id", "Representative"))
            period = row.get("period", "period unavailable")
            labels[identifier] = f"{rep} · {period}"
        selected = st.selectbox(
            "Drill-down selection",
            list(labels),
            format_func=lambda value: labels[value],
            key="cr_queue_drilldown_selection",
            help="The selection is retained when you open Rep Anomaly Drill-down from the sidebar.",
        )
        st.session_state["commercial_review_selected_observation"] = selected
        st.caption("Selection saved. Open **Rep Anomaly Drill-down** in the commercial review navigation.")
    _download(
        "Download filtered review queue",
        queue[columns] if columns else queue,
        "dashboard_manager_review_queue_filtered.csv",
        "cr_download_queue",
    )


def _selected_rep_rows(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    if summary.empty:
        return summary, None
    observation = _first_column(summary, ("observation_id",))
    rep = _first_column(summary, ("rep_id", "rep_name"))
    period = _first_column(summary, ("period",))
    options: list[str]
    labels: dict[str, str]
    key_column = observation
    work = summary.copy()
    if key_column:
        work["__dashboard_key"] = work[key_column].astype(str)
    else:
        key_column = "__dashboard_key"
        work[key_column] = np.arange(len(work)).astype(str)
    options = work[key_column].tolist()
    labels = {}
    for _, row in work.iterrows():
        value = str(row[key_column])
        labels[value] = f"{row.get('rep_name', row.get('rep_id', 'Representative'))} · {row.get('period', 'period unavailable')}"
    saved = str(st.session_state.get("commercial_review_selected_observation", ""))
    default = options.index(saved) if saved in options else 0
    selected = st.selectbox(
        "Representative-period",
        options,
        index=default,
        format_func=lambda value: labels.get(value, value),
        key="cr_drilldown_observation",
    )
    selected_row = work.loc[work[key_column].eq(str(selected))].iloc[0]
    if rep:
        history = work.loc[work[rep].astype(str).eq(str(selected_row[rep]))].copy()
    else:
        history = work.loc[work[key_column].eq(str(selected))].copy()
    if period:
        history["__period_sort"] = pd.to_datetime(history[period], errors="coerce")
        history = history.sort_values(["__period_sort", key_column], kind="stable")
    return history.drop(columns=["__period_sort"], errors="ignore"), selected_row


def _render_timeline(selected: pd.Series, period_column: str | None) -> None:
    if not period_column or period_column not in selected:
        st.info("Period-close and payout timeline is unavailable for this row.")
        return
    period = pd.to_datetime(selected.get(period_column), errors="coerce")
    if pd.isna(period):
        st.info("Period-close and payout timeline is unavailable for this row.")
        return
    period_start = period.to_period("M").start_time
    period_close = period.to_period("M").end_time.normalize()
    events = [
        {"event": "Incentive period opens", "date": period_start, "source": "derived from period"},
        {"event": "Incentive period closes", "date": period_close, "source": "derived from period"},
    ]
    payout_column = next((column for column in ("payout_date", "incentive_payout_date") if column in selected.index), None)
    if payout_column:
        payout = pd.to_datetime(selected.get(payout_column), errors="coerce")
        if pd.notna(payout):
            events.append({"event": "Synthetic incentive payout", "date": payout, "source": payout_column})
    timeline = pd.DataFrame(events)
    _plot(
        px.scatter(timeline, x="date", y="event", color="source", title="Period-close and payout timeline"),
        "cr_drill_timeline",
    )
    if len(events) == 2:
        st.caption("Payout date is not present in this compact semantic export; no date was inferred or fabricated.")


def _render_rep_drilldown(
    tables: Mapping[str, pd.DataFrame], render_table: TableRenderer, section: SectionRenderer
) -> None:
    section(
        "Representative anomaly drill-down",
        "Review the commercial context, peer baseline, and PCA contribution evidence before validating the source records.",
    )
    summary = _manager_safe(_frame(tables, "dashboard_rep_period_summary.csv"))
    if summary.empty:
        _empty_panel("Representative-period summary is unavailable.")
        return
    history, selected = _selected_rep_rows(summary)
    if selected is None:
        return
    period = _first_column(history, ("period",))
    cards = st.columns(4)
    cards[0].metric("Anomaly score", _metric_value(selected, "anomaly_score", "decimal"), help=DEFINITIONS["Anomaly score"])
    cards[1].metric("Attainment", _metric_value(selected, "attainment_pct", "percent_already"), help=DEFINITIONS["Attainment"])
    cards[2].metric("Incentive residual", _metric_value(selected, "incentive_calculation_residual", "money"), help=DEFINITIONS["Incentive residual"])
    cards[3].metric("Capacity utilization", _metric_value(selected, "capacity_utilization_pct", "percent_already"), help=DEFINITIONS["Capacity utilization"])

    left, right = st.columns(2)
    with left:
        if not _line_chart(history, period, ["gross_sales", "net_sales", "target_sales"], "Sales and target trend", "cr_drill_sales"):
            st.info("Sales/target trend fields are unavailable.")
        if not _line_chart(history, period, ["attainment_pct"], "Target attainment trend", "cr_drill_attainment"):
            st.info("Attainment trend is unavailable.")
        if not _line_chart(
            history,
            period,
            ["average_discount_pct", "return_rate", "post_incentive_return_rate", "end_of_period_sales_share"],
            "Discount, return, and period-close indicators",
            "cr_drill_commercial",
        ):
            st.info("Discount/return/period-close indicators are unavailable.")
    with right:
        if not _line_chart(
            history,
            period,
            ["final_incentive_paid", "expected_incentive"],
            "Expected versus paid synthetic incentive",
            "cr_drill_incentive",
        ):
            st.info("Incentive trend is unavailable.")
        if not _line_chart(
            history,
            period,
            ["completed_visit_count", "average_visit_duration", "crm_interaction_count"],
            "Visit and CRM activity",
            "cr_drill_activity",
        ):
            st.info("Visit/CRM activity trend is unavailable.")
        if not _line_chart(
            history,
            period,
            ["claimed_expense_amount", "distance_claim_ratio", "impossible_travel_count"],
            "Travel and expense deviation",
            "cr_drill_expense",
        ):
            st.info("Travel/expense deviation is unavailable.")

    observation = _first_column(history, ("observation_id",))
    selected_id = str(selected.get(observation)) if observation else ""
    peer = _manager_safe(_frame(tables, "dashboard_peer_comparison.csv"))
    contributions = _manager_safe(_frame(tables, "dashboard_feature_contributions.csv"))
    if observation and observation in peer:
        peer = peer.loc[peer[observation].astype(str).eq(selected_id)]
    elif "rep_id" in peer and "rep_id" in selected:
        peer = peer.loc[peer.rep_id.astype(str).eq(str(selected.rep_id))]
    if observation and observation in contributions:
        contributions = contributions.loc[contributions[observation].astype(str).eq(selected_id)]

    left, right = st.columns(2)
    with left:
        st.subheader("Peer comparison")
        required = {"metric_name", "actual_value", "peer_median_value"}
        if required.issubset(peer.columns) and not peer.empty:
            peer_chart = peer[["metric_name", "actual_value", "peer_median_value"]].head(12).melt(
                id_vars="metric_name", var_name="comparison", value_name="value"
            )
            _plot(px.bar(peer_chart, x="metric_name", y="value", color="comparison", barmode="group", title="Actual versus peer median"), "cr_drill_peer")
        else:
            st.info("Peer comparison is unavailable for this observation.")
        render_table(peer.head(20), height=300)
    with right:
        st.subheader("Top score contributions")
        feature = _first_column(contributions, ("name", "feature", "feature_name"))
        value = _first_column(contributions, ("contribution", "contribution_value"))
        if feature and value and not contributions.empty:
            plot = contributions.copy()
            plot[value] = pd.to_numeric(plot[value], errors="coerce")
            if "driver_rank" in plot:
                plot = plot.sort_values("driver_rank")
            plot = plot.head(10)
            _plot(px.bar(plot, x=value, y=feature, orientation="h", title="Largest PCA reconstruction deviations"), "cr_drill_contributions")
        else:
            st.info("Feature-contribution evidence is unavailable for this observation.")
        st.caption("Contributions describe reconstruction error; they are not causal explanations.")
        render_table(contributions.head(20), height=300)

    st.subheader("Period-close context")
    _render_timeline(selected, period)
    detail_columns = _available(
        history,
        (
            "period", "gross_sales", "net_sales", "target_sales", "attainment_pct", "final_incentive_paid",
            "expected_incentive", "incentive_calculation_residual", "average_discount_pct", "return_rate",
            "post_incentive_return_rate", "end_of_period_sales_share", "completed_visit_count",
            "average_visit_duration", "distance_claim_ratio", "capacity_utilization_pct", "fte_gap",
        ),
    )
    render_table(history[detail_columns] if detail_columns else history, height=360)
    _download("Download representative history", history.drop(columns=["__dashboard_key"], errors="ignore"), "dashboard_rep_history.csv", "cr_download_rep")


def _organization_summary(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    work = frame.copy()
    flag = _first_column(work, ("manager_review_flag", "threshold_flag"))
    work["__review_flag"] = _as_bool(work[flag]) if flag else False
    grouped = work.groupby(group_column, dropna=False, observed=True)
    summary = grouped.size().rename("observations").to_frame()
    rep = _first_column(work, ("rep_id", "rep_name"))
    if rep:
        summary["team_size_reps"] = grouped[rep].nunique()
    else:
        summary["team_size_reps"] = summary["observations"]
    summary["review_candidates"] = grouped["__review_flag"].sum()
    summary["flag_rate"] = summary["review_candidates"] / summary["observations"].clip(lower=1)
    for source, target in (
        ("anomaly_score", "mean_anomaly_score"),
        ("gross_sales", "gross_sales"),
        ("net_sales", "net_sales"),
        ("final_incentive_paid", "total_incentive_paid"),
        ("peer_adjusted_performance", "mean_peer_adjusted_performance"),
    ):
        if source in work:
            work[source] = pd.to_numeric(work[source], errors="coerce")
            grouped = work.groupby(group_column, dropna=False, observed=True)
            summary[target] = grouped[source].sum() if source in {"gross_sales", "net_sales", "final_incentive_paid"} else grouped[source].mean()
    return summary.reset_index().sort_values(["flag_rate", "observations"], ascending=[False, False], kind="stable")


def _render_team_manager(
    tables: Mapping[str, pd.DataFrame], render_table: TableRenderer, section: SectionRenderer
) -> None:
    section(
        "Team and manager context",
        "Flag rates and score distributions account for observation volume; raw candidate counts are shown only with their denominator.",
    )
    summary = _manager_safe(_frame(tables, "dashboard_rep_period_summary.csv"))
    if summary.empty:
        _empty_panel("Representative-period summary is unavailable.")
        return
    peer = _manager_safe(_frame(tables, "dashboard_peer_comparison.csv"))
    if {"observation_id", "peer_z_score"}.issubset(peer.columns) and "observation_id" in summary:
        peer_rollup = peer.groupby("observation_id", observed=True).peer_z_score.mean().rename("peer_adjusted_performance")
        summary = summary.merge(peer_rollup, on="observation_id", how="left", validate="one_to_one")
    manager = _first_column(summary, ("manager_name", "manager_id"))
    team = _first_column(summary, ("team_name", "team_id"))
    if not manager and not team:
        _empty_panel("Manager and team identifiers are unavailable.")
        return
    level_options = {"Manager": manager, "Team": team}
    level_options = {label: column for label, column in level_options.items() if column}
    level = st.selectbox("Summarize by", list(level_options), key="cr_org_level")
    group_column = level_options[level]
    grouped = _organization_summary(summary, group_column)
    st.caption("Ordered by flag rate, not by raw review-candidate count.")
    render_table(grouped, height=390)

    left, right = st.columns(2)
    with left:
        x = "flag_rate"
        y = "mean_anomaly_score" if "mean_anomaly_score" in grouped else "review_candidates"
        size = "team_size_reps" if "team_size_reps" in grouped else None
        _plot(
            px.scatter(
                grouped,
                x=x,
                y=y,
                size=size,
                hover_name=group_column,
                title=f"{level} flag rate versus anomaly score",
                labels={x: "Flag rate (denominator-adjusted)"},
            ),
            "cr_org_rate",
        )
    with right:
        if "anomaly_score" in summary:
            _plot(px.box(summary, x=group_column, y="anomaly_score", points="outliers", title=f"Anomaly-score distribution by {level.lower()}"), "cr_org_distribution")
        else:
            st.info("Anomaly-score distribution is unavailable.")

    period = _first_column(summary, ("period",))
    flag = _first_column(summary, ("manager_review_flag", "threshold_flag"))
    if period and flag:
        heat = summary[[group_column, period, flag]].copy()
        heat["flag"] = _as_bool(heat[flag]).astype(float)
        heat = heat.pivot_table(index=group_column, columns=period, values="flag", aggfunc="mean", fill_value=0)
        if not heat.empty:
            _plot(
                go.Figure(
                    data=go.Heatmap(
                        z=heat.to_numpy(float),
                        x=[str(value) for value in heat.columns],
                        y=[str(value) for value in heat.index],
                        colorbar=dict(title="Flag rate"),
                        hovertemplate=f"{level}: %{{y}}<br>Period: %{{x}}<br>Flag rate: %{{z:.1%}}<extra></extra>",
                    ),
                    layout=dict(title=f"Period flag-rate heatmap by {level.lower()}"),
                ),
                "cr_org_heatmap",
            )
    _download("Download organization summary", grouped, "dashboard_team_manager_summary.csv", "cr_download_org")


def _balancing_opportunities(capacity: pd.DataFrame) -> pd.DataFrame:
    if capacity.empty or "fte_gap" not in capacity:
        return pd.DataFrame()
    work = capacity.copy()
    work["fte_gap"] = pd.to_numeric(work["fte_gap"], errors="coerce")
    group_columns = _available(work, ("period", "team_id", "team_name", "manager_id", "manager_name"))
    grouping = [column for column in group_columns if column in {"period", "team_id", "team_name"}]
    if not grouping:
        work["__all"] = "All"
        grouping = ["__all"]
    receiver_name = _first_column(work, ("territory_name", "rep_name", "territory_id", "rep_id"))
    if not receiver_name:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    grouper: Any = grouping[0] if len(grouping) == 1 else grouping
    for group_value, part in work.groupby(grouper, dropna=False, observed=True):
        receivers = part.loc[part.fte_gap.gt(0)].sort_values("fte_gap", ascending=False)
        donors = part.loc[part.fte_gap.lt(0)].sort_values("fte_gap")
        if receivers.empty or donors.empty:
            continue
        values = group_value if isinstance(group_value, tuple) else (group_value,)
        common = dict(zip(grouping, values))
        for (_, receiver), (_, donor) in zip(receivers.iterrows(), donors.iterrows()):
            rows.append(
                {
                    **common,
                    "potential_receiver": receiver[receiver_name],
                    "potential_donor": donor[receiver_name],
                    "receiver_fte_gap": receiver.fte_gap,
                    "donor_spare_fte": abs(donor.fte_gap),
                    "illustrative_balance_fte": min(receiver.fte_gap, abs(donor.fte_gap)),
                }
            )
    return pd.DataFrame(rows).drop(columns=["__all"], errors="ignore").sort_values(
        "illustrative_balance_fte", ascending=False, kind="stable"
    ) if rows else pd.DataFrame()


def _territory_capacity_view(
    filtered_rep_capacity: pd.DataFrame,
    territory_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Select whole territory-period rows represented by the active rep filters."""
    if territory_summary.empty:
        return filtered_rep_capacity.copy()
    result = territory_summary.copy()
    territory_key = _first_column(
        filtered_rep_capacity, ("territory_id", "territory_name")
    )
    if not territory_key or territory_key not in result:
        return result.iloc[0:0].copy() if filtered_rep_capacity.empty else result
    keys = [territory_key]
    if "period" in filtered_rep_capacity and "period" in result:
        keys.append("period")
    allowed = filtered_rep_capacity[keys].astype(str).drop_duplicates()
    comparable = result.copy()
    comparable[keys] = comparable[keys].astype(str)
    return comparable.merge(allowed, on=keys, how="inner", validate="many_to_one")


def _filter_capacity_allocation(
    rep_capacity: pd.DataFrame,
    allocation: pd.DataFrame,
    selections: Mapping[str, Iterable[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply identical semantic selections to allocation rows and linked reps."""
    filtered = allocation.copy()
    for column, selected in selections.items():
        values = {str(value) for value in selected}
        if values and column in filtered:
            filtered = filtered.loc[filtered[column].astype(str).isin(values)].copy()
    if filtered.empty:
        return rep_capacity.iloc[0:0].copy(), filtered
    if "capacity_record_id" in filtered and "capacity_record_id" in rep_capacity:
        record_ids = set(filtered["capacity_record_id"].dropna().astype(str))
        reps = rep_capacity.loc[
            rep_capacity["capacity_record_id"].astype(str).isin(record_ids)
        ].copy()
    else:
        keys = [column for column in ("rep_id", "period") if column in filtered and column in rep_capacity]
        if not keys:
            reps = rep_capacity.copy()
        else:
            selected_keys = pd.MultiIndex.from_frame(filtered[keys].astype(str).drop_duplicates())
            reps = rep_capacity.loc[
                pd.MultiIndex.from_frame(rep_capacity[keys].astype(str)).isin(selected_keys)
            ].copy()
    return reps, filtered


def _render_capacity(
    tables: Mapping[str, pd.DataFrame], render_table: TableRenderer, section: SectionRenderer
) -> None:
    section(
        "Field-force capacity overview",
        "Required workload and available capacity are deterministic planning estimates, separate from anomaly-review priority.",
    )
    capacity = _manager_safe(_frame(tables, "dashboard_capacity_summary.csv"))
    customer = _manager_safe(_frame(tables, "dashboard_capacity_customer_drilldown.csv"))
    territory_allocation = _manager_safe(
        _frame(tables, "dashboard_capacity_territory_allocation.csv")
    )
    territory_summary = _manager_safe(
        _frame(tables, "dashboard_capacity_territory_summary.csv")
    )
    if capacity.empty:
        _empty_panel("Capacity summary is unavailable.")
        return
    specs = [
        (("manager_name", "manager_id"), "Manager"),
        (("team_name", "team_id"), "Team"),
        (("territory_name", "territory_id"), "Allocated territory"),
        (("period",), "Period"),
        (("capacity_risk_band", "workload_risk_band"), "Rep-period capacity risk"),
    ]
    containers = st.columns(3)
    if not territory_allocation.empty:
        original_allocation = territory_allocation.copy()
        selections: dict[str, list[str]] = {}
        for index, (candidates, label) in enumerate(specs):
            column = _first_column(territory_allocation, candidates)
            if not column:
                continue
            options = sorted(
                territory_allocation[column].dropna().astype(str).unique(),
                key=str.casefold,
            )
            selected = containers[index % len(containers)].multiselect(
                label, options, key=f"cr_capacity_filter_{index}"
            )
            selections[column] = selected
            if selected:
                territory_allocation = territory_allocation.loc[
                    territory_allocation[column].astype(str).isin(selected)
                ].copy()
        capacity, territory_allocation = _filter_capacity_allocation(
            capacity, original_allocation, selections
        )
        territory_capacity = (
            build_capacity_territory_summary(territory_allocation)
            if not territory_allocation.empty
            else territory_summary.iloc[0:0].copy()
        )
        capacity_totals = territory_allocation
        st.caption(
            "All controls use rep-territory-period allocation rows. Visit/travel workload is "
            "attributed to the actual synthetic visit territory and coverage workload to the "
            "owned-customer territory; shared roster availability is allocated proportionally."
        )
    else:
        original = capacity.copy()
        for index, (candidates, label) in enumerate(specs):
            capacity = _categorical_filter(
                capacity,
                _first_column(original, candidates),
                label,
                f"cr_capacity_filter_{index}",
                containers[index % len(containers)],
            )
        territory_capacity = _territory_capacity_view(capacity, territory_summary)
        capacity_totals = capacity
        st.caption(
            "The optional allocation fact is unavailable; filters use compact rep-period rows."
        )
    risk = _first_column(capacity_totals, ("capacity_risk_band", "workload_risk_band"))
    overload = _first_column(capacity_totals, ("capacity_overload_flag", "overload_flag"))
    cards = st.columns(4)
    cards[0].metric("Required hours", f"{_sum(capacity_totals, 'required_total_hours'):,.1f}")
    cards[1].metric("Available hours", f"{_sum(capacity_totals, 'available_field_hours'):,.1f}")
    cards[2].metric("Positive FTE gap", f"{_sum(capacity_totals.assign(fte_gap=pd.to_numeric(capacity_totals.get('fte_gap'), errors='coerce').clip(lower=0)) if 'fte_gap' in capacity_totals else capacity_totals, 'fte_gap'):,.2f}", help=DEFINITIONS["FTE gap"])
    if overload:
        overloaded_rows = capacity_totals.loc[_as_bool(capacity_totals[overload])]
    elif risk:
        overloaded_rows = capacity_totals.loc[
            capacity_totals[risk].astype(str).str.casefold().isin({"high", "critical"})
        ]
    else:
        overloaded_rows = capacity_totals.iloc[0:0]
    if "capacity_record_id" in overloaded_rows:
        overloaded = int(overloaded_rows["capacity_record_id"].nunique())
    elif {"rep_id", "period"}.issubset(overloaded_rows.columns):
        overloaded = int(overloaded_rows[["rep_id", "period"]].drop_duplicates().shape[0])
    else:
        overloaded = len(overloaded_rows)
    cards[3].metric("Overloaded rep-periods", f"{overloaded:,}", help=DEFINITIONS["Capacity utilization"])

    territory = _first_column(
        territory_capacity, ("territory_name", "territory_id", "rep_name", "rep_id")
    )
    territory_risk = _first_column(
        territory_capacity, ("capacity_risk_band", "workload_risk_band")
    )
    left, right = st.columns(2)
    with left:
        if territory and {"required_total_hours", "available_field_hours"}.issubset(territory_capacity.columns):
            workload = territory_capacity.groupby(territory, dropna=False, observed=True)[["required_total_hours", "available_field_hours"]].sum().reset_index()
            _plot(px.bar(workload, x=territory, y=["required_total_hours", "available_field_hours"], barmode="group", title="Territory required versus allocated available hours"), "cr_capacity_hours")
        else:
            st.info("Required/available hour comparison is unavailable.")
        if territory and {"required_fte", "available_fte"}.issubset(territory_capacity.columns):
            fte = territory_capacity.groupby(territory, dropna=False, observed=True)[["required_fte", "available_fte"]].sum().reset_index()
            _plot(px.bar(fte, x=territory, y=["required_fte", "available_fte"], barmode="group", title="Territory required versus allocated available FTE"), "cr_capacity_fte")
    with right:
        utilization = _first_column(territory_capacity, ("capacity_utilization_pct", "utilization_pct"))
        if territory and utilization:
            _plot(px.scatter(territory_capacity, x=territory, y=utilization, color=territory_risk, hover_data=_available(territory_capacity, ("period", "rep_count", "fractional_rep_equivalent", "fte_gap")), title="Territory-period capacity utilization"), "cr_capacity_utilization")
        else:
            st.info("Capacity-utilization chart is unavailable.")
        coverage_gap = _first_column(
            territory_capacity,
            ("priority_customer_coverage_gap", "customer_coverage_gap"),
        )
        if territory and coverage_gap:
            coverage = (
                territory_capacity.groupby(territory, dropna=False, observed=True)[
                    coverage_gap
                ]
                .sum()
                .reset_index()
            )
            _plot(
                px.bar(
                    coverage,
                    x=territory,
                    y=coverage_gap,
                    title="High-priority customer coverage gap by territory",
                ),
                "cr_capacity_coverage_gap",
            )
        else:
            st.info("Customer coverage-gap chart is unavailable.")

    st.subheader("Potential workload-balancing opportunities")
    opportunities = _balancing_opportunities(capacity)
    if opportunities.empty:
        st.info("No within-team donor/receiver pair is supported by the current filters.")
    else:
        render_table(opportunities.head(25), height=330)
        st.caption("Illustrative pairing uses modeled FTE gaps only; travel, skills, labor rules, and manager validation remain required.")

    if not customer.empty and {"rep_id", "period"}.issubset(customer.columns):
        if capacity.empty:
            customer = customer.iloc[0:0].copy()
        elif {"rep_id", "period"}.issubset(capacity.columns):
            selected_keys = pd.MultiIndex.from_frame(
                capacity[["rep_id", "period"]].astype(str).drop_duplicates()
            )
            customer = customer.loc[
                pd.MultiIndex.from_frame(customer[["rep_id", "period"]].astype(str)).isin(
                    selected_keys
                )
            ].copy()
        if not territory_allocation.empty:
            customer_territory = _first_column(
                customer, ("territory_id", "territory_name")
            )
            allocation_territory = _first_column(
                territory_allocation, ("territory_id", "territory_name")
            )
            if customer_territory and allocation_territory:
                allowed_territories = set(
                    territory_allocation[allocation_territory]
                    .dropna()
                    .astype(str)
                )
                customer = customer.loc[
                    customer[customer_territory]
                    .astype(str)
                    .isin(allowed_territories)
                ].copy()
    columns = _available(
        capacity,
        (
            "rep_name", "manager_name", "team_name", "territory_name", "period", "required_total_hours",
            "available_field_hours", "capacity_utilization_pct", "required_fte", "available_fte", "fte_gap",
            "priority_customer_coverage_gap", "workload_per_active_customer", "capacity_risk_band",
        ),
    )
    render_table(capacity[columns] if columns else capacity, height=420)
    if not territory_allocation.empty:
        st.caption(
            "The representative table shows whole rep-period records linked to the selected "
            "allocation rows; cards and territory charts use the allocated quantities."
        )
    with st.expander("Customer coverage drill-down"):
        render_table(customer.head(1000), height=420)
        st.caption("The on-screen drill-down is capped at 1,000 rows; the download contains the full semantic table.")
        _download("Download customer coverage", customer, "dashboard_capacity_customer_drilldown.csv", "cr_download_coverage")
    if not territory_summary.empty:
        with st.expander("Allocated territory-period capacity"):
            render_table(territory_capacity, height=360)
            _download(
                "Download territory capacity",
                territory_capacity,
                "dashboard_capacity_territory_summary_filtered.csv",
                "cr_download_territory_capacity",
            )
            if not territory_allocation.empty:
                _download(
                    "Download filtered territory allocation",
                    territory_allocation,
                    "dashboard_capacity_territory_allocation_filtered.csv",
                    "cr_download_territory_allocation",
                )
    _download("Download filtered capacity summary", capacity, "dashboard_capacity_summary_filtered.csv", "cr_download_capacity")
    st.warning("Capacity risk is a workload-planning signal, not a performance rating or automated staffing decision.")


def _curve(frame: pd.DataFrame, curve_type: str) -> pd.DataFrame:
    if frame.empty or "curve_type" not in frame:
        return pd.DataFrame()
    return frame.loc[frame.curve_type.astype(str).str.casefold().eq(curve_type.casefold())].copy()


def _render_benchmark(
    tables: Mapping[str, pd.DataFrame], render_table: TableRenderer, section: SectionRenderer
) -> None:
    section(
        "Finalized PCA controlled benchmark",
        "Evaluation-only labels are visible on this page. The dashboard reads persisted results and never fits or refits a model.",
    )
    st.warning(
        "Benchmark mode: labels identify controlled synthetic injections only. Precision, recall, ROC, lift, and subgroup recall "
        "measure benchmark recovery—not verified production performance or misconduct."
    )
    metrics = _frame(tables, "dashboard_model_metrics.csv")
    type_metrics = _frame(tables, "dashboard_anomaly_type_metrics.csv")
    kpi = _frame(tables, "dashboard_kpi_summary.csv")
    threshold = np.nan
    if not kpi.empty:
        threshold = _number(kpi.iloc[0].get("selected_threshold", np.nan), np.nan)
    if not np.isfinite(threshold) and "threshold" in metrics:
        values = pd.to_numeric(metrics.threshold, errors="coerce").dropna()
        threshold = float(values.iloc[0]) if len(values) else np.nan
    columns = st.columns(3)
    columns[0].metric("Manager-review threshold", f"{threshold:,.4f}" if np.isfinite(threshold) else "—", help="Frozen on unlabelled validation scores; labels did not select it.")
    if not kpi.empty:
        columns[1].metric("Test precision", _metric_value(kpi.iloc[0], "test_precision_at_selected_threshold", "percent"))
        columns[2].metric("Test recall", _metric_value(kpi.iloc[0], "test_recall_at_selected_threshold", "percent"))
    else:
        columns[1].metric("Test precision", "—")
        columns[2].metric("Test recall", "—")

    distribution = _frame(tables, "dashboard_score_distribution.csv")
    curves = _frame(tables, "dashboard_model_curve.csv")
    left, right = st.columns(2)
    with left:
        if {"population", "score_lower", "count"}.issubset(distribution.columns):
            shown = distribution.copy()
            if "split" in shown and shown.split.astype(str).eq("test").any():
                shown = shown.loc[shown.split.astype(str).eq("test")]
            _plot(px.line(shown, x="score_lower", y="count", color="population", markers=True, title="Clean versus injected score distribution"), "cr_benchmark_distribution")
        else:
            st.info("Optional clean-versus-injected score distribution is unavailable.")
        pr = _curve(curves, "precision_recall")
        if {"recall", "precision"}.issubset(pr.columns) and not pr.empty:
            _plot(px.line(pr, x="recall", y="precision", title="Precision–recall curve"), "cr_benchmark_pr")
        else:
            st.info("Optional precision–recall curve is unavailable.")
    with right:
        roc = _curve(curves, "roc")
        if {"false_positive_rate", "true_positive_rate"}.issubset(roc.columns) and not roc.empty:
            fig = px.line(roc, x="false_positive_rate", y="true_positive_rate", title="ROC curve")
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
            _plot(fig, "cr_benchmark_roc")
        else:
            st.info("Optional ROC curve is unavailable.")
        lift = _curve(curves, "lift")
        if {"review_fraction", "lift"}.issubset(lift.columns) and not lift.empty:
            _plot(px.line(lift, x="review_fraction", y="lift", title="Lift at review depth"), "cr_benchmark_lift")
        else:
            st.info("Optional lift curve is unavailable.")

    left, right = st.columns(2)
    for container, grouping, title, key in (
        (left, "anomaly_type", "Recall by controlled anomaly type", "cr_benchmark_type"),
        (right, "severity", "Recall by controlled severity", "cr_benchmark_severity"),
    ):
        subset = type_metrics.loc[type_metrics.grouping.astype(str).eq(grouping)] if "grouping" in type_metrics else pd.DataFrame()
        metric = _first_column(subset, ("recall_at_threshold", "recall_at_top5pct", "detection_rate_at_threshold"))
        label = _first_column(subset, ("value", grouping))
        with container:
            if metric and label and not subset.empty:
                _plot(px.bar(subset, x=label, y=metric, title=title), key)
            else:
                st.info(f"{title} is unavailable.")

    confusion = _frame(tables, "dashboard_confusion_matrix.csv")
    variance = _frame(tables, "dashboard_pca_variance.csv")
    left, right = st.columns(2)
    with left:
        if {"actual", "predicted", "count"}.issubset(confusion.columns) and not confusion.empty:
            matrix = confusion.pivot_table(index="actual", columns="predicted", values="count", aggfunc="sum", fill_value=0)
            _plot(
                go.Figure(
                    data=go.Heatmap(
                        z=matrix.to_numpy(float),
                        x=[str(value) for value in matrix.columns],
                        y=[str(value) for value in matrix.index],
                        text=matrix.to_numpy(),
                        texttemplate="%{text}",
                        colorscale="Blues",
                    ),
                    layout=dict(title="Confusion matrix at selected threshold", xaxis_title="Predicted", yaxis_title="Actual"),
                ),
                "cr_benchmark_confusion",
            )
        else:
            st.info("Optional confusion matrix is unavailable.")
    with right:
        cumulative = _first_column(variance, ("cumulative_explained_variance", "cumulative_variance"))
        component = _first_column(variance, ("component", "pca_component"))
        if cumulative and component and not variance.empty:
            _plot(px.line(variance, x=component, y=cumulative, markers=True, title="PCA cumulative explained variance"), "cr_benchmark_variance")
        else:
            st.info("Optional PCA variance table is unavailable.")

    st.subheader("Persisted benchmark metrics")
    render_table(metrics, height=420)
    _download("Download benchmark metrics", metrics, "dashboard_model_metrics.csv", "cr_download_model_metrics")
    _download("Download subgroup metrics", type_metrics, "dashboard_anomaly_type_metrics.csv", "cr_download_type_metrics")


def _manifest_value(manifest: pd.DataFrame, candidates: Iterable[str]) -> str:
    if manifest.empty:
        return "Unavailable"
    column = _first_column(manifest, candidates)
    if not column:
        return "Unavailable"
    value = manifest.iloc[0][column]
    return "Unavailable" if pd.isna(value) else str(value)


def _render_health(
    tables: Mapping[str, pd.DataFrame], render_table: TableRenderer, section: SectionRenderer
) -> None:
    section(
        "Data and model health",
        "Run provenance, dataset coverage, quality checks, and score stability from the persisted semantic layer.",
    )
    quality = _frame(tables, "dashboard_data_quality.csv")
    manifest = _frame(tables, "dashboard_run_manifest.csv")
    stability = _frame(tables, "dashboard_period_stability.csv")
    rep_period = _manager_safe(_frame(tables, "dashboard_rep_period_summary.csv"))
    cards = st.columns(4)
    cards[0].metric("Model run timestamp", _manifest_value(manifest, ("execution_timestamp", "run_timestamp")))
    cards[1].metric("Seed", _manifest_value(manifest, ("random_seed", "seed")))
    cards[2].metric("Model / version", _manifest_value(manifest, ("finalized_model_name", "model_version", "model_name")))
    cards[3].metric("Config version", _manifest_value(manifest, ("configuration_hash", "config_version", "configuration_file")))
    st.caption(f"Input fingerprint: {_manifest_value(manifest, ('input_file_hash', 'input_file_fingerprint', 'source_sha256'))}")

    row_counts = pd.DataFrame(
        [
            {"dashboard_dataset": filename, "rows": len(tables.get(filename, pd.DataFrame()))}
            for filename in REQUIRED_DASHBOARD_FILES
        ]
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Dataset row counts")
        render_table(row_counts, height=390)
    with right:
        st.subheader("Date coverage")
        period = _first_column(rep_period, ("period",))
        if period:
            dates = pd.to_datetime(rep_period[period], errors="coerce").dropna()
            coverage = pd.DataFrame(
                [{"dataset": "dashboard_rep_period_summary.csv", "date_start": dates.min(), "date_end": dates.max(), "periods": dates.nunique()}]
            )
            render_table(coverage, height=150)
        else:
            st.info("Period coverage is unavailable.")

        lineage_rows = []
        for filename in REQUIRED_DASHBOARD_FILES:
            frame = tables.get(filename, pd.DataFrame())
            if isinstance(frame, pd.DataFrame) and "data_lineage" in frame:
                for lineage, count in frame.data_lineage.fillna("unspecified").astype(str).value_counts().items():
                    lineage_rows.append({"dashboard_dataset": filename, "data_lineage": lineage, "rows": int(count)})
        st.subheader("Synthetic versus observed lineage")
        if lineage_rows:
            render_table(pd.DataFrame(lineage_rows), height=250)
        else:
            st.info("Lineage counts are unavailable in the compact tables.")

    if quality.empty:
        st.info("Data-quality checks are unavailable.")
    else:
        check = _first_column(quality, ("check_name", "check", "metric"))
        st.subheader("Missing values and referential integrity")
        if check:
            names = quality[check].fillna("").astype(str)
            focused = quality.loc[
                names.str.contains("missing", case=False)
                | names.str.startswith("fk__")
                | names.str.contains("referential", case=False)
            ]
            render_table(focused if not focused.empty else quality, height=420)
        else:
            render_table(quality, height=420)
        status = _first_column(quality, ("status",))
        if status:
            counts = quality[status].fillna("unspecified").astype(str).value_counts().rename_axis("status").reset_index(name="checks")
            _plot(px.bar(counts, x="status", y="checks", color="status", title="Quality-check status"), "cr_health_quality")

    if not stability.empty and {"period", "mean_anomaly_score"}.issubset(stability.columns):
        color = "population" if "population" in stability else None
        _plot(px.line(stability, x="period", y="mean_anomaly_score", color=color, markers=True, title="Period score stability"), "cr_health_stability")
        render_table(stability, height=320)
    else:
        st.info("Optional period-stability output is unavailable.")
    st.subheader("Run manifest")
    render_table(manifest, height=260)
    _download("Download data-quality checks", quality, "dashboard_data_quality.csv", "cr_download_quality")
    _download("Download run manifest", manifest, "dashboard_run_manifest.csv", "cr_download_manifest")


def render_commercial_review(
    root: str | Path,
    render_table: TableRenderer,
    section: SectionRenderer,
) -> None:
    """Render the seven-page commercial-review workspace from persisted CSVs.

    Parameters
    ----------
    root:
        Repository root containing ``data/dashboard``.
    render_table, section:
        Existing application render helpers. Reusing them keeps this additive
        workspace visually consistent with the established Streamlit app.
    """
    repository_root = Path(root).expanduser().resolve()
    tables, missing, errors = _load_tables(repository_root)

    st.sidebar.caption("Commercial-review semantic layer · read only")
    page = st.sidebar.radio("Commercial review workspace", PAGES, key="commercial_review_workspace")

    if missing:
        st.info(
            "Commercial review dashboard-ready datasets are not yet complete. "
            "Run the commercial-review pipeline, then refresh this page. Missing: "
            + ", ".join(missing)
        )
        return
    if errors:
        st.warning("Some dashboard CSVs could not be read: " + " | ".join(errors))
    empty_required = [filename for filename in REQUIRED_DASHBOARD_FILES if tables.get(filename, pd.DataFrame()).empty]
    if empty_required:
        st.warning("Some required dashboard datasets are empty: " + ", ".join(empty_required))

    renderers = {
        "Executive Overview": _render_executive,
        "Manager Review Queue": _render_manager_queue,
        "Rep Anomaly Drill-down": _render_rep_drilldown,
        "Team and Manager View": _render_team_manager,
        "Capacity Overview": _render_capacity,
        "Model Benchmark View": _render_benchmark,
        "Data and Model Health": _render_health,
    }
    renderers[page](tables, render_table, section)
    _definitions()
    _responsible_use()


__all__ = [
    "OPTIONAL_DASHBOARD_FILES",
    "PAGES",
    "REQUIRED_DASHBOARD_FILES",
    "render_commercial_review",
]
