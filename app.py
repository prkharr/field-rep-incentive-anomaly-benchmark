"""Streamlit dashboard for the field-representative anomaly benchmark.

The application is deliberately read-only: it presents artifacts produced by
``run_pipeline.py`` and never fits or re-fits a model.  Column resolution is kept
defensive because the pipeline can adapt to a provided pharmaceutical dataset.
"""

from __future__ import annotations

import html
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Field Rep Anomaly Command Center",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded",
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
METRICS_DIR = ROOT / "artifacts" / "metrics"
PLOTS_DIR = ROOT / "artifacts" / "plots"

TABLE_PATHS = {
    "scored_observations": DATA_DIR / "scored_observations.csv",
    "rep_risk_summary": DATA_DIR / "rep_risk_summary.csv",
    "clustering_benchmark": METRICS_DIR / "clustering_benchmark.csv",
    "anomaly_metrics": METRICS_DIR / "anomaly_metrics.csv",
    "cluster_profiles": METRICS_DIR / "cluster_profiles_kmeans.csv",
    "model_selection": METRICS_DIR / "model_selection.csv",
}


st.markdown(
    """
    <style>
      :root {
        --ink: #10253f;
        --muted: #61738a;
        --blue: #155eef;
        --cyan: #0aa6a6;
        --panel: #f7faff;
        --line: #dce6f2;
      }
      .stApp { background: #fbfdff; color: var(--ink); }
      [data-testid="stSidebar"] { background: #f4f8fc; border-right: 1px solid var(--line); }
      [data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 5px 18px rgba(16, 37, 63, 0.045);
      }
      [data-testid="stMetricLabel"] { color: var(--muted); }
      [data-testid="stMetricValue"] { color: var(--ink); }
      .hero {
        background: linear-gradient(115deg, #0c2748 0%, #164c88 63%, #087f8c 100%);
        border-radius: 18px;
        padding: 23px 28px;
        margin-bottom: 20px;
        color: white;
        box-shadow: 0 12px 28px rgba(15, 54, 95, 0.16);
      }
      .hero-kicker { font-size: 0.76rem; letter-spacing: 0.13em; text-transform: uppercase; opacity: 0.76; }
      .hero-title { font-size: 1.75rem; font-weight: 720; margin: 4px 0 2px; }
      .hero-subtitle { font-size: 0.96rem; opacity: 0.85; }
      .status-pill {
        display: inline-block;
        margin-top: 12px;
        padding: 5px 10px;
        border: 1px solid rgba(255,255,255,.28);
        border-radius: 999px;
        background: rgba(255,255,255,.11);
        font-size: .78rem;
      }
      .section-kicker { color: var(--blue); text-transform: uppercase; letter-spacing: .1em; font-size: .72rem; font-weight: 700; }
      .section-title { color: var(--ink); font-size: 1.28rem; font-weight: 680; margin: 1px 0 2px; }
      .section-caption { color: var(--muted); font-size: .9rem; margin-bottom: 10px; }
      .model-card, .note-card {
        background: white;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 17px 19px;
        min-height: 128px;
      }
      .model-card h4, .note-card h4 { color: var(--ink); margin: 0 0 6px; }
      .model-card p, .note-card p { color: var(--muted); margin: 0; font-size: .91rem; line-height: 1.5; }
      .quiet { color: var(--muted); font-size: .82rem; }
      div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
      .stTabs [data-baseweb="tab-list"] { gap: 4px; }
      .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; }
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _read_csv_cached(path: str, modified_ns: int) -> pd.DataFrame:
    """Read a CSV; ``modified_ns`` makes the cache refresh after a pipeline run."""
    del modified_ns
    return pd.read_csv(path, low_memory=False)


@st.cache_data(show_spinner=False)
def _read_json_cached(path: str, modified_ns: int) -> Any:
    del modified_ns
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> tuple[pd.DataFrame, str | None]:
    if not path.exists():
        return pd.DataFrame(), None
    try:
        return _read_csv_cached(str(path), path.stat().st_mtime_ns), None
    except Exception as exc:  # a corrupt partial artifact should not crash the app
        return pd.DataFrame(), f"{path.name}: {exc}"


def read_json(path: Path) -> tuple[Any | None, str | None]:
    if not path.exists():
        return None, None
    try:
        return _read_json_cached(str(path), path.stat().st_mtime_ns), None
    except Exception as exc:
        return None, f"{path.name}: {exc}"


def token(value: Any) -> str:
    """Return a comparison-safe token for a column, model, or metric name."""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def find_column(frame: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    if frame.empty and len(frame.columns) == 0:
        return None
    lookup = {token(column): str(column) for column in frame.columns}
    for alias in aliases:
        if token(alias) in lookup:
            return lookup[token(alias)]
    return None


def canonical_model(value: Any) -> str | None:
    name = token(value)
    if "kmeans" in name:
        return "K-Means"
    if "dbscan" in name:
        return "DBSCAN"
    return str(value).strip() if value is not None and str(value).strip() else None


FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "rep": ("rep_id", "representative_id", "field_rep_id", "sales_rep_id", "rep"),
    "country": ("country", "country_name"),
    "city": ("city", "city_name"),
    "territory": ("territory_id", "territory", "territory_name", "geography"),
    "team": ("sales_team", "team", "team_name"),
    "manager": ("manager_id", "sales_manager", "manager", "manager_name"),
    "product": ("product_name", "product", "brand", "brand_name"),
    "product_class": ("product_class", "therapy_area", "therapeutic_class", "class"),
    "channel": ("channel", "sales_channel"),
    "subchannel": ("subchannel", "sub_channel", "sub-channel", "sales_subchannel"),
    "latitude": ("latitude", "lat", "mean_latitude"),
    "longitude": ("longitude", "lon", "lng", "long", "mean_longitude"),
    "date": ("date", "month_date", "year_month", "period", "observation_month", "month"),
    "sales": ("total_sales", "sales", "sales_value", "revenue", "net_sales"),
    "quantity": ("total_quantity", "quantity", "units", "volume"),
    "customers": ("unique_customers", "customer_count", "customers"),
    "activity": ("total_calls", "calls", "call_count", "field_calls"),
    "attainment": ("target_attainment_pct", "sales_target_attainment_pct", "target_attainment"),
    "incentive": (
        "actual_incentive_paid",
        "total_incentive_paid",
        "actual_incentive",
        "incentive_paid",
        "incentive",
    ),
    "manual_override": ("manual_override_amount", "payout_adjustment", "incentive_adjustment"),
}


NUMERIC_FIELDS = {
    "latitude",
    "longitude",
    "sales",
    "quantity",
    "customers",
    "activity",
    "attainment",
    "incentive",
    "manual_override",
}


def model_score_column(frame: pd.DataFrame, model: str) -> str | None:
    if canonical_model(model) == "K-Means":
        aliases = (
            "kmeans_anomaly_score",
            "anomaly_score_kmeans",
            "kmeans_score",
            "centroid_anomaly_score",
            "centroid_distance_score",
        )
        model_token = "kmeans"
    else:
        aliases = (
            "dbscan_anomaly_score",
            "anomaly_score_dbscan",
            "dbscan_score",
            "density_anomaly_score",
            "knn_anomaly_score",
        )
        model_token = "dbscan"
    direct = find_column(frame, aliases)
    if direct:
        return direct
    for column in frame.columns:
        name = token(column)
        if model_token in name and "score" in name and ("anomaly" in name or "outlier" in name):
            return str(column)
    return find_column(frame, ("anomaly_score", "risk_score", "final_anomaly_score", "review_score"))


def model_cluster_column(frame: pd.DataFrame, model: str) -> str | None:
    if canonical_model(model) == "K-Means":
        aliases = ("kmeans_cluster", "cluster_kmeans", "kmeans_label", "cluster_id", "cluster")
        model_token = "kmeans"
    else:
        aliases = ("dbscan_cluster", "cluster_dbscan", "dbscan_label", "density_cluster")
        model_token = "dbscan"
    direct = find_column(frame, aliases)
    if direct:
        return direct
    for column in frame.columns:
        name = token(column)
        if model_token in name and ("cluster" in name or "label" in name):
            return str(column)
    return None


def model_flag_column(frame: pd.DataFrame, model: str) -> str | None:
    if canonical_model(model) == "K-Means":
        aliases = (
            "kmeans_anomaly_flag",
            "kmeans_flag",
            "is_kmeans_anomaly",
            "kmeans_review_flag",
        )
        model_token = "kmeans"
    else:
        aliases = (
            "dbscan_anomaly_flag",
            "dbscan_flag",
            "is_dbscan_anomaly",
            "dbscan_noise_flag",
            "dbscan_review_flag",
        )
        model_token = "dbscan"
    direct = find_column(frame, aliases)
    if direct:
        return direct
    for column in frame.columns:
        name = token(column)
        if model_token in name and ("flag" in name or "candidate" in name):
            return str(column)
    # Deliberately exclude injected_anomaly_flag: it is evaluation ground truth.
    return find_column(
        frame,
        ("anomaly_flag", "predicted_anomaly_flag", "review_flag", "high_risk_flag", "flagged_anomaly"),
    )


def driver_columns(frame: pd.DataFrame, model: str) -> list[str]:
    model_name = "kmeans" if canonical_model(model) == "K-Means" else "dbscan"
    direct_aliases = (
        f"{model_name}_top_anomaly_drivers",
        f"{model_name}_anomaly_reasons",
        f"{model_name}_top_drivers",
        "top_anomaly_drivers",
        "anomaly_reasons",
        "top_drivers",
        "explanation",
        "feature_contributions",
    )
    direct = find_column(frame, direct_aliases)
    if direct:
        return [direct]
    candidates: list[str] = []
    for column in frame.columns:
        name = token(column)
        is_explanation = any(part in name for part in ("driver", "reason", "topfeature", "contribution"))
        is_relevant_model = model_name in name or ("kmeans" not in name and "dbscan" not in name)
        if is_explanation and is_relevant_model:
            candidates.append(str(column))
    return candidates[:5]


def to_flag(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip().str.lower()
    truthy = {"true", "t", "yes", "y", "anomaly", "flagged", "noise", "outlier"}
    return ((numeric.fillna(0) > 0) | text.isin(truthy)).fillna(False).astype(bool)


def combine_driver_text(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series("Not available in exported artifacts", index=frame.index, dtype="string")

    def combine(row: pd.Series) -> str:
        values: list[str] = []
        for value in row:
            if pd.isna(value):
                continue
            text_value = str(value).strip()
            if text_value and text_value.lower() not in {"nan", "none", "[]", "{}"}:
                values.append(text_value)
        return " · ".join(dict.fromkeys(values)) if values else "Not available in exported artifacts"

    return frame[columns].apply(combine, axis=1).astype("string")


def prepare_frame(raw: pd.DataFrame, model: str) -> tuple[pd.DataFrame, str]:
    """Create app-only canonical view columns; the underlying artifact is unchanged."""
    frame = raw.copy()
    for field, aliases in FIELD_ALIASES.items():
        source = find_column(frame, aliases)
        if source is None:
            if field in NUMERIC_FIELDS:
                frame[f"__{field}"] = pd.Series(float("nan"), index=frame.index, dtype="float64")
            elif field == "date":
                frame[f"__{field}"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
            else:
                frame[f"__{field}"] = pd.Series(pd.NA, index=frame.index, dtype="string")
        elif field == "date":
            frame[f"__{field}"] = pd.to_datetime(frame[source], errors="coerce")
        elif field in NUMERIC_FIELDS:
            frame[f"__{field}"] = pd.to_numeric(frame[source], errors="coerce")
        else:
            frame[f"__{field}"] = frame[source].astype("string").str.strip()

    score_column = model_score_column(frame, model)
    if score_column:
        frame["__score"] = pd.to_numeric(frame[score_column], errors="coerce")
    else:
        frame["__score"] = pd.Series(float("nan"), index=frame.index)

    cluster_column = model_cluster_column(frame, model)
    frame["__cluster"] = frame[cluster_column] if cluster_column else pd.Series(pd.NA, index=frame.index)

    flag_column = model_flag_column(frame, model)
    if flag_column:
        frame["__flag"] = to_flag(frame[flag_column])
        flag_basis = f"exported flag: {flag_column}"
    elif canonical_model(model) == "DBSCAN" and cluster_column:
        frame["__flag"] = pd.to_numeric(frame[cluster_column], errors="coerce").eq(-1)
        flag_basis = f"DBSCAN noise label (-1): {cluster_column}"
    elif frame["__score"].notna().any():
        threshold = frame["__score"].quantile(0.95)
        frame["__flag"] = frame["__score"].ge(threshold) & frame["__score"].notna()
        flag_basis = "dashboard fallback: top 5% of exported score"
    else:
        frame["__flag"] = False
        flag_basis = "unavailable (no prediction flag or score artifact)"

    frame["__drivers"] = combine_driver_text(frame, driver_columns(frame, model))
    frame["__model"] = canonical_model(model) or model
    return frame, flag_basis


def parse_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    text_value = str(value).strip()
    if not text_value or text_value.lower() in {"nan", "none", "na", "n/a", "—", "-"}:
        return None
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text_value.replace(",", ""))
    if not match:
        return None
    numeric = float(match.group(0))
    return numeric / 100.0 if "%" in text_value else numeric


def metric_value(table: pd.DataFrame, model: str, aliases: Iterable[str]) -> Any | None:
    """Read either wide (Metric/K-Means/DBSCAN) or long metric tables."""
    if table.empty:
        return None
    aliases_token = {token(alias) for alias in aliases}
    metric_column = find_column(table, ("metric", "measure", "metric_name", "name"))

    # Wide benchmark: one row per metric, one column per model.
    if metric_column:
        model_column_wide = next(
            (str(column) for column in table.columns if canonical_model(column) == canonical_model(model)),
            None,
        )
        if model_column_wide:
            mask = table[metric_column].map(token).isin(aliases_token)
            if mask.any():
                return table.loc[mask, model_column_wide].iloc[0]

    # Long metrics: one row per model with metrics in columns.
    model_column = find_column(table, ("model", "algorithm", "model_name", "method"))
    subset = table
    if model_column:
        mask = table[model_column].map(canonical_model).eq(canonical_model(model))
        if mask.any():
            subset = table.loc[mask]
        else:
            subset = table.iloc[0:0]
    if not subset.empty:
        direct_metric = next((str(column) for column in subset.columns if token(column) in aliases_token), None)
        if direct_metric:
            return subset[direct_metric].iloc[0]
        if metric_column:
            value_column = find_column(subset, ("value", "metric_value", "score"))
            if value_column:
                mask = subset[metric_column].map(token).isin(aliases_token)
                if mask.any():
                    return subset.loc[mask, value_column].iloc[0]

    # One-row table with explicit model prefixes, e.g. kmeans_precision.
    model_token = "kmeans" if canonical_model(model) == "K-Means" else "dbscan"
    for column in table.columns:
        name = token(column)
        if model_token in name and any(alias in name for alias in aliases_token):
            return table[column].iloc[0]
    return None


def combined_metric(
    model: str,
    aliases: Iterable[str],
    anomaly_metrics: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> Any | None:
    value = metric_value(anomaly_metrics, model, aliases)
    return value if value is not None else metric_value(benchmark, model, aliases)


def selection_from_json(payload: Any) -> dict[str, str | None]:
    result: dict[str, str | None] = {"segmentation": None, "anomaly": None}

    def walk(value: Any, trail: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                key_token = token(key)
                full = f"{trail}{key_token}"
                if isinstance(child, str):
                    model = canonical_model(child)
                    if model in {"K-Means", "DBSCAN"}:
                        if "anomaly" in full or "detection" in full:
                            result["anomaly"] = model
                        elif "segment" in full or "cluster" in full:
                            result["segmentation"] = model
                walk(child, full)
        elif isinstance(value, list):
            for child in value:
                walk(child, trail)

    walk(payload)
    return result


def selection_from_table(table: pd.DataFrame) -> dict[str, str | None]:
    result: dict[str, str | None] = {"segmentation": None, "anomaly": None}
    if table.empty:
        return result
    model_column = find_column(table, ("model", "algorithm", "model_name", "method"))
    for goal, aliases in {
        "segmentation": ("best_segmentation_model", "segmentation_model"),
        "anomaly": ("best_anomaly_model", "best_anomaly_detection_model", "anomaly_model"),
    }.items():
        direct = find_column(table, aliases)
        if direct:
            # The pipeline's model-selection table stores these as boolean
            # winner markers, while alternative exporters may store the model
            # name directly in the same column.
            if model_column:
                winner_mask = to_flag(table[direct])
                if winner_mask.any():
                    result[goal] = canonical_model(table.loc[winner_mask, model_column].iloc[0])
            if result[goal] is None and table[direct].notna().any():
                candidate = canonical_model(table[direct].dropna().iloc[0])
                if candidate in {"K-Means", "DBSCAN"}:
                    result[goal] = candidate

    objective_column = find_column(table, ("objective", "selection_objective", "use_case", "task", "purpose"))
    winner_column = find_column(table, ("selected_model", "winner", "best_model", "recommended_model", "model"))
    if objective_column and winner_column:
        for _, row in table.iterrows():
            objective = token(row[objective_column])
            model = canonical_model(row[winner_column])
            if model not in {"K-Means", "DBSCAN"}:
                continue
            if "anomaly" in objective or "detection" in objective:
                result["anomaly"] = model
            elif "segment" in objective or "cluster" in objective:
                result["segmentation"] = model

    key_column = find_column(table, ("key", "selection", "recommendation"))
    value_column = find_column(table, ("value", "selected_model", "winner"))
    if key_column and value_column:
        for _, row in table.iterrows():
            key = token(row[key_column])
            model = canonical_model(row[value_column])
            if model not in {"K-Means", "DBSCAN"}:
                continue
            if "anomaly" in key:
                result["anomaly"] = model
            elif "segment" in key or "cluster" in key:
                result["segmentation"] = model
    return result


def merge_selection(*selections: dict[str, str | None]) -> dict[str, str | None]:
    merged: dict[str, str | None] = {"segmentation": None, "anomaly": None}
    for selection in selections:
        for key in merged:
            if selection.get(key):
                merged[key] = selection[key]
    return merged


def compact_number(value: Any, currency: bool = False) -> str:
    numeric = parse_number(value)
    if numeric is None:
        return "—"
    prefix = "$" if currency else ""
    absolute = abs(numeric)
    if absolute >= 1_000_000_000:
        return f"{prefix}{numeric / 1_000_000_000:.2f}B"
    if absolute >= 1_000_000:
        return f"{prefix}{numeric / 1_000_000:.2f}M"
    if absolute >= 1_000:
        return f"{prefix}{numeric / 1_000:.1f}K"
    return f"{prefix}{numeric:,.0f}"


def rate_text(value: Any) -> str:
    numeric = parse_number(value)
    if numeric is None:
        return "—"
    # Exported ``*_pct`` fields are commonly stored on a 0-100 scale and may
    # legitimately exceed 100 (for example, 117% target attainment).
    if abs(numeric) > 1.0:
        numeric /= 100.0
    return f"{numeric:.1%}"


def lift_text(value: Any) -> str:
    numeric = parse_number(value)
    return "—" if numeric is None else f"{numeric:.2f}×"


def mean_or_none(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.mean()) if values.notna().any() else None


def sum_or_none(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.sum()) if values.notna().any() else None


def available_values(frame: pd.DataFrame, field: str) -> list[str]:
    column = f"__{field}"
    if column not in frame or frame[column].isna().all():
        return []
    values = frame[column].dropna().astype(str).str.strip()
    values = values[~values.str.lower().isin({"", "nan", "none", "<na>"})]
    return sorted(values.unique().tolist(), key=str.casefold)


def apply_category_filters(frame: pd.DataFrame, selections: dict[str, list[str]]) -> pd.DataFrame:
    filtered = frame
    for field, selected in selections.items():
        if selected:
            filtered = filtered[filtered[f"__{field}"].astype(str).isin(selected)]
    return filtered


def render_filter_row(
    frame: pd.DataFrame,
    specs: list[tuple[str, str]],
    key_prefix: str,
) -> dict[str, list[str]]:
    columns = st.columns(len(specs))
    selections: dict[str, list[str]] = {}
    for container, (field, label) in zip(columns, specs):
        options = available_values(frame, field)
        with container:
            selections[field] = st.multiselect(
                label,
                options,
                placeholder="All",
                key=f"{key_prefix}_{field}",
                disabled=not options,
            )
    return selections


def section(title: str, caption: str, kicker: str = "ANALYSIS") -> None:
    st.markdown(
        f"<div class='section-kicker'>{html.escape(kicker)}</div>"
        f"<div class='section-title'>{html.escape(title)}</div>"
        f"<div class='section-caption'>{html.escape(caption)}</div>",
        unsafe_allow_html=True,
    )


def render_metrics(items: list[tuple[str, str, str | None]], per_row: int = 5) -> None:
    for start in range(0, len(items), per_row):
        row = items[start : start + per_row]
        columns = st.columns(per_row)
        for container, (label, value, help_text) in zip(columns, row):
            container.metric(label, value, help=help_text)


def missing_data_panel(detail: str) -> None:
    st.info(
        f"{detail}\n\nRun `python run_pipeline.py` from the repository root, then use "
        "**Refresh artifacts** in the sidebar. This dashboard reads outputs only and will not retrain models."
    )


def plot_path(*required_tokens: str) -> Path | None:
    if not PLOTS_DIR.exists():
        return None
    required = [token(value) for value in required_tokens]
    for path in sorted(PLOTS_DIR.glob("*.png")):
        stem = token(path.stem)
        if all(value in stem for value in required):
            return path
    return None


def investigation_table(frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    columns = [
        ("Rep", "__rep"),
        ("Territory", "__territory"),
        ("Product", "__product"),
        ("Manager", "__manager"),
        ("Team", "__team"),
        ("Sales", "__sales"),
        ("Target Attainment", "__attainment"),
        ("Incentive", "__incentive"),
        ("Anomaly Score", "__score"),
        ("Model", "__model"),
        ("Top Anomaly Drivers", "__drivers"),
    ]
    for display, internal in columns:
        output[display] = frame[internal] if internal in frame else pd.NA
    return output.reset_index(drop=True)


def render_table(frame: pd.DataFrame, height: int = 430) -> None:
    st.dataframe(
        frame,
        use_container_width=True,
        hide_index=True,
        height=height,
        column_config={
            "Sales": st.column_config.NumberColumn(format="$%.0f"),
            "Incentive": st.column_config.NumberColumn(format="$%.0f"),
            "Target Attainment": st.column_config.NumberColumn(format="%.1f%%"),
            "Anomaly Score": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.3f"),
            "Anomaly Rate": st.column_config.NumberColumn(format="%.1f%%"),
            "Average Score": st.column_config.NumberColumn(format="%.3f"),
        },
    )


if (ROOT / 'data/dashboard/dashboard_metadata.json').exists() or (METRICS_DIR / 'final_anomaly_model_benchmark.csv').exists():
    workspace = st.sidebar.radio('Dataset workspace', ['Real commercial extension', 'Legacy synthetic demo'])
    if workspace == 'Real commercial extension':
        from field_rep_anomaly.extended_dashboard import render_extended
        render_extended(ROOT, render_table, section)
        st.stop()
    st.warning('Legacy synthetic demonstration: its numbers are NOT comparable with the real-data extension.')

tables: dict[str, pd.DataFrame] = {}
load_errors: list[str] = []
for table_name, table_path in TABLE_PATHS.items():
    tables[table_name], error = read_csv(table_path)
    if error:
        load_errors.append(error)

selection_json_candidates = (
    METRICS_DIR / "model_selection.json",
    ROOT / "artifacts" / "reports" / "model_selection.json",
)
selection_json_path = next(
    (path for path in selection_json_candidates if path.exists()),
    selection_json_candidates[0],
)
selection_json, selection_json_error = read_json(selection_json_path)
if selection_json_error:
    load_errors.append(selection_json_error)

selection = merge_selection(
    selection_from_table(tables["model_selection"]),
    selection_from_json(selection_json) if selection_json is not None else {"segmentation": None, "anomaly": None},
)

scored_raw = tables["scored_observations"]
rep_summary_raw = tables["rep_risk_summary"]
benchmark = tables["clustering_benchmark"]
anomaly_metrics = tables["anomaly_metrics"]
profiles = tables["cluster_profiles"]

base_raw = scored_raw if not scored_raw.empty else rep_summary_raw
best_anomaly_model = selection["anomaly"]
best_segmentation_model = selection["segmentation"]


# Sidebar --------------------------------------------------------------------
st.sidebar.markdown("### ⚕ Review Command Center")
st.sidebar.caption("Field representative incentive anomaly benchmark")

pages = (
    "Executive Overview",
    "Geographic View",
    "Product View",
    "Model Benchmark",
    "Cluster Explorer",
    "Anomaly Investigation",
    "Methodology & Limitations",
)
page = st.sidebar.radio("Workspace", pages, label_visibility="collapsed")

model_options = ["K-Means", "DBSCAN"]
default_model = best_anomaly_model if best_anomaly_model in model_options else "K-Means"
selected_model = st.sidebar.selectbox(
    "Risk scoring model",
    model_options,
    index=model_options.index(default_model),
    help="Changes the exported anomaly score, flag, cluster label, and explanations displayed. It does not run a model.",
)

base_view, flag_basis = prepare_frame(base_raw, selected_model)

if not base_view.empty and base_view["__date"].notna().any():
    minimum_date = base_view["__date"].min().date()
    maximum_date = base_view["__date"].max().date()
    selected_dates = st.sidebar.date_input(
        "Observation period",
        value=(minimum_date, maximum_date),
        min_value=minimum_date,
        max_value=maximum_date,
    )
    if isinstance(selected_dates, (list, tuple)) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
        base_view = base_view[
            base_view["__date"].dt.date.between(start_date, end_date, inclusive="both")
        ]

st.sidebar.markdown("---")
non_selection_paths = [
    path for name, path in TABLE_PATHS.items() if name != "model_selection"
]
artifact_count = sum(path.exists() for path in non_selection_paths) + int(
    TABLE_PATHS["model_selection"].exists() or selection_json_path.exists()
)
if not base_raw.empty:
    st.sidebar.success(f"Ready · {len(base_raw):,} scored rows")
else:
    st.sidebar.warning("Awaiting pipeline outputs")
st.sidebar.caption(f"{artifact_count}/6 core artifact groups present")
st.sidebar.caption(f"Flag basis: {flag_basis}")

if st.sidebar.button("Refresh artifacts", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

with st.sidebar.expander("Artifact status"):
    for name, path in TABLE_PATHS.items():
        if path.exists():
            rows = len(tables[name])
            st.markdown(f"✓ **{name.replace('_', ' ').title()}** · {rows:,} rows")
        else:
            st.markdown(f"○ {name.replace('_', ' ').title()} · pending")
    if selection_json_path.exists():
        st.markdown("✓ **Model Selection JSON**")
    if load_errors:
        st.error("\n".join(load_errors))


status_text = "Artifacts loaded" if not base_raw.empty else "Pipeline outputs pending"
best_text = best_anomaly_model or "selection pending"
st.markdown(
    f"<div class='hero'><div class='hero-kicker'>Pharmaceutical commercial analytics · Decision support</div>"
    f"<div class='hero-title'>{html.escape(page)}</div>"
    f"<div class='hero-subtitle'>Peer-relative field-representative behavior, incentive review, and K-Means vs DBSCAN evidence.</div>"
    f"<div class='status-pill'>{html.escape(status_text)} · Best anomaly model: {html.escape(best_text)}</div></div>",
    unsafe_allow_html=True,
)


def executive_overview() -> None:
    section(
        "Portfolio risk at a glance",
        "A model-aware summary of the currently selected observation period. Flags are review candidates—not findings of fraud or payment error.",
        "EXECUTIVE OVERVIEW",
    )
    if base_view.empty:
        missing_data_panel("No scored observations or representative risk summary is available yet.")
        render_metrics(
            [
                ("Representatives analyzed", "—", None),
                ("Sales", "—", None),
                ("Incentive payout", "—", None),
                ("Flagged anomalies", "—", None),
                ("Anomaly rate", "—", None),
                ("Best model", best_text, None),
                ("Recall", "—", None),
                ("Precision", "—", None),
                ("Lift@5%", "—", None),
            ]
        )
        return

    representative_count = (
        int(base_view["__rep"].nunique(dropna=True))
        if base_view["__rep"].notna().any()
        else len(base_view)
    )
    sales = sum_or_none(base_view["__sales"])
    incentive = sum_or_none(base_view["__incentive"])
    flagged = int(base_view["__flag"].sum())
    anomaly_rate = float(base_view["__flag"].mean()) if len(base_view) else None
    metric_model = best_anomaly_model or selected_model
    recall = combined_metric(metric_model, ("recall",), anomaly_metrics, benchmark)
    precision = combined_metric(metric_model, ("precision",), anomaly_metrics, benchmark)
    lift = combined_metric(
        metric_model,
        ("lift_at_5pct", "lift_at_5_percent", "lift_5", "lift@5%", "lift@5"),
        anomaly_metrics,
        benchmark,
    )

    render_metrics(
        [
            ("Representatives analyzed", f"{representative_count:,}", "Unique representative IDs in the filtered view."),
            ("Sales", compact_number(sales, currency=True), "Sum of exported sales at the analytical grain."),
            ("Incentive payout", compact_number(incentive, currency=True), "Actual incentive paid in the filtered view."),
            ("Flagged anomalies", f"{flagged:,}", f"Based on {flag_basis}."),
            ("Anomaly rate", rate_text(anomaly_rate), "Flagged observations divided by observations analyzed."),
            ("Best model", best_anomaly_model or "Pending", "Weighted anomaly-detection selection, not simply silhouette."),
            ("Recall", rate_text(recall), f"Injected-anomaly recall for {metric_model}."),
            ("Precision", rate_text(precision), f"Injected-anomaly precision for {metric_model}."),
            ("Lift@5%", lift_text(lift), "Concentration of injected anomalies in the highest-ranked 5%."),
        ]
    )

    st.markdown("<br>", unsafe_allow_html=True)
    chart_left, chart_right = st.columns([1.15, 0.85])
    with chart_left:
        section(
            "Anomaly-score distribution",
            f"Continuous exported risk scores for {selected_model}; darker observations are currently flagged.",
            "RISK DISTRIBUTION",
        )
        score_data = base_view[base_view["__score"].notna()].copy()
        if not score_data.empty:
            score_data["Review status"] = score_data["__flag"].map({True: "Flagged", False: "Not flagged"})
            figure = px.histogram(
                score_data,
                x="__score",
                color="Review status",
                nbins=32,
                barmode="overlay",
                color_discrete_map={"Flagged": "#e5484d", "Not flagged": "#3b82f6"},
                labels={"__score": "Anomaly score", "count": "Observations"},
            )
            figure.update_layout(margin=dict(l=5, r=5, t=10, b=5), legend_title_text="")
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("No continuous anomaly score is present in the scored artifact.")

    with chart_right:
        section(
            "Review queue composition",
            "Current flags versus the remainder of the scored population.",
            "OPERATING VIEW",
        )
        composition = pd.DataFrame(
            {
                "Status": ["Flagged", "Not flagged"],
                "Observations": [flagged, max(len(base_view) - flagged, 0)],
            }
        )
        figure = px.pie(
            composition,
            names="Status",
            values="Observations",
            hole=0.68,
            color="Status",
            color_discrete_map={"Flagged": "#e5484d", "Not flagged": "#d9e7f6"},
        )
        figure.update_traces(textposition="outside", textinfo="percent+label")
        figure.update_layout(margin=dict(l=5, r=5, t=10, b=5), showlegend=False)
        st.plotly_chart(figure, use_container_width=True)

    section(
        "Commercial relationship and priority queue",
        "Sales, actual incentive payout, and the highest-ranked observations help reviewers distinguish business scale from unusual behavior.",
        "INVESTIGATION CONTEXT",
    )
    scatter_col, queue_col = st.columns([1.05, 0.95])
    with scatter_col:
        scatter = base_view.dropna(subset=["__sales", "__incentive"]).copy()
        if len(scatter) > 5000:
            scatter = scatter.sample(5000, random_state=42)
        if not scatter.empty:
            scatter["Review status"] = scatter["__flag"].map({True: "Flagged", False: "Not flagged"})
            scatter["Rep"] = scatter["__rep"].fillna("Unknown")
            scatter["Product"] = scatter["__product"].fillna("Unknown")
            figure = px.scatter(
                scatter,
                x="__sales",
                y="__incentive",
                color="Review status",
                hover_name="Rep",
                hover_data={"Product": True, "__score": ":.3f", "__sales": ":,.0f", "__incentive": ":,.0f"},
                color_discrete_map={"Flagged": "#e5484d", "Not flagged": "#2f80ed"},
                labels={"__sales": "Sales", "__incentive": "Actual incentive", "__score": "Anomaly score"},
                opacity=0.68,
            )
            figure.update_layout(margin=dict(l=5, r=5, t=10, b=5), legend_title_text="")
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("Sales and incentive fields are needed for the commercial relationship view.")
    with queue_col:
        ranked = base_view.sort_values("__score", ascending=False, na_position="last").head(10)
        render_table(investigation_table(ranked), height=390)


def geographic_view() -> None:
    section(
        "Where risk concentrates",
        "Cross-filter exported results by geography and organization, then compare anomaly concentration with commercial scale.",
        "GEOGRAPHIC VIEW",
    )
    if base_view.empty:
        missing_data_panel("Geographic analysis requires scored observations.")
        return

    selections = render_filter_row(
        base_view,
        [
            ("country", "Country"),
            ("city", "City"),
            ("territory", "Territory"),
            ("team", "Sales team"),
            ("manager", "Sales manager"),
        ],
        "geo",
    )
    filtered = apply_category_filters(base_view, selections)
    if filtered.empty:
        st.warning("No observations match the selected geographic filters.")
        return

    territory_count = filtered["__territory"].nunique(dropna=True)
    flagged = int(filtered["__flag"].sum())
    render_metrics(
        [
            ("Territories", f"{territory_count:,}", None),
            ("Sales", compact_number(sum_or_none(filtered["__sales"]), currency=True), None),
            ("Incentive payout", compact_number(sum_or_none(filtered["__incentive"]), currency=True), None),
            ("Flagged anomalies", f"{flagged:,}", None),
            ("Anomaly rate", rate_text(filtered["__flag"].mean()), None),
        ]
    )

    available_levels = [
        (label, field)
        for label, field in (("Territory", "territory"), ("City", "city"), ("Country", "country"))
        if filtered[f"__{field}"].notna().any()
    ]
    if not available_levels:
        st.info("Country, city, or territory columns are not available in this artifact.")
        return
    level_label = st.selectbox("Summarize by", [label for label, _ in available_levels], key="geo_level")
    level_field = dict(available_levels)[level_label]

    grouped = (
        filtered.dropna(subset=[f"__{level_field}"])
        .groupby(f"__{level_field}", dropna=False)
        .agg(
            Latitude=("__latitude", "mean"),
            Longitude=("__longitude", "mean"),
            Sales=("__sales", "sum"),
            Incentive=("__incentive", "sum"),
            Observations=("__flag", "size"),
            Flagged=("__flag", "sum"),
            **{"Anomaly Rate": ("__flag", "mean"), "Average Score": ("__score", "mean")},
        )
        .reset_index()
        .rename(columns={f"__{level_field}": level_label})
    )
    grouped["Review volume"] = grouped["Flagged"].fillna(0).astype(float) + 1.0

    map_col, rank_col = st.columns([1.35, 0.65])
    with map_col:
        section(
            "Anomaly concentration map",
            "Marker size reflects flagged review volume; color reflects the anomaly rate.",
            "SPATIAL SIGNAL",
        )
        mapped = grouped.dropna(subset=["Latitude", "Longitude"])
        if not mapped.empty:
            figure = px.scatter_mapbox(
                mapped,
                lat="Latitude",
                lon="Longitude",
                size="Review volume",
                color="Anomaly Rate",
                hover_name=level_label,
                hover_data={
                    "Review volume": False,
                    "Flagged": ":,.0f",
                    "Observations": ":,.0f",
                    "Sales": ":,.0f",
                    "Incentive": ":,.0f",
                    "Anomaly Rate": ":.1%",
                    "Average Score": ":.3f",
                },
                color_continuous_scale="RdYlBu_r",
                size_max=32,
                zoom=1.1,
                height=510,
            )
            figure.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("Latitude and longitude are unavailable; use the ranked geographic view to compare locations.")
    with rank_col:
        section(
            "Highest-rate geographies",
            "Minimum one scored observation; validate volume before escalation.",
            "RANKING",
        )
        rank = grouped.sort_values(["Anomaly Rate", "Flagged"], ascending=False).head(15)
        figure = px.bar(
            rank.sort_values("Anomaly Rate"),
            x="Anomaly Rate",
            y=level_label,
            orientation="h",
            color="Anomaly Rate",
            color_continuous_scale="Blues",
            text_auto=".1%",
        )
        figure.update_layout(margin=dict(l=5, r=5, t=10, b=5), coloraxis_showscale=False, height=510)
        st.plotly_chart(figure, use_container_width=True)

    section(
        "Geographic review table",
        "Sort and compare risk, sales, incentive, and review volume at the selected geographic level.",
        "DETAIL",
    )
    table = grouped.sort_values(["Anomaly Rate", "Average Score"], ascending=False).drop(
        columns=["Review volume"], errors="ignore"
    )
    table["Anomaly Rate"] = table["Anomaly Rate"] * 100.0
    render_table(table, height=390)


def product_view() -> None:
    section(
        "Product performance and review signal",
        "Evaluate sales, target attainment, incentive payout, and anomaly rates across the exported product hierarchy.",
        "PRODUCT VIEW",
    )
    if base_view.empty:
        missing_data_panel("Product analysis requires scored observations.")
        return

    selections = render_filter_row(
        base_view,
        [
            ("product", "Product"),
            ("product_class", "Product class"),
            ("channel", "Channel"),
            ("subchannel", "Sub-channel"),
        ],
        "product",
    )
    filtered = apply_category_filters(base_view, selections)
    if filtered.empty:
        st.warning("No observations match the selected product filters.")
        return

    render_metrics(
        [
            ("Products", f"{filtered['__product'].nunique(dropna=True):,}", None),
            ("Sales", compact_number(sum_or_none(filtered["__sales"]), currency=True), None),
            ("Quantity", compact_number(sum_or_none(filtered["__quantity"])), None),
            ("Target attainment", rate_text(mean_or_none(filtered["__attainment"])), None),
            ("Incentive payout", compact_number(sum_or_none(filtered["__incentive"]), currency=True), None),
            ("Anomaly rate", rate_text(filtered["__flag"].mean()), None),
        ],
        per_row=6,
    )

    dimensions = [
        ("Product", "product"),
        ("Product class", "product_class"),
        ("Channel", "channel"),
        ("Sub-channel", "subchannel"),
    ]
    dimensions = [(label, field) for label, field in dimensions if filtered[f"__{field}"].notna().any()]
    if not dimensions:
        st.info("No product hierarchy fields are present in the scored artifact.")
        return
    dimension_label = st.selectbox("Compare by", [label for label, _ in dimensions], key="product_dimension")
    dimension_field = dict(dimensions)[dimension_label]

    grouped = (
        filtered.dropna(subset=[f"__{dimension_field}"])
        .groupby(f"__{dimension_field}", dropna=False)
        .agg(
            Sales=("__sales", "sum"),
            Quantity=("__quantity", "sum"),
            Incentive=("__incentive", "sum"),
            **{
                "Target Attainment": ("__attainment", "mean"),
                "Observations": ("__flag", "size"),
                "Flagged": ("__flag", "sum"),
                "Anomaly Rate": ("__flag", "mean"),
                "Average Score": ("__score", "mean"),
            },
        )
        .reset_index()
        .rename(columns={f"__{dimension_field}": dimension_label})
    )

    sales_col, relationship_col = st.columns([1.05, 0.95])
    with sales_col:
        section(
            "Commercial scale with risk overlay",
            "Bars show sales; color shows the anomaly rate for the selected comparison level.",
            "PORTFOLIO",
        )
        top = grouped.nlargest(20, "Sales").sort_values("Sales")
        figure = px.bar(
            top,
            x="Sales",
            y=dimension_label,
            orientation="h",
            color="Anomaly Rate",
            color_continuous_scale="RdYlBu_r",
            hover_data={"Incentive": ":,.0f", "Target Attainment": ":.1f", "Flagged": ":,.0f"},
        )
        figure.update_layout(margin=dict(l=5, r=5, t=10, b=5), height=500)
        st.plotly_chart(figure, use_container_width=True)
    with relationship_col:
        section(
            "Sales versus incentive",
            "Bubble size reflects observation volume and color reflects anomaly rate.",
            "RELATIONSHIP",
        )
        relationship = grouped.dropna(subset=["Sales", "Incentive"])
        if not relationship.empty:
            figure = px.scatter(
                relationship,
                x="Sales",
                y="Incentive",
                size="Observations",
                color="Anomaly Rate",
                hover_name=dimension_label,
                hover_data={"Target Attainment": ":.1f", "Flagged": ":,.0f", "Average Score": ":.3f"},
                color_continuous_scale="RdYlBu_r",
                size_max=38,
                height=500,
            )
            figure.update_layout(margin=dict(l=5, r=5, t=10, b=5))
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("Sales and incentive fields are required for this relationship view.")

    section(
        "Product review table",
        "Use the anomaly rate together with volume, attainment, and incentive—not as a standalone conclusion.",
        "DETAIL",
    )
    detail = grouped.sort_values(["Anomaly Rate", "Sales"], ascending=False).copy()
    detail["Anomaly Rate"] *= 100.0
    render_table(detail, height=410)

    comparison_plot = plot_path("product", "anomaly")
    if comparison_plot:
        with st.expander("Open saved product anomaly comparison"):
            st.image(str(comparison_plot), use_column_width=True)


def model_benchmark_view() -> None:
    section(
        "K-Means versus DBSCAN",
        "Segmentation quality and anomaly-detection utility are evaluated separately. The selected model is based on weighted evidence, not a single clustering metric.",
        "MODEL BENCHMARK",
    )

    if benchmark.empty and anomaly_metrics.empty:
        missing_data_panel("The clustering benchmark and anomaly metrics artifacts are not available yet.")
    segmentation_model = best_segmentation_model or "Pending"
    anomaly_model = best_anomaly_model or "Pending"
    seg_col, anomaly_col = st.columns(2)
    with seg_col:
        st.markdown(
            f"<div class='model-card'><h4>Best segmentation model · {html.escape(segmentation_model)}</h4>"
            "<p>Balances separation, cluster balance, stability, runtime, interpretability, and operational usefulness.</p></div>",
            unsafe_allow_html=True,
        )
    with anomaly_col:
        st.markdown(
            f"<div class='model-card'><h4>Best anomaly model · {html.escape(anomaly_model)}</h4>"
            "<p>Balances precision, recall, F2, PR-AUC, Lift@5%, stability, runtime, and reviewer usefulness.</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    metric_aliases = {
        "Precision": ("precision",),
        "Recall": ("recall",),
        "F1": ("f1", "f1_score"),
        "F2": ("f2", "f2_score"),
        "PR-AUC": ("pr_auc", "prauc", "average_precision"),
    }
    chart_rows: list[dict[str, Any]] = []
    for model in ("K-Means", "DBSCAN"):
        for label, aliases in metric_aliases.items():
            numeric = parse_number(combined_metric(model, aliases, anomaly_metrics, benchmark))
            if numeric is None:
                continue
            if numeric > 1.0 and numeric <= 100:
                numeric /= 100.0
            chart_rows.append({"Model": model, "Metric": label, "Value": numeric})
    chart_data = pd.DataFrame(chart_rows)
    if not chart_data.empty:
        section(
            "Detection effectiveness",
            "Metrics use controlled injected anomalies as demo ground truth; production performance may differ.",
            "EVALUATION",
        )
        figure = px.bar(
            chart_data,
            x="Metric",
            y="Value",
            color="Model",
            barmode="group",
            text_auto=".1%",
            color_discrete_map={"K-Means": "#2f80ed", "DBSCAN": "#08a0a0"},
        )
        figure.update_yaxes(tickformat=".0%", range=[0, min(1.05, max(chart_data["Value"].max() * 1.18, 0.25))])
        figure.update_layout(margin=dict(l=5, r=5, t=10, b=5), legend_title_text="")
        st.plotly_chart(figure, use_container_width=True)

    if not benchmark.empty:
        section(
            "Final benchmark table",
            "Full quality, stability, anomaly ranking, runtime, strength, and limitation evidence exported by the pipeline.",
            "COMPARISON",
        )
        render_table(benchmark, height=min(680, max(300, len(benchmark) * 35 + 60)))

    if not anomaly_metrics.empty:
        with st.expander("Detailed anomaly and ranking metrics"):
            render_table(anomaly_metrics, height=min(620, max(260, len(anomaly_metrics) * 35 + 60)))
    if not tables["model_selection"].empty:
        with st.expander("Weighted model-selection evidence"):
            render_table(tables["model_selection"], height=300)
    elif selection_json is not None:
        with st.expander("Weighted model-selection evidence"):
            st.json(selection_json)

    kmeans_col, dbscan_col = st.columns(2)
    with kmeans_col:
        st.markdown(
            "<div class='note-card'><h4>K-Means · centroid behavior</h4>"
            "<p>Useful for stable, business-readable segmentation. Anomaly rank is distance from the assigned centroid. It favors compact clusters and requires a chosen k.</p></div>",
            unsafe_allow_html=True,
        )
    with dbscan_col:
        st.markdown(
            "<div class='note-card'><h4>DBSCAN · local density</h4>"
            "<p>Finds irregular shapes and labels sparse points as noise. It is sensitive to eps, min_samples, scaling, and density variation; fixed inputs and parameters are deterministic.</p></div>",
            unsafe_allow_html=True,
        )

    saved_benchmark = plot_path("benchmark")
    if saved_benchmark:
        with st.expander("Open saved benchmark visualization"):
            st.image(str(saved_benchmark), use_column_width=True)


def derive_profiles_from_scored(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or frame["__cluster"].isna().all():
        return pd.DataFrame()
    return (
        frame.dropna(subset=["__cluster"])
        .groupby("__cluster", dropna=False)
        .agg(
            population=("__cluster", "size"),
            total_sales=("__sales", "sum"),
            average_incentive=("__incentive", "mean"),
            average_target_attainment=("__attainment", "mean"),
            average_activity=("__activity", "mean"),
            anomaly_rate=("__flag", "mean"),
        )
        .reset_index()
        .rename(columns={"__cluster": "cluster"})
    )


def cluster_explorer() -> None:
    section(
        "Business cluster profiles",
        "Explore K-Means segments, their commercial characteristics, business interpretations, and position in two-dimensional PCA space.",
        "CLUSTER EXPLORER",
    )
    kmeans_view, _ = prepare_frame(base_raw, "K-Means")
    profile_data = profiles.copy()
    if profile_data.empty:
        profile_data = derive_profiles_from_scored(kmeans_view)

    pca_one = find_column(kmeans_view, ("pca_1", "pca1", "pc1", "pca_component_1"))
    pca_two = find_column(kmeans_view, ("pca_2", "pca2", "pc2", "pca_component_2"))
    pca_plot = plot_path("pca", "kmeans") or plot_path("pca")

    if profile_data.empty and not pca_plot and not (pca_one and pca_two):
        missing_data_panel("K-Means cluster profiles and PCA artifacts are not available yet.")
        return

    if not profile_data.empty:
        cluster_column = find_column(profile_data, ("cluster", "cluster_id", "kmeans_cluster", "label"))
        if cluster_column:
            cluster_options = profile_data[cluster_column].dropna().astype(str).unique().tolist()
            cluster_options = sorted(cluster_options, key=lambda item: (not item.lstrip("-").isdigit(), item))
            selected_clusters = st.multiselect(
                "Clusters",
                cluster_options,
                default=cluster_options,
                key="cluster_profile_filter",
            )
            shown_profiles = profile_data[profile_data[cluster_column].astype(str).isin(selected_clusters)].copy()
        else:
            shown_profiles = profile_data.copy()

        population_column = find_column(shown_profiles, ("population", "count", "size", "n_observations"))
        anomaly_column = find_column(shown_profiles, ("anomaly_rate", "flag_rate", "injected_anomaly_rate"))
        render_metrics(
            [
                ("Clusters", f"{len(shown_profiles):,}", None),
                (
                    "Profiled population",
                    compact_number(shown_profiles[population_column].sum()) if population_column else "—",
                    None,
                ),
                (
                    "Average anomaly rate",
                    rate_text(mean_or_none(shown_profiles[anomaly_column])) if anomaly_column else "—",
                    None,
                ),
            ],
            per_row=3,
        )

        heatmap_col, profile_col = st.columns([1.05, 0.95])
        with heatmap_col:
            section(
                "Relative profile heatmap",
                "Each metric is standardized across clusters to highlight above- and below-average behavior.",
                "PROFILE SHAPE",
            )
            numeric_columns: list[str] = []
            numeric_values: dict[str, pd.Series] = {}
            for column in shown_profiles.columns:
                if column == cluster_column:
                    continue
                converted = pd.to_numeric(shown_profiles[column], errors="coerce")
                if converted.notna().mean() >= 0.7 and converted.nunique(dropna=True) > 1:
                    numeric_columns.append(str(column))
                    numeric_values[str(column)] = converted
            preferred_tokens = (
                "sales",
                "quantity",
                "customer",
                "activity",
                "attainment",
                "incentive",
                "opportunity",
                "workload",
                "anomalyrate",
            )
            ordered = sorted(
                numeric_columns,
                key=lambda name: (
                    next((index for index, part in enumerate(preferred_tokens) if part in token(name)), 99),
                    name,
                ),
            )[:12]
            if ordered:
                matrix = pd.DataFrame({column: numeric_values[column] for column in ordered}, index=shown_profiles.index)
                standardized = (matrix - matrix.mean()) / matrix.std(ddof=0).replace(0, 1)
                labels = (
                    shown_profiles[cluster_column].astype(str).map(lambda value: f"Cluster {value}").tolist()
                    if cluster_column
                    else [f"Cluster {index}" for index in range(len(shown_profiles))]
                )
                figure = go.Figure(
                    data=go.Heatmap(
                        z=standardized.to_numpy(),
                        x=[str(column).replace("_", " ").title() for column in ordered],
                        y=labels,
                        colorscale="RdBu",
                        zmid=0,
                        colorbar=dict(title="z-score"),
                        hovertemplate="%{y}<br>%{x}: %{z:.2f}<extra></extra>",
                    )
                )
                figure.update_layout(margin=dict(l=5, r=5, t=10, b=5), height=max(330, 52 * len(labels)))
                st.plotly_chart(figure, use_container_width=True)
            else:
                saved_heatmap = plot_path("cluster", "profile", "heatmap")
                if saved_heatmap:
                    st.image(str(saved_heatmap), use_column_width=True)
                else:
                    st.info("The profile artifact does not contain enough varying numeric fields for a heatmap.")
        with profile_col:
            section(
                "Profile statistics",
                "Business-readable cluster statistics exported from actual fitted results.",
                "DETAIL",
            )
            render_table(shown_profiles, height=max(330, min(560, 55 * len(shown_profiles) + 80)))

        interpretation_column = find_column(
            shown_profiles,
            ("business_interpretation", "interpretation", "business_label", "cluster_label", "segment_name"),
        )
        if interpretation_column:
            section(
                "Business interpretation",
                "Labels are derived from observed profile statistics and should be validated with commercial stakeholders.",
                "MEANING",
            )
            card_columns = st.columns(min(3, max(1, len(shown_profiles))))
            for index, (_, row) in enumerate(shown_profiles.iterrows()):
                cluster_name = row[cluster_column] if cluster_column else index
                interpretation = row[interpretation_column]
                with card_columns[index % len(card_columns)]:
                    st.markdown(
                        f"<div class='note-card'><h4>Cluster {html.escape(str(cluster_name))}</h4>"
                        f"<p>{html.escape(str(interpretation))}</p></div><br>",
                        unsafe_allow_html=True,
                    )

    section(
        "PCA cluster landscape",
        "Two components provide a visual diagnostic only; model fitting uses the configured feature space.",
        "PROJECTION",
    )
    if pca_one and pca_two and not kmeans_view.empty:
        pca_data = kmeans_view.dropna(subset=[pca_one, pca_two]).copy()
        if len(pca_data) > 6000:
            pca_data = pca_data.sample(6000, random_state=42)
        if not pca_data.empty:
            pca_data["Cluster"] = pca_data["__cluster"].astype(str)
            pca_data["Rep"] = pca_data["__rep"].fillna("Unknown")
            pca_data["Product"] = pca_data["__product"].fillna("Unknown")
            figure = px.scatter(
                pca_data,
                x=pca_one,
                y=pca_two,
                color="Cluster",
                hover_name="Rep",
                hover_data={"Product": True, "__score": ":.3f"},
                opacity=0.66,
                labels={pca_one: "PCA component 1", pca_two: "PCA component 2", "__score": "Anomaly score"},
            )
            figure.update_layout(margin=dict(l=5, r=5, t=10, b=5), legend_title_text="Cluster")
            st.plotly_chart(figure, use_container_width=True)
    elif pca_plot:
        st.image(str(pca_plot), use_column_width=True)
    else:
        st.info("The pipeline has not exported PCA coordinates or a PCA plot yet.")


def anomaly_investigation() -> None:
    section(
        "Ranked review queue",
        "Prioritize observations by exported model score, retain commercial context, and download the filtered queue for governed investigation.",
        "ANOMALY INVESTIGATION",
    )
    if base_view.empty:
        missing_data_panel("A scored observation or representative risk summary artifact is required for investigation.")
        return

    selections = render_filter_row(
        base_view,
        [
            ("country", "Country"),
            ("territory", "Territory"),
            ("product", "Product"),
            ("team", "Sales team"),
        ],
        "investigation",
    )
    filtered = apply_category_filters(base_view, selections)
    control_one, control_two, control_three = st.columns([0.8, 1.2, 0.8])
    with control_one:
        flagged_only = st.toggle("Flagged only", value=bool(filtered["__flag"].any()), key="flagged_only")
    with control_two:
        valid_scores = filtered["__score"].dropna()
        if not valid_scores.empty and float(valid_scores.min()) < float(valid_scores.max()):
            minimum_score = st.slider(
                "Minimum anomaly score",
                min_value=float(valid_scores.min()),
                max_value=float(valid_scores.max()),
                value=float(valid_scores.min()),
                step=max((float(valid_scores.max()) - float(valid_scores.min())) / 100.0, 0.001),
                format="%.3f",
            )
        else:
            minimum_score = None
            st.caption("Continuous score filter unavailable")
    with control_three:
        maximum_rows = max(1, min(1000, len(filtered)))
        rows_to_show = st.number_input(
            "Rows to show",
            min_value=1,
            max_value=maximum_rows,
            value=min(100, maximum_rows),
            step=25 if maximum_rows >= 25 else 1,
        )

    if flagged_only:
        filtered = filtered[filtered["__flag"]]
    if minimum_score is not None:
        filtered = filtered[filtered["__score"].ge(minimum_score)]
    ranked = filtered.sort_values("__score", ascending=False, na_position="last").head(int(rows_to_show))
    output = investigation_table(ranked)

    render_metrics(
        [
            ("Queue rows", f"{len(output):,}", None),
            ("Unique reps", f"{ranked['__rep'].nunique(dropna=True):,}", None),
            ("Flagged in queue", f"{int(ranked['__flag'].sum()):,}", None),
            ("Average score", f"{mean_or_none(ranked['__score']):.3f}" if ranked["__score"].notna().any() else "—", None),
        ],
        per_row=4,
    )

    export_col, note_col = st.columns([0.27, 0.73])
    with export_col:
        st.download_button(
            "Download review queue (CSV)",
            data=output.to_csv(index=False).encode("utf-8"),
            file_name=f"anomaly_review_queue_{token(selected_model)}.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=output.empty,
        )
    with note_col:
        st.caption(
            "Export contains the filtered, ranked dashboard view. Investigation outcomes should be recorded in a governed case-management process."
        )

    if output.empty:
        st.warning("No observations match the current review controls.")
        return
    render_table(output, height=520)

    section(
        "Observation drill-down",
        "Inspect the exported model explanation and core commercial measures for one ranked observation.",
        "EXPLAINABILITY",
    )
    choices = [f"#{index + 1} · {row['Rep']} · {row['Product']}" for index, (_, row) in enumerate(output.iterrows())]
    selected_choice = st.selectbox("Ranked observation", choices, key="drilldown_observation")
    selected_index = choices.index(selected_choice)
    selected_row = output.iloc[selected_index]
    detail_left, detail_right = st.columns([0.38, 0.62])
    with detail_left:
        st.metric("Anomaly score", f"{parse_number(selected_row['Anomaly Score']):.3f}" if parse_number(selected_row["Anomaly Score"]) is not None else "—")
        st.metric("Sales", compact_number(selected_row["Sales"], currency=True))
        st.metric("Incentive", compact_number(selected_row["Incentive"], currency=True))
    with detail_right:
        st.markdown("#### Primary exported drivers")
        driver_text = str(selected_row["Top Anomaly Drivers"])
        if driver_text == "Not available in exported artifacts":
            st.info("Feature-level driver details were not present in the loaded artifact. No explanation has been fabricated by the dashboard.")
        else:
            st.write(driver_text)
        st.caption(
            f"Rep: {selected_row['Rep']} · Territory: {selected_row['Territory']} · Manager: {selected_row['Manager']} · Team: {selected_row['Team']}"
        )


def methodology_view() -> None:
    section(
        "How to interpret this prototype",
        "The dashboard is the presentation layer for a reproducible benchmark. It reads persisted outputs and performs no training or parameter tuning.",
        "METHODOLOGY",
    )
    method_tab, metric_tab, limitation_tab, artifact_tab = st.tabs(
        ["Analytical design", "Models & metrics", "Limitations", "Artifact inventory"]
    )
    with method_tab:
        st.markdown(
            """
            #### Data foundation

            The pipeline first discovers and audits a provided pharmaceutical commercial CSV. If no qualifying source is available, it creates a clearly identified fallback transaction dataset. Original sales data remains the commercial foundation; provenance records distinguish source, derived, defaulted, and synthetic fields.

            #### Analytical grain and enrichment

            Transactions are mapped deterministically to synthetic field representatives and aggregated at approximately **Representative × Product × Territory × Month**. Synthetic activity, target, incentive, capacity, and opportunity fields are generated with business relationships to observed sales, customers, geography, and organization—not as independent noise.

            #### Feature engineering and leakage controls

            Features cover sales dynamics, customer coverage, product concentration, activity efficiency, incentive alignment, peer-relative deviation, territory opportunity, workload, and rep capacity. Injected labels and anomaly types are evaluation labels only and are excluded from model features. Missing-value handling, configured clipping, and scaling are fitted inside the modeling workflow and persisted.

            #### Controlled evaluation

            Multiple anomaly patterns are injected at variable severity so ranking and classification performance can be measured. These labels create useful demo ground truth; they do not turn unsupervised clustering into a production fraud classifier.
            """
        )
    with metric_tab:
        st.markdown(
            """
            #### K-Means

            Candidate values of k are compared using inertia, silhouette, Davies–Bouldin, Calinski–Harabasz, cluster balance, stability, runtime, and interpretability. Anomaly score is normalized distance from the assigned centroid; feature contributions are based on squared standardized distance.

            #### DBSCAN

            Candidate `eps` and `min_samples` configurations are assessed after inspecting k-nearest-neighbor distance. Meaningless one-cluster, all-noise, and excessive-noise solutions are rejected. Label `-1` is the primary noise candidate; the continuous score captures local sparsity. DBSCAN is deterministic for fixed data and parameters.

            #### Benchmark and selection

            Detection metrics include precision, recall, F1, F2, specificity, balanced accuracy, ROC-AUC, PR-AUC, and confusion counts. Ranking metrics include Precision/Recall/Lift at 1%, 5%, and 10%, plus top-decile capture. Weighted selection produces separate recommendations for **segmentation** and **anomaly detection** because the best model can differ by use case.
            """
        )
    with limitation_tab:
        st.warning("An anomaly is a prompt for review—not evidence of fraud, misconduct, or an incorrect incentive payment.")
        st.markdown(
            """
            - Synthetic representative mapping and enrichment support a hackathon demonstration, not causal inference about real employees.
            - Performance against injected anomalies may overstate production performance and depends on how closely injected patterns resemble real review cases.
            - Scores are peer-relative and sensitive to feature definitions, scaling, seasonality, sparse peer groups, and data quality.
            - K-Means favors compact groups; DBSCAN is sensitive to density variation and its `eps`/`min_samples` configuration.
            - A threshold should reflect investigation capacity, false-positive cost, governance policy, and temporal validation—not just contamination.
            - Geographic concentration can reflect market structure or missingness. Small populations require additional care.
            - Incentive and activity fields must be reconciled with authoritative source systems before any operational decision.
            - Clustering alone should **not** decide sales-force hiring. Future planning should combine forecasting, capacity modeling, optimization, scenario simulation, and geographic analysis.
            """
        )
    with artifact_tab:
        rows: list[dict[str, Any]] = []
        for name, path in TABLE_PATHS.items():
            rows.append(
                {
                    "Artifact": name.replace("_", " ").title(),
                    "Status": "Available" if path.exists() else "Pending",
                    "Rows": len(tables[name]) if path.exists() else None,
                    "Last modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if path.exists() else None,
                    "Path": str(path.relative_to(ROOT)),
                }
            )
        rows.append(
            {
                "Artifact": "Model Selection JSON",
                "Status": "Available" if selection_json_path.exists() else "Pending",
                "Rows": None,
                "Last modified": datetime.fromtimestamp(selection_json_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if selection_json_path.exists() else None,
                "Path": str(selection_json_path.relative_to(ROOT)),
            }
        )
        render_table(pd.DataFrame(rows), height=370)
        if load_errors:
            st.error("Artifacts with read errors:\n\n" + "\n\n".join(load_errors))
        st.caption("Expected plot assets are read from artifacts/plots/*.png when an interactive equivalent is unavailable.")


if page == "Executive Overview":
    executive_overview()
elif page == "Geographic View":
    geographic_view()
elif page == "Product View":
    product_view()
elif page == "Model Benchmark":
    model_benchmark_view()
elif page == "Cluster Explorer":
    cluster_explorer()
elif page == "Anomaly Investigation":
    anomaly_investigation()
else:
    methodology_view()


st.markdown("---")
st.caption(
    "Decision-support prototype · Field Representative Incentive Anomaly Detection · "
    "Results must be reviewed with commercial, data-governance, HR, legal, and compliance stakeholders."
)
