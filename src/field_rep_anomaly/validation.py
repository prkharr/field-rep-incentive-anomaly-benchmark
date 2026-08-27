"""Data-quality profiling and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def build_data_quality_report(data: pd.DataFrame, source_metadata: Mapping[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return a machine-readable quality summary and one row per column."""
    column_rows: list[dict[str, Any]] = []
    for column in data.columns:
        series = data[column]
        row: dict[str, Any] = {
            "column": str(column),
            "dtype": str(series.dtype),
            "non_null_count": int(series.notna().sum()),
            "missing_count": int(series.isna().sum()),
            "missing_pct": float(series.isna().mean() * 100.0),
            "cardinality": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if not clean.empty:
                row.update(
                    {
                        "min": float(clean.min()),
                        "p25": float(clean.quantile(0.25)),
                        "median": float(clean.median()),
                        "mean": float(clean.mean()),
                        "p75": float(clean.quantile(0.75)),
                        "max": float(clean.max()),
                        "std": float(clean.std(ddof=1)) if len(clean) > 1 else 0.0,
                    }
                )
        else:
            top = series.dropna().astype(str).value_counts()
            if not top.empty:
                row["most_common_value"] = top.index[0]
                row["most_common_count"] = int(top.iloc[0])
        column_rows.append(row)
    column_profile = pd.DataFrame(column_rows)

    dates = pd.to_datetime(data.get("date"), errors="coerce") if "date" in data else pd.Series(dtype="datetime64[ns]")
    duplicate_count = int(data.duplicated().sum())
    coverage: dict[str, Any] = {}
    for field in ("country", "city", "product_name", "product_class", "sales_manager", "sales_team", "channel", "subchannel"):
        if field in data:
            values = data[field].dropna().astype(str)
            coverage[field] = {
                "count": int(values.nunique()),
                "examples": sorted(values.unique().tolist())[:20],
            }
    report = {
        "source": {
            "source_type": source_metadata.get("source_type"),
            "source_path": source_metadata.get("source_path"),
            "fallback_used": bool(source_metadata.get("fallback_used")),
            "fallback_reason": source_metadata.get("fallback_reason"),
        },
        "shape": {"rows": int(data.shape[0]), "columns": int(data.shape[1])},
        "exact_duplicate_rows": duplicate_count,
        "exact_duplicate_pct": float(100.0 * duplicate_count / max(len(data), 1)),
        "total_missing_cells": int(data.isna().sum().sum()),
        "date_coverage": {
            "start": str(dates.min().date()) if not dates.dropna().empty else None,
            "end": str(dates.max().date()) if not dates.dropna().empty else None,
            "distinct_months": int(dates.dt.to_period("M").nunique()) if not dates.dropna().empty else 0,
            "invalid_dates": int(dates.isna().sum()) if len(dates) else 0,
        },
        "coverage": coverage,
        "schema_audit": source_metadata.get("schema_audit", {}),
        "discovery_audit": source_metadata.get("discovery_audit", []),
    }
    return report, column_profile


def validate_canonical_data(data: pd.DataFrame) -> list[str]:
    """Validate model-critical fields; return non-fatal warnings."""
    required = {"customer", "product_name", "sales", "date", "city", "country"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Canonical dataset is missing required columns: {', '.join(missing)}")
    if len(data) < 100:
        raise ValueError("At least 100 transaction rows are required for a defensible benchmark.")
    warnings: list[str] = []
    if data["sales"].isna().mean() > 0.20:
        raise ValueError("More than 20% of sales values are missing.")
    if (pd.to_numeric(data["sales"], errors="coerce") < 0).any():
        warnings.append("Negative sales values were retained and will be handled by robust preprocessing.")
    if data.duplicated().any():
        warnings.append("Exact duplicate source rows were detected and retained for auditability.")
    if data["date"].isna().any():
        warnings.append("Rows with invalid dates will be excluded from monthly aggregation.")
    return warnings


def write_data_quality_report(
    report: Mapping[str, Any], column_profile: pd.DataFrame, reports_dir: Path, metrics_dir: Path
) -> None:
    """Persist JSON, Markdown, and column-level CSV quality evidence."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "data_quality_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    column_profile.to_csv(metrics_dir / "data_quality_columns.csv", index=False)
    source = report["source"]
    fallback_banner = (
        "**Fallback used:** Yes. No qualifying original pharma CSV was available; all downstream results are demo-only."
        if source["fallback_used"]
        else "**Fallback used:** No. A provided CSV is the commercial-data foundation."
    )
    coverage_lines = []
    for field, details in report["coverage"].items():
        examples = ", ".join(details["examples"][:8])
        coverage_lines.append(f"- `{field}`: {details['count']} distinct ({examples})")
    missing_top = column_profile.sort_values("missing_pct", ascending=False).head(10)
    missing_lines = [f"- `{row.column}`: {row.missing_count} ({row.missing_pct:.2f}%)" for row in missing_top.itertuples()]
    markdown = f"""# Data-quality report

{fallback_banner}

## Provenance

- Source type: `{source['source_type']}`
- Source path: `{source['source_path']}`
- Reason: {source.get('fallback_reason') or 'Not applicable'}

## Structure

- Shape: **{report['shape']['rows']:,} rows × {report['shape']['columns']:,} columns**
- Exact duplicates: **{report['exact_duplicate_rows']:,} ({report['exact_duplicate_pct']:.2f}%)**
- Total missing cells: **{report['total_missing_cells']:,}**
- Date coverage: **{report['date_coverage']['start']} to {report['date_coverage']['end']}** ({report['date_coverage']['distinct_months']} months)
- Invalid dates: **{report['date_coverage']['invalid_dates']:,}**

## Highest missingness

{chr(10).join(missing_lines)}

## Coverage

{chr(10).join(coverage_lines)}

## Field lineage

Original source columns and their canonical mappings are recorded in `data_quality_report.json`. Synthetic enrichment and anomaly-label fields are added only after this source-level profile.
"""
    (reports_dir / "data_quality_report.md").write_text(markdown, encoding="utf-8")
