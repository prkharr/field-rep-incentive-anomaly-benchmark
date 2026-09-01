"""Source profiling, normalization, stable identifiers, and lineage."""

from __future__ import annotations

import calendar
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SOURCE_ALIASES = {
    "Distributor": "distributor",
    "Customer Name": "customer_name",
    "City": "city",
    "Country": "country",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Channel": "channel",
    "Sub-channel": "sub_channel",
    "Product Name": "product_name",
    "Product Class": "product_class",
    "Quantity": "quantity",
    "Price": "price",
    "Sales": "sales",
    "Month": "month",
    "Year": "year",
    "Name of Sales Rep": "rep_name",
    "Manager": "manager_name",
    "Sales Team": "team_name",
}

REQUIRED_SOURCE_COLUMNS = set(SOURCE_ALIASES)
REQUIRED_MODEL_FIELDS = [
    "period",
    "rep_name",
    "manager_name",
    "team_name",
    "customer_name",
    "product_name",
    "product_class",
    "quantity",
    "price",
    "sales",
]


def snake_case(value: str) -> str:
    """Return a stable snake-case field name."""
    text = re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip())
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    return re.sub(r"_+", "_", text).strip("_").lower()


def stable_id(prefix: str, *values: Any, length: int = 12) -> str:
    """Build a deterministic surrogate identifier from normalized values."""
    payload = "|".join("" if pd.isna(v) else str(v).strip().casefold() for v in values)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:length]}"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _monthly_date(year: pd.Series, month: pd.Series) -> pd.Series:
    names = {name.casefold(): number for number, name in enumerate(calendar.month_name) if name}
    names.update({name.casefold(): number for number, name in enumerate(calendar.month_abbr) if name})
    numeric = pd.to_numeric(month, errors="coerce")
    numeric = numeric.fillna(month.astype(str).str.strip().str.casefold().map(names))
    return pd.to_datetime(
        {"year": pd.to_numeric(year, errors="coerce"), "month": numeric, "day": 1},
        errors="coerce",
    )


def _stable_day(source_row_id: pd.Series, low: int, high: int, seed: int) -> pd.Series:
    span = max(high - low + 1, 1)
    values = source_row_id.map(
        lambda value: int(hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()[:12], 16)
    )
    return low + values.mod(span)


def _mapping(frame: pd.DataFrame, id_column: str, name_column: str) -> pd.DataFrame:
    return (
        frame[[id_column, name_column]]
        .drop_duplicates()
        .sort_values([name_column, id_column], kind="mergesort")
        .reset_index(drop=True)
        .assign(data_lineage="source_observed")
    )


def profile_and_normalize_source(
    input_path: str | Path,
    seed: int = 42,
    currency_code: str = "UNK",
    generated_day_min: int = 1,
    generated_day_max: int = 27,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame], pd.DataFrame]:
    """Profile the actual CSV and return a leakage-neutral normalized modeling copy.

    The source file is read only. Exact duplicates and incomplete required rows are
    described in the profile and excluded only from the returned modeling copy.
    """
    path = Path(input_path)
    raw = pd.read_csv(path)
    missing_columns = REQUIRED_SOURCE_COLUMNS - set(raw.columns)
    if missing_columns:
        raise ValueError(f"Source is missing required columns: {sorted(missing_columns)}")

    renamed = raw.rename(columns={column: snake_case(column) for column in raw.columns})
    renamed = renamed.rename(
        columns={snake_case(source): target for source, target in SOURCE_ALIASES.items()}
    ).copy()
    renamed["source_file_row_number"] = np.arange(2, len(renamed) + 2, dtype=np.int64)
    renamed["source_row_id"] = [
        stable_id("SRC", path.name, row_number, length=16)
        for row_number in renamed["source_file_row_number"]
    ]

    raw_duplicate_mask = raw.duplicated(keep="first")
    renamed["period"] = _monthly_date(renamed["year"], renamed["month"])
    for column in ["quantity", "price", "sales", "latitude", "longitude"]:
        renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
    invalid_mask = renamed[REQUIRED_MODEL_FIELDS].isna().any(axis=1)
    invalid_mask |= ~np.isfinite(renamed[["quantity", "price", "sales"]]).all(axis=1)

    normalized = renamed.loc[~raw_duplicate_mask & ~invalid_mask].copy()
    text_columns = [
        "distributor",
        "customer_name",
        "city",
        "country",
        "channel",
        "sub_channel",
        "product_name",
        "product_class",
        "rep_name",
        "manager_name",
        "team_name",
    ]
    for column in text_columns:
        normalized[column] = normalized[column].astype(str).str.strip()

    day = _stable_day(
        normalized["source_row_id"], generated_day_min, generated_day_max, seed
    )
    normalized["transaction_date"] = normalized["period"] + pd.to_timedelta(day - 1, unit="D")
    normalized["rep_id"] = normalized["rep_name"].map(lambda value: stable_id("REP", value))
    normalized["manager_id"] = normalized["manager_name"].map(
        lambda value: stable_id("MGR", value)
    )
    normalized["team_id"] = normalized["team_name"].map(lambda value: stable_id("TEAM", value))
    normalized["customer_id"] = normalized["customer_name"].map(
        lambda value: stable_id("CUST", value)
    )
    normalized["product_id"] = normalized["product_name"].map(
        lambda value: stable_id("PROD", value)
    )
    normalized["territory_name"] = normalized["team_name"] + " / " + normalized["country"]
    normalized["territory_id"] = normalized["territory_name"].map(
        lambda value: stable_id("TERR", value)
    )
    normalized["transaction_id"] = normalized["source_row_id"].map(
        lambda value: stable_id("TXN", value, length=16)
    )
    normalized["currency_code"] = str(currency_code)
    normalized["data_lineage"] = "source_observed"

    canonical_order = [
        "source_row_id",
        "source_file_row_number",
        "transaction_id",
        "transaction_date",
        "period",
        "rep_id",
        "rep_name",
        "manager_id",
        "manager_name",
        "team_id",
        "team_name",
        "territory_id",
        "territory_name",
        "customer_id",
        "customer_name",
        "product_id",
        "product_name",
        "product_class",
        "distributor",
        "city",
        "country",
        "latitude",
        "longitude",
        "channel",
        "sub_channel",
        "quantity",
        "price",
        "sales",
        "currency_code",
        "data_lineage",
    ]
    normalized = normalized[canonical_order].sort_values(
        ["period", "source_file_row_number"], kind="mergesort"
    ).reset_index(drop=True)

    date_min = normalized["period"].min()
    date_max = normalized["period"].max()
    duplicate_count = int(raw_duplicate_mask.sum())
    invalid_count = int(invalid_mask[~raw_duplicate_mask].sum())
    dimensions_expected = [254082, 18]
    profile: dict[str, Any] = {
        "source_file": path.name,
        "source_sha256": file_sha256(path),
        "input_rows": int(len(raw)),
        "input_columns": int(len(raw.columns)),
        "expected_brief_rows": dimensions_expected[0],
        "expected_brief_columns": dimensions_expected[1],
        "matches_expected_dimensions": bool(list(raw.shape) == dimensions_expected),
        "modeling_rows": int(len(normalized)),
        "date_min": str(date_min.date()),
        "date_max": str(date_max.date()),
        "representatives": int(normalized["rep_id"].nunique()),
        "managers": int(normalized["manager_id"].nunique()),
        "teams": int(normalized["team_id"].nunique()),
        "customers": int(normalized["customer_id"].nunique()),
        "products": int(normalized["product_id"].nunique()),
        "territories": int(normalized["territory_id"].nunique()),
        "geographies": int(normalized[["city", "country"]].drop_duplicates().shape[0]),
        "source_cardinalities": {
            "representatives": int(raw["Name of Sales Rep"].nunique(dropna=True)),
            "managers": int(raw["Manager"].nunique(dropna=True)),
            "teams": int(raw["Sales Team"].nunique(dropna=True)),
            "customers": int(raw["Customer Name"].nunique(dropna=True)),
            "products": int(raw["Product Name"].nunique(dropna=True)),
            "product_classes": int(raw["Product Class"].nunique(dropna=True)),
            "countries": int(raw["Country"].nunique(dropna=True)),
            "cities": int(raw["City"].nunique(dropna=True)),
        },
        "exact_duplicate_rows": duplicate_count,
        "exact_duplicate_rate": float(duplicate_count / max(len(raw), 1)),
        "incomplete_rows_excluded": invalid_count,
        "missing_value_rates": {
            str(column): float(value)
            for column, value in raw.isna().mean().sort_index().items()
        },
        "monetary_unit_assumption": (
            f"Source has no reliable currency field. Numeric monetary values are preserved; "
            f"generated datasets use currency_code={currency_code}. No currency conversion performed."
        ),
        "transaction_date_method": (
            f"Deterministic synthetic day {generated_day_min}-{generated_day_max} within source month; seed={seed}."
        ),
        "source_sales_total": float(normalized["sales"].sum()),
        "nonpositive_source_sales_rows": int(normalized["sales"].le(0).sum()),
    }

    mappings = {
        "rep_mapping": _mapping(normalized, "rep_id", "rep_name"),
        "manager_mapping": _mapping(normalized, "manager_id", "manager_name"),
        "team_mapping": _mapping(normalized, "team_id", "team_name"),
        "customer_mapping": _mapping(normalized, "customer_id", "customer_name"),
        "product_mapping": _mapping(normalized, "product_id", "product_name"),
        "territory_mapping": _mapping(normalized, "territory_id", "territory_name"),
    }
    quality_rows = [
        {
            "check_name": "input_dimensions",
            "status": "pass",
            "value": f"{len(raw)} x {len(raw.columns)}",
            "detail": "Actual source dimensions; no shape was hardcoded.",
        },
        {
            "check_name": "expected_dimensions_match",
            "status": "pass" if profile["matches_expected_dimensions"] else "warning",
            "value": str(profile["matches_expected_dimensions"]),
            "detail": f"Brief expected approximately {dimensions_expected[0]} x {dimensions_expected[1]}.",
        },
        {
            "check_name": "exact_duplicates",
            "status": "warning" if duplicate_count else "pass",
            "value": duplicate_count,
            "detail": "Duplicates remain in the unchanged source and are excluded from the modeling copy.",
        },
        {
            "check_name": "incomplete_required_rows",
            "status": "warning" if invalid_count else "pass",
            "value": invalid_count,
            "detail": "Incomplete required rows remain in source and are excluded from modeling.",
        },
        {
            "check_name": "source_file_sha256",
            "status": "pass",
            "value": profile["source_sha256"],
            "detail": path.name,
        },
    ]
    for column, rate in profile["missing_value_rates"].items():
        quality_rows.append(
            {
                "check_name": f"missing_rate__{snake_case(column)}",
                "status": "warning" if rate > 0 else "pass",
                "value": rate,
                "detail": "Rate in unchanged source file.",
            }
        )
    return normalized, profile, mappings, pd.DataFrame(quality_rows)


def profile_json(profile: dict[str, Any]) -> str:
    return json.dumps(profile, indent=2, sort_keys=True, default=str)
