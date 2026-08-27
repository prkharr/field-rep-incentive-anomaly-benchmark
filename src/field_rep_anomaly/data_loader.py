"""Dataset discovery, schema adaptation, and explicit fallback generation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .config import resolve_path


COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "distributor": ("distributor", "distributor_name", "supplier"),
    "customer": ("customer", "customer_name", "account", "account_name", "hcp"),
    "city": ("city", "customer_city", "town"),
    "country": ("country", "customer_country", "nation"),
    "latitude": ("latitude", "lat"),
    "longitude": ("longitude", "lon", "lng", "long"),
    "channel": ("channel", "sales_channel"),
    "subchannel": ("subchannel", "sub_channel", "sub-channel", "sales_subchannel"),
    "product_name": ("product_name", "product", "brand", "brand_name", "sku"),
    "product_class": ("product_class", "class", "therapy_area", "therapeutic_class"),
    "quantity": ("quantity", "qty", "units", "volume"),
    "price": ("price", "unit_price", "selling_price", "asp"),
    "sales": ("sales", "revenue", "net_sales", "sales_value", "amount"),
    "month": ("month", "sales_month", "period", "date", "transaction_date"),
    "year": ("year", "sales_year"),
    "sales_manager": ("sales_manager", "manager", "manager_name"),
    "sales_team": ("sales_team", "team", "team_name"),
}


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def stable_int(value: str, seed: int = 42) -> int:
    digest = hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def canonicalise_columns(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Map common source column variants to canonical names without guessing by position."""
    normalised = {_normalise_name(column): column for column in data.columns}
    rename: dict[str, str] = {}
    mapping: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            source = normalised.get(_normalise_name(alias))
            if source is not None and canonical not in mapping:
                rename[source] = canonical
                mapping[canonical] = source
                break
    result = data.rename(columns=rename).copy()
    return result, mapping


def _candidate_score(path: Path) -> tuple[int, int]:
    try:
        columns = pd.read_csv(path, nrows=0).columns
    except Exception:
        return (0, 0)
    normalised = {_normalise_name(column) for column in columns}
    matched = sum(
        any(_normalise_name(alias) in normalised for alias in aliases)
        for aliases in COLUMN_ALIASES.values()
    )
    core = sum(
        any(_normalise_name(alias) in normalised for alias in COLUMN_ALIASES[name])
        for name in ("customer", "product_name", "sales", "quantity", "month")
    )
    return (core, matched)


def discover_pharma_csv(
    config: Mapping[str, Any], explicit_path: str | Path | None = None
) -> tuple[Path | None, list[dict[str, Any]]]:
    """Search configured workspace roots for a plausible source CSV.

    Generated pipeline outputs are excluded so repeat runs cannot silently ingest their
    own artifacts. Candidate selection is based on header matches, not file names alone.
    """
    audit: list[dict[str, Any]] = []
    if explicit_path:
        explicit = Path(explicit_path).expanduser().resolve()
        if not explicit.exists():
            raise FileNotFoundError(f"Explicit input does not exist: {explicit}")
        return explicit, [{"path": str(explicit), "source": "explicit", "selected": True}]

    repo_root = Path(str(config["_repo_root"])).resolve()
    roots: list[Path] = [resolve_path(config, config["paths"]["raw_dir"])]
    roots.extend(resolve_path(config, value) for value in config["paths"].get("search_roots", []))
    excluded = {
        (repo_root / "data" / "synthetic").resolve(),
        (repo_root / "data" / "processed").resolve(),
        (repo_root / "artifacts").resolve(),
        (repo_root / ".venv").resolve(),
    }
    candidates: list[tuple[tuple[int, int], int, Path]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            audit.append({"root": str(root), "status": "missing"})
            continue
        for path in root.rglob("*.csv"):
            resolved = path.resolve()
            if resolved in seen or any(parent == resolved or parent in resolved.parents for parent in excluded):
                continue
            seen.add(resolved)
            score = _candidate_score(resolved)
            size = resolved.stat().st_size
            is_plausible = score[0] >= 3 and score[1] >= 7
            audit.append(
                {
                    "path": str(resolved),
                    "core_header_matches": score[0],
                    "total_header_matches": score[1],
                    "size_bytes": size,
                    "plausible": is_plausible,
                }
            )
            if is_plausible:
                candidates.append((score, size, resolved))
    if not candidates:
        return None, audit
    candidates.sort(key=lambda item: (item[0][0], item[0][1], item[1], str(item[2])), reverse=True)
    selected = candidates[0][2]
    for entry in audit:
        if entry.get("path") == str(selected):
            entry["selected"] = True
    return selected, audit


def generate_fallback_transactions(
    rows: int = 9000, start: str = "2024-01-01", months: int = 18, seed: int = 42
) -> pd.DataFrame:
    """Generate a coherent transaction-level pharma dataset used only as an explicit fallback."""
    rng = np.random.default_rng(seed)
    locations = pd.DataFrame(
        [
            ("Mumbai", "India", 19.0760, 72.8777, 1.35),
            ("Delhi", "India", 28.6139, 77.2090, 1.28),
            ("Bengaluru", "India", 12.9716, 77.5946, 1.22),
            ("London", "United Kingdom", 51.5072, -0.1276, 1.25),
            ("Manchester", "United Kingdom", 53.4808, -2.2426, 0.92),
            ("Birmingham", "United Kingdom", 52.4862, -1.8904, 0.88),
            ("New York", "United States", 40.7128, -74.0060, 1.42),
            ("Chicago", "United States", 41.8781, -87.6298, 1.08),
            ("Houston", "United States", 29.7604, -95.3698, 1.02),
            ("Sao Paulo", "Brazil", -23.5505, -46.6333, 1.18),
            ("Rio de Janeiro", "Brazil", -22.9068, -43.1729, 0.98),
            ("Brasilia", "Brazil", -15.7939, -47.8828, 0.86),
        ],
        columns=["City", "Country", "Latitude", "Longitude", "opportunity"],
    )
    products = pd.DataFrame(
        [
            ("Cardiovex", "Cardiovascular", 82.0, 1.15),
            ("Glycoban", "Diabetes", 64.0, 1.20),
            ("Respira", "Respiratory", 48.0, 0.95),
            ("Oncora", "Oncology", 210.0, 0.72),
            ("Neurocalm", "Neurology", 96.0, 0.90),
            ("Immunara", "Immunology", 155.0, 0.78),
            ("Dermasol", "Dermatology", 38.0, 1.05),
            ("Renapro", "Renal", 118.0, 0.82),
        ],
        columns=["Product Name", "Product Class", "base_price", "demand"],
    )
    location_weights = locations["opportunity"].to_numpy()
    location_weights = location_weights / location_weights.sum()
    product_weights = products["demand"].to_numpy()
    product_weights = product_weights / product_weights.sum()
    location_idx = rng.choice(len(locations), size=rows, p=location_weights)
    product_idx = rng.choice(len(products), size=rows, p=product_weights)
    month_dates = pd.date_range(start=start, periods=months, freq="MS")
    month_idx = rng.choice(len(month_dates), size=rows)

    location = locations.iloc[location_idx].reset_index(drop=True)
    product = products.iloc[product_idx].reset_index(drop=True)
    dates = pd.Series(month_dates[month_idx])
    seasonality = 1.0 + 0.12 * np.sin(2 * np.pi * dates.dt.month.to_numpy() / 12.0)
    trend = 1.0 + 0.012 * month_idx
    demand = location["opportunity"].to_numpy() * product["demand"].to_numpy() * seasonality * trend
    quantity = np.maximum(1, rng.poisson(7.5 * demand)).astype(int)
    price = product["base_price"].to_numpy() * rng.lognormal(mean=0.0, sigma=0.075, size=rows)
    sales = quantity * price * rng.normal(1.0, 0.025, size=rows)

    city_codes = pd.Categorical(location["City"], categories=locations["City"]).codes
    customer_num = rng.integers(1, 46, size=rows)
    customer = [f"CUST_{city:02d}_{number:03d}" for city, number in zip(city_codes, customer_num)]
    team_names = np.array(["Team_Apex", "Team_Bridge", "Team_Catalyst", "Team_Delta"])
    team = team_names[city_codes % len(team_names)]
    manager = np.array([f"MGR_{(code % 8) + 1:02d}" for code in city_codes])
    channel = rng.choice(["Hospital", "Retail", "Clinic"], size=rows, p=[0.42, 0.34, 0.24])
    subchannel_map = {
        "Hospital": ["Public Hospital", "Private Hospital"],
        "Retail": ["Independent Pharmacy", "Pharmacy Chain"],
        "Clinic": ["Specialist Clinic", "Primary Care"],
    }
    subchannel = np.array([rng.choice(subchannel_map[value]) for value in channel])

    result = pd.DataFrame(
        {
            "Distributor": [f"DIST_{(code % 6) + 1:02d}" for code in city_codes],
            "Customer": customer,
            "City": location["City"].to_numpy(),
            "Country": location["Country"].to_numpy(),
            "Latitude": location["Latitude"].to_numpy() + rng.normal(0, 0.025, rows),
            "Longitude": location["Longitude"].to_numpy() + rng.normal(0, 0.025, rows),
            "Channel": channel,
            "Sub-channel": subchannel,
            "Product Name": product["Product Name"].to_numpy(),
            "Product Class": product["Product Class"].to_numpy(),
            "Quantity": quantity,
            "Price": np.round(price, 2),
            "Sales": np.round(sales, 2),
            "Month": dates.dt.month_name().to_numpy(),
            "Year": dates.dt.year.to_numpy(),
            "Sales Manager": manager,
            "Sales Team": team,
        }
    )
    return result.sort_values(["Year", "Month", "Country", "City", "Customer"], kind="stable").reset_index(drop=True)


def adapt_schema(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Canonicalise a source table, derive safe essentials, and preserve a field audit."""
    original_columns = [str(column) for column in data.columns]
    frame, mapping = canonicalise_columns(data)
    numeric_fields = ("latitude", "longitude", "quantity", "price", "sales")
    for field in numeric_fields:
        if field in frame:
            frame[field] = pd.to_numeric(frame[field], errors="coerce")
    if "sales" not in frame and {"quantity", "price"}.issubset(frame):
        frame["sales"] = frame["quantity"] * frame["price"]
        mapping["sales"] = "DERIVED: quantity * price"
    if "price" not in frame and {"sales", "quantity"}.issubset(frame):
        frame["price"] = frame["sales"].div(frame["quantity"].replace(0, np.nan))
        mapping["price"] = "DERIVED: sales / quantity"
    if "quantity" not in frame and {"sales", "price"}.issubset(frame):
        frame["quantity"] = frame["sales"].div(frame["price"].replace(0, np.nan))
        mapping["quantity"] = "DERIVED: sales / price"
    if "sales" not in frame:
        raise ValueError("Source must contain sales or both quantity and price.")

    if "month" in frame:
        raw_month = frame["month"]
        if "year" in frame:
            year = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
            month_text = raw_month.astype(str).str.strip()
            month_number = pd.to_numeric(month_text, errors="coerce")
            month_from_name = pd.to_datetime(month_text.str[:3], format="%b", errors="coerce").dt.month
            month_number = month_number.fillna(month_from_name).astype("Int64")
            combined = pd.to_datetime(
                {"year": year, "month": month_number, "day": pd.Series(1, index=frame.index)}, errors="coerce"
            )
            parsed = combined
        else:
            parsed = pd.to_datetime(raw_month, errors="coerce")
        frame["date"] = parsed.dt.to_period("M").dt.to_timestamp()
    elif "year" in frame:
        frame["date"] = pd.to_datetime(
            pd.to_numeric(frame["year"], errors="coerce").astype("Int64").astype(str) + "-01-01",
            errors="coerce",
        )
    else:
        raise ValueError("Source must contain a month/date field or a year field.")
    if frame["date"].isna().all():
        raise ValueError("No valid dates could be derived from the source month/year fields.")

    defaults = {
        "distributor": "Unknown Distributor",
        "customer": None,
        "city": "Unknown City",
        "country": "Unknown Country",
        "channel": "Unknown Channel",
        "subchannel": "Unknown Sub-channel",
        "product_name": "Unknown Product",
        "product_class": "Unknown Class",
        "sales_manager": "Unknown Manager",
        "sales_team": "Unknown Team",
        "quantity": np.nan,
        "price": np.nan,
        "latitude": np.nan,
        "longitude": np.nan,
    }
    for field, default in defaults.items():
        if field not in frame:
            if field == "customer":
                frame[field] = [f"SYNTH_CUSTOMER_{index:07d}" for index in range(len(frame))]
                mapping[field] = "SYNTHETIC: row-level customer placeholder"
            else:
                frame[field] = default
                mapping[field] = f"DEFAULT: {default}"
    for field in ("distributor", "customer", "city", "country", "channel", "subchannel", "product_name", "product_class", "sales_manager", "sales_team"):
        frame[field] = frame[field].fillna(defaults.get(field) or "Unknown").astype(str).str.strip()

    audit = {
        "original_columns": original_columns,
        "canonical_mapping": mapping,
        "canonical_columns": [str(column) for column in frame.columns],
        "original_fields": sorted([name for name, source in mapping.items() if not str(source).startswith(("DERIVED", "SYNTHETIC", "DEFAULT"))]),
        "derived_or_defaulted_fields": sorted([name for name, source in mapping.items() if str(source).startswith(("DERIVED", "SYNTHETIC", "DEFAULT"))]),
    }
    return frame, audit


def load_or_generate_data(
    config: Mapping[str, Any], synthetic_dir: Path, explicit_path: str | Path | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the best source candidate or create and persist an explicit fallback."""
    selected, discovery = discover_pharma_csv(config, explicit_path=explicit_path)
    seed = int(config["project"]["seed"])
    if selected is None:
        fallback_path = synthetic_dir / "fallback_pharma_sales.csv"
        raw = generate_fallback_transactions(
            rows=int(config["data"]["fallback_rows"]),
            start=str(config["data"]["fallback_start"]),
            months=int(config["data"]["fallback_months"]),
            seed=seed,
        )
        raw.to_csv(fallback_path, index=False)
        source_type = "fallback_synthetic"
        source_path = fallback_path
        fallback_used = True
    else:
        source_path = selected
        raw = pd.read_csv(source_path, low_memory=False)
        max_rows = config["data"].get("max_input_rows")
        if max_rows:
            raw = raw.head(int(max_rows)).copy()
        source_type = "provided_csv"
        fallback_used = False
    canonical, schema_audit = adapt_schema(raw)
    source_display = str(source_path.resolve())
    if fallback_used:
        source_display = source_path.resolve().relative_to(Path(str(config["_repo_root"])).resolve()).as_posix()
    metadata = {
        "source_type": source_type,
        "source_path": source_display,
        "fallback_used": fallback_used,
        "fallback_reason": "No qualifying pharma CSV was found in configured workspace search roots." if fallback_used else None,
        "discovery_audit": discovery,
        "schema_audit": schema_audit,
        "seed": seed,
        "rows_loaded": int(len(canonical)),
    }
    with (synthetic_dir / "provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)
    return canonical, metadata
