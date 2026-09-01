"""Output, hashing, manifest, and generated data-dictionary helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_sha256(config: dict[str, Any]) -> str:
    payload = yaml.safe_dump(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return target


def write_full_table(
    frame: pd.DataFrame,
    directory: str | Path,
    name: str,
    write_parquet: bool = True,
    write_csv_gz: bool = False,
) -> list[Path]:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    if write_parquet:
        parquet = target / f"{name}.parquet"
        frame.to_parquet(parquet, index=False, compression="snappy")
        files.append(parquet)
    if write_csv_gz or not write_parquet:
        csv = target / f"{name}.csv.gz"
        frame.to_csv(csv, index=False, compression="gzip")
        files.append(csv)
    return files


def write_dashboard_csv(frame: pd.DataFrame, directory: str | Path, name: str) -> Path:
    target = Path(directory) / name
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return target


def git_commit(root: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(root), check=True,
            capture_output=True, text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_worktree_dirty(root: str | Path) -> bool | None:
    """Return whether tracked or untracked repository content is present."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(root),
            check=True,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def implementation_sha256(paths: list[str | Path]) -> str:
    """Hash named implementation/config files with stable relative labels."""
    digest = hashlib.sha256()
    resolved = sorted((Path(path).resolve() for path in paths), key=lambda path: path.as_posix())
    for path in resolved:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _definition(column: str) -> str:
    exact = {
        "data_lineage": "Observed/synthetic provenance category for this row.",
        "source_row_id": "Stable identifier for the unchanged source CSV row.",
        "currency_code": "Currency code; UNK means the source supplied no reliable currency field.",
        "period": "First calendar day representing the monthly scoring period.",
        "ground_truth_label": "Controlled synthetic evaluation label; excluded from all model features.",
        "anomaly_score": "Training-reference percentile of PCA reconstruction error; higher is more unusual.",
        "fte_gap": "Required FTE minus available FTE; evidence for workload review, not an employment decision.",
        "utilization_pct": "Required workload hours divided by available field hours, expressed as percent.",
    }
    if column in exact:
        return exact[column]
    if column.endswith("_id"):
        return f"Stable identifier for {column[:-3].replace('_', ' ')}."
    if column.endswith("_flag"):
        return f"Boolean indicator for {column[:-5].replace('_', ' ')}."
    if column.endswith("_pct"):
        return f"Percentage value for {column[:-4].replace('_', ' ')}."
    if column.endswith("_rate") or column.endswith("_ratio") or column.endswith("_share"):
        return f"Ratio for {column.replace('_', ' ')}; zero-denominator handling is explicit."
    return column.replace("_", " ").capitalize() + "."


def generate_data_dictionary(
    path: str | Path, datasets: dict[str, pd.DataFrame]
) -> Path:
    """Document every emitted field from the actual generated schemas."""
    lines = [
        "# Commercial review generated-data dictionary",
        "",
        "All fields below are generated from the executed schema. `source_observed` rows retain source facts; "
        "`synthetic_derived`, `synthetic_normal`, and `synthetic_injected` identify controlled additions. "
        "`UNK` currency means no reliable source currency was available, so no conversion was attempted.",
        "",
    ]
    for dataset_name in sorted(datasets):
        frame = datasets[dataset_name]
        lines.extend([f"## `{dataset_name}`", "", f"Executed rows: {len(frame):,}.", "", "| Field | Type | Definition |", "|---|---|---|"])
        for column in frame.columns:
            lines.append(f"| `{column}` | `{frame[column].dtype}` | {_definition(column)} |")
        lines.append("")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def build_output_manifest_rows(output_files: list[Path]) -> pd.DataFrame:
    rows = []
    for path in output_files:
        suffixes = "".join(path.suffixes)
        row_count: int | None = None
        if path.suffix == ".csv":
            row_count = len(pd.read_csv(path))
        elif suffixes.endswith(".csv.gz"):
            row_count = len(pd.read_csv(path, compression="gzip"))
        elif path.suffix == ".parquet":
            # Reading a Parquet file with ``columns=[]`` returns an empty
            # zero-row pandas frame even when the file contains records.  The
            # footer is authoritative and avoids loading a large fact table.
            import pyarrow.parquet as pq

            row_count = int(pq.ParquetFile(path).metadata.num_rows)
        rows.append(
            {
                "output_file": path.as_posix(),
                "output_format": suffixes.lstrip("."),
                "output_rows": row_count,
                "output_bytes": path.stat().st_size,
                "output_sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)
