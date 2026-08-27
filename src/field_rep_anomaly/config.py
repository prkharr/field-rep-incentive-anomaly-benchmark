"""Configuration loading and path resolution."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating inputs."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_update(dict(result[key]), value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML configuration and attach an absolute repository root."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    required = {"project", "paths", "data", "anomalies", "preprocessing", "models"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Configuration is missing sections: {', '.join(missing)}")
    config["_config_path"] = str(config_path)
    config["_repo_root"] = str(config_path.parent.parent)
    return config


def resolve_path(config: Mapping[str, Any], value: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    path = Path(value)
    if not path.is_absolute():
        path = Path(str(config["_repo_root"])) / path
    return path.resolve()


def ensure_project_directories(config: Mapping[str, Any]) -> dict[str, Path]:
    """Create and return the configured data/artifact directories."""
    paths: dict[str, Path] = {}
    for key in ("raw_dir", "synthetic_dir", "processed_dir", "artifacts_dir"):
        paths[key] = resolve_path(config, config["paths"][key])
        paths[key].mkdir(parents=True, exist_ok=True)
    artifacts = paths["artifacts_dir"]
    for name in ("metrics", "plots", "models", "reports"):
        (artifacts / name).mkdir(parents=True, exist_ok=True)
    return paths
