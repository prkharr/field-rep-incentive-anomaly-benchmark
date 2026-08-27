"""Shared deterministic fixtures for the fast unit test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from field_rep_anomaly.anomaly_injection import inject_controlled_anomalies
from field_rep_anomaly.config import load_config
from field_rep_anomaly.data_loader import adapt_schema, generate_fallback_transactions
from field_rep_anomaly.feature_engineering import engineer_features
from field_rep_anomaly.synthetic_enrichment import build_enriched_analytical_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_config() -> dict:
    return load_config(PROJECT_ROOT / "configs" / "config.yaml")


@pytest.fixture(scope="session")
def canonical_transactions():
    raw = generate_fallback_transactions(rows=600, start="2024-01-01", months=6, seed=17)
    canonical, _ = adapt_schema(raw)
    return canonical


@pytest.fixture(scope="session")
def enrichment_bundle(canonical_transactions):
    return build_enriched_analytical_dataset(
        canonical_transactions,
        reps_per_territory=3,
        seed=17,
    )


@pytest.fixture(scope="session")
def analytical_data(enrichment_bundle):
    return enrichment_bundle[0]


@pytest.fixture(scope="session")
def injected_bundle(analytical_data):
    return inject_controlled_anomalies(
        analytical_data,
        injection_rate=0.06,
        severity_min=0.35,
        severity_max=1.0,
        seed=17,
    )


@pytest.fixture(scope="session")
def injected_data(injected_bundle):
    return injected_bundle[0]


@pytest.fixture(scope="session")
def engineered_data(injected_data):
    return engineer_features(injected_data)
