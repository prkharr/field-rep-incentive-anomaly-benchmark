"""Leakage-safe numeric preprocessing with persisted fitted transformers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from .feature_engineering import validate_model_features


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each fitted numeric feature to training-set quantiles."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X: np.ndarray, y: Any = None) -> "QuantileClipper":
        values = np.asarray(X, dtype=float)
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("Clip quantiles must satisfy 0 <= lower < upper <= 1.")
        self.lower_bounds_ = np.nanquantile(values, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(values, self.upper, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)


@dataclass
class FittedPreprocessor:
    pipeline: Pipeline
    feature_names: list[str]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.pipeline.transform(frame[self.feature_names]), dtype=float)


def build_preprocessor(settings: Mapping[str, Any]) -> Pipeline:
    """Build median-imputation, optional clipping, and configured scaling."""
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if bool(settings.get("clip_outliers", False)):
        steps.append(
            (
                "clipper",
                QuantileClipper(
                    lower=float(settings.get("clip_lower_quantile", 0.01)),
                    upper=float(settings.get("clip_upper_quantile", 0.99)),
                ),
            )
        )
    scaler_name = str(settings.get("scaler", "robust")).lower()
    if scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "robust":
        scaler = RobustScaler(quantile_range=(25.0, 75.0))
    else:
        raise ValueError("preprocessing.scaler must be 'standard' or 'robust'.")
    steps.append(("scaler", scaler))
    return Pipeline(steps)


def fit_preprocessor(frame: pd.DataFrame, settings: Mapping[str, Any]) -> tuple[FittedPreprocessor, np.ndarray]:
    features = [str(feature) for feature in settings["features"]]
    validate_model_features(frame, features)
    pipeline = build_preprocessor(settings)
    matrix = np.asarray(pipeline.fit_transform(frame[features]), dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Preprocessing produced non-finite values.")
    fitted = FittedPreprocessor(pipeline=pipeline, feature_names=features)
    return fitted, matrix


def persist_preprocessor(preprocessor: FittedPreprocessor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
