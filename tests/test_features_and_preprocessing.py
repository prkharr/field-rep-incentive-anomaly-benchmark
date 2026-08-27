"""Feature-integrity and preprocessing tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler, StandardScaler

from field_rep_anomaly.feature_engineering import LABEL_COLUMNS, engineer_features, validate_model_features
from field_rep_anomaly.preprocessing import QuantileClipper, fit_preprocessor


def test_configured_features_are_finite_numeric_and_independent_of_labels(
    injected_data, engineered_data, project_config
):
    features = list(project_config["preprocessing"]["features"])
    validate_model_features(engineered_data, features)

    assert len(features) == len(set(features))
    assert set(features).isdisjoint(LABEL_COLUMNS)
    assert all(pd.api.types.is_numeric_dtype(engineered_data[name]) for name in features)
    assert np.isfinite(engineered_data[features].to_numpy(dtype=float)).all()

    labels_changed = injected_data.copy()
    labels_changed["injected_anomaly_flag"] = ~labels_changed["injected_anomaly_flag"]
    labels_changed["anomaly_type"] = "deliberately_changed_label"
    labels_changed["anomaly_severity"] = 999.0
    reengineered = engineer_features(labels_changed)
    pd.testing.assert_frame_equal(engineered_data[features], reengineered[features])

    with pytest.raises(ValueError, match="leakage"):
        validate_model_features(engineered_data, features + ["injected_anomaly_flag"])


@pytest.mark.parametrize(
    ("scaler_name", "scaler_type"),
    [("standard", StandardScaler), ("robust", RobustScaler)],
)
def test_supported_scalers_impute_and_return_finite_matrices(scaler_name, scaler_type):
    frame = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, np.nan, 4.0, 100.0],
            "feature_b": [10.0, 11.0, 12.0, np.nan, 14.0],
        }
    )
    settings = {
        "features": ["feature_a", "feature_b"],
        "scaler": scaler_name,
        "clip_outliers": False,
    }

    fitted, matrix = fit_preprocessor(frame, settings)

    assert isinstance(fitted.pipeline.named_steps["scaler"], scaler_type)
    assert "clipper" not in fitted.pipeline.named_steps
    assert fitted.feature_names == settings["features"]
    assert matrix.shape == (5, 2)
    assert np.isfinite(matrix).all()
    np.testing.assert_allclose(fitted.transform(frame), matrix)
    if scaler_name == "standard":
        np.testing.assert_allclose(matrix.mean(axis=0), 0.0, atol=1e-12)
    else:
        np.testing.assert_allclose(np.median(matrix, axis=0), 0.0, atol=1e-12)


def test_optional_quantile_clipping_uses_fitted_training_bounds():
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 100.0], "y": [-20.0, 0.0, 1.0, 2.0, 3.0]})
    settings = {
        "features": ["x", "y"],
        "scaler": "robust",
        "clip_outliers": True,
        "clip_lower_quantile": 0.20,
        "clip_upper_quantile": 0.80,
    }

    fitted, matrix = fit_preprocessor(frame, settings)
    clipper = fitted.pipeline.named_steps["clipper"]

    assert isinstance(clipper, QuantileClipper)
    assert np.isfinite(matrix).all()
    extremes = np.array([[-1_000.0, 1_000.0], [1_000.0, -1_000.0]])
    clipped = clipper.transform(extremes)
    assert np.all(clipped >= clipper.lower_bounds_)
    assert np.all(clipped <= clipper.upper_bounds_)


def test_unknown_scaler_is_rejected():
    frame = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="standard.*robust"):
        fit_preprocessor(frame, {"features": ["x"], "scaler": "minmax"})
