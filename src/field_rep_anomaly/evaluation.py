"""Anomaly-classification and ranking evaluation against injected labels."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


def top_fraction_flags(scores: np.ndarray, fraction: float) -> np.ndarray:
    """Flag an exact top fraction with deterministic tie handling."""
    values = np.asarray(scores, dtype=float)
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")
    count = max(1, int(np.ceil(len(values) * fraction)))
    order = np.argsort(-np.nan_to_num(values, nan=-np.inf), kind="stable")
    flags = np.zeros(len(values), dtype=bool)
    flags[order[:count]] = True
    return flags


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(y_true, dtype=bool)
    prediction = np.asarray(y_pred, dtype=bool)
    scores = np.asarray(y_score, dtype=float)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[False, True]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    has_two_classes = len(np.unique(truth)) == 2
    return {
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "f2": float(fbeta_score(truth, prediction, beta=2, zero_division=0)),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)) if has_two_classes else np.nan,
        "roc_auc": float(roc_auc_score(truth, scores)) if has_two_classes else np.nan,
        "pr_auc": float(average_precision_score(truth, scores)) if truth.any() else np.nan,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "predicted_anomalies": int(prediction.sum()),
        "predicted_anomaly_pct": float(prediction.mean() * 100.0),
    }

def ranking_metrics(y_true: np.ndarray, y_score: np.ndarray, cutoffs: Iterable[float] = (0.01, 0.05, 0.10)) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=bool)
    scores = np.asarray(y_score, dtype=float)
    prevalence = float(truth.mean())
    positives = int(truth.sum())
    result: dict[str, float] = {}
    for cutoff in cutoffs:
        flags = top_fraction_flags(scores, float(cutoff))
        captured = int((truth & flags).sum())
        precision = captured / max(int(flags.sum()), 1)
        recall = captured / positives if positives else np.nan
        lift = precision / prevalence if prevalence > 0 else np.nan
        label = f"{int(round(float(cutoff) * 100))}pct"
        result[f"precision_at_{label}"] = float(precision)
        result[f"recall_at_{label}"] = float(recall)
        result[f"lift_at_{label}"] = float(lift)
    result["top_decile_capture"] = result.get("recall_at_10pct", np.nan)
    return result


def evaluate_anomaly_model(
    model_name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    cutoffs: Iterable[float] = (0.01, 0.05, 0.10),
    classification_rule: str = "",
) -> dict[str, float | int | str]:
    return {
        "model": model_name,
        "classification_rule": classification_rule,
        **classification_metrics(y_true, y_score, y_pred),
        **ranking_metrics(y_true, y_score, cutoffs=cutoffs),
    }
