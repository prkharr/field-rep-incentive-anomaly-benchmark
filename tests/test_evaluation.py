"""Hand-checkable classification and ranking metric tests."""

from __future__ import annotations

import numpy as np

from field_rep_anomaly.evaluation import (
    classification_metrics,
    evaluate_anomaly_model,
    ranking_metrics,
    top_fraction_flags,
)


def test_classification_metrics_match_hand_calculated_confusion_matrix():
    truth = np.array([True, True, False, False])
    scores = np.array([0.9, 0.4, 0.8, 0.1])
    predicted = np.array([True, False, True, False])

    metrics = classification_metrics(truth, scores, predicted)

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_negatives"] == 1
    for name in ("precision", "recall", "f1", "f2", "specificity", "balanced_accuracy"):
        assert np.isclose(metrics[name], 0.5)
    assert np.isclose(metrics["roc_auc"], 0.75)
    assert np.isclose(metrics["pr_auc"], 5.0 / 6.0)
    assert metrics["predicted_anomalies"] == 2
    assert np.isclose(metrics["predicted_anomaly_pct"], 50.0)


def test_ranking_metrics_and_exact_stable_top_fraction_are_hand_checkable():
    truth = np.array([True, False, True, False])
    scores = np.array([0.9, 0.8, 0.4, 0.1])

    metrics = ranking_metrics(truth, scores, cutoffs=(0.25, 0.50))

    assert metrics["precision_at_25pct"] == 1.0
    assert metrics["recall_at_25pct"] == 0.5
    assert metrics["lift_at_25pct"] == 2.0
    assert metrics["precision_at_50pct"] == 0.5
    assert metrics["recall_at_50pct"] == 0.5
    assert metrics["lift_at_50pct"] == 1.0

    tied = top_fraction_flags(np.array([0.9, 0.9, 0.8, 0.1]), fraction=0.50)
    assert tied.tolist() == [True, True, False, False]
    assert int(tied.sum()) == 2


def test_evaluate_anomaly_model_combines_identity_confusion_and_ranking():
    truth = np.array([True, False, False, True, False])
    scores = np.array([0.95, 0.10, 0.50, 0.80, 0.20])
    predicted = top_fraction_flags(scores, 0.40)

    result = evaluate_anomaly_model(
        "hand-check",
        truth,
        scores,
        predicted,
        cutoffs=(0.40,),
        classification_rule="top 40%",
    )

    assert result["model"] == "hand-check"
    assert result["classification_rule"] == "top 40%"
    assert result["true_positives"] == 2
    assert result["false_positives"] == 0
    assert result["precision_at_40pct"] == 1.0
    assert result["recall_at_40pct"] == 1.0
