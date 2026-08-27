"""DBSCAN clustering with density-distance anomaly scoring."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from .base import BaseClusteringModel
from .kmeans import _ecdf


class DBSCANClusteringModel(BaseClusteringModel):
    """DBSCAN batch model.

    scikit-learn DBSCAN has no native out-of-sample ``predict``. This class uses the
    documented approximation of assigning a new point to its nearest core sample only
    when that sample lies within ``eps``; otherwise the point is noise (-1).
    """

    def __init__(self, eps: float, min_samples: int = 5, metric: str = "euclidean"):
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.metric = metric
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric)

    def fit(self, X: np.ndarray) -> "DBSCANClusteringModel":
        matrix = np.asarray(X, dtype=float)
        self.X_fit_ = matrix.copy()
        self.labels_ = self.model.fit_predict(matrix)
        neighbor_count = min(max(2, self.min_samples), len(matrix))
        self.neighbor_model_ = NearestNeighbors(n_neighbors=neighbor_count, metric=self.metric).fit(matrix)
        distances, _ = self.neighbor_model_.kneighbors(matrix)
        self.training_neighbor_distances_ = distances[:, -1]
        self.distance_reference_ = np.sort(self.training_neighbor_distances_)
        base = _ecdf(self.training_neighbor_distances_, self.distance_reference_)
        # Keep every DBSCAN noise point above every core/border point while retaining ranking.
        self.training_scores_ = np.where(self.labels_ == -1, 0.75 + 0.25 * base, 0.75 * base)
        core_indices = getattr(self.model, "core_sample_indices_", np.array([], dtype=int))
        self.core_samples_ = matrix[core_indices] if len(core_indices) else np.empty((0, matrix.shape[1]))
        self.core_labels_ = self.labels_[core_indices] if len(core_indices) else np.empty(0, dtype=int)
        if len(self.core_samples_):
            self.core_neighbor_model_ = NearestNeighbors(n_neighbors=1, metric=self.metric).fit(self.core_samples_)
        else:
            self.core_neighbor_model_ = None
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        if self.core_neighbor_model_ is None:
            return np.full(len(matrix), -1, dtype=int)
        distance, index = self.core_neighbor_model_.kneighbors(matrix)
        labels = self.core_labels_[index[:, 0]].astype(int)
        labels[distance[:, 0] > self.eps] = -1
        return labels

    def neighbor_distances(self, X: np.ndarray) -> np.ndarray:
        distances, _ = self.neighbor_model_.kneighbors(np.asarray(X, dtype=float))
        return distances[:, -1]

    def score_anomaly(self, X: np.ndarray) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        if matrix.shape == self.X_fit_.shape and np.allclose(matrix, self.X_fit_):
            return self.training_scores_.copy()
        base = _ecdf(self.neighbor_distances(matrix), self.distance_reference_)
        predicted = self.predict(matrix)
        return np.clip(np.where(predicted == -1, 0.75 + 0.25 * base, 0.75 * base), 0.0, 1.0)

    def get_params(self) -> dict[str, Any]:
        return {
            "algorithm": "DBSCAN",
            "eps": self.eps,
            "min_samples": self.min_samples,
            "metric": self.metric,
            "prediction_note": "Nearest-core assignment within eps; sklearn DBSCAN has no native predict.",
        }
