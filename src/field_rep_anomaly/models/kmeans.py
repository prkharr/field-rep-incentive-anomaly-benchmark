"""K-Means segmentation with centroid-distance anomaly scoring."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from .base import BaseClusteringModel


def _ecdf(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    reference = np.sort(np.asarray(reference, dtype=float))
    if len(reference) == 0:
        return np.zeros(len(values), dtype=float)
    return np.searchsorted(reference, values, side="right") / len(reference)


class KMeansClusteringModel(BaseClusteringModel):
    def __init__(self, n_clusters: int, random_state: int = 42, n_init: int = 20, max_iter: int = 500):
        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
            max_iter=self.max_iter,
        )

    def fit(self, X: np.ndarray) -> "KMeansClusteringModel":
        matrix = np.asarray(X, dtype=float)
        self.labels_ = self.model.fit_predict(matrix)
        self.cluster_centers_ = self.model.cluster_centers_
        residual = matrix - self.cluster_centers_[self.labels_]
        self.training_distances_ = np.linalg.norm(residual, axis=1)
        self.distance_reference_ = np.sort(self.training_distances_)
        self.training_scores_ = _ecdf(self.training_distances_, self.distance_reference_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(np.asarray(X, dtype=float))

    def distances(self, X: np.ndarray) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        labels = self.predict(matrix)
        return np.linalg.norm(matrix - self.cluster_centers_[labels], axis=1)

    def score_anomaly(self, X: np.ndarray) -> np.ndarray:
        return np.clip(_ecdf(self.distances(X), self.distance_reference_), 0.0, 1.0)

    def feature_contributions(self, X: np.ndarray) -> np.ndarray:
        """Return each feature's share of squared distance to the assigned centroid."""
        matrix = np.asarray(X, dtype=float)
        labels = self.predict(matrix)
        squared = np.square(matrix - self.cluster_centers_[labels])
        totals = squared.sum(axis=1, keepdims=True)
        return np.divide(squared, totals, out=np.zeros_like(squared), where=totals > 0)

    def get_params(self) -> dict[str, Any]:
        return {
            "algorithm": "K-Means",
            "n_clusters": self.n_clusters,
            "random_state": self.random_state,
            "n_init": self.n_init,
            "max_iter": self.max_iter,
            "inertia": float(self.model.inertia_) if hasattr(self.model, "inertia_") else None,
        }
