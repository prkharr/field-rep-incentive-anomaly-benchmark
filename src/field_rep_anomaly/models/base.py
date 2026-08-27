"""Common clustering model contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseClusteringModel(ABC):
    """Interface shared by segmentation/anomaly clustering implementations."""

    labels_: np.ndarray

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseClusteringModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def score_anomaly(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        raise NotImplementedError

    def get_cluster_profiles(self, X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
        frame = pd.DataFrame(np.asarray(X), columns=feature_names)
        frame["cluster"] = self.labels_
        return frame.groupby("cluster", observed=True).mean(numeric_only=True).reset_index()
