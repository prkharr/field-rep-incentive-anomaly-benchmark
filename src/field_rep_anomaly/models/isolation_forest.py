"""Continuous Isolation Forest score: higher means more unusual."""
import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestAnomaly:
    def __init__(self, random_state=42, **params):
        self.model = IsolationForest(random_state=random_state, n_jobs=1, **params)

    def fit(self, X):
        self.model.fit(X)
        self.center_ = np.median(X, axis=0)
        return self

    def raw_score(self, X):
        return -self.model.score_samples(X)

    def contributions(self, X):
        # Signed score reduction after replacing one feature by its TRAIN median.
        base = self.raw_score(X)
        out = np.zeros_like(X)
        for j in range(X.shape[1]):
            counterfactual = np.array(X, copy=True)
            counterfactual[:, j] = self.center_[j]
            out[:, j] = np.maximum(base - self.raw_score(counterfactual), 0)
        return out
