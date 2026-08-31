"""Small sklearn dense reconstruction model, trained on X -> X."""
import numpy as np
from sklearn.neural_network import MLPRegressor


class AutoencoderAnomaly:
    def __init__(self, hidden_layer_sizes=(32, 8, 32), random_state=42, max_iter=250, alpha=.1):
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, random_state=random_state,
                                  max_iter=max_iter, alpha=alpha, early_stopping=True,
                                  validation_fraction=.15, n_iter_no_change=20, batch_size=64,
                                  learning_rate_init=.001)

    def fit(self, X):
        # Internal early stopping uses a subset of TRAIN only, never chronological val/test.
        self.model.fit(X, X)
        return self

    def contributions(self, X):
        return np.square(X - self.model.predict(X))

    def raw_score(self, X):
        return self.contributions(X).mean(axis=1)
