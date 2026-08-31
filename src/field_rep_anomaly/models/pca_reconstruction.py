"""Linear reconstruction control for the nonlinear autoencoder."""
import numpy as np
from sklearn.decomposition import PCA


class PCAReconstruction:
    def __init__(self, n_components=.9, random_state=42):
        self.model = PCA(n_components=n_components, svd_solver='full', random_state=random_state)

    def fit(self, X):
        self.model.fit(X)
        return self

    def contributions(self, X):
        return np.square(X - self.model.inverse_transform(self.model.transform(X)))

    def raw_score(self, X):
        return self.contributions(X).mean(axis=1)
