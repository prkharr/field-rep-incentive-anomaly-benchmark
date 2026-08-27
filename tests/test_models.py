"""Tests for the shared clustering contract and anomaly scores."""

from __future__ import annotations

import numpy as np
import pytest

from field_rep_anomaly.models.base import BaseClusteringModel
from field_rep_anomaly.models.dbscan import DBSCANClusteringModel
from field_rep_anomaly.models.kmeans import KMeansClusteringModel


@pytest.fixture
def compact_clusters():
    rng = np.random.default_rng(314)
    left = rng.normal(loc=(-1.0, -1.0), scale=0.08, size=(24, 2))
    right = rng.normal(loc=(1.0, 1.0), scale=0.08, size=(24, 2))
    return np.vstack([left, right, [[4.0, 4.0], [5.0, 5.0]]])


@pytest.mark.parametrize(
    "model",
    [
        KMeansClusteringModel(n_clusters=2, random_state=7, n_init=5, max_iter=100),
        DBSCANClusteringModel(eps=0.30, min_samples=4),
    ],
    ids=["kmeans", "dbscan"],
)
def test_models_honor_common_contract_and_scores_are_probability_bounded(model, compact_clusters):
    fitted = model.fit(compact_clusters)
    scores = fitted.score_anomaly(compact_clusters)
    query_scores = fitted.score_anomaly(np.array([[-1.0, -1.0], [9.0, 9.0]]))

    assert isinstance(fitted, BaseClusteringModel)
    assert fitted.labels_.shape == (len(compact_clusters),)
    assert fitted.predict(compact_clusters).shape == fitted.labels_.shape
    assert scores.shape == fitted.labels_.shape
    assert np.isfinite(scores).all()
    assert np.isfinite(query_scores).all()
    assert ((0.0 <= scores) & (scores <= 1.0)).all()
    assert ((0.0 <= query_scores) & (query_scores <= 1.0)).all()
    assert isinstance(fitted.get_params()["algorithm"], str)


def test_kmeans_feature_contributions_sum_to_distance_share(compact_clusters):
    model = KMeansClusteringModel(n_clusters=2, random_state=7, n_init=5, max_iter=100).fit(compact_clusters)
    contributions = model.feature_contributions(compact_clusters)
    labels = model.predict(compact_clusters)
    squared_distance = np.square(compact_clusters - model.cluster_centers_[labels]).sum(axis=1)
    nonzero = squared_distance > 1e-15

    assert contributions.shape == compact_clusters.shape
    assert ((0.0 <= contributions) & (contributions <= 1.0)).all()
    np.testing.assert_allclose(contributions[nonzero].sum(axis=1), 1.0)
    np.testing.assert_allclose(contributions[~nonzero].sum(axis=1), 0.0)


def test_kmeans_distance_is_assigned_centroid_euclidean_l2(compact_clusters):
    model = KMeansClusteringModel(n_clusters=2, random_state=7, n_init=5, max_iter=100).fit(compact_clusters)
    labels = model.predict(compact_clusters)
    expected = np.sqrt(np.square(compact_clusters - model.cluster_centers_[labels]).sum(axis=1))
    sklearn_assigned = model.model.transform(compact_clusters)[np.arange(len(compact_clusters)), labels]

    np.testing.assert_allclose(model.distances(compact_clusters), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(model.distances(compact_clusters), sklearn_assigned, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.square(expected).sum(), model.model.inertia_, rtol=1e-12, atol=1e-12)
    assert np.array_equal(labels, np.argmin(model.model.transform(compact_clusters), axis=1))


def test_dbscan_ranks_noise_above_cluster_members(compact_clusters):
    model = DBSCANClusteringModel(eps=0.30, min_samples=4).fit(compact_clusters)
    scores = model.score_anomaly(compact_clusters)
    noise = model.labels_ == -1

    assert noise.any()
    assert (~noise).any()
    assert scores[noise].min() >= scores[~noise].max()
