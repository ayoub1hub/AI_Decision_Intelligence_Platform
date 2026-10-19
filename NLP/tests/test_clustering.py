"""Tests for clustering."""

from __future__ import annotations

import numpy as np

from nlp_engine.modeling import cluster_texts, representative_docs


def _make_blobs() -> np.ndarray:
    """Create 3 well-separated blobs of 2D points."""
    rng = np.random.default_rng(42)
    blob1 = rng.normal(loc=[0, 0], scale=0.3, size=(30, 2))
    blob2 = rng.normal(loc=[5, 5], scale=0.3, size=(30, 2))
    blob3 = rng.normal(loc=[10, 0], scale=0.3, size=(30, 2))
    return np.vstack([blob1, blob2, blob3])


class TestClusterTexts:
    def test_returns_expected_shape(self) -> None:
        X = _make_blobs()
        result = cluster_texts(X, n_clusters=3)
        assert result.labels.shape == (90,)
        assert result.n_clusters == 3

    def test_silhouette_high_for_separated_blobs(self) -> None:
        X = _make_blobs()
        result = cluster_texts(X, n_clusters=3)
        # Well-separated blobs → high silhouette
        assert result.silhouette > 0.7

    def test_cluster_sizes_sum_to_n(self) -> None:
        X = _make_blobs()
        result = cluster_texts(X, n_clusters=3)
        assert sum(result.cluster_sizes.values()) == 90

    def test_summary_string(self) -> None:
        X = _make_blobs()
        result = cluster_texts(X, n_clusters=3)
        s = result.summary()
        assert "Clustering Report" in s
        assert "Silhouette" in s


class TestRepresentativeDocs:
    def test_returns_n_per_cluster(self) -> None:
        X = _make_blobs()
        result = cluster_texts(X, n_clusters=3)
        reps = representative_docs(X, result.labels, n_per_cluster=2)
        assert len(reps) == 3
        for indices in reps.values():
            assert len(indices) == 2
