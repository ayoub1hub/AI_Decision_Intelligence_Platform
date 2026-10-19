"""Clustering for text data.

Discovers groups of similar documents without labels. Useful for:

- Topic discovery
- Exploratory analysis
- Data organization

Uses KMeans on BERT embeddings (or TF-IDF) and provides:

- Cluster assignments
- Silhouette score (cluster quality)
- Top terms per cluster (for TF-IDF)
- Representative documents per cluster
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

__all__ = ["ClusterResult", "cluster_texts"]

logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    """Result of clustering."""

    n_clusters: int
    labels: np.ndarray
    silhouette: float
    inertia: float
    cluster_sizes: dict[int, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "Clustering Report",
            "─" * 40,
            f"N clusters    : {self.n_clusters}",
            f"Silhouette    : {self.silhouette:.4f}",
            f"Inertia (SSE) : {self.inertia:.2f}",
            "",
            "Cluster sizes:",
        ]
        for k, size in sorted(self.cluster_sizes.items()):
            lines.append(f"  Cluster {k} : {size} documents")
        return "\n".join(lines)


def cluster_texts(
    X: np.ndarray,
    *,
    n_clusters: int = 5,
    random_state: int = 42,
) -> ClusterResult:
    """Cluster text embeddings with KMeans.

    Parameters
    ----------
    X : np.ndarray, shape (n_docs, dim)
        Document embeddings (dense).
    n_clusters : int, default 5
        Number of clusters.
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    ClusterResult
        With cluster labels, silhouette score, inertia.
    """
    logger.info("Clustering %d documents into %d clusters", X.shape[0], n_clusters)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    labels = kmeans.fit_predict(X)

    # Silhouette score (needs at least 2 clusters)
    if n_clusters >= 2:
        silhouette = float(silhouette_score(X, labels))
    else:
        silhouette = 0.0

    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    sizes = dict(zip(unique.tolist(), counts.tolist()))

    result = ClusterResult(
        n_clusters=n_clusters,
        labels=labels,
        silhouette=silhouette,
        inertia=float(kmeans.inertia_),
        cluster_sizes=sizes,
    )

    logger.info(
        "Silhouette: %.4f | Inertia: %.2f",
        result.silhouette,
        result.inertia,
    )
    return result


def representative_docs(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    n_per_cluster: int = 3,
) -> dict[int, list[int]]:
    """Return the indices of the most representative documents per cluster.

    A document is representative if it's closest to its cluster centroid.
    """
    unique_labels = sorted(set(labels.tolist()))
    result: dict[int, list[int]] = {}

    for k in unique_labels:
        mask = labels == k
        indices = np.where(mask)[0]
        centroid = X[mask].mean(axis=0)

        # Distance to centroid
        distances = np.linalg.norm(X[indices] - centroid, axis=1)
        order = np.argsort(distances)[:n_per_cluster]

        result[k] = indices[order].tolist()

    return result
