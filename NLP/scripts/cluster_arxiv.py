"""Cluster arXiv embeddings to discover topics.

Usage:
    python scripts/cluster_arxiv.py --n-clusters 5
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from nlp_engine.logging_config import setup_logging
from nlp_engine.modeling import cluster_texts, representative_docs
from nlp_engine.vectorization import EmbeddingModel

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-cache", default="data/processed/arxiv_embeddings.npz")
    parser.add_argument("--data", default="data/raw/arxiv_labeled.csv")
    parser.add_argument("--n-clusters", type=int, default=5)
    args = parser.parse_args()

    setup_logging(level="INFO")

    # Load embeddings
    logger.info("Loading embeddings from %s", args.emb_cache)
    X, _ = EmbeddingModel.load_embeddings(args.emb_cache)

    # Load data for analysis
    df = pd.read_csv(args.data)

    # Cluster
    result = cluster_texts(X, n_clusters=args.n_clusters)
    print("\n" + result.summary())

    # Compare to true labels
    print("\n=== Cluster vs true category ===")
    df_cluster = df.copy()
    df_cluster["cluster"] = result.labels

    contingency = pd.crosstab(df_cluster["cluster"], df_cluster["label"])
    print(contingency)

    # Representative docs
    print("\n=== Representative documents per cluster ===")
    reps = representative_docs(X, result.labels, n_per_cluster=2)
    for k, indices in reps.items():
        print(f"\nCluster {k}:")
        for idx in indices:
            title = df.iloc[idx]["title"][:80]
            label = df.iloc[idx]["label"]
            print(f"  [{label}] {title}...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
