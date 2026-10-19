"""Interactive semantic search on arXiv embeddings.

Usage:
    python scripts/search_arxiv.py "neural networks for vision"
"""

from __future__ import annotations

import logging
import sys

import pandas as pd

from nlp_engine.logging_config import setup_logging
from nlp_engine.search import SemanticSearch
from nlp_engine.vectorization import EmbeddingModel

logger = logging.getLogger(__name__)


def main() -> int:
    setup_logging(level="INFO")

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "neural networks for vision"

    # Load embeddings + metadata
    X, _ = EmbeddingModel.load_embeddings("data/processed/arxiv_embeddings.npz")
    df = pd.read_csv("data/raw/arxiv_labeled.csv")
    texts = (df["title"].fillna("") + " | " + df["abstract"].fillna("")).tolist()

    # Build search index
    search = SemanticSearch(X, texts)

    # Search
    print(f"\nQuery : {query}\n")
    results = search.search(query, top_k=5)

    print("=" * 70)
    print("Top 5 results:")
    print("=" * 70)
    for i, r in enumerate(results, 1):
        row = df.iloc[r.index]
        print(f"\n{i}. [{r.score:.4f}] [{row['label']}]")
        print(f"   {row['title'][:80]}")
        print(f"   {row['abstract'][:150]}...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
