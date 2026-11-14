"""Interactive semantic search over arXiv.

Usage:
    python scripts/search.py "neural networks for vision"
    python scripts/search.py "transformers" --top-k 3
    python scripts/search.py "optimization" --label cs.LG
"""

from __future__ import annotations

import argparse
import logging
import sys

from rag_engine.logging_config import setup_logging
from rag_engine.retriever import Retriever

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="+", help="Search query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--label", default=None, help="Filter by arXiv category")
    parser.add_argument("--index", default="data/index/arxiv")
    args = parser.parse_args()

    setup_logging(level="INFO")
    query = " ".join(args.query)

    retriever = Retriever(args.index)
    results = retriever.retrieve(query, top_k=args.top_k, label_filter=args.label)

    print(f"\nQuery: {query}\n")
    print("=" * 70)
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. [{doc.score:.4f}] [{doc.label}]")
        print(f"   {doc.title}")
        print(f"   {doc.abstract[:150]}...")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
