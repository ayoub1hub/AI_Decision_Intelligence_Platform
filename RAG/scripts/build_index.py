"""Build the FAISS index from arXiv embeddings + metadata.

Usage:
    python scripts/build_index.py
"""

from __future__ import annotations

import logging
import sys

from rag_engine.logging_config import setup_logging
from rag_engine.retriever import build_index

logger = logging.getLogger(__name__)


def main() -> int:
    setup_logging(level="INFO")

    logger.info("Building FAISS index from arXiv data")
    index, df = build_index(
        embeddings_path="data/processed/arxiv_embeddings.npz",
        csv_path="data/raw/arxiv_labeled.csv",
        index_path="data/index/arxiv",
    )

    print("\n✅ Index built")
    print(f"   Vectors   : {index.ntotal}")
    print(f"   Dimensions: {index.d}")
    print(f"   Metadata  : {len(df)} rows")
    print("   Saved to  : data/index/arxiv.faiss")
    return 0


if __name__ == "__main__":
    sys.exit(main())
