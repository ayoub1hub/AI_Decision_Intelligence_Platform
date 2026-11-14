"""Build and persist a FAISS index from arXiv embeddings.

The index is created once and reused for all queries. It stores the
embeddings and a parallel list of metadata (title, abstract, label).

Usage:

    from rag_engine.retriever import build_index

    build_index(
        embeddings_path="data/processed/arxiv_embeddings.npz",
        csv_path="data/raw/arxiv_labeled.csv",
        index_path="data/index/arxiv",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

__all__ = ["build_index", "load_index_metadata"]

logger = logging.getLogger(__name__)


def build_index(
    embeddings_path: str | Path,
    csv_path: str | Path,
    index_path: str | Path,
) -> tuple[faiss.Index, pd.DataFrame]:
    """Build a FAISS index from embeddings + metadata.

    Parameters
    ----------
    embeddings_path : str or Path
        Path to the .npz file with keys ``embeddings`` and ``texts``.
    csv_path : str or Path
        Path to the arXiv CSV (title, abstract, label, ...).
    index_path : str or Path
        Base path for the index (will create ``.faiss`` and ``.parquet``).

    Returns
    -------
    (faiss.Index, pd.DataFrame)
        The index and the metadata DataFrame (same order as embeddings).
    """
    embeddings_path = Path(embeddings_path)
    csv_path = Path(csv_path)
    index_path = Path(index_path)

    # 1. Load embeddings
    logger.info("Loading embeddings from %s", embeddings_path)
    data = np.load(embeddings_path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)

    # 2. Load metadata
    logger.info("Loading metadata from %s", csv_path)
    df = pd.read_csv(csv_path)

    # Sanity check
    if len(embeddings) != len(df):
        raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(df)} rows in CSV")

    # 3. Build FAISS index (inner product = cosine on normalized vectors)
    dim = embeddings.shape[1]
    logger.info("Building FAISS index (dim=%d, n=%d)", dim, len(embeddings))
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # 4. Persist
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path.with_suffix(".faiss")))
    df.to_parquet(index_path.with_suffix(".parquet"), index=False)

    logger.info("Index saved: %s.faiss", index_path)
    logger.info("Metadata saved: %s.parquet", index_path)
    return index, df


def load_index_metadata(index_path: str | Path) -> tuple[faiss.Index, pd.DataFrame]:
    """Load a previously built FAISS index + metadata."""
    index_path = Path(index_path)
    index = faiss.read_index(str(index_path.with_suffix(".faiss")))
    df = pd.read_parquet(index_path.with_suffix(".parquet"))
    logger.info("Loaded index (%d vectors) + metadata (%d rows)", index.ntotal, len(df))
    return index, df
