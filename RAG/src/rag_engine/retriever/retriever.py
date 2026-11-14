"""Semantic retriever over an arXiv corpus.

Loads a pre-built FAISS index and metadata, encodes queries with
Sentence Transformers, and returns the top-k most similar documents.

Usage:

    from rag_engine.retriever import Retriever

    retriever = Retriever(
        index_path="data/index/arxiv",
        model_name="all-MiniLM-L6-v2",
    )
    docs = retriever.retrieve("neural networks for vision", top_k=5)
    for d in docs:
        print(d.score, d.title)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

__all__ = ["RetrievedDoc", "Retriever"]

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDoc:
    """A single retrieved document."""

    index: int
    score: float
    title: str
    abstract: str
    label: str
    arxiv_id: str | None = None

    def to_context(self) -> str:
        """Format for LLM prompt."""
        return f"[{self.label}] {self.title}\n{self.abstract}"


class Retriever:
    """Semantic retriever over a FAISS index."""

    def __init__(
        self,
        index_path: str | Path,
        *,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        index_path = Path(index_path)

        logger.info("Loading FAISS index from %s", index_path)
        self.index: faiss.Index = faiss.read_index(str(index_path.with_suffix(".faiss")))
        self.metadata: pd.DataFrame = pd.read_parquet(index_path.with_suffix(".parquet"))

        logger.info(
            "Index loaded (%d vectors, %d metadata rows)",
            self.index.ntotal,
            len(self.metadata),
        )

        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name, device=device)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        label_filter: str | None = None,
    ) -> list[RetrievedDoc]:
        """Retrieve the top-k documents most similar to a query.

        Parameters
        ----------
        query : str
            The user query.
        top_k : int, default 5
            Number of results to return.
        label_filter : str, optional
            If set, only return documents with this exact label
            (e.g. "cs.CV"). WARNING: this may return fewer than top_k.

        Returns
        -------
        list of RetrievedDoc
        """
        if not query.strip():
            return []

        # Encode query (L2-normalized)
        q_emb = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # If filtering by label, retrieve more and filter
        search_k = top_k * 5 if label_filter else top_k
        scores, indices = self.index.search(q_emb, search_k)

        results: list[RetrievedDoc] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            row = self.metadata.iloc[int(idx)]

            if label_filter and row.get("label") != label_filter:
                continue

            results.append(
                RetrievedDoc(
                    index=int(idx),
                    score=float(score),
                    title=str(row.get("title", "")),
                    abstract=str(row.get("abstract", "")),
                    label=str(row.get("label", "")),
                    arxiv_id=str(row["arxiv_id"]) if "arxiv_id" in row else None,
                )
            )
            if len(results) >= top_k:
                break

        logger.info("Retrieved %d docs for query: %s", len(results), query[:50])
        return results

    def __repr__(self) -> str:
        return f"Retriever(n_docs={self.index.ntotal}, model={self.model})"
