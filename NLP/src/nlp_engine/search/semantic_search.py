"""Semantic search over text corpora using FAISS.

Provides fast similarity search on BERT embeddings.

Usage:

    from nlp_engine.search import SemanticSearch

    search = SemanticSearch.from_embeddings(X, texts)
    results = search.search("neural networks for vision", top_k=5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from nlp_engine.vectorization import EmbeddingModel

__all__ = ["SearchResult", "SemanticSearch"]

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result."""

    text: str
    score: float
    index: int


class SemanticSearch:
    """FAISS-based semantic search over text embeddings."""

    def __init__(
        self,
        embeddings: np.ndarray,
        texts: list[str],
        *,
        model: EmbeddingModel | None = None,
    ) -> None:
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Embeddings ({len(embeddings)}) and texts ({len(texts)}) "
                "must have the same length."
            )
        self.embeddings = embeddings.astype(np.float32)
        self.texts = texts
        self.model = model or EmbeddingModel()

        # Build FAISS index (inner product = cosine on normalized vectors)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        logger.info("Built FAISS index with %d vectors", self.index.ntotal)

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return the top-k most similar texts to the query."""
        query_emb = self.model.encode([query])[0].astype(np.float32)
        scores, indices = self.index.search(query_emb.reshape(1, -1), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append(
                    SearchResult(
                        text=self.texts[int(idx)],
                        score=float(score),
                        index=int(idx),
                    )
                )
        return results

    def save(self, path: str | Path) -> None:
        """Save the FAISS index + texts to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path.with_suffix(".faiss")))
        np.save(path.with_suffix(".texts.npy"), np.array(self.texts, dtype=object))
        logger.info("Index + texts saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> SemanticSearch:
        """Load a FAISS index + texts."""
        path = Path(path)
        index = faiss.read_index(str(path.with_suffix(".faiss")))
        texts = list(np.load(path.with_suffix(".texts.npy"), allow_pickle=True))

        # Reconstruct embeddings from index
        n = index.ntotal
        embeddings = np.vstack([index.reconstruct(i) for i in range(n)])

        instance = cls.__new__(cls)
        instance.embeddings = embeddings
        instance.texts = texts
        instance.model = EmbeddingModel()
        instance.index = index
        logger.info("Loaded index with %d vectors from %s", n, path)
        return instance
