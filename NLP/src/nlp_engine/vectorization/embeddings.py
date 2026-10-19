"""Sentence embeddings using Sentence Transformers.

Transforms text into dense 384-dimensional vectors that capture
semantic meaning. Unlike TF-IDF, embeddings understand that
"neural network" and "deep learning" are related concepts.

Usage:

    from nlp_engine.vectorization import EmbeddingModel

    model = EmbeddingModel()              # loads all-MiniLM-L6-v2
    X = model.encode(texts)                # (n_docs, 384) dense matrix
    sim = model.similarity(X[0], X[1])     # cosine similarity
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

__all__ = ["EmbeddingModel"]

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingModel:
    """Wrapper around Sentence Transformers for text embeddings.

    Parameters
    ----------
    model_name : str, default "all-MiniLM-L6-v2"
        Hugging Face model name. Downloaded on first use (~80 MB).
    normalize : bool, default True
        L2-normalize vectors so cosine similarity = dot product.
    batch_size : int, default 32
        Batch size for encoding.
    device : str, optional
        "cpu" or "cuda". Auto-detected if None.
    """

    def __init__(
        self,
        *,
        model_name: str = _DEFAULT_MODEL,
        normalize: bool = True,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.normalize = normalize
        self.batch_size = batch_size
        self.device = device
        self._model = None  # lazy loaded

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------
    def _load_model(self):
        """Load the Sentence Transformer model (lazy, cached)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading Sentence Transformer: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Model loaded. Embedding dim: %d", self.dimension)
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding dimension (e.g. 384 for all-MiniLM-L6-v2)."""
        return self._load_model().get_embedding_dimension()

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------
    def encode(
        self,
        texts: list[str],
        *,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode a list of texts into a dense matrix."""
        if not texts:
            return np.zeros((0, self.dimension))

        logger.info("Encoding %d texts (batch_size=%d)", len(texts), self.batch_size)
        embeddings = self._load_model().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        logger.info("Encoded shape: %s", embeddings.shape)
        return embeddings

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------
    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors (assumes L2-normalized)."""
        return float(np.dot(a, b))

    @staticmethod
    def similarity_matrix(X: np.ndarray) -> np.ndarray:
        """Pairwise cosine similarity matrix (n, n) for normalized X."""
        return X @ X.T

    def most_similar(
        self,
        query: np.ndarray,
        corpus: np.ndarray,
        *,
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """Return top-k most similar corpus items to query."""
        scores = corpus @ query
        order = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in order]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        texts: list[str],
        path: str | Path,
    ) -> None:
        """Save embeddings + source texts to a .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            embeddings=embeddings,
            texts=np.array(texts, dtype=object),
            model_name=self.model_name,
        )
        logger.info("Embeddings saved to %s", path)

    @staticmethod
    def load_embeddings(path: str | Path) -> tuple[np.ndarray, list[str]]:
        """Load embeddings + source texts from a .npz file."""
        data = np.load(path, allow_pickle=True)
        embeddings = data["embeddings"]
        texts = list(data["texts"])
        logger.info("Loaded %d embeddings from %s", len(texts), path)
        return embeddings, texts

    def __repr__(self) -> str:
        loaded = self._model is not None
        return (
            f"EmbeddingModel(model_name={self.model_name!r}, "
            f"loaded={loaded}, normalize={self.normalize})"
        )
