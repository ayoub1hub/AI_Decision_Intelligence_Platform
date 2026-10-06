"""TF-IDF vectorization for text.

Wraps scikit-learn's TfidfVectorizer with sensible defaults and
save/load functionality.

Usage:

    from nlp_engine.vectorization import TfidfModel

    model = TfidfModel(max_features=5000, ngram_range=(1, 2))
    X = model.fit_transform(texts)          # sparse matrix (n_docs, vocab)
    top = model.top_terms(10)                # top 10 terms per category

    model.save("models/tfidf.pkl")
    model2 = TfidfModel.load("models/tfidf.pkl")
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

__all__ = ["TfidfModel"]

logger = logging.getLogger(__name__)


class TfidfModel:
    """TF-IDF vectorizer wrapper with save/load and analysis helpers.

    Parameters
    ----------
    max_features : int, default 5000
        Maximum vocabulary size (keeps the most frequent terms).
    ngram_range : tuple of int, default (1, 2)
        Range of n-grams (unigrams + bigrams).
    min_df : int or float, default 2
        Ignore terms appearing in fewer than `min_df` documents.
    max_df : float, default 0.95
        Ignore terms appearing in more than `max_df` fraction of docs.
    sublinear_tf : bool, default True
        Apply sublinear scaling to TF (log(1+tf)) — reduces impact of
        very frequent terms.

    Examples
    --------
    >>> model = TfidfModel(max_features=1000)
    >>> X = model.fit_transform(["neural network", "deep learning"])
    >>> X.shape
    (2, 4)
    """

    def __init__(
        self,
        *,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: float = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf

        self._vectorizer: TfidfVectorizer | None = None
        self._vocabulary: dict[str, int] | None = None

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------
    def _build_vectorizer(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            lowercase=False,  # already done in preprocessing
            token_pattern=r"\S+",  # we assume pre-tokenized (space-joined)
        )

    def fit(self, texts: list[str]) -> TfidfModel:
        """Fit the vectorizer on a corpus of texts."""
        logger.info("Fitting TF-IDF on %d documents", len(texts))
        self._vectorizer = self._build_vectorizer()
        self._vectorizer.fit(texts)
        self._vocabulary = self._vectorizer.vocabulary_
        logger.info("Vocabulary size: %d terms", len(self._vocabulary))
        return self

    def transform(self, texts: list[str]) -> spmatrix:
        """Transform texts into a sparse TF-IDF matrix."""
        if self._vectorizer is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._vectorizer.transform(texts)

    def fit_transform(self, texts: list[str]) -> spmatrix:
        """Fit and transform in one call."""
        self.fit(texts)
        return self.transform(texts)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    @property
    def vocabulary_size(self) -> int:
        if self._vocabulary is None:
            return 0
        return len(self._vocabulary)

    def top_terms(self, n: int = 20) -> list[str]:
        """Return the first N terms of the vocabulary (by index order)."""
        if self._vectorizer is None:
            raise RuntimeError("Model not fitted.")
        # get_feature_names_out returns terms ordered by index
        return list(self._vectorizer.get_feature_names_out()[:n])

    def top_terms_in_row(self, X: spmatrix, row_idx: int, n: int = 10) -> list[tuple[str, float]]:
        """Top N terms (with scores) for a specific document."""
        if self._vectorizer is None:
            raise RuntimeError("Model not fitted.")
        names = self._vectorizer.get_feature_names_out()

        row = X.getrow(row_idx)
        indices = row.indices
        values = row.data
        order = np.argsort(values)[::-1][:n]
        return [(str(names[indices[i]]), float(values[i])) for i in order]

    def top_terms_by_class(
        self,
        X: spmatrix,
        labels: list[str],
        n: int = 10,
    ) -> dict[str, list[tuple[str, float]]]:
        """Average TF-IDF per class, return top terms per label."""
        if self._vectorizer is None:
            raise RuntimeError("Model not fitted.")
        names = self._vectorizer.get_feature_names_out()
        unique = sorted(set(labels))

        result: dict[str, list[tuple[str, float]]] = {}
        for label in unique:
            mask = np.array([lbl == label for lbl in labels])
            mean_vec = np.asarray(X[mask].mean(axis=0)).ravel()
            order = np.argsort(mean_vec)[::-1][:n]
            result[label] = [(str(names[i]), float(mean_vec[i])) for i in order if mean_vec[i] > 0]
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Save the fitted model to disk (pickle)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        logger.info("TF-IDF model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> TfidfModel:
        """Load a fitted model from disk."""
        with Path(path).open("rb") as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise TypeError(f"Expected TfidfModel, got {type(model).__name__}")
        logger.info("TF-IDF model loaded from %s", path)
        return model

    def __repr__(self) -> str:
        fitted = self._vectorizer is not None
        return (
            f"TfidfModel(fitted={fitted}, vocab={self.vocabulary_size}, "
            f"max_features={self.max_features}, ngram_range={self.ngram_range})"
        )
