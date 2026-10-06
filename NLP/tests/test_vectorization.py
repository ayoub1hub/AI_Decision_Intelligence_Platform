"""Tests for TF-IDF vectorization."""

from __future__ import annotations

from pathlib import Path

import pytest
from scipy.sparse import issparse

from nlp_engine.vectorization import TfidfModel

# Corpus for tests (preprocessed — lowercase, space-separated tokens)
_CORPUS = [
    "neural network deep learn",
    "neural network train gradient",
    "deep learn transformer attention",
    "transformer attention mechanism",
    "gradient descent optim",
    "optim adam gradient",
]


class TestTfidfFit:
    def test_fit_sets_vocabulary(self) -> None:
        model = TfidfModel(max_features=100, min_df=1)
        model.fit(_CORPUS)
        assert model.vocabulary_size > 0
        assert model.vocabulary_size <= 100

    def test_fit_returns_self(self) -> None:
        model = TfidfModel(min_df=1)
        assert model.fit(_CORPUS) is model

    def test_transform_returns_sparse(self) -> None:
        model = TfidfModel(min_df=1)
        X = model.fit_transform(_CORPUS)
        assert issparse(X)
        assert X.shape[0] == len(_CORPUS)

    def test_fit_transform_equals_fit_then_transform(self) -> None:
        m1 = TfidfModel(min_df=1)
        X1 = m1.fit_transform(_CORPUS)

        m2 = TfidfModel(min_df=1)
        m2.fit(_CORPUS)
        X2 = m2.transform(_CORPUS)

        assert (X1 != X2).nnz == 0

    def test_transform_before_fit_raises(self) -> None:
        model = TfidfModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.transform(["hello"])


class TestTfidfContent:
    def test_common_terms_get_lower_idf(self) -> None:
        """'neural' appears in 2 docs, 'transformer' in 2 docs too."""
        model = TfidfModel(min_df=1)
        X = model.fit_transform(_CORPUS)

        # Document 0
        top = model.top_terms_in_row(X, 0, n=3)
        terms = [t for t, _ in top]
        # 'neural' should be there but not the most informative alone
        assert "neural" in terms

    def test_empty_text(self) -> None:
        model = TfidfModel(min_df=1)
        model.fit(_CORPUS)
        X = model.transform([""])
        assert X.shape[0] == 1
        assert X.nnz == 0


class TestTfidfAnalysis:
    def test_top_terms(self) -> None:
        model = TfidfModel(min_df=1)
        model.fit(_CORPUS)
        terms = model.top_terms(5)
        assert len(terms) == 5
        assert all(isinstance(t, str) for t in terms)

    def test_top_terms_in_row(self) -> None:
        model = TfidfModel(min_df=1)
        X = model.fit_transform(_CORPUS)
        top = model.top_terms_in_row(X, 0, n=3)
        assert len(top) == 3
        # Score should be decreasing
        scores = [s for _, s in top]
        assert scores == sorted(scores, reverse=True)

    def test_top_terms_by_class(self) -> None:
        labels = ["A", "A", "B", "B", "C", "C"]
        model = TfidfModel(min_df=1)
        X = model.fit_transform(_CORPUS)
        by_class = model.top_terms_by_class(X, labels, n=3)
        assert set(by_class.keys()) == {"A", "B", "C"}
        for terms in by_class.values():
            assert len(terms) <= 3
            assert len(terms) > 0


class TestTfidfPersistence:
    def test_save_and_load(self, tmp_path: Path) -> None:
        model = TfidfModel(min_df=1)
        model.fit(_CORPUS)
        path = tmp_path / "tfidf.pkl"
        model.save(path)
        assert path.exists()

        loaded = TfidfModel.load(path)
        assert loaded.vocabulary_size == model.vocabulary_size
        assert loaded.max_features == model.max_features

        # Transform with loaded model gives same result
        X1 = model.transform(_CORPUS)
        X2 = loaded.transform(_CORPUS)
        assert (X1 != X2).nnz == 0

    def test_load_wrong_type_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "wrong.pkl"
        import pickle

        with path.open("wb") as f:
            pickle.dump({"not": "a model"}, f)
        with pytest.raises(TypeError):
            TfidfModel.load(path)


class TestTfidfNgrams:
    def test_bigrams_are_captured(self) -> None:
        model = TfidfModel(min_df=1, ngram_range=(1, 2), max_features=1000)
        model.fit(_CORPUS)
        terms = set(model.top_terms(model.vocabulary_size))
        # Bigrams contain a space
        bigrams = [t for t in terms if " " in t]
        assert len(bigrams) > 0
