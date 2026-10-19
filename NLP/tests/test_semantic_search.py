"""Tests for semantic search."""

from __future__ import annotations

import pytest

from nlp_engine.search import SemanticSearch

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def search_index() -> SemanticSearch:
    """Build a small search index on real embeddings."""
    texts = [
        "Neural networks for image classification.",
        "Transformers for natural language processing.",
        "Gradient descent optimization for deep learning.",
        "Attention mechanisms in machine translation.",
        "Convolutional networks for computer vision.",
    ]
    from nlp_engine.vectorization import EmbeddingModel

    model = EmbeddingModel()
    X = model.encode(texts)
    return SemanticSearch(X, texts, model=model)


class TestSemanticSearch:
    def test_search_returns_top_k(self, search_index: SemanticSearch) -> None:
        results = search_index.search("deep learning", top_k=3)
        assert len(results) == 3

    def test_search_scores_descending(self, search_index: SemanticSearch) -> None:
        results = search_index.search("neural networks", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_finds_relevant(self, search_index: SemanticSearch) -> None:
        results = search_index.search("CNN for vision", top_k=1)
        # Should match "Convolutional networks for computer vision"
        assert "Convolutional" in results[0].text

    def test_search_transformer(self, search_index: SemanticSearch) -> None:
        results = search_index.search("BERT language model", top_k=1)
        assert "Transformers" in results[0].text or "Attention" in results[0].text


class TestPersistence:
    def test_save_and_load(self, search_index: SemanticSearch, tmp_path) -> None:
        path = tmp_path / "index"
        search_index.save(path)
        assert path.with_suffix(".faiss").exists()

        loaded = SemanticSearch.load(path)
        results = loaded.search("deep learning", top_k=2)
        assert len(results) == 2
