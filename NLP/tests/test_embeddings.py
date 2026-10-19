"""Tests for embedding model."""

from __future__ import annotations

import numpy as np
import pytest

from nlp_engine.vectorization import EmbeddingModel

# Mark all tests as slow (they load a model)
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def model() -> EmbeddingModel:
    """Load the model once for all tests in this module."""
    return EmbeddingModel()


class TestEmbeddingEncode:
    def test_encode_returns_correct_shape(self, model: EmbeddingModel) -> None:
        texts = ["hello world", "bonjour le monde", "hola mundo"]
        X = model.encode(texts)
        assert X.shape == (3, model.dimension)

    def test_encode_empty(self, model: EmbeddingModel) -> None:
        X = model.encode([])
        assert X.shape[0] == 0

    def test_normalized_vectors(self, model: EmbeddingModel) -> None:
        """Vectors should have unit L2 norm."""
        X = model.encode(["hello world"])
        norm = float(np.linalg.norm(X[0]))
        assert abs(norm - 1.0) < 1e-5


class TestSemanticSimilarity:
    """Key advantage over TF-IDF: semantic understanding."""

    def test_similar_sentences_are_close(self, model: EmbeddingModel) -> None:
        X = model.encode(
            [
                "The cat is sleeping on the sofa.",
                "A feline is resting on the couch.",  # semantically similar
                "The stock market crashed yesterday.",  # unrelated
            ]
        )
        sim_related = model.similarity(X[0], X[1])
        sim_unrelated = model.similarity(X[0], X[2])
        assert sim_related > sim_unrelated
        assert sim_related > 0.5

    def test_synonyms_are_close(self, model: EmbeddingModel) -> None:
        X = model.encode(
            [
                "I love machine learning.",
                "I enjoy artificial intelligence.",
                "The weather is nice today.",
            ]
        )
        sim_syn = model.similarity(X[0], X[1])
        sim_unrel = model.similarity(X[0], X[2])
        assert sim_syn > sim_unrel

    def test_similarity_matrix_shape(self, model: EmbeddingModel) -> None:
        X = model.encode(["a", "b", "c"])
        S = model.similarity_matrix(X)
        assert S.shape == (3, 3)
        # Diagonal should be ~1
        assert np.allclose(np.diag(S), 1.0, atol=1e-5)


class TestMostSimilar:
    def test_most_similar_returns_sorted(self, model: EmbeddingModel) -> None:
        corpus = model.encode(
            [
                "neural networks",
                "deep learning",
                "quantum computing",
                "computer vision",
            ]
        )
        query = model.encode(["machine learning"])[0]
        results = model.most_similar(query, corpus, top_k=3)
        assert len(results) == 3
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_most_similar_finds_relevant(self, model: EmbeddingModel) -> None:
        corpus = model.encode(
            [
                "The cat is on the sofa.",
                "Quantum entanglement is fascinating.",
                "Neural networks learn representations.",
            ]
        )
        query = model.encode(["A feline on the couch"])[0]
        results = model.most_similar(query, corpus, top_k=1)
        assert results[0][0] == 0


class TestPersistence:
    def test_save_and_load(self, model: EmbeddingModel, tmp_path) -> None:
        texts = ["hello", "world", "foo", "bar"]
        X = model.encode(texts)

        path = tmp_path / "embeddings.npz"
        model.save_embeddings(X, texts, path)
        assert path.exists()

        loaded_X, loaded_texts = model.load_embeddings(path)
        assert loaded_texts == texts
        assert np.allclose(loaded_X, X, atol=1e-6)
