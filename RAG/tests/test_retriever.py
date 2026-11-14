"""Tests for the retriever."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rag_engine.retriever import RetrievedDoc, Retriever, build_index

pytestmark = pytest.mark.slow


@pytest.fixture
def fake_index(tmp_path: Path) -> Path:
    """Create a fake embeddings + CSV + index for testing."""
    n_docs = 20
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((n_docs, 384)).astype(np.float32)
    # Normalize
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    texts = [f"Document {i} about topic {i % 5}" for i in range(n_docs)]
    df = pd.DataFrame(
        {
            "title": texts,
            "abstract": [f"Abstract for doc {i}" for i in range(n_docs)],
            "label": [f"cat_{i % 5}" for i in range(n_docs)],
        }
    )

    # Save
    embeddings_path = tmp_path / "embeddings.npz"
    csv_path = tmp_path / "data.csv"
    np.savez(embeddings_path, embeddings=embeddings, texts=texts)
    df.to_csv(csv_path, index=False)

    # Build
    index_base = tmp_path / "index"
    build_index(embeddings_path, csv_path, index_base)

    return index_base


class TestBuildIndex:
    def test_creates_files(self, fake_index: Path) -> None:
        assert fake_index.with_suffix(".faiss").exists()
        assert fake_index.with_suffix(".parquet").exists()


class TestRetriever:
    def test_retrieve_returns_top_k(self, fake_index: Path) -> None:
        retriever = Retriever(fake_index)
        docs = retriever.retrieve("topic 0", top_k=3)
        assert len(docs) == 3
        assert all(isinstance(d, RetrievedDoc) for d in docs)

    def test_retrieve_scores_descending(self, fake_index: Path) -> None:
        retriever = Retriever(fake_index)
        docs = retriever.retrieve("topic 0", top_k=5)
        scores = [d.score for d in docs]
        assert scores == sorted(scores, reverse=True)

    def test_empty_query(self, fake_index: Path) -> None:
        retriever = Retriever(fake_index)
        assert retriever.retrieve("") == []

    def test_label_filter(self, fake_index: Path) -> None:
        retriever = Retriever(fake_index)
        docs = retriever.retrieve("topic 0", top_k=3, label_filter="cat_0")
        assert all(d.label == "cat_0" for d in docs)

    def test_to_context(self, fake_index: Path) -> None:
        retriever = Retriever(fake_index)
        docs = retriever.retrieve("topic 0", top_k=1)
        ctx = docs[0].to_context()
        assert "[" in ctx  # label in brackets
