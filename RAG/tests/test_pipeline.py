"""Tests for the RAG pipeline (mocked LLM + retriever)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rag_engine.generator import LLMResponse
from rag_engine.pipeline import RAGAnswer, RAGPipeline
from rag_engine.retriever import RetrievedDoc


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def mock_retriever() -> MagicMock:
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        RetrievedDoc(index=0, score=0.9, title="Paper A", abstract="About A.", label="cs.LG"),
        RetrievedDoc(index=1, score=0.8, title="Paper B", abstract="About B.", label="cs.CV"),
    ]
    return retriever


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.model = "test-model"
    llm.generate.return_value = LLMResponse(
        content="Answer based on [1] and [2].",
        model="test-model",
        tokens_in=100,
        tokens_out=20,
    )
    return llm


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
class TestRAGPipeline:
    def test_ask_returns_answer(self, mock_retriever: MagicMock, mock_llm: MagicMock) -> None:
        pipeline = RAGPipeline(retriever=mock_retriever, llm=mock_llm, top_k=2)
        answer = pipeline.ask("What is A?")

        assert isinstance(answer, RAGAnswer)
        assert answer.text == "Answer based on [1] and [2]."
        assert len(answer.sources) == 2
        assert answer.tokens_in == 100
        assert answer.tokens_out == 20
        assert answer.tokens_total == 120
        assert answer.query == "What is A?"

    def test_ask_calls_retriever_and_llm(
        self, mock_retriever: MagicMock, mock_llm: MagicMock
    ) -> None:
        pipeline = RAGPipeline(retriever=mock_retriever, llm=mock_llm, top_k=2)
        pipeline.ask("What is A?")

        mock_retriever.retrieve.assert_called_once()
        mock_llm.generate.assert_called_once()

    def test_ask_empty_results(self, mock_llm: MagicMock) -> None:
        retriever = MagicMock()
        retriever.retrieve.return_value = []

        pipeline = RAGPipeline(retriever=retriever, llm=mock_llm, top_k=5)
        answer = pipeline.ask("Query with no results")

        assert "No relevant documents" in answer.text
        assert answer.sources == []
        # LLM should NOT be called if no docs
        mock_llm.generate.assert_not_called()

    def test_ask_passes_label_filter(self, mock_retriever: MagicMock, mock_llm: MagicMock) -> None:
        pipeline = RAGPipeline(retriever=mock_retriever, llm=mock_llm, top_k=2)
        pipeline.ask("Q", label_filter="cs.LG")

        call_kwargs = mock_retriever.retrieve.call_args.kwargs
        assert call_kwargs["label_filter"] == "cs.LG"

    def test_ask_overrides_top_k(self, mock_retriever: MagicMock, mock_llm: MagicMock) -> None:
        pipeline = RAGPipeline(retriever=mock_retriever, llm=mock_llm, top_k=5)
        pipeline.ask("Q", top_k=3)

        call_kwargs = mock_retriever.retrieve.call_args.kwargs
        assert call_kwargs["top_k"] == 3
