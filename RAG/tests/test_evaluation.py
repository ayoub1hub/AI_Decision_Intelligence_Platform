"""Tests for RAG evaluation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rag_engine.evaluation import (
    context_precision,
    evaluate,
    load_test_questions,
    score_answer_relevance,
    score_faithfulness,
)
from rag_engine.generator import LLMResponse
from rag_engine.retriever import RetrievedDoc


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.model = "test-model"
    llm.generate.return_value = LLMResponse(content="0.85", model="test-model")
    return llm


@pytest.fixture
def mock_docs() -> list[RetrievedDoc]:
    return [
        RetrievedDoc(index=0, score=0.9, title="Paper A", abstract="About A.", label="cs.LG"),
        RetrievedDoc(index=1, score=0.7, title="Paper B", abstract="About B.", label="cs.CV"),
    ]


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
class TestFaithfulness:
    def test_returns_score(self, mock_llm: MagicMock, mock_docs: list) -> None:
        score = score_faithfulness("Answer", mock_docs, mock_llm)
        assert 0.0 <= score <= 1.0

    def test_empty_docs_returns_zero(self, mock_llm: MagicMock) -> None:
        assert score_faithfulness("Answer", [], mock_llm) == 0.0


class TestAnswerRelevance:
    def test_returns_score(self, mock_llm: MagicMock) -> None:
        score = score_answer_relevance("Q?", "Answer", mock_llm)
        assert 0.0 <= score <= 1.0


class TestContextPrecision:
    def test_returns_mean_score(self, mock_docs: list) -> None:
        score = context_precision(mock_docs)
        assert abs(score - 0.8) < 1e-9  # (0.9 + 0.7) / 2

    def test_empty_docs_returns_zero(self) -> None:
        assert context_precision([]) == 0.0


# ----------------------------------------------------------------------
# Evaluator
# ----------------------------------------------------------------------
class TestEvaluate:
    def test_full_evaluation(self, mock_llm: MagicMock, mock_docs: list) -> None:
        # Mock the pipeline
        pipeline = MagicMock()
        pipeline.llm = mock_llm

        answer = MagicMock()
        answer.text = "Answer"
        answer.sources = mock_docs
        pipeline.ask.return_value = answer

        questions = [
            {"question": "Q1"},
            {"question": "Q2", "label": "cs.CV"},
        ]
        report = evaluate(pipeline, questions)

        assert report.n == 2
        assert report.mean_faithfulness == 0.85
        assert report.mean_relevance == 0.85
        assert report.mean_precision > 0.0

    def test_summary_string(self, mock_llm: MagicMock, mock_docs: list) -> None:
        pipeline = MagicMock()
        pipeline.llm = mock_llm

        answer = MagicMock()
        answer.text = "Answer"
        answer.sources = mock_docs
        pipeline.ask.return_value = answer

        report = evaluate(pipeline, [{"question": "Q1"}])
        s = report.summary()
        assert "RAG Evaluation Report" in s
        assert "Faithfulness" in s


class TestLoadQuestions:
    def test_loads_yaml(self, tmp_path) -> None:
        path = tmp_path / "test.yaml"
        path.write_text(
            "questions:\n  - question: 'What is A?'\n  - question: 'What is B?'\n",
            encoding="utf-8",
        )
        questions = load_test_questions(path)
        assert len(questions) == 2
        assert questions[0]["question"] == "What is A?"
