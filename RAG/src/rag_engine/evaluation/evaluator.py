"""RAG evaluator: runs metrics on a set of test questions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from rag_engine.evaluation.metrics import (
    context_precision,
    score_answer_relevance,
    score_faithfulness,
)
from rag_engine.pipeline import RAGPipeline

__all__ = ["EvalResult", "EvaluationReport", "evaluate"]

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of evaluating one question."""

    question: str
    answer: str
    n_sources: int
    faithfulness: float
    answer_relevance: float
    context_precision: float

    @property
    def overall(self) -> float:
        return (self.faithfulness + self.answer_relevance + self.context_precision) / 3


@dataclass
class EvaluationReport:
    """Aggregated report."""

    results: list[EvalResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def mean_faithfulness(self) -> float:
        return sum(r.faithfulness for r in self.results) / self.n if self.n else 0.0

    @property
    def mean_relevance(self) -> float:
        return sum(r.answer_relevance for r in self.results) / self.n if self.n else 0.0

    @property
    def mean_precision(self) -> float:
        return sum(r.context_precision for r in self.results) / self.n if self.n else 0.0

    @property
    def mean_overall(self) -> float:
        return sum(r.overall for r in self.results) / self.n if self.n else 0.0

    def summary(self) -> str:
        lines = [
            "RAG Evaluation Report",
            "─" * 50,
            f"N questions           : {self.n}",
            f"Faithfulness (avg)    : {self.mean_faithfulness:.4f}",
            f"Answer relevance (avg): {self.mean_relevance:.4f}",
            f"Context precision (avg): {self.mean_precision:.4f}",
            f"Overall score         : {self.mean_overall:.4f}",
            "",
            "Per-question:",
        ]
        for i, r in enumerate(self.results, 1):
            lines.append(
                f"  [{i}] {r.question[:60]:<60} "
                f"F={r.faithfulness:.2f} R={r.answer_relevance:.2f} P={r.context_precision:.2f}"
            )
        return "\n".join(lines)


def load_test_questions(path: str | Path) -> list[dict]:
    """Load test questions from a YAML file."""
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw.get("questions", [])


def evaluate(
    pipeline: RAGPipeline,
    questions: list[dict],
) -> EvaluationReport:
    """Evaluate a RAG pipeline on a set of questions.

    Parameters
    ----------
    pipeline : RAGPipeline
        The pipeline to evaluate.
    questions : list of dict
        Each dict must have a "question" key (and optionally "label").

    Returns
    -------
    EvaluationReport
    """
    report = EvaluationReport()

    for i, q in enumerate(questions, 1):
        question = q["question"]
        label = q.get("label")

        logger.info("[%d/%d] Evaluating: %s", i, len(questions), question[:60])

        # Run RAG
        answer = pipeline.ask(question, label_filter=label)

        # Metrics
        faith = score_faithfulness(answer.text, answer.sources, pipeline.llm)
        relev = score_answer_relevance(question, answer.text, pipeline.llm)
        prec = context_precision(answer.sources)

        report.results.append(
            EvalResult(
                question=question,
                answer=answer.text,
                n_sources=len(answer.sources),
                faithfulness=faith,
                answer_relevance=relev,
                context_precision=prec,
            )
        )

    return report
