"""RAG evaluation utilities."""

from rag_engine.evaluation.evaluator import (
    EvalResult,
    EvaluationReport,
    evaluate,
    load_test_questions,
)
from rag_engine.evaluation.metrics import (
    context_precision,
    score_answer_relevance,
    score_faithfulness,
)

__all__ = [
    "EvalResult",
    "EvaluationReport",
    "context_precision",
    "evaluate",
    "load_test_questions",
    "score_answer_relevance",
    "score_faithfulness",
]
