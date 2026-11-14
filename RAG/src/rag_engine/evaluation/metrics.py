"""RAG evaluation metrics.

LLM-as-judge metrics:

- Faithfulness: is every claim in the answer supported by the context?
- Answer Relevance: does the answer address the question?

Both return a score in [0, 1].

Reference:
    RAGAS: https://docs.ragas.io/
"""

from __future__ import annotations

import logging

from rag_engine.generator.llm_client import LLMClient
from rag_engine.retriever import RetrievedDoc

__all__ = ["context_precision", "score_answer_relevance", "score_faithfulness"]

logger = logging.getLogger(__name__)


_FAITHFULNESS_PROMPT = """You are an evaluator. Score how faithful the answer is to the context.

Context:
{context}

Answer:
{answer}

For each factual claim in the answer, check if it is supported by the context.
Return a single number between 0.0 and 1.0, where:
- 1.0 = every claim is supported by the context
- 0.5 = some claims are unsupported
- 0.0 = most claims are hallucinated

Return ONLY the number, no explanation."""


_RELEVANCE_PROMPT = """You are an evaluator. Score how relevant the answer is to the question.

Question:
{question}

Answer:
{answer}

Return a single number between 0.0 and 1.0, where:
- 1.0 = the answer directly addresses the question
- 0.5 = the answer is somewhat related but misses the point
- 0.0 = the answer is off-topic

Return ONLY the number, no explanation."""


def _parse_score(raw: str) -> float:
    """Extract a float from LLM output, clamped to [0, 1]."""
    raw = raw.strip().lower()
    try:
        # Find the first number in the output
        for token in raw.replace(",", ".").split():
            try:
                v = float(token)
                return max(0.0, min(1.0, v))
            except ValueError:
                continue
        # Fallback: try the whole string
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        logger.warning("Could not parse score from: %r, defaulting to 0.5", raw)
        return 0.5


def score_faithfulness(
    answer: str,
    docs: list[RetrievedDoc],
    llm: LLMClient,
) -> float:
    """Score how faithful an answer is to the retrieved context.

    Parameters
    ----------
    answer : str
        The generated answer.
    docs : list of RetrievedDoc
        The retrieved documents.
    llm : LLMClient
        The LLM to use as judge.

    Returns
    -------
    float in [0, 1]
    """
    if not docs:
        return 0.0

    context = "\n\n".join(f"[{i}] {d.title}\n{d.abstract}" for i, d in enumerate(docs, 1))
    prompt = _FAITHFULNESS_PROMPT.format(context=context, answer=answer)

    resp = llm.generate([{"role": "user", "content": prompt}])
    return _parse_score(resp.content)


def score_answer_relevance(
    question: str,
    answer: str,
    llm: LLMClient,
) -> float:
    """Score how relevant an answer is to the question."""
    prompt = _RELEVANCE_PROMPT.format(question=question, answer=answer)
    resp = llm.generate([{"role": "user", "content": prompt}])
    return _parse_score(resp.content)


def context_precision(docs: list[RetrievedDoc]) -> float:
    """Average retrieval score (proxy for relevance).

    A simple, deterministic metric: the mean of the top-k similarity scores.
    """
    if not docs:
        return 0.0
    return sum(d.score for d in docs) / len(docs)
