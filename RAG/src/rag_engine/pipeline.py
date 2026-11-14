"""RAG pipeline orchestrator.

Wraps Retriever + LLMClient in a single `RAGPipeline` class:

    pipeline = RAGPipeline.from_config("configs/dev.yaml")
    answer = pipeline.ask("What are vision transformers?")

    print(answer.text)
    print(answer.sources)        # list of RetrievedDoc
    print(answer.tokens_total)   # int
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from rag_engine.config import Config, load_config
from rag_engine.generator import LLMClient, build_rag_prompt
from rag_engine.retriever import Retriever, RetrievedDoc

__all__ = ["RAGPipeline", "RAGAnswer"]

logger = logging.getLogger(__name__)


@dataclass
class RAGAnswer:
    """Answer returned by the RAG pipeline."""

    text: str
    sources: list[RetrievedDoc] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    query: str = ""
    model: str = ""

    @property
    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out


class RAGPipeline:
    """End-to-end RAG pipeline (retrieve → prompt → generate)."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        llm: LLMClient,
        top_k: int = 5,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    @classmethod
    def from_config(cls, config_path: str | Path = "configs/dev.yaml") -> "RAGPipeline":
        """Build a pipeline from a YAML config file."""
        cfg: Config = load_config(config_path)

        logger.info("Building pipeline from %s", config_path)

        retriever = Retriever(
            cfg.retriever.index_path,
            model_name=cfg.retriever.model_name,
        )
        llm = LLMClient(
            provider=cfg.llm.provider,
            model=cfg.llm.model,
            api_key_env=cfg.llm.api_key_env,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
        )

        return cls(
            retriever=retriever,
            llm=llm,
            top_k=cfg.retriever.top_k,
        )

    def ask(
        self,
        query: str,
        *,
        top_k: int | None = None,
        label_filter: str | None = None,
    ) -> RAGAnswer:
        """Run the full RAG pipeline for a query.

        Parameters
        ----------
        query : str
            User question.
        top_k : int, optional
            Override the default top_k.
        label_filter : str, optional
            Restrict retrieval to a specific arXiv category.

        Returns
        -------
        RAGAnswer
            With ``text`` (answer), ``sources``, and token counts.
        """
        k = top_k or self.top_k
        logger.info("RAG ask: '%s' (top_k=%d, filter=%s)", query, k, label_filter)

        # 1. Retrieve
        docs = self.retriever.retrieve(query, top_k=k, label_filter=label_filter)
        if not docs:
            return RAGAnswer(
                text="No relevant documents found.",
                sources=[],
                query=query,
                model=self.llm.model,
            )

        # 2. Build prompt
        messages = build_rag_prompt(query, docs)

        # 3. Generate
        response = self.llm.generate(messages)

        return RAGAnswer(
            text=response.content,
            sources=docs,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            query=query,
            model=response.model,
        )

    def __repr__(self) -> str:
        return (
            f"RAGPipeline(model={self.llm.model!r}, "
            f"top_k={self.top_k}, docs={self.retriever.index.ntotal})"
        )