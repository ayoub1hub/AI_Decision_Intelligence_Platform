"""End-to-end RAG: ask a question over arXiv.

Usage:
    python scripts/ask.py "What is the Drosophila connectome?"
    python scripts/ask.py "transformers for vision" --label cs.CV --top-k 3
"""

from __future__ import annotations

import argparse
import logging
import sys

from rag_engine.config import load_config
from rag_engine.generator import LLMClient, build_rag_prompt
from rag_engine.logging_config import setup_logging
from rag_engine.retriever import Retriever

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="+", help="Your question")
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    setup_logging(level="INFO")
    cfg = load_config(args.config)
    query = " ".join(args.query)
    top_k = args.top_k or cfg.retriever.top_k

    # 1. Retrieve
    logger.info("Retrieving top-%d docs for: %s", top_k, query)
    retriever = Retriever(
        cfg.retriever.index_path,
        model_name=cfg.retriever.model_name,
    )
    docs = retriever.retrieve(query, top_k=top_k, label_filter=args.label)

    if not docs:
        print("\n❌ No documents retrieved. Try a different query.")
        return 1

    # 2. Build prompt
    messages = build_rag_prompt(query, docs)

    # 3. Generate
    logger.info("Calling LLM (%s / %s)", cfg.llm.provider, cfg.llm.model)
    llm = LLMClient(
        provider=cfg.llm.provider,
        model=cfg.llm.model,
        api_key_env=cfg.llm.api_key_env,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
    )
    response = llm.generate(messages)

    # 4. Print
    print("\n" + "=" * 70)
    print(f"Question: {query}")
    print("=" * 70)
    print()
    print(response.content)
    print()
    print("─" * 70)
    print(
        f"Tokens: {response.tokens_in} in, {response.tokens_out} out ({response.total_tokens} total)"
    )
    print()
    print("Sources:")
    for i, doc in enumerate(docs, 1):
        print(f"  [{i}] [{doc.label}] {doc.title[:80]}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
