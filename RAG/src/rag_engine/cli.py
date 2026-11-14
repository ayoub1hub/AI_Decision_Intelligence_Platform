"""Command-line interface for the RAG engine.

Usage:
    python -m rag_engine ask "What are vision transformers?"
    python -m rag_engine search "neural networks for vision" --top-k 5
    python -m rag_engine build-index
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rag_engine.config import load_config
from rag_engine.logging_config import setup_logging
from rag_engine.pipeline import RAGPipeline
from rag_engine.retriever import Retriever, build_index

logger = logging.getLogger(__name__)


def cmd_ask(args: argparse.Namespace) -> int:
    """Run a full RAG query."""
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_file=cfg.logging.log_file)

    pipeline = RAGPipeline.from_config(args.config)
    query = " ".join(args.query)

    answer = pipeline.ask(
        query,
        top_k=args.top_k or cfg.retriever.top_k,
        label_filter=args.label,
    )

    print("\n" + "=" * 70)
    print(f"Question: {query}")
    print("=" * 70)
    print()
    print(answer.text)
    print()
    print("─" * 70)
    print(
        f"Tokens: {answer.tokens_in} in, "
        f"{answer.tokens_out} out ({answer.tokens_total} total)"
    )
    print()
    print("Sources:")
    for i, doc in enumerate(answer.sources, 1):
        print(f"  [{i}] [{doc.label}] {doc.title[:80]}")
    print()
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Only run the retriever (no LLM call)."""
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_file=cfg.logging.log_file)

    retriever = Retriever(
        cfg.retriever.index_path,
        model_name=cfg.retriever.model_name,
    )
    query = " ".join(args.query)
    docs = retriever.retrieve(
        query,
        top_k=args.top_k or cfg.retriever.top_k,
        label_filter=args.label,
    )

    print(f"\nQuery: {query}\n")
    print("=" * 70)
    for i, doc in enumerate(docs, 1):
        print(f"\n{i}. [{doc.score:.4f}] [{doc.label}]")
        print(f"   {doc.title}")
        print(f"   {doc.abstract[:150]}...")
    print()
    return 0


def cmd_build_index(args: argparse.Namespace) -> int:
    """Build the FAISS index from embeddings + metadata."""
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_file=cfg.logging.log_file)

    index, df = build_index(
        embeddings_path=cfg.retriever.embeddings_cache,
        csv_path=cfg.data.arxiv_csv,
        index_path=cfg.retriever.index_path,
    )

    print(f"\n✅ Index built")
    print(f"   Vectors   : {index.ntotal}")
    print(f"   Dimensions: {index.d}")
    print(f"   Metadata  : {len(df)} rows")
    print(f"   Saved to  : {cfg.retriever.index_path}.faiss")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag_engine",
        description="RAG engine CLI",
    )
    parser.add_argument(
        "--config", default="configs/dev.yaml", help="Path to config file"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ask
    p_ask = sub.add_parser("ask", help="Full RAG query (retrieve + generate)")
    p_ask.add_argument("query", nargs="+")
    p_ask.add_argument("--top-k", type=int, default=None)
    p_ask.add_argument("--label", default=None)
    p_ask.set_defaults(func=cmd_ask)

    # search
    p_search = sub.add_parser("search", help="Only retrieve (no LLM)")
    p_search.add_argument("query", nargs="+")
    p_search.add_argument("--top-k", type=int, default=None)
    p_search.add_argument("--label", default=None)
    p_search.set_defaults(func=cmd_search)

    # build-index
    p_build = sub.add_parser("build-index", help="Build FAISS index")
    p_build.set_defaults(func=cmd_build_index)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())