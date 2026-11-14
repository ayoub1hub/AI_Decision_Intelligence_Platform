"""Evaluate the RAG pipeline on a test set.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --questions data/processed/test_questions.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys

from rag_engine.config import load_config
from rag_engine.evaluation import evaluate, load_test_questions
from rag_engine.logging_config import setup_logging
from rag_engine.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="data/processed/test_questions.yaml")
    parser.add_argument("--config", default="configs/dev.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_file=cfg.logging.log_file)

    # Load questions
    questions = load_test_questions(args.questions)
    logger.info("Loaded %d test questions", len(questions))

    # Build pipeline
    pipeline = RAGPipeline.from_config(args.config)

    # Evaluate
    report = evaluate(pipeline, questions)

    # Print
    print("\n" + "=" * 60)
    print(report.summary())
    print("=" * 60)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
