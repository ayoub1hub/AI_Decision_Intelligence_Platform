"""Download a sample of arXiv papers to data/raw/.

Usage:
    python scripts/download_arxiv.py --query "cat:cs.LG" --max 200
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from data_engine.ingestion.api_loader import fetch_arxiv
from data_engine.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download arXiv papers.")
    parser.add_argument("--query", default="cat:cs.LG", help="arXiv query")
    parser.add_argument("--max", type=int, default=200, help="Number of papers")
    parser.add_argument(
        "--out", default="data/raw/arxiv_fetched.csv", help="Output CSV path"
    )
    args = parser.parse_args()

    setup_logging(level="INFO")

    try:
        df = fetch_arxiv(query=args.query, max_results=args.max)
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    logger.info("Saved %d papers to %s", len(df), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())