"""Download a labeled arXiv dataset for classification.

Fetches N papers from several categories, adds a `label` column,
and saves a combined CSV to data/raw/.

Usage:
    python scripts/generate_dataset.py --per-cat 50 --out data/raw/arxiv_labeled.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from nlp_engine.ingestion.arxiv_client import fetch_arxiv  # ← on va créer ça
from nlp_engine.logging_config import setup_logging  # ← et ça

logger = logging.getLogger(__name__)

# Categories to fetch (balanced dataset)
CATEGORIES = [
    "cs.LG",  # Machine Learning
    "cs.CL",  # Computation and Language
    "cs.CV",  # Computer Vision
    "cs.AI",  # Artificial Intelligence
    "cs.NE",  # Neural and Evolutionary Computing
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-cat", type=int, default=50)
    parser.add_argument("--out", default="data/raw/arxiv_labeled.csv")
    args = parser.parse_args()

    setup_logging(level="INFO")

    all_dfs = []
    for cat in CATEGORIES:
        logger.info("Fetching %d papers from %s", args.per_cat, cat)
        df = fetch_arxiv(query=f"cat:{cat}", max_results=args.per_cat)
        df["label"] = cat
        all_dfs.append(df)
        logger.info("Got %d papers from %s", len(df), cat)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset="id")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False, encoding="utf-8")

    logger.info("Saved %d labeled papers to %s", len(combined), out)
    print(f"\n✅ {len(combined)} papers saved to {out}")
    print(f"   Categories: {combined['label'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
