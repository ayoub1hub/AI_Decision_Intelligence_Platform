"""Command-line interface for the data engine.

Usage:
    python -m data_engine run --config configs/dev.yaml
    python -m data_engine download-arxiv --query "cat:cs.LG" --max 100
    python -m data_engine query --db data/processed/arxiv_clean.db --sql "SELECT 1"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from requests import RequestException

from data_engine.ingestion.api_loader import fetch_arxiv
from data_engine.logging_config import setup_logging
from data_engine.pipeline import run_pipeline
from data_engine.storage.sql_store import query as sql_query

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Sub-commands
# ----------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    """Run the full pipeline."""
    result = run_pipeline(args.config)
    if result.get("status") != "success":
        return 1
    print("\n✅ Pipeline completed successfully.")
    print(f"   Parquet : {result.get('output_parquet')}")
    print(f"   DuckDB  : {result.get('output_db')}")
    print(f"   Report  : {result.get('report_md')}")  # ← md
    print(f"   HTML    : {result.get('report_html')}")  # ← html
    return 0


def cmd_download_arxiv(args: argparse.Namespace) -> int:
    """Download papers from the arXiv API."""
    setup_logging(level="INFO")
    try:
        df = fetch_arxiv(query=args.query, max_results=args.max)
    except (RequestException, ValueError) as exc:
        logger.error("Download failed: %s", exc)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"✅ Saved {len(df)} papers to {out}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    """Run a SQL query against a DuckDB file."""
    setup_logging(level="WARNING")
    result = sql_query(args.db, args.sql)
    print(result.to_string(index=False))
    return 0


# ----------------------------------------------------------------------
# Argument parser
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="data_engine",
        description="Data Intelligence Pipeline CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Run the full pipeline")
    p_run.add_argument("--config", default="configs/dev.yaml", help="Path to config YAML")
    p_run.set_defaults(func=cmd_run)

    # download-arxiv
    p_dl = sub.add_parser("download-arxiv", help="Download papers from arXiv")
    p_dl.add_argument("--query", default="cat:cs.LG", help="arXiv query")
    p_dl.add_argument("--max", type=int, default=100, help="Number of papers")
    p_dl.add_argument("--out", default="data/raw/arxiv_fetched.csv", help="Output CSV")
    p_dl.set_defaults(func=cmd_download_arxiv)

    # query
    p_q = sub.add_parser("query", help="Run SQL against a DuckDB file")
    p_q.add_argument("--db", required=True, help="Path to DuckDB file")
    p_q.add_argument("--sql", required=True, help="SQL query")
    p_q.set_defaults(func=cmd_query)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
