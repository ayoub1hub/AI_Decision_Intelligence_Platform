"""arXiv API client (adapted from Data Engineering module).

Fetches papers from the arXiv API and returns a pandas DataFrame.
"""

from __future__ import annotations

import logging
import time
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd
import requests

__all__ = ["fetch_arxiv"]

logger = logging.getLogger(__name__)

_ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def _parse_entry(entry: ET.Element) -> dict[str, Any]:
    def _text(tag: str) -> str | None:
        el = entry.find(f"atom:{tag}", _ARXIV_NS)
        return el.text.strip() if el is not None and el.text else None

    authors = [
        a.find("atom:name", _ARXIV_NS).text.strip()
        for a in entry.findall("atom:author", _ARXIV_NS)
        if a.find("atom:name", _ARXIV_NS) is not None
    ]

    return {
        "id": _text("id"),
        "title": _text("title"),
        "abstract": _text("summary"),
        "authors": "; ".join(authors) if authors else None,
        "published_date": _text("published"),
    }


def fetch_arxiv(
    query: str = "cat:cs.LG",
    *,
    max_results: int = 50,
    timeout: int = 30,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch papers from the arXiv API."""
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Fetching arXiv: %s (attempt %d)", query, attempt)
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning("Attempt %d failed: %s", attempt, exc)
            if attempt < max_retries:
                time.sleep(2**attempt)
    else:
        raise last_exc or RuntimeError("Unknown fetch failure")

    root = ET.fromstring(resp.content)
    entries = root.findall("atom:entry", _ARXIV_NS)

    if not entries:
        return pd.DataFrame(columns=["id", "title", "abstract", "authors", "published_date"])

    rows = [_parse_entry(e) for e in entries]
    df = pd.DataFrame(rows)
    if "published_date" in df.columns:
        df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce", utc=True)
    return df
