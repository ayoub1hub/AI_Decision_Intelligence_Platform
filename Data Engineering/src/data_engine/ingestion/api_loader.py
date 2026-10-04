"""API ingestion.

Fetches data from a REST API and returns a pandas DataFrame. Uses
`requests` with retries, timeouts, and pagination support.

Currently supports the arXiv API (http://export.arxiv.org/api/query).
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

# arXiv API namespaces (Atom feed)
_ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def _parse_arxiv_entry(entry: ET.Element) -> dict[str, Any]:
    """Parse a single Atom <entry> into a flat dict."""

    def _text(tag: str, ns: str = "atom") -> str | None:
        el = entry.find(f"{ns}:{tag}", _ARXIV_NS)
        return el.text.strip() if el is not None and el.text else None

    # Authors
    authors = [
        a.find("atom:name", _ARXIV_NS).text.strip()
        for a in entry.findall("atom:author", _ARXIV_NS)
        if a.find("atom:name", _ARXIV_NS) is not None
    ]

    # Categories
    categories = [
        c.attrib["term"] for c in entry.findall("atom:category", _ARXIV_NS) if "term" in c.attrib
    ]

    return {
        "id": _text("id"),
        "title": _text("title"),
        "abstract": _text("summary"),
        "authors": "; ".join(authors) if authors else None,
        "categories": "; ".join(categories) if categories else None,
        "published_date": _text("published"),
        "updated_date": _text("updated"),
    }


def fetch_arxiv(
    query: str = "cat:cs.LG",
    *,
    max_results: int = 50,
    start: int = 0,
    timeout: int = 30,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch papers from the arXiv API.

    Parameters
    ----------
    query : str, default "cat:cs.LG"
        arXiv query string (e.g. ``"cat:cs.LG"``, ``"all:transformer"``).
    max_results : int, default 50
        Number of papers to fetch (arXiv caps at 2000 per request).
    start : int, default 0
        Offset for pagination.
    timeout : int, default 30
        HTTP timeout in seconds.
    max_retries : int, default 3
        Retry count on transient failures.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: id, title, abstract, authors,
        categories, published_date, updated_date.

    Raises
    ------
    requests.HTTPError
        On non-2xx HTTP responses after retries.
    ValueError
        If the response cannot be parsed.
    """
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Fetching arXiv: query=%s, max=%d, attempt=%d",
                query,
                max_results,
                attempt,
            )
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning("Attempt %d failed: %s", attempt, exc)
            if attempt < max_retries:
                time.sleep(2**attempt)  # exponential backoff
    else:
        raise last_exc if last_exc else RuntimeError("Unknown fetch failure")

    # Parse XML
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as exc:
        raise ValueError(f"Failed to parse arXiv response: {exc}") from exc

    entries = root.findall("atom:entry", _ARXIV_NS)
    if not entries:
        logger.warning("No entries returned for query: %s", query)
        return pd.DataFrame(
            columns=[
                "id",
                "title",
                "abstract",
                "authors",
                "categories",
                "published_date",
                "updated_date",
            ]
        )

    rows = [_parse_arxiv_entry(e) for e in entries]
    df = pd.DataFrame(rows)

    # Convert dates
    for col in ("published_date", "updated_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    logger.info("Fetched %d papers from arXiv", len(df))
    return df
