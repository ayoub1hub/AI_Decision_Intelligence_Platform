"""Feature engineering for arXiv papers.

Derives useful columns from raw fields:

- arxiv_id       : clean identifier (without URL prefix)
- title_length   : number of characters in the title
- abstract_length: number of characters in the abstract
- n_authors      : number of authors
- n_categories   : number of categories
- primary_category: first category (e.g. "cs.LG")
- days_to_update : days between published and updated
- published_year : year of publication
"""

from __future__ import annotations

import logging

import pandas as pd

__all__ = ["engineer_features"]

logger = logging.getLogger(__name__)


def _clean_arxiv_id(s: pd.Series) -> pd.Series:
    """Extract '2610.02207v1' from 'http://arxiv.org/abs/2610.02207v1'."""
    return s.astype(str).str.extract(r"abs/(.+?)$", expand=False)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to an arXiv DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least: id, title, abstract, authors,
        categories, published_date, updated_date.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with new feature columns appended.
    """
    df = df.copy()
    n0 = len(df.columns)

    # 1. Clean arXiv ID
    if "id" in df.columns:
        df["arxiv_id"] = _clean_arxiv_id(df["id"])
    else:
        df["arxiv_id"] = None

    # 2. Title & abstract lengths
    df["title_length"] = df["title"].fillna("").astype(str).str.len()
    df["abstract_length"] = df["abstract"].fillna("").astype(str).str.len()

    # 3. Author count (careful: NaN → 0, else count(';') + 1)
    authors_clean = df["authors"].fillna("").astype(str)
    df["n_authors"] = authors_clean.apply(lambda s: 0 if s.strip() == "" else s.count(";") + 1)

    # 4. Category count
    cats_clean = df["categories"].fillna("").astype(str)
    df["n_categories"] = cats_clean.apply(lambda s: 0 if s.strip() == "" else s.count(";") + 1)

    # 5. Primary category (first category)
    df["primary_category"] = cats_clean.str.split(";").str[0].str.strip().replace("", None)

    # 6. Days to update
    if "published_date" in df.columns and "updated_date" in df.columns:
        delta = df["updated_date"] - df["published_date"]
        df["days_to_update"] = delta.dt.days.fillna(0).astype(int)
    else:
        df["days_to_update"] = 0

    # 7. Published year
    if "published_date" in df.columns:
        df["published_year"] = df["published_date"].dt.year
    else:
        df["published_year"] = None

    n_new = len(df.columns) - n0
    logger.info("Engineered %d new features", n_new)
    return df
