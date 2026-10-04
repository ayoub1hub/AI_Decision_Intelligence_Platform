"""Cleaning transformations.

Basic cleaning steps for an ingested DataFrame:

- Drop fully duplicated rows.
- Strip whitespace from string columns.
- Drop rows with missing required values (configurable).
"""

from __future__ import annotations

import logging

import pandas as pd

__all__ = ["clean_basic"]

logger = logging.getLogger(__name__)


def clean_basic(
    df: pd.DataFrame,
    *,
    required_columns: list[str] | None = None,
    strip_strings: bool = True,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """Apply basic cleaning steps.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    required_columns : list of str, optional
        Columns that must not contain NaN. Rows with NaN in these are
        dropped.
    strip_strings : bool, default True
        Strip leading/trailing whitespace from all string columns.
    drop_duplicates : bool, default True
        Drop fully duplicated rows.

    Returns
    -------
    pd.DataFrame
        Cleaned copy.
    """
    df = df.copy()
    n0 = len(df)

    if strip_strings:
        text_cols = df.select_dtypes(include=["object", "string", "str"]).columns
        for col in text_cols:
            df[col] = df[col].astype(str).str.strip()

    if drop_duplicates:
        df = df.drop_duplicates()
        if len(df) < n0:
            logger.info("Dropped %d duplicate rows", n0 - len(df))

    if required_columns:
        n1 = len(df)
        df = df.dropna(subset=required_columns)
        if len(df) < n1:
            logger.info("Dropped %d rows with missing required values", n1 - len(df))

    logger.info("Cleaning done: %d rows remaining (from %d)", len(df), n0)
    return df.reset_index(drop=True)
