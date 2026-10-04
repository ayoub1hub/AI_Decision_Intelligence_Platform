"""CSV ingestion.

Loads a CSV file into a pandas DataFrame. Handles common encoding
issues and validates that the file exists before opening it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

__all__ = ["load_csv"]

logger = logging.getLogger(__name__)


def load_csv(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    parse_dates: list[str] | None = None,
    dtype: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.
    encoding : str, default "utf-8"
        File encoding. Try "latin-1" if UTF-8 fails.
    parse_dates : list of str, optional
        Columns to parse as datetime.
    dtype : dict of str -> str, optional
        Force specific dtypes for specific columns (e.g. ``{"id": str}``).

    Returns
    -------
    pd.DataFrame
        Loaded data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    logger.info("Loading CSV: %s", path)
    df = pd.read_csv(
        path,
        encoding=encoding,
        parse_dates=parse_dates,
        dtype=dtype,
    )

    if df.empty:
        raise ValueError(f"CSV file is empty: {path}")

    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df
