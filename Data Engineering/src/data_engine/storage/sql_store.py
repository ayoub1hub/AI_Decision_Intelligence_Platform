"""SQL storage with DuckDB.

DuckDB is an in-process analytical database — no server needed.
It speaks full SQL and reads/writes Parquet natively.

Usage:

    from data_engine.storage.sql_store import save_to_duckdb, query

    save_to_duckdb(df, "data/processed/arxiv.db", table="papers")
    top = query("data/processed/arxiv.db",
                "SELECT primary_category, COUNT(*) AS n FROM papers GROUP BY 1 ORDER BY n DESC")
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

__all__ = ["list_tables", "query", "save_to_duckdb"]

logger = logging.getLogger(__name__)


def save_to_duckdb(
    df: pd.DataFrame,
    db_path: str | Path,
    *,
    table: str = "papers",
    replace: bool = True,
) -> None:
    """Save a DataFrame to a DuckDB table.

    Parameters
    ----------
    df : pd.DataFrame
        Data to save.
    db_path : str or Path
        Path to the DuckDB file (created if not exists).
    table : str, default "papers"
        Table name.
    replace : bool, default True
        Drop the table first if it exists.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(db_path)) as conn:
        if replace:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute(f"CREATE TABLE {table} AS SELECT * FROM df")
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    logger.info("Saved %d rows to %s (table=%s)", n, db_path, table)


def query(db_path: str | Path, sql: str) -> pd.DataFrame:
    """Run a SQL query and return the result as a DataFrame.

    Parameters
    ----------
    db_path : str or Path
        Path to the DuckDB file.
    sql : str
        SQL query.

    Returns
    -------
    pd.DataFrame
        Query result.
    """
    with duckdb.connect(str(db_path), read_only=True) as conn:
        result = conn.execute(sql).fetchdf()

    logger.info("Query returned %d rows", len(result))
    return result


def list_tables(db_path: str | Path) -> list[str]:
    """List all tables in the DuckDB file."""
    with duckdb.connect(str(db_path), read_only=True) as conn:
        rows = conn.execute("SHOW TABLES").fetchall()
    return [r[0] for r in rows]
