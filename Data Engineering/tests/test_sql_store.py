"""Tests for DuckDB storage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_engine.storage.sql_store import list_tables, query, save_to_duckdb


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "primary_category": ["cs.LG", "cs.LG", "cs.CL", "cs.CV"],
            "n_authors": [2, 3, 1, 4],
            "title_length": [10, 20, 15, 30],
        }
    )


class TestSaveToDuckDB:
    def test_creates_table(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        save_to_duckdb(_sample_df(), db, table="papers")
        assert "papers" in list_tables(db)

    def test_idempotent_replace(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        save_to_duckdb(_sample_df(), db, table="papers")
        save_to_duckdb(_sample_df(), db, table="papers")  # must not fail
        assert len(query(db, "SELECT * FROM papers")) == 4


class TestQuery:
    def test_groupby(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        save_to_duckdb(_sample_df(), db, table="papers")

        result = query(
            db,
            "SELECT primary_category, COUNT(*) AS n FROM papers GROUP BY 1 ORDER BY n DESC",
        )
        assert "primary_category" in result.columns
        assert result.iloc[0]["primary_category"] == "cs.LG"
        assert result.iloc[0]["n"] == 2

    def test_aggregation(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        save_to_duckdb(_sample_df(), db, table="papers")

        result = query(db, "SELECT AVG(n_authors) AS avg_authors FROM papers")
        assert abs(result.iloc[0]["avg_authors"] - 2.5) < 1e-9


class TestListTables:
    def test_empty_db(self, tmp_path: Path) -> None:
        db = tmp_path / "empty.db"
        # Create the DB file first
        save_to_duckdb(pd.DataFrame({"x": [1]}), db, table="tmp")
        assert "tmp" in list_tables(db)
