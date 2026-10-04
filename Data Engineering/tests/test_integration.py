"""Integration tests — end-to-end pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_engine.pipeline import run_pipeline
from data_engine.storage.sql_store import list_tables, query


@pytest.fixture
def mini_project(tmp_path: Path, monkeypatch) -> Path:
    """Create a minimal project structure and return its root."""
    root = tmp_path / "mini_project"
    root.mkdir()

    # Data directories
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "processed").mkdir(parents=True)
    (root / "configs" / "schemas").mkdir(parents=True)
    (root / "logs").mkdir()

    # Sample CSV
    csv = root / "data" / "raw" / "sample.csv"
    csv.write_text(
        "id,title,abstract,authors,categories,published_date,updated_date\n"
        "2401.00001,A,Abstract A,Alice;Bob,cs.LG,2024-01-01,2024-01-05\n"
        "2401.00002,B,Abstract B,Carol,cs.CL,2024-01-02,2024-01-02\n"
        "2401.00003,C,Abstract C,Dave,cs.AI,2024-01-03,2024-01-03\n",
        encoding="utf-8",
    )

    # Schema
    schema = root / "configs" / "schemas" / "v1.yaml"
    schema.write_text(
        "columns:\n"
        "  - {name: id, dtype: string, required: true, unique: true}\n"
        "  - {name: title, dtype: string, required: true}\n"
        "  - {name: abstract, dtype: string, required: true}\n"
        "  - {name: published_date, dtype: datetime, required: false}\n"
        "  - {name: updated_date, dtype: datetime, required: false}\n",
        encoding="utf-8",
    )

    # Config
    cfg = root / "configs" / "dev.yaml"
    cfg.write_text(
        "environment: dev\n"
        "data:\n"
        "  raw_dir: data/raw\n"
        "  processed_dir: data/processed\n"
        "  input_file: sample.csv\n"
        "  output_file: sample.parquet\n"
        "  dtypes: {id: string}\n"
        "  parse_dates: [published_date, updated_date]\n"
        "schema:\n"
        "  path: configs/schemas/v1.yaml\n"
        "logging:\n"
        "  level: INFO\n"
        "  log_file: null\n",
        encoding="utf-8",
    )

    # Change working directory so relative paths work
    monkeypatch.chdir(root)
    return root


class TestEndToEnd:
    def test_pipeline_creates_outputs(self, mini_project: Path) -> None:
        result = run_pipeline("configs/dev.yaml")

        assert result["status"] == "success"
        assert result["rows_in"] == 3
        assert result["rows_out"] == 3

        parquet_path = Path(result["output_parquet"])
        db_path = Path(result["output_db"])
        assert parquet_path.exists()
        assert db_path.exists()

    def test_features_added(self, mini_project: Path) -> None:
        result = run_pipeline("configs/dev.yaml")
        df = pd.read_parquet(result["output_parquet"])

        for col in [
            "arxiv_id",
            "title_length",
            "abstract_length",
            "n_authors",
            "n_categories",
            "primary_category",
            "days_to_update",
            "published_year",
        ]:
            assert col in df.columns, f"Missing feature: {col}"

        # Sanity checks
        assert df["n_authors"].tolist() == [2, 1, 1]
        assert df["primary_category"].tolist() == ["cs.LG", "cs.CL", "cs.AI"]

    def test_duckdb_queryable(self, mini_project: Path) -> None:
        result = run_pipeline("configs/dev.yaml")
        db = result["output_db"]

        assert "papers" in list_tables(db)
        n = query(db, "SELECT COUNT(*) AS n FROM papers").iloc[0]["n"]
        assert n == 3

        by_cat = query(
            db,
            "SELECT primary_category, COUNT(*) AS n FROM papers GROUP BY 1 ORDER BY 1",
        )
        assert list(by_cat["primary_category"]) == ["cs.AI", "cs.CL", "cs.LG"]
