"""Tests for the ingestion + validation layers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_engine.ingestion.csv_loader import load_csv
from data_engine.validation.schema import (
    ColumnSpec,
    DatasetSchema,
    validate,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Write a small CSV file and return its path."""
    csv = tmp_path / "sample.csv"
    csv.write_text(
        "id,title,abstract,published_date\n"
        "1,Paper A,Abstract A,2024-01-01\n"
        "2,Paper B,Abstract B,2024-01-02\n"
        "3,Paper C,Abstract C,2024-01-03\n",
        encoding="utf-8",
    )
    return csv


@pytest.fixture
def sample_schema() -> DatasetSchema:
    return DatasetSchema(
        columns=[
            ColumnSpec("id", "string", required=True, unique=True),
            ColumnSpec("title", "string", required=True),
            ColumnSpec("abstract", "string", required=True),
            ColumnSpec("published_date", "datetime", required=False),
        ]
    )


# ----------------------------------------------------------------------
# Ingestion
# ----------------------------------------------------------------------
class TestLoadCSV:
    def test_loads_correctly(self, sample_csv: Path) -> None:
        df = load_csv(sample_csv)
        assert len(df) == 3
        assert list(df.columns) == ["id", "title", "abstract", "published_date"]

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_csv("nonexistent.csv")

    def test_empty_file(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.csv"
        empty.write_text("id,title\n", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            load_csv(empty)

    def test_parse_dates(self, sample_csv: Path) -> None:
        df = load_csv(sample_csv, parse_dates=["published_date"])
        assert "datetime" in str(df["published_date"].dtype)


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
class TestValidation:
    def test_valid_dataframe(self, sample_csv: Path, sample_schema: DatasetSchema) -> None:
        df = load_csv(
            sample_csv,
            parse_dates=["published_date"],
            dtype={"id": str},
        )
        report = validate(df, sample_schema)
        assert report.passed
        assert report.missing_columns == []
        assert report.duplicate_ids == 0

    def test_missing_column(self, sample_schema: DatasetSchema) -> None:
        df = pd.DataFrame({"id": ["1"], "title": ["a"]})
        report = validate(df, sample_schema)
        assert not report.passed
        assert "abstract" in report.missing_columns

    def test_duplicate_ids(self, sample_schema: DatasetSchema) -> None:
        df = pd.DataFrame(
            {
                "id": ["1", "1"],
                "title": ["a", "b"],
                "abstract": ["x", "y"],
                "published_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            }
        )
        report = validate(df, sample_schema)
        assert not report.passed
        assert report.duplicate_ids == 1

    def test_report_summary(self, sample_csv: Path, sample_schema: DatasetSchema) -> None:
        df = load_csv(
            sample_csv,
            parse_dates=["published_date"],
            dtype={"id": str},
        )
        report = validate(df, sample_schema)
        summary = report.summary()
        assert "Dataset Quality Report" in summary
        assert "PASS" in summary
