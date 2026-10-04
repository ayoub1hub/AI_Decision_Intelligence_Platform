"""Tests for quality checks."""

from __future__ import annotations

import pandas as pd

from data_engine.validation.checks import run_checks


class TestMissingCheck:
    def test_detects_missing(self) -> None:
        df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", None]})
        report = run_checks(df)
        assert report.missing_by_col == {"a": 1, "b": 1}


class TestOutlierCheck:
    def test_detects_outliers(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]})
        report = run_checks(df)
        assert report.outliers_by_col.get("x", 0) >= 1


class TestConsistencyCheck:
    def test_updated_before_published(self) -> None:
        df = pd.DataFrame(
            {
                "published_date": pd.to_datetime(["2024-01-10"]),
                "updated_date": pd.to_datetime(["2024-01-01"]),  # avant !
            }
        )
        report = run_checks(df)
        assert not report.passed
        assert any("updated_date" in i for i in report.consistency_issues)

    def test_short_title(self) -> None:
        df = pd.DataFrame({"title": ["Hi"]})
        report = run_checks(df)
        assert any("title" in i for i in report.consistency_issues)


class TestCardinality:
    def test_counts_unique(self) -> None:
        df = pd.DataFrame({"cat": ["a", "a", "b", "c"]})
        report = run_checks(df)
        assert report.cardinality["cat"] == 3
