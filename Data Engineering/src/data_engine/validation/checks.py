"""Data quality checks.

Complementary to schema validation: checks that go beyond structural
correctness (missing values, outliers, cardinality, cross-column
consistency).

Each check returns a list of issues (as strings) so all problems can
be collected in one pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

__all__ = ["QualityReport", "run_checks"]

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Aggregated quality report."""

    n_rows: int = 0
    n_cols: int = 0
    missing_by_col: dict[str, int] = field(default_factory=dict)
    outliers_by_col: dict[str, int] = field(default_factory=dict)
    cardinality: dict[str, int] = field(default_factory=dict)
    consistency_issues: list[str] = field(default_factory=list)
    passed: bool = True

    def summary(self) -> str:
        lines = [
            "Extended Quality Report",
            "─" * 40,
            f"Rows                : {self.n_rows}",
            f"Columns             : {self.n_cols}",
            f"Columns with NaN    : {sum(1 for v in self.missing_by_col.values() if v > 0)}",
            f"Columns with outliers: {sum(1 for v in self.outliers_by_col.values() if v > 0)}",
            f"Consistency issues  : {len(self.consistency_issues)}",
            f"Status              : {'PASS' if self.passed else 'WARN'}",
        ]
        if self.consistency_issues:
            lines.append("")
            lines.append("Issues:")
            for issue in self.consistency_issues:
                lines.append(f"  - {issue}")
        return "\n".join(lines)


def _check_missing(df: pd.DataFrame) -> dict[str, int]:
    """Missing value count per column (only columns with > 0)."""
    missing = df.isna().sum()
    return {col: int(n) for col, n in missing.items() if n > 0}


def _check_outliers(df: pd.DataFrame, iqr_factor: float = 1.5) -> dict[str, int]:
    """Count outliers per numeric column using the IQR rule."""
    outliers: dict[str, int] = {}
    for col in df.select_dtypes(include="number").columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        low = q1 - iqr_factor * iqr
        high = q3 + iqr_factor * iqr
        n = int(((df[col] < low) | (df[col] > high)).sum())
        if n > 0:
            outliers[col] = n
    return outliers


def _check_cardinality(df: pd.DataFrame) -> dict[str, int]:
    """Number of unique values per column."""
    return {col: int(df[col].nunique(dropna=True)) for col in df.columns}


def _check_consistency(df: pd.DataFrame) -> list[str]:
    """Cross-column consistency checks."""
    issues: list[str] = []

    # Dates: updated >= published
    if "published_date" in df.columns and "updated_date" in df.columns:
        bad = (df["updated_date"] < df["published_date"]).sum()
        if bad > 0:
            issues.append(f"{int(bad)} rows have updated_date < published_date")

    # Titles shouldn't be too short
    if "title" in df.columns:
        short = (df["title"].fillna("").str.len() < 5).sum()
        if short > 0:
            issues.append(f"{int(short)} rows have title shorter than 5 chars")

    return issues


def run_checks(df: pd.DataFrame) -> QualityReport:
    """Run all quality checks on a DataFrame."""
    report = QualityReport(n_rows=len(df), n_cols=len(df.columns))

    report.missing_by_col = _check_missing(df)
    report.outliers_by_col = _check_outliers(df)
    report.cardinality = _check_cardinality(df)
    report.consistency_issues = _check_consistency(df)

    # Global status: WARN if any consistency issue
    report.passed = len(report.consistency_issues) == 0

    logger.info("Quality checks complete: %s", "PASS" if report.passed else "WARN")
    return report
