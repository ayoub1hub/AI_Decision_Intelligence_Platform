"""Dataset schema definition and validation.

Loads a schema from a YAML file and checks a DataFrame against it.
Returns a report with all issues found (missing columns, wrong types,
etc.) instead of crashing on the first error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

__all__ = ["ColumnSpec", "DatasetSchema", "ValidationReport", "load_schema", "validate"]

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Schema model
# ----------------------------------------------------------------------
@dataclass
class ColumnSpec:
    """Specification for one column."""

    name: str
    dtype: str
    required: bool = True
    unique: bool = False


@dataclass
class DatasetSchema:
    """A collection of column specs."""

    columns: list[ColumnSpec]

    def names(self) -> set[str]:
        return {c.name for c in self.columns}

    def required_names(self) -> set[str]:
        return {c.name for c in self.columns if c.required}


@dataclass
class ValidationReport:
    """Result of a validation run."""

    n_rows: int = 0
    n_cols: int = 0
    missing_columns: list[str] = field(default_factory=list)
    wrong_dtypes: list[tuple[str, str, str]] = field(default_factory=list)
    duplicate_ids: int = 0
    passed: bool = True

    def summary(self) -> str:
        lines = [
            "Dataset Quality Report",
            "─" * 40,
            f"Rows            : {self.n_rows}",
            f"Columns         : {self.n_cols}",
            f"Missing columns : {len(self.missing_columns)}",
            f"Wrong dtypes    : {len(self.wrong_dtypes)}",
            f"Duplicate IDs   : {self.duplicate_ids}",
            f"Status          : {'PASS' if self.passed else 'FAIL'}",
        ]
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_schema(path: str | Path) -> DatasetSchema:
    """Load a DatasetSchema from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    columns = [ColumnSpec(**c) for c in raw.get("columns", [])]
    logger.info("Loaded schema with %d columns", len(columns))
    return DatasetSchema(columns=columns)


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def _is_dtype_ok(col_spec_dtype: str, actual: str) -> bool:
    """Check if the actual pandas dtype matches the expected one.

    Handles pandas 2.x StringDtype (`str`) as equivalent to `object`.
    """
    actual = actual.lower()

    if col_spec_dtype == "string":
        # pandas uses either "object" or "str" (StringDtype) for text
        return "object" in actual or actual.startswith("str")

    if col_spec_dtype == "int":
        return "int" in actual

    if col_spec_dtype == "float":
        return "float" in actual

    if col_spec_dtype == "datetime":
        return "datetime" in actual

    if col_spec_dtype == "bool":
        return "bool" in actual

    return True  # unknown dtype → don't fail


def validate(df: pd.DataFrame, schema: DatasetSchema) -> ValidationReport:
    """Validate a DataFrame against a schema."""
    report = ValidationReport(n_rows=len(df), n_cols=len(df.columns))

    # 1. Missing columns
    missing = schema.required_names() - set(df.columns)
    report.missing_columns = sorted(missing)

    # 2. Wrong dtypes
    for col_spec in schema.columns:
        if col_spec.name not in df.columns:
            continue
        actual = str(df[col_spec.name].dtype)
        if not _is_dtype_ok(col_spec.dtype, actual):
            report.wrong_dtypes.append((col_spec.name, col_spec.dtype, actual))

    # 3. Unique columns
    for col_spec in schema.columns:
        if col_spec.unique and col_spec.name in df.columns:
            n_dup = int(df[col_spec.name].duplicated().sum())
            report.duplicate_ids += n_dup

    # 4. Global pass / fail
    report.passed = (
        not report.missing_columns and not report.wrong_dtypes and report.duplicate_ids == 0
    )

    logger.info("Validation complete: %s", "PASS" if report.passed else "FAIL")
    return report
