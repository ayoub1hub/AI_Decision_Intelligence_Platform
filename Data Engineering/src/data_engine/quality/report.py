"""Quality report generation.

Combines schema validation and quality checks into a single
Markdown report, saved to disk.
"""

from __future__ import annotations

import logging
from pathlib import Path

from data_engine.validation.checks import QualityReport
from data_engine.validation.schema import ValidationReport

__all__ = ["build_report", "save_report"]

logger = logging.getLogger(__name__)


def build_report(
    schema_report: ValidationReport,
    quality_report: QualityReport,
    *,
    pipeline_status: str = "success",
) -> str:
    """Build a Markdown report from both validation and quality reports."""
    lines = [
        "# Data Pipeline Report",
        "",
        f"**Pipeline status** : `{pipeline_status}`",
        "",
        "## Schema validation",
        "",
        "```",
        schema_report.summary(),
        "```",
        "",
        "## Extended quality checks",
        "",
        "```",
        quality_report.summary(),
        "```",
        "",
    ]

    if quality_report.missing_by_col:
        lines += [
            "## Missing values by column",
            "",
            "| Column | Missing |",
            "|---|---|",
        ]
        for col, n in quality_report.missing_by_col.items():
            lines.append(f"| {col} | {n} |")
        lines.append("")

    if quality_report.outliers_by_col:
        lines += [
            "## Outliers by column (IQR rule)",
            "",
            "| Column | Outliers |",
            "|---|---|",
        ]
        for col, n in quality_report.outliers_by_col.items():
            lines.append(f"| {col} | {n} |")
        lines.append("")

    if quality_report.cardinality:
        lines += [
            "## Cardinality",
            "",
            "| Column | Unique values |",
            "|---|---|",
        ]
        for col, n in quality_report.cardinality.items():
            lines.append(f"| {col} | {n} |")
        lines.append("")

    return "\n".join(lines)


def save_report(content: str, path: str | Path) -> None:
    """Save report content to a file (creating parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("Report saved to %s", path)
