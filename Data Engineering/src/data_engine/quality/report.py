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


def build_html_report(
    schema_report,
    quality_report,
    *,
    pipeline_status: str = "success",
) -> str:
    """Build a minimal HTML report (no external CSS)."""
    status_color = "#28a745" if pipeline_status == "success" else "#dc3545"

    schema_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in {
            "Rows": schema_report.n_rows,
            "Columns": schema_report.n_cols,
            "Missing columns": len(schema_report.missing_columns),
            "Wrong dtypes": len(schema_report.wrong_dtypes),
            "Duplicate IDs": schema_report.duplicate_ids,
        }.items()
    )

    quality_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in {
            "Columns with NaN": sum(1 for v in quality_report.missing_by_col.values() if v > 0),
            "Columns with outliers": sum(
                1 for v in quality_report.outliers_by_col.values() if v > 0
            ),
            "Consistency issues": len(quality_report.consistency_issues),
        }.items()
    )

    cardinality_rows = "".join(
        f"<tr><td>{col}</td><td>{n}</td></tr>" for col, n in quality_report.cardinality.items()
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Data Pipeline Report</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1em; }}
  h1 {{ color: #333; }}
  .status {{ display: inline-block; padding: 0.3em 0.8em; border-radius: 6px;
             background: {status_color}; color: white; font-weight: bold; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ border: 1px solid #ddd; padding: 0.5em; text-align: left; }}
  th {{ background: #f4f4f4; }}
  h2 {{ border-bottom: 2px solid #eee; padding-bottom: 0.3em; margin-top: 2em; }}
</style>
</head>
<body>
  <h1>Data Pipeline Report</h1>
  <p>Pipeline status : <span class="status">{pipeline_status}</span></p>

  <h2>Schema Validation</h2>
  <table>{schema_rows}</table>

  <h2>Extended Quality Checks</h2>
  <table>{quality_rows}</table>

  <h2>Cardinality</h2>
  <table>
    <tr><th>Column</th><th>Unique values</th></tr>
    {cardinality_rows}
  </table>
</body>
</html>
"""
