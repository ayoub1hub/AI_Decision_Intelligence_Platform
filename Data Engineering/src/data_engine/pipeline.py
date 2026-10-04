"""Pipeline orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path

from data_engine.config import load_config
from data_engine.ingestion.csv_loader import load_csv
from data_engine.logging_config import setup_logging
from data_engine.quality.report import build_html_report, build_report, save_report
from data_engine.storage.sql_store import save_to_duckdb
from data_engine.transformation.cleaning import clean_basic
from data_engine.transformation.features import engineer_features
from data_engine.validation.checks import run_checks
from data_engine.validation.schema import load_schema, validate

__all__ = ["run_pipeline"]

logger = logging.getLogger(__name__)


def run_pipeline(config_path: str | Path = "configs/dev.yaml") -> dict:
    """Execute the full pipeline end to end."""
    # 1. Config
    cfg = load_config(config_path)

    # 2. Logging
    setup_logging(
        level=cfg.logging.level,
        fmt=cfg.logging.format,
        datefmt=cfg.logging.datefmt,
        log_file=cfg.logging.log_file,
    )
    logger.info("Starting pipeline (env=%s)", cfg.environment)

    # 3. Ingestion
    raw_path = Path(cfg.data.raw_dir) / cfg.data.input_file
    df = load_csv(
        raw_path,
        parse_dates=cfg.data.parse_dates,
        dtype=cfg.data.dtypes,
    )

    # 4. Schema validation
    schema = load_schema(cfg.schema_.path)
    schema_report = validate(df, schema)
    logger.info("\n%s", schema_report.summary())

    if not schema_report.passed:
        logger.error("Schema validation FAILED. Aborting pipeline.")
        return {"status": "failed", "report": schema_report.summary()}

    # 5. Quality checks
    quality_report = run_checks(df)
    logger.info("\n%s", quality_report.summary())

    # 6. Cleaning
    required = [c.name for c in schema.columns if c.required]
    df_clean = clean_basic(df, required_columns=required)

    # 7. Feature engineering
    df_features = engineer_features(df_clean)
    logger.info("Features: %s", list(df_features.columns))

    # 8. Storage — Parquet
    out_path = Path(cfg.data.processed_dir) / cfg.data.output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(out_path, index=False)
    logger.info("Saved processed data to %s", out_path)

    # 9. Storage — DuckDB
    db_path = out_path.with_suffix(".db")
    save_to_duckdb(df_features, db_path, table="papers")
    logger.info("Saved to DuckDB: %s", db_path)

    # 10. Reports (Markdown + HTML)
    report_md = build_report(schema_report, quality_report, pipeline_status="success")
    report_path = Path("logs") / "quality_report.md"
    save_report(report_md, report_path)

    report_html = build_html_report(schema_report, quality_report, pipeline_status="success")
    html_path = Path("logs") / "quality_report.html"
    save_report(report_html, html_path)

    logger.info("Reports saved: %s, %s", report_path, html_path)
    logger.info("Pipeline completed successfully")

    return {
        "status": "success",
        "rows_in": len(df),
        "rows_out": len(df_features),
        "output_parquet": str(out_path),
        "output_db": str(db_path),
        "report_md": str(report_path),
        "report_html": str(html_path),
    }


if __name__ == "__main__":
    run_pipeline()
