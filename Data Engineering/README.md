# Data Intelligence Pipeline

A production-grade Data Engineering module for the **AI Decision Intelligence Platform**. Handles ingestion, validation, cleaning, feature engineering, and storage.

## Pipeline

```
CSV / API
   │
   ▼
Ingestion (pandas)
   │
   ▼
Schema validation (Pydantic)
   │
   ▼
Quality checks (missing, outliers, consistency)
   │
   ▼
Cleaning (strip, dedup, dropna)
   │
   ▼
Feature engineering (8 new columns)
   │
   ▼
Storage ─┬─► Parquet (files)
         └─► DuckDB (SQL queries)
   │
   ▼
Quality report (Markdown + HTML)
```

## Features

- **Ingestion**: CSV, arXiv API (with retries + backoff)
- **Validation**: schema (columns, dtypes, uniqueness) + extended checks (missing, outliers, consistency, cardinality)
- **Feature engineering**: domain-specific features (e.g. arXiv ID extraction, lengths, categories)
- **Storage**: dual — Parquet (efficient files) + DuckDB (SQL analytics)
- **CLI**: `data_engine run`, `data_engine download-arxiv`, `data_engine query`
- **Reports**: Markdown + HTML quality reports
- **CI**: GitHub Actions (lint + tests)

## Quick Start

```bash
pip install -e ".[dev]"

# Run the full pipeline
python -m data_engine run --config configs/dev.yaml

# Or use the CLI
python -m data_engine download-arxiv --query "cat:cs.LG" --max 100
python -m data_engine run
python -m data_engine query --db data/processed/arxiv_clean.db --sql "SELECT * FROM papers LIMIT 5"
```

## CLI

```bash
# Run the full pipeline
python -m data_engine run --config configs/dev.yaml

# Download papers from arXiv
python -m data_engine download-arxiv --query "cat:cs.LG" --max 100

# Query the DuckDB database
python -m data_engine query --db data/processed/arxiv_clean.db --sql "SELECT primary_category, COUNT(*) FROM papers GROUP BY 1"
```

## Examples

### Programmatic use

```python
from data_engine.pipeline import run_pipeline

result = run_pipeline("configs/dev.yaml")
print(result["rows_in"])
print(result["rows_out"])
print(result["output_parquet"])
print(result["output_db"])
```

### Query after pipeline

```python
from data_engine.storage.sql_store import query

db = "data/processed/arxiv_clean.db"
result = query(db, """
    SELECT primary_category, COUNT(*) AS n, AVG(n_authors) AS avg_authors
    FROM papers
    GROUP BY 1
    ORDER BY n DESC
""")
print(result)
```

## Configuration

Config files in `configs/`:

- `dev.yaml` — development (local paths, verbose logs)
- `prod.yaml` — production (absolute paths, warning-level logs)
- `schemas/arxiv_v1.yaml` — schema for arXiv papers

Example `configs/dev.yaml`:

```yaml
environment: dev
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  input_file: "arxiv_sample.csv"
  output_file: "arxiv_clean.parquet"
  dtypes:
    id: "string"
  parse_dates:
    - published_date
    - updated_date
schema:
  path: "configs/schemas/arxiv_v1.yaml"
logging:
  level: "INFO"
  log_file: "logs/pipeline.log"
```

## Structure

```
src/data_engine/
├── config.py                 — load YAML config
├── logging_config.py         — setup logging
├── pipeline.py               — orchestrator
├── cli.py                    — CLI entrypoint
├── __main__.py               — python -m data_engine
├── ingestion/
│   ├── csv_loader.py         — load CSV
│   └── api_loader.py         — fetch from arXiv
├── validation/
│   ├── schema.py             — schema validation
│   └── checks.py             — quality checks
├── transformation/
│   ├── cleaning.py           — strip, dedup, dropna
│   └── features.py           — 8 features for arXiv
├── storage/
│   └── sql_store.py          — DuckDB storage + queries
└── quality/
    └── report.py             — Markdown + HTML reports
```

## Tests

**31 tests** covering:

- Ingestion (CSV, API with mocked HTTP)
- Schema validation
- Quality checks (missing, outliers, consistency)
- Feature engineering
- DuckDB storage + queries
- End-to-end integration (full pipeline)

```bash
pytest -v
```

## Quality Reports

Every pipeline run generates:

- `logs/quality_report.md` — Markdown report
- `logs/quality_report.html` — HTML report (open in browser)

Each report includes:

- Schema validation summary (rows, columns, missing, duplicates)
- Extended quality checks (NaN, outliers, consistency issues)
- Cardinality per column

## Dependencies

- `pandas` — data manipulation
- `pyarrow` — Parquet support
- `pydantic` — config + schema validation
- `pyyaml` — config files
- `requests` — arXiv API
- `duckdb` — SQL analytics on local files

## CI

Tests run on every push via **GitHub Actions**:

- Python 3.11 + 3.12
- `ruff` lint + format check
- `pytest` with coverage

## License

MIT