# Data Intelligence Pipeline

Data Engineering layer for the **AI Decision Intelligence Platform**.

## Pipeline

```
CSV → load → validate → clean → Parquet
```

## Quick Start

```bash
pip install -e ".[dev]"
python -m data_engine.pipeline
pytest
```

## Structure

- `src/data_engine/` — code source
- `configs/` — configuration (dev, prod, schemas)
- `data/` — données (raw, processed, external)
- `tests/` — tests
- `notebooks/` — exploration
- `scripts/` — utilitaires