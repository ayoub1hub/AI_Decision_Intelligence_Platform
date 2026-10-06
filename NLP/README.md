# NLP Engine

NLP module for the **AI Decision Intelligence Platform**.

## Pipeline

```
Text → Clean → Tokenize → Vectorize → Model → Search
```

## Quick Start

```bash
pip install -e ".[dev]"
pytest
```

## Structure

- `src/nlp_engine/preprocessing/` — nettoyage, tokenisation
- `src/nlp_engine/vectorization/` — TF-IDF, embeddings
- `src/nlp_engine/modeling/` — classification, clustering
- `src/nlp_engine/search/` — recherche sémantique