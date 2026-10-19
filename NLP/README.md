# NLP Engine

NLP module for the **AI Decision Intelligence Platform**.

## Pipeline

```
Text → Preprocess → Vectorize → Model → Search
```

## Features

- **Preprocessing**: HTML/URLs/LaTeX cleaning, tokenization, lemmatization
- **Vectorization**: TF-IDF (sparse) + BERT embeddings (dense, 384-dim)
- **Classification**: arXiv category prediction (SVM 78% with BERT)
- **Clustering**: KMeans topic discovery
- **Search**: FAISS semantic search

## Quick Start

```bash
pip install -e ".[dev]"
pytest
```

## Tests

65 tests covering preprocessing, vectorization, classification, clustering, and search.

## Results

| Task | Method | Score |
|---|---|---|
| Classification (TF-IDF) | SVM | 73% |
| Classification (BERT) | SVM | **78%** |
| Search | BERT + FAISS | top-k relevant |

## Structure

- `src/nlp_engine/preprocessing/` — cleaning, tokenization
- `src/nlp_engine/vectorization/` — TF-IDF, BERT
- `src/nlp_engine/modeling/` — classifier, clustering
- `src/nlp_engine/search/` — semantic search (FAISS)
- `scripts/` — training, clustering, search CLI
- `tests/` — 65 tests