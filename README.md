# AI Decision Intelligence Platform

A unified platform for building end-to-end AI systems, from numerical foundations to agentic applications.

## Vision

```
                    AI DECISION INTELLIGENCE PLATFORM
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
   DATA LAYER              MODEL LAYER              APPLICATION LAYER
        │                        │                        │
   Data Engineering         NLP / CV / ML              RAG / LLM
   Knowledge Base           Numerical Lab              Agentic AI
                                                       PoC / Vertex AI
```

## Modules

| # | Module | Description | Status |
|---|---|---|---|
| 1 | [Numerical Computing Lab](./Numerical%20Computing%20Lab/) | From-scratch numerical methods | ✅ Terminé |
| 2 | [Data Engineering](./Data%20Engineering/) | Data Intelligence Pipeline | ✅ Terminé |
| 3 | [NLP](./NLP/) | Natural Language Processing | ✅ Terminé |
| 4 | Knowledge Base | Structured knowledge | ⏳ À venir |
| 5 | RAG | Retrieval-Augmented Generation | ⏳ À venir |
| 6 | Machine Learning | Classical ML | ⏳ À venir |
| 7 | Computer Vision | Image processing | ⏳ À venir |
| 8 | LLM | Large Language Models | ⏳ À venir |
| 9 | Agentic AI | Autonomous agents | ⏳ À venir |
| 10 | PoC / Vertex AI | Cloud deployment | ⏳ À venir |

## Philosophy

Each module:

- Is a **standalone** installable Python package (`pip install -e`)
- Has its own **tests** (`pytest`)
- Has its own **README** documenting usage
- Can be used **independently**, but is designed to **compose** with others

## Quick Start

```bash
git clone https://github.com/ayoub1hub/AI_Decision_Intelligence_Platform.git
cd AI_Decision_Intelligence_Platform
```

## Status

### ✅ Numerical Computing Lab

- 9 numerical methods: root finding (Dichotomy, Newton, Secant), linear systems (Gauss, LU, Jacobi, Gauss-Seidel), ODE (Euler, RK4, RK45)
- **83 tests**, benchmarked against SciPy
- Convergence order validated empirically (Dichotomy ~1, Newton ~2, Secant ~1.618)

### ✅ Data Engineering

- Full pipeline: CSV/API ingestion → schema validation → cleaning → feature engineering
- Storage: **Parquet** + **DuckDB** (dual)
- CLI: `run`, `download-arxiv`, `query`
- **31 tests**, HTML quality reports, GitHub Actions CI

### ✅ NLP

- Preprocessing: HTML/URLs/LaTeX removal, tokenization, lemmatization
- Vectorization: **TF-IDF** (sparse) + **BERT embeddings** (dense, all-MiniLM-L6-v2)
- Classification: arXiv category prediction (5 classes) — **SVM 78% accuracy with BERT**
- Clustering: KMeans topic discovery on BERT embeddings
- Semantic search: FAISS index for retrieval
- **65 tests**, TF-IDF vs BERT comparison (BERT +5% accuracy)

## Tech Stack

| Layer | Tools |
|---|---|
| **Numerical** | NumPy, SciPy (for validation only) |
| **Data** | Pandas, PyArrow, DuckDB, Pydantic |
| **NLP** | NLTK, scikit-learn, Sentence Transformers, FAISS |
| **Dev** | pytest, ruff, mypy, GitHub Actions |

## License

MIT (or your choice)
