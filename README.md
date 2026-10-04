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
| 2 | [Data Engineering](./Data%20Engineering/) | Data Intelligence Pipeline | 🚧 En cours |
| 3 | NLP | Natural Language Processing | ⏳ À venir |
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

- **Numerical Computing Lab** — 9 numerical methods, 83 tests, benchmarked vs SciPy
- **Data Engineering** — J1 complete (CSV ingestion, validation, cleaning, Parquet storage)