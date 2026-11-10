# RAG Engine

Retrieval-Augmented Generation module for the **AI Decision Intelligence Platform**.

## Pipeline

```
Question → Retrieve (FAISS) → Generate (LLM) → Answer
```

## Quick Start

```bash
pip install -e ".[dev]"
python -m rag_engine ask "What is a transformer?"
```

## Features

- **Retrieval**: FAISS-based semantic search over arXiv papers
- **Generation**: LLM-agnostic (Groq, OpenAI, Ollama)
- **Evaluation**: Faithfulness, relevance, context precision

## Structure

- `src/rag_engine/retriever/` — FAISS index + retrieval
- `src/rag_engine/generator/` — LLM client + prompt templates
- `src/rag_engine/pipeline.py` — orchestrator
- `scripts/` — build index, ask
- `tests/` — unit + integration tests

## LLM Providers

| Provider | Cost | Speed | Setup |
|---|---|---|---|
| **Groq** | Free | Very fast | API key |
| **OpenAI** | Paid | Fast | API key |
| **Ollama** | Free | Slow (CPU) | Local install |