# RAG Engine

Retrieval-Augmented Generation module for the **AI Decision Intelligence Platform**.

Combines **FAISS-based retrieval** over arXiv papers with **LLM-based generation**, producing **grounded answers with inline citations**.

## Pipeline
User question
│
▼
┌────────────────────────────────────────────────────┐
│ 1. RETRIEVE (FAISS) │ ← top-k papers from 205 arXiv abstracts
│ BERT embeddings │
└────────────────────────────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────┐
│ 2. PROMPT BUILDER │ ← context + question + rules
│ System + Context + Q │
└────────────────────────────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────┐
│ 3. GENERATE (LLM) │ ← Groq / OpenAI / Ollama
│ Grounded answer │
└────────────────────────────────────────────────────┘
│
▼
Answer with sources [1], [2], ...

text

## Features

- **Retrieval**: FAISS semantic search over 205 arXiv papers (BERT embeddings, 384-dim)
- **Generation**: unified client for **Groq**, **OpenAI**, and **Ollama** (one API, three backends)
- **Citations**: answers cite sources inline as `[1]`, `[2]`, ... matching the retrieved docs
- **Prompt engineering**: system prompt enforces "answer only from context" and "say I don't know if not in context"
- **CLI**: `ask`, `search`, `build-index` — all via `python -m rag_engine`
- **Retry logic**: tenacity with exponential backoff for LLM calls
- **Tests**: unit + pipeline tests, all mocked (no API calls in CI)
- **Python 3.12** compatible

## Quick Start

### 1. Install

```bash
cd RAG
pip install -e ".[dev]"
```