"""RAG configuration (YAML + Pydantic)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

__all__ = ["Config", "LLMConfig", "RetrieverConfig", "load_config"]


class LLMConfig(BaseModel):
    """LLM settings."""

    provider: str = "groq"  # "groq" | "openai" | "ollama"
    model: str = "openai/gpt-oss-20b"  # model name or path
    temperature: float = 0.0
    max_tokens: int = 512
    api_key_env: str = "GROQ_API_KEY"  # env var name


class RetrieverConfig(BaseModel):
    """Retriever settings."""

    index_path: str = "data/index/arxiv.faiss"
    embeddings_cache: str = "data/processed/arxiv_embeddings.npz"
    top_k: int = 5
    model_name: str = "all-MiniLM-L6-v2"


class DataConfig(BaseModel):
    """Data paths."""

    arxiv_csv: str = "data/raw/arxiv_labeled.csv"


class LoggingConfig(BaseModel):
    """Logging settings."""

    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"
    log_file: str | None = None


class Config(BaseModel):
    """Root config."""

    environment: str = Field(default="dev")
    llm: LLMConfig
    retriever: RetrieverConfig
    data: DataConfig
    logging: LoggingConfig


def load_config(path: str | Path = "configs/dev.yaml") -> Config:
    """Load a YAML config file into a typed Config object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(**raw)
