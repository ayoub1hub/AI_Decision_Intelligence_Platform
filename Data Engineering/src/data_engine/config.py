"""Configuration loading for the data engine.

Reads a YAML file and validates it with Pydantic. The resulting
`Config` object gives typed access to all settings:

    cfg = load_config("configs/dev.yaml")
    print(cfg.data.input_file)   # "arxiv_sample.csv"
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

__all__ = ["Config", "DataConfig", "LoggingConfig", "load_config"]


class DataConfig(BaseModel):
    """Paths and file names for the data layer."""

    raw_dir: str
    processed_dir: str
    input_file: str
    output_file: str
    dtypes: dict[str, str] = {}
    parse_dates: list[str] = []


class SchemaConfig(BaseModel):
    """Where to find the schema definition."""

    path: str


class LoggingConfig(BaseModel):
    """Logging settings."""

    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"
    log_file: str | None = None


class Config(BaseModel):
    """Root configuration object."""

    environment: str = Field(default="dev")
    data: DataConfig
    schema_: SchemaConfig = Field(alias="schema")
    logging: LoggingConfig

    model_config = {"populate_by_name": True}


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file (e.g. ``configs/dev.yaml``).

    Returns
    -------
    Config
        Validated configuration object.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(**raw)
