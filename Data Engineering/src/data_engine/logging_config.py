"""Centralized logging configuration.

Call `setup_logging(...)` once at the start of your program.
After that, every module can just do:

    import logging
    logger = logging.getLogger(__name__)
    logger.info("hello")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

__all__ = ["setup_logging"]


def setup_logging(
    *,
    level: str = "INFO",
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    log_file: str | Path | None = None,
) -> None:
    """Configure root logging with console + optional file handler.

    Parameters
    ----------
    level : str, default "INFO"
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    fmt : str
        Log message format.
    datefmt : str
        Date format for `%(asctime)s`.
    log_file : str or Path, optional
        If provided, log records are also written to this file.

    Notes
    -----
    This function configures the ROOT logger. It should be called ONCE
    at program start. Calling it again will reset the configuration.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )
