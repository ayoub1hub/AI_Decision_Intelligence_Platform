"""Entry point for `python -m rag_engine`."""

from __future__ import annotations

import sys

from rag_engine.cli import main

if __name__ == "__main__":
    sys.exit(main())