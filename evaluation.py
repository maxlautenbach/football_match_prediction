"""Thin CLI wrapper — prefer `uv run python -m eval.compare`."""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "scripts"))

from eval.compare import main

if __name__ == "__main__":
    main()
