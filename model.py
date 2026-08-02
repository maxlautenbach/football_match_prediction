"""Stable public Model adapter.

`evaluation.py` / `scripts/predict.py` expect:
- `Model()` with no args → loads production `artifacts/`
- `Model.predict(X: pd.DataFrame) -> List[str]` where each entry is "H:A"

Algorithm-specific code lives under `models/<model_type>/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from models.contract import validate_predictions
from models.registry import detect_model_type, load_model

BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"


class Model:
    def __init__(self, artifacts_dir: Path | str | None = None, *, require_ready: bool = True):
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else ARTIFACTS_DIR
        self.ready = False
        self.model_type: str | None = None
        self._impl = None
        self._load_error: Exception | None = None

        try:
            self.model_type = detect_model_type(self.artifacts_dir)
            self._impl = load_model(self.model_type, self.artifacts_dir)
            self.ready = True
        except Exception as e:
            self._load_error = e
            msg = f"[model] Failed to load artifacts from {self.artifacts_dir}: {e}"
            if require_ready:
                raise RuntimeError(msg) from e
            print(f"WARNING: {msg}")

    def predict(self, X: pd.DataFrame) -> List[str]:
        if not self.ready or self._impl is None:
            err = self._load_error or RuntimeError("Model artifacts not loaded")
            raise RuntimeError(
                f"Model not ready (artifacts={self.artifacts_dir}): {err}"
            ) from (self._load_error if isinstance(self._load_error, Exception) else None)
        preds = self._impl.predict(X)
        return validate_predictions(list(preds), len(X))
