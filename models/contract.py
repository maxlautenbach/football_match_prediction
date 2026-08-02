"""Shared prediction contract for all Kicktipp model types."""

from __future__ import annotations

import re
from typing import List, Protocol, runtime_checkable

import pandas as pd

BUNDLE_SCHEMA_VERSION = 1

REQUIRED_COLUMNS = (
    "Team Home",
    "Team Away",
    "Saison",
    "Spieltag",
    "Wochentag",
)

_SCORE_RE = re.compile(r"^\d+:\d+$")


@runtime_checkable
class Predictor(Protocol):
    def predict(self, X: pd.DataFrame) -> List[str]:
        """Return one 'H:A' score string per input row."""


def validate_predictions(preds: List[str], n_expected: int) -> List[str]:
    """Ensure prediction length and 'H:A' formatting; raise on violation."""
    if len(preds) != n_expected:
        raise ValueError(f"Expected {n_expected} predictions, got {len(preds)}")
    for i, p in enumerate(preds):
        if not isinstance(p, str) or not _SCORE_RE.match(p):
            raise ValueError(f"Invalid prediction at index {i}: {p!r} (expected 'H:A')")
    return preds


def write_bundle_json(
    bundle_dir,
    *,
    model_type: str,
    recipe_name: str,
    holdout_season: int,
    train_seasons: str,
    files: list[str],
    extra: dict | None = None,
) -> None:
    """Write bundle.json metadata into a trained bundle directory."""
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    bundle_dir = Path(bundle_dir)
    payload = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "model_type": model_type,
        "recipe_name": recipe_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "holdout_season": int(holdout_season),
        "train_seasons": train_seasons,
        "required_columns": list(REQUIRED_COLUMNS),
        "files": sorted(files),
    }
    if extra:
        payload.update(extra)
    (bundle_dir / "bundle.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_bundle_json(bundle_dir) -> dict:
    """Load and lightly validate bundle.json."""
    import json
    from pathlib import Path

    path = Path(bundle_dir) / "bundle.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing bundle.json in {bundle_dir}")
    meta = json.loads(path.read_text(encoding="utf-8"))
    for key in ("schema_version", "model_type", "files"):
        if key not in meta:
            raise ValueError(f"bundle.json missing required key: {key}")
    return meta
