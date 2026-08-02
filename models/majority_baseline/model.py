"""Load and predict with a majority-class baseline bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd


class MajorityBaselineModel:
    def __init__(self, bundle_dir: Path | str):
        art = Path(bundle_dir)
        meta = json.loads((art / "majority.json").read_text(encoding="utf-8"))
        self.majority_class = str(meta["majority_class"])
        self.liga = str(meta.get("liga", "bl1"))
        self.bundle_dir = art

    def predict(self, X: pd.DataFrame) -> List[str]:
        return [self.majority_class] * len(X)


def load(bundle_dir: Path | str) -> MajorityBaselineModel:
    return MajorityBaselineModel(bundle_dir)
