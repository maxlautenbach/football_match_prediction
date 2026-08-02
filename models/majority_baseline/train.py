"""Train a majority-class baseline into a tiny bundle directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from eval.baselines import majority_class
from models.contract import write_bundle_json

MODEL_TYPE = "majority_baseline"

BUNDLE_FILES = ["bundle.json", "majority.json"]


def train(
    train_df_all: pd.DataFrame,
    holdout_df: pd.DataFrame,
    mv_df_raw: pd.DataFrame,
    bundle_dir: Path,
    *,
    params: Mapping[str, Any],
    recipe_name: str,
    holdout_season: int,
) -> dict[str, Any]:
    del holdout_df, mv_df_raw  # unused — interface matches other trainers

    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    liga = str(params.get("liga", "bl1")).lower()
    results = train_df_all["Ergebnis"]
    if "Liga" in train_df_all.columns:
        mask = train_df_all["Liga"].astype(str).str.lower() == liga
        filtered = train_df_all.loc[mask, "Ergebnis"]
        if len(filtered) > 0:
            results = filtered

    maj = majority_class(results)
    train_seasons = f"{int(train_df_all['Saison'].min())}-{int(train_df_all['Saison'].max())}"

    majority_meta = {
        "majority_class": maj,
        "liga": liga,
        "n_train": int(len(results)),
        "holdout_season": holdout_season,
        "train_seasons": train_seasons,
    }
    (bundle_dir / "majority.json").write_text(
        json.dumps(majority_meta, indent=2) + "\n", encoding="utf-8"
    )

    write_bundle_json(
        bundle_dir,
        model_type=MODEL_TYPE,
        recipe_name=recipe_name,
        holdout_season=holdout_season,
        train_seasons=train_seasons,
        files=["majority.json"],
        extra={"majority_class": maj},
    )

    print(f"[train] Majority baseline '{maj}' written to {bundle_dir}")
    return {
        "model_type": MODEL_TYPE,
        "majority_class": maj,
        "n_train": len(results),
        "liga": liga,
        "train_seasons": train_seasons,
    }
