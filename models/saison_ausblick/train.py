"""Train saison_ausblick score-maxing priors into a bundle directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from models.contract import write_bundle_json
from models.saison_ausblick.model import MODEL_TYPE, predict_season_row

BUNDLE_FILES = ["bundle.json", "prior.json", "team_features.csv"]


def train(
    train_labels: pd.DataFrame,
    holdout_labels: pd.DataFrame,
    team_features: pd.DataFrame,
    bundle_dir: Path,
    *,
    params: Mapping[str, Any],
    recipe_name: str,
    holdout_season: int,
) -> dict[str, Any]:
    """
    Fit is parameteric (recipe priors). Bundle stores params + team features
    needed for holdout / future prediction.
    """
    del holdout_labels  # unused for fitting — evaluation is orchestrator-side

    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    params = dict(params)
    train_seasons_list = sorted(int(s) for s in train_labels["Saison"].unique())
    train_seasons = (
        f"{train_seasons_list[0]}-{train_seasons_list[-1]}" if train_seasons_list else ""
    )

    # Keep features for train + holdout seasons that exist in the table
    seasons_needed = set(train_seasons_list) | {int(holdout_season)}
    feats = team_features[team_features["Saison"].isin(seasons_needed)].copy()
    feats.to_csv(bundle_dir / "team_features.csv", index=False)

    prior = {
        "params": params,
        "n_train_seasons": len(train_seasons_list),
        "train_seasons": train_seasons,
        "holdout_season": int(holdout_season),
        "model_type": MODEL_TYPE,
    }
    (bundle_dir / "prior.json").write_text(
        json.dumps(prior, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    write_bundle_json(
        bundle_dir,
        model_type=MODEL_TYPE,
        recipe_name=recipe_name,
        holdout_season=holdout_season,
        train_seasons=train_seasons,
        files=["prior.json", "team_features.csv"],
        extra={
            "required_columns": ["Saison"],
            "output_fields": [
                "champion",
                "herbstmeister",
                "bottom3",
                "top_scorer_team",
            ],
            "points_per_correct": 6,
            "max_score": 24,
        },
    )

    # Quick train-season expected scores for logging
    train_expected: dict[int, float] = {}
    for season in train_seasons_list:
        season_feat = feats[feats["Saison"] == season]
        if season_feat.empty:
            continue
        pred = predict_season_row(season_feat, params)
        train_expected[int(season)] = float(pred["expected_saison_score"])

    print(f"[train] saison_ausblick prior written to {bundle_dir}")
    return {
        "model_type": MODEL_TYPE,
        "n_train": len(train_seasons_list),
        "n_train_seasons": len(train_seasons_list),
        "train_seasons": train_seasons,
        "mean_train_expected_score": (
            float(sum(train_expected.values()) / len(train_expected))
            if train_expected
            else 0.0
        ),
        "_train_expected_by_season": train_expected,
    }
