"""Load and predict with a Dixon-Coles rating bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from models.common.kicktipp import (
    build_kicktipp_points_matrix,
    kicktipp_optimal_score_from_joint,
)
from models.common.teams import normalize_team_name
from models.dixon_coles.ratings import dixon_coles_joint_probs


def _checkpoint_key(season: int, matchday: int) -> int:
    return int(season) * 1000 + int(matchday)


class DixonColesModel:
    def __init__(self, bundle_dir: Path | str):
        art = Path(bundle_dir)
        config = json.loads((art / "dixon_coles.json").read_text(encoding="utf-8"))
        self.goal_cap = int(config.get("goal_cap", 7))

        ratings: pd.DataFrame = joblib.load(art / "ratings.joblib")
        checkpoints: pd.DataFrame = joblib.load(art / "checkpoints.joblib")
        if checkpoints.empty:
            raise ValueError(f"Empty checkpoint table in {art}")

        checkpoints = checkpoints.sort_values(["Saison", "Spieltag"]).reset_index(drop=True)
        self._checkpoints = checkpoints
        self._keys = np.array(
            [_checkpoint_key(s, m) for s, m in zip(checkpoints["Saison"], checkpoints["Spieltag"])],
            dtype=np.int64,
        )

        self._ratings_by_checkpoint: List[Dict[str, Tuple[float, float]]] = []
        grouped = {
            _checkpoint_key(s, m): g
            for (s, m), g in ratings.groupby(["Saison", "Spieltag"], sort=False)
        }
        for key in self._keys:
            group = grouped.get(int(key))
            if group is None:
                self._ratings_by_checkpoint.append({})
                continue
            self._ratings_by_checkpoint.append(
                dict(
                    zip(
                        group["team_norm"],
                        zip(group["attack"].astype(float), group["defence"].astype(float)),
                    )
                )
            )

        self._points_matrix = build_kicktipp_points_matrix(self.goal_cap)
        self.config = config
        self.bundle_dir = art

    def _checkpoint_for(self, season: int, matchday: int) -> int:
        """Index of the latest checkpoint at or before the requested matchday."""
        pos = int(np.searchsorted(self._keys, _checkpoint_key(season, matchday), side="right")) - 1
        return max(pos, 0)

    def predict(self, X: pd.DataFrame) -> List[str]:
        home = X["Team Home"].map(normalize_team_name)
        away = X["Team Away"].map(normalize_team_name)
        seasons = X["Saison"].astype(int)
        matchdays = X["Spieltag"].astype(int)

        preds: List[str] = []
        for team_home, team_away, season, matchday in zip(home, away, seasons, matchdays):
            pos = self._checkpoint_for(season, matchday)
            ckpt = self._checkpoints.iloc[pos]
            table = self._ratings_by_checkpoint[pos]
            fallback = (float(ckpt["newcomer_attack"]), float(ckpt["newcomer_defence"]))

            attack_home, defence_home = table.get(team_home, fallback)
            attack_away, defence_away = table.get(team_away, fallback)

            intercept = float(ckpt["intercept"])
            lam = float(np.exp(intercept + float(ckpt["home_advantage"]) + attack_home - defence_away))
            mu = float(np.exp(intercept + attack_away - defence_home))

            joint = dixon_coles_joint_probs(lam, mu, float(ckpt["rho"]), self.goal_cap)
            h, a = kicktipp_optimal_score_from_joint(joint, self.goal_cap, self._points_matrix)
            preds.append(f"{h}:{a}")
        return preds


def load(bundle_dir: Path | str) -> DixonColesModel:
    return DixonColesModel(bundle_dir)
