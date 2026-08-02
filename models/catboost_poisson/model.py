"""Load and predict with a CatBoost Poisson Kicktipp bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from models.catboost_poisson.features import build_features
from models.common.kicktipp import (
    build_kicktipp_points_matrix,
    kicktipp_optimal_score,
    poisson_mode_score,
)


class CatBoostPoissonModel:
    def __init__(self, bundle_dir: Path | str):
        art = Path(bundle_dir)
        meta = json.loads((art / "meta.json").read_text(encoding="utf-8"))
        self.feature_columns: List[str] = list(meta["feature_columns"])
        self.cat_feature_names: List[str] = list(meta["cat_feature_names"])
        self.goal_cap = int(meta.get("goal_cap", 7))
        self.decode_goal_cap = int(self.goal_cap)
        self.home_lambda_scale = float(meta.get("home_lambda_scale", 1.0))
        self.away_lambda_scale = float(meta.get("away_lambda_scale", 1.0))

        self.mv_alias_map = json.loads((art / "mv_alias_map.json").read_text(encoding="utf-8"))
        self.team_aggs = joblib.load(art / "team_aggs.joblib")
        self.market_values = joblib.load(art / "market_values.joblib")
        self.team_form = joblib.load(art / "team_form.joblib")
        self.team_elo = joblib.load(art / "team_elo.joblib")

        self.home_model = CatBoostRegressor()
        self.home_model.load_model(str(art / "home_goals.cbm"))
        self.away_model = CatBoostRegressor()
        self.away_model.load_model(str(art / "away_goals.cbm"))

        self._kicktipp_points = build_kicktipp_points_matrix(self.decode_goal_cap)
        self.bundle_dir = art

    def predict_lambdas(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Calibrated per-row home/away goal expectations."""
        feats = build_features(
            X,
            self.market_values,
            self.mv_alias_map,
            self.team_aggs,
            self.team_form,
            self.team_elo,
            feature_columns=self.feature_columns,
        )
        lam_home = np.asarray(self.home_model.predict(feats), dtype=float)
        lam_away = np.asarray(self.away_model.predict(feats), dtype=float)
        return lam_home * self.home_lambda_scale, lam_away * self.away_lambda_scale

    def predict(self, X: pd.DataFrame) -> List[str]:
        lam_home, lam_away = self.predict_lambdas(X)

        preds: List[str] = []
        for lh, la in zip(lam_home, lam_away):
            if self._kicktipp_points is not None:
                h, a = kicktipp_optimal_score(lh, la, self.decode_goal_cap, self._kicktipp_points)
            else:
                h, a = poisson_mode_score(lh, la, self.decode_goal_cap)
            preds.append(f"{h}:{a}")
        return preds


def load(bundle_dir: Path | str) -> CatBoostPoissonModel:
    return CatBoostPoissonModel(bundle_dir)
