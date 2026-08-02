"""Load and predict with market-adjusted Dixon-Coles ratings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd

from models.common.kicktipp import (
    build_kicktipp_points_matrix,
    kicktipp_optimal_score_from_joint,
)
from models.common.teams import normalize_team_name
from models.dixon_coles import model as dc_model
from models.dixon_coles.ratings import dixon_coles_joint_probs


class MarketDixonColesModel:
    """Dixon-Coles with a decaying, season-relative squad-value adjustment."""

    def __init__(self, bundle_dir: Path | str):
        art = Path(bundle_dir)
        config = json.loads((art / "market_dixon_coles.json").read_text(encoding="utf-8"))

        self.goal_cap = int(config.get("goal_cap", 7))
        self.market_attack_weight = float(config["market_attack_weight"])
        self.market_defence_weight = float(config["market_defence_weight"])
        self.market_decay_matchdays = float(config["market_decay_matchdays"])
        self.probability_temperature = float(config.get("probability_temperature", 1.0))

        if self.market_decay_matchdays <= 0:
            raise ValueError("market_decay_matchdays must be positive")
        if self.probability_temperature <= 0:
            raise ValueError("probability_temperature must be positive")

        self.dixon_coles = dc_model.load(art / "dixon_coles")
        self.mv_alias_map = json.loads(
            (art / "mv_alias_map.json").read_text(encoding="utf-8")
        )
        market_values: pd.DataFrame = joblib.load(art / "market_values.joblib")
        self._market_z = {
            (str(team), int(season)): float(value)
            for team, season, value in market_values[
                ["team_norm", "Saison", "market_value_z"]
            ].itertuples(index=False, name=None)
        }

        self._points_matrix = build_kicktipp_points_matrix(self.goal_cap)
        self.config = config
        self.bundle_dir = art

    def _market_strengths(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        home = X["Team Home"].map(normalize_team_name)
        away = X["Team Away"].map(normalize_team_name)
        seasons = X["Saison"].astype(int)

        home_z = np.fromiter(
            (
                self._market_z.get((self.mv_alias_map.get(team, team), int(season)), 0.0)
                for team, season in zip(home, seasons)
            ),
            dtype=float,
            count=len(X),
        )
        away_z = np.fromiter(
            (
                self._market_z.get((self.mv_alias_map.get(team, team), int(season)), 0.0)
                for team, season in zip(away, seasons)
            ),
            dtype=float,
            count=len(X),
        )
        return home_z, away_z

    def predict_lambdas(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return market-adjusted home/away goal rates and Dixon-Coles rho."""
        lam_home, lam_away, rho = self.dixon_coles.predict_lambdas(X)
        home_z, away_z = self._market_strengths(X)
        matchdays = X["Spieltag"].astype(float).to_numpy()
        decay = np.exp(
            -np.maximum(matchdays - 1.0, 0.0) / self.market_decay_matchdays
        )

        home_adjustment = (
            self.market_attack_weight * home_z
            - self.market_defence_weight * away_z
        ) * decay
        away_adjustment = (
            self.market_attack_weight * away_z
            - self.market_defence_weight * home_z
        ) * decay

        # The clipping is inactive for normal z-scores but keeps malformed
        # market-value inputs from creating infinite Poisson rates.
        lam_home = np.asarray(lam_home, dtype=float) * np.exp(
            np.clip(home_adjustment, -2.0, 2.0)
        )
        lam_away = np.asarray(lam_away, dtype=float) * np.exp(
            np.clip(away_adjustment, -2.0, 2.0)
        )
        return lam_home, lam_away, np.asarray(rho, dtype=float)

    def predict(self, X: pd.DataFrame) -> List[str]:
        lam_home, lam_away, rho = self.predict_lambdas(X)

        predictions: List[str] = []
        for home_rate, away_rate, correlation in zip(lam_home, lam_away, rho):
            joint = dixon_coles_joint_probs(
                float(home_rate),
                float(away_rate),
                float(correlation),
                self.goal_cap,
            )
            if self.probability_temperature != 1.0:
                joint = np.power(joint, self.probability_temperature)
                joint /= float(joint.sum())
            home_goals, away_goals = kicktipp_optimal_score_from_joint(
                joint,
                self.goal_cap,
                self._points_matrix,
            )
            predictions.append(f"{home_goals}:{away_goals}")
        return predictions


def load(bundle_dir: Path | str) -> MarketDixonColesModel:
    return MarketDixonColesModel(bundle_dir)
