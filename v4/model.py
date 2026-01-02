"""Football match prediction model.

`evaluation.py` expects:
- `Model()` with no args
- `Model.predict(X: pd.DataFrame) -> List[str]` where each entry is "H:A"
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"


def _normalize_team_name(name: str) -> str:
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return ""
    s = str(name).strip()
    s = " ".join(s.split())
    s = s.replace(". ", ".")
    s = s.replace("’", "'")
    s = s.replace("–", "-")
    s = s.replace("—", "-")
    return s


def _poisson_mode_score(lam_home: float, lam_away: float, goal_cap: int) -> Tuple[int, int]:
    lam_home = float(max(lam_home, 1e-8))
    lam_away = float(max(lam_away, 1e-8))

    ks = np.arange(goal_cap + 1)
    log_fact = np.array([math.lgamma(k + 1) for k in ks])

    logp_h = -lam_home + ks * math.log(lam_home) - log_fact
    logp_a = -lam_away + ks * math.log(lam_away) - log_fact

    grid = logp_h.reshape(-1, 1) + logp_a.reshape(1, -1)
    idx = int(np.argmax(grid))
    h = idx // (goal_cap + 1)
    a = idx % (goal_cap + 1)
    return int(h), int(a)


def _build_kicktipp_points_matrix(goal_cap: int) -> np.ndarray:
    k = goal_cap + 1
    n = k * k
    pts = np.zeros((n, n), dtype=np.float32)

    def outcome(h: int, a: int) -> int:
        return 1 if h > a else (0 if h == a else -1)

    for ph in range(k):
        for pa in range(k):
            p_idx = ph * k + pa
            p_diff = ph - pa
            p_out = outcome(ph, pa)
            for th in range(k):
                for ta in range(k):
                    t_idx = th * k + ta
                    if ph == th and pa == ta:
                        pts[p_idx, t_idx] = 5.0
                    elif p_diff == (th - ta):
                        pts[p_idx, t_idx] = 3.0
                    elif p_out == outcome(th, ta):
                        pts[p_idx, t_idx] = 1.0
    return pts


def _poisson_probs(lam: float, goal_cap: int) -> np.ndarray:
    lam = float(max(lam, 1e-8))
    ks = np.arange(goal_cap + 1)
    log_fact = np.array([math.lgamma(k + 1) for k in ks])
    logp = -lam + ks * math.log(lam) - log_fact
    m = float(np.max(logp))
    p = np.exp(logp - m)
    p = p / float(np.sum(p))
    return p


def _kicktipp_optimal_score(
    lam_home: float,
    lam_away: float,
    goal_cap: int,
    points_matrix: np.ndarray,
) -> Tuple[int, int]:
    p_home = _poisson_probs(lam_home, goal_cap)
    p_away = _poisson_probs(lam_away, goal_cap)
    p_true = np.outer(p_home, p_away).reshape(-1)
    exp_pts = points_matrix @ p_true
    idx = int(np.argmax(exp_pts))
    k = goal_cap + 1
    return idx // k, idx % k


class Model:
    def __init__(self):
        self.ready = False
        self.goal_cap = 7
        self.decode_goal_cap = 7
        self.home_lambda_scale = 1.0
        self.away_lambda_scale = 1.0
        self._kicktipp_points: np.ndarray | None = None
        self.feature_columns: List[str] = []
        self.cat_feature_names: List[str] = []
        self.mv_alias_map: Dict[str, str] = {}
        self.team_aggs: pd.DataFrame | None = None
        self.team_form: pd.DataFrame | None = None
        self.team_elo: pd.DataFrame | None = None
        self.market_values: pd.DataFrame | None = None
        self.home_model: CatBoostRegressor | None = None
        self.away_model: CatBoostRegressor | None = None

        try:
            meta = json.loads((ARTIFACTS_DIR / "meta.json").read_text(encoding="utf-8"))
            self.feature_columns = list(meta["feature_columns"])
            self.cat_feature_names = list(meta["cat_feature_names"])
            self.goal_cap = int(meta.get("goal_cap", 7))
            self.decode_goal_cap = int(self.goal_cap)
            self.home_lambda_scale = float(meta.get("home_lambda_scale", 1.0))
            self.away_lambda_scale = float(meta.get("away_lambda_scale", 1.0))

            self.mv_alias_map = json.loads(
                (ARTIFACTS_DIR / "mv_alias_map.json").read_text(encoding="utf-8")
            )
            self.team_aggs = joblib.load(ARTIFACTS_DIR / "team_aggs.joblib")
            self.market_values = joblib.load(ARTIFACTS_DIR / "market_values.joblib")
            self.team_form = joblib.load(ARTIFACTS_DIR / "team_form.joblib")
            self.team_elo = joblib.load(ARTIFACTS_DIR / "team_elo.joblib")

            self.home_model = CatBoostRegressor()
            self.home_model.load_model(ARTIFACTS_DIR / "home_goals.cbm")
            self.away_model = CatBoostRegressor()
            self.away_model.load_model(ARTIFACTS_DIR / "away_goals.cbm")

            self._kicktipp_points = _build_kicktipp_points_matrix(self.decode_goal_cap)

            self.ready = True
        except Exception as e:
            # Keep evaluation runnable even if artifacts are missing.
            # Train first via: `uv run python train.py`.
            print(f"[model] WARNING: could not load artifacts: {e}")
            print("[model] Using fallback baseline (all '0:0').")

    def _build_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if (
            self.market_values is None
            or self.team_aggs is None
            or self.team_form is None
            or self.team_elo is None
        ):
            raise RuntimeError("Artifacts not loaded")

        df = X.copy()
        df["team_home_norm"] = df["Team Home"].map(_normalize_team_name)
        df["team_away_norm"] = df["Team Away"].map(_normalize_team_name)

        df["team_home_mv"] = df["team_home_norm"].map(lambda x: self.mv_alias_map.get(x, x))
        df["team_away_mv"] = df["team_away_norm"].map(lambda x: self.mv_alias_map.get(x, x))

        mv = self.market_values.copy()
        mv = mv.rename(columns={"MarketValue": "market_value", "team_norm": "team_norm"})

        home_mv = mv.rename(columns={"team_norm": "team_home_mv"})
        away_mv = mv.rename(columns={"team_norm": "team_away_mv"})

        df = df.merge(
            home_mv[["team_home_mv", "Saison", "market_value"]].rename(
                columns={"market_value": "home_market_value"}
            ),
            on=["team_home_mv", "Saison"],
            how="left",
        )
        df = df.merge(
            away_mv[["team_away_mv", "Saison", "market_value"]].rename(
                columns={"market_value": "away_market_value"}
            ),
            on=["team_away_mv", "Saison"],
            how="left",
        )

        df["home_market_value"] = df["home_market_value"].astype(float)
        df["away_market_value"] = df["away_market_value"].astype(float)
        df["mv_diff"] = df["home_market_value"] - df["away_market_value"]
        eps = 1e-6
        df["mv_ratio_log"] = np.log((df["home_market_value"] + eps) / (df["away_market_value"] + eps))

        aggs = self.team_aggs.copy()
        df = df.merge(
            aggs.add_prefix("home_").rename(columns={"home_team_norm": "team_home_norm"}),
            on="team_home_norm",
            how="left",
        )
        df = df.merge(
            aggs.add_prefix("away_").rename(columns={"away_team_norm": "team_away_norm"}),
            on="team_away_norm",
            how="left",
        )

        for col in [
            "home_home_gf_mean",
            "home_home_ga_mean",
            "away_away_gf_mean",
            "away_away_ga_mean",
        ]:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())

        # Elo ratings per matchday
        te = self.team_elo.copy()
        home_te = te.rename(columns={"team_norm": "team_home_norm", "elo": "home_elo"})
        away_te = te.rename(columns={"team_norm": "team_away_norm", "elo": "away_elo"})
        df = df.merge(home_te, on=["team_home_norm", "Saison", "Spieltag"], how="left")
        df = df.merge(away_te, on=["team_away_norm", "Saison", "Spieltag"], how="left")
        df["home_elo"] = df["home_elo"].fillna(1500.0)
        df["away_elo"] = df["away_elo"].fillna(1500.0)
        df["elo_diff"] = df["home_elo"] - df["away_elo"]

        # Team form features (season-to-date + recent window), keyed by (team, Saison, Spieltag)
        tf = self.team_form.copy()
        home_tf = tf.rename(columns={"team_norm": "team_home_norm"})
        home_tf = home_tf.rename(
            columns={c: f"home_{c}" for c in home_tf.columns if c not in {"team_home_norm", "Saison", "Spieltag"}}
        )
        away_tf = tf.rename(columns={"team_norm": "team_away_norm"})
        away_tf = away_tf.rename(
            columns={c: f"away_{c}" for c in away_tf.columns if c not in {"team_away_norm", "Saison", "Spieltag"}}
        )

        df = df.merge(home_tf, on=["team_home_norm", "Saison", "Spieltag"], how="left")
        df = df.merge(away_tf, on=["team_away_norm", "Saison", "Spieltag"], how="left")

        for col in [
            "home_season_points_per_game",
            "home_season_gf_per_game",
            "home_season_ga_per_game",
            "home_recent_points_per_game",
            "home_recent_gf_per_game",
            "home_recent_ga_per_game",
            "away_season_points_per_game",
            "away_season_gf_per_game",
            "away_season_ga_per_game",
            "away_recent_points_per_game",
            "away_recent_gf_per_game",
            "away_recent_ga_per_game",
        ]:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())

        features = pd.DataFrame(
            {
                "team_home": df["team_home_norm"],
                "team_away": df["team_away_norm"],
                "saison": df["Saison"].astype(int),
                "spieltag": df["Spieltag"].astype(int),
                "wochentag": df["Wochentag"].astype(str),
                "home_market_value": df["home_market_value"],
                "away_market_value": df["away_market_value"],
                "mv_diff": df["mv_diff"],
                "mv_ratio_log": df["mv_ratio_log"],
                "home_home_gf_mean": df["home_home_gf_mean"],
                "home_home_ga_mean": df["home_home_ga_mean"],
                "away_away_gf_mean": df["away_away_gf_mean"],
                "away_away_ga_mean": df["away_away_ga_mean"],
                "home_season_points_per_game": df["home_season_points_per_game"],
                "home_season_gf_per_game": df["home_season_gf_per_game"],
                "home_season_ga_per_game": df["home_season_ga_per_game"],
                "home_recent_points_per_game": df["home_recent_points_per_game"],
                "home_recent_gf_per_game": df["home_recent_gf_per_game"],
                "home_recent_ga_per_game": df["home_recent_ga_per_game"],
                "away_season_points_per_game": df["away_season_points_per_game"],
                "away_season_gf_per_game": df["away_season_gf_per_game"],
                "away_season_ga_per_game": df["away_season_ga_per_game"],
                "away_recent_points_per_game": df["away_recent_points_per_game"],
                "away_recent_gf_per_game": df["away_recent_gf_per_game"],
                "away_recent_ga_per_game": df["away_recent_ga_per_game"],
                "home_elo": df["home_elo"],
                "away_elo": df["away_elo"],
                "elo_diff": df["elo_diff"],
            }
        )

        # Ensure exact column order
        return features[self.feature_columns]

    def predict(self, X: pd.DataFrame) -> List[str]:
        if not self.ready or self.home_model is None or self.away_model is None:
            return ["0:0"] * len(X)

        feats = self._build_features(X)
        lam_home = np.asarray(self.home_model.predict(feats), dtype=float)
        lam_away = np.asarray(self.away_model.predict(feats), dtype=float)

        lam_home = lam_home * self.home_lambda_scale
        lam_away = lam_away * self.away_lambda_scale

        preds: List[str] = []
        for lh, la in zip(lam_home, lam_away):
            if self._kicktipp_points is not None:
                h, a = _kicktipp_optimal_score(lh, la, self.decode_goal_cap, self._kicktipp_points)
            else:
                h, a = _poisson_mode_score(lh, la, self.decode_goal_cap)
            preds.append(f"{h}:{a}")
        return preds

