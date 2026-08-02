"""Market-value / history priors with Kicktipp saison score-maxing decode."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from models.common.teams import normalize_team_name

MODEL_TYPE = "saison_ausblick"
POINTS_PER_CORRECT = 6


def _softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    t = max(float(temperature), 1e-6)
    x = np.asarray(scores, dtype=float) / t
    x = x - np.max(x)
    ex = np.exp(x)
    s = ex.sum()
    if s <= 0 or not np.isfinite(s):
        return np.ones_like(ex) / len(ex)
    return ex / s


def strength_scores(
    feat: pd.DataFrame,
    *,
    mv_weight: float,
    prior_rank_weight: float,
) -> np.ndarray:
    """Higher = stronger (championship / herbst / scorer prior base)."""
    log_mv = feat["log_mv"].to_numpy(dtype=float)
    # Invert rank so rank 1 contributes positively
    prior_bonus = (19.0 - feat["prior_rank"].to_numpy(dtype=float)) / 18.0
    return mv_weight * log_mv + prior_rank_weight * prior_bonus


def scorer_scores(
    feat: pd.DataFrame,
    *,
    mv_weight: float,
    gf_weight: float,
    prior_rank_weight: float,
) -> np.ndarray:
    log_mv = feat["log_mv"].to_numpy(dtype=float)
    prior_gf = feat["prior_gf"].to_numpy(dtype=float)
    prior_gf_z = (prior_gf - prior_gf.mean()) / (prior_gf.std() + 1e-6)
    prior_bonus = (19.0 - feat["prior_rank"].to_numpy(dtype=float)) / 18.0
    return mv_weight * log_mv + gf_weight * prior_gf_z + prior_rank_weight * prior_bonus


def bottom_inclusion_probs(relegation_scores: np.ndarray, temperature: float) -> np.ndarray:
    """Probability a team finishes in places 16–18 (softmax over weak scores)."""
    return _softmax(relegation_scores, temperature)


def exact_bottom3_probs(
    teams: list[str],
    inclusion_p: np.ndarray,
) -> dict[tuple[str, str, str], float]:
    """
    Approximate P(exact unordered trio) proportional to product of inclusion probs.

    Renormalize over all C(n,3) combinations.
    """
    n = len(teams)
    raw: dict[tuple[str, str, str], float] = {}
    total = 0.0
    for combo in itertools.combinations(range(n), 3):
        p = float(inclusion_p[combo[0]] * inclusion_p[combo[1]] * inclusion_p[combo[2]])
        key = tuple(sorted(teams[i] for i in combo))
        raw[key] = p
        total += p
    if total <= 0:
        uniform = 1.0 / max(len(raw), 1)
        return {k: uniform for k in raw}
    return {k: v / total for k, v in raw.items()}


def encode_bottom3_tuple(teams: tuple[str, str, str] | list[str]) -> str:
    return "|".join(sorted(normalize_team_name(t) for t in teams))


def predict_season_row(
    feat: pd.DataFrame,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Score-maxing tips + per-question probabilities for one season's teams."""
    if feat.empty:
        raise ValueError("Empty team features for season")

    feat = feat.copy()
    feat["Team"] = feat["Team"].map(normalize_team_name)
    teams = feat["Team"].tolist()

    champ_temp = float(params.get("champion_temperature", 0.85))
    herbst_temp = float(params.get("herbst_temperature", 1.1))
    bottom_temp = float(params.get("bottom_temperature", 0.9))
    scorer_temp = float(params.get("scorer_temperature", 1.0))
    mv_w = float(params.get("mv_weight", 1.0))
    rank_w = float(params.get("prior_rank_weight", 0.35))
    gf_w = float(params.get("gf_weight", 0.45))

    strong = strength_scores(feat, mv_weight=mv_w, prior_rank_weight=rank_w)
    p_champ = _softmax(strong, champ_temp)
    p_herbst = _softmax(strong, herbst_temp)

    # Weak teams: negative strength
    p_bottom_inc = bottom_inclusion_probs(-strong, bottom_temp)
    p_bottom3 = exact_bottom3_probs(teams, p_bottom_inc)

    scorer = scorer_scores(
        feat, mv_weight=mv_w, gf_weight=gf_w, prior_rank_weight=rank_w * 0.5
    )
    p_scorer = _softmax(scorer, scorer_temp)

    champ_idx = int(np.argmax(p_champ))
    herbst_idx = int(np.argmax(p_herbst))
    scorer_idx = int(np.argmax(p_scorer))
    best_trio = max(p_bottom3.items(), key=lambda kv: kv[1])

    return {
        "champion": teams[champ_idx],
        "herbstmeister": teams[herbst_idx],
        "bottom3": encode_bottom3_tuple(best_trio[0]),
        "top_scorer_team": teams[scorer_idx],
        "p_champion": float(p_champ[champ_idx]),
        "p_herbstmeister": float(p_herbst[herbst_idx]),
        "p_bottom3": float(best_trio[1]),
        "p_top_scorer_team": float(p_scorer[scorer_idx]),
        "expected_saison_score": float(
            POINTS_PER_CORRECT
            * (
                p_champ[champ_idx]
                + p_herbst[herbst_idx]
                + best_trio[1]
                + p_scorer[scorer_idx]
            )
        ),
        "team_probs": {
            "champion": {t: float(p) for t, p in zip(teams, p_champ)},
            "herbstmeister": {t: float(p) for t, p in zip(teams, p_herbst)},
            "bottom_inclusion": {t: float(p) for t, p in zip(teams, p_bottom_inc)},
            "top_scorer_team": {t: float(p) for t, p in zip(teams, p_scorer)},
        },
    }


class SaisonAusblickModel:
    def __init__(self, bundle_dir: Path | str):
        art = Path(bundle_dir)
        self.bundle_dir = art
        self.bundle = json.loads((art / "bundle.json").read_text(encoding="utf-8"))
        self.prior = json.loads((art / "prior.json").read_text(encoding="utf-8"))
        self.params = dict(self.prior.get("params") or {})
        feats_path = art / "team_features.csv"
        if feats_path.exists():
            self.team_features = pd.read_csv(feats_path)
        else:
            self.team_features = pd.DataFrame()

    def predict_saison(
        self,
        season: int,
        team_features: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        feats = team_features if team_features is not None else self.team_features
        if feats is None or feats.empty:
            raise ValueError("No team features available for prediction")
        season_feat = feats[feats["Saison"] == season].copy()
        if season_feat.empty:
            raise ValueError(f"No team features for season {season}")
        out = predict_season_row(season_feat, self.params)
        out["Saison"] = int(season)
        return out

    def predict_many(
        self,
        seasons: list[int],
        team_features: pd.DataFrame | None = None,
    ) -> list[dict[str, Any]]:
        return [self.predict_saison(s, team_features=team_features) for s in seasons]

    def predict(self, X: pd.DataFrame) -> list[str]:
        """
        MLflow / registry adapter: one row per season with column Saison.

        Returns a JSON string per row with the four tips (not match H:A).
        """
        if "Saison" not in X.columns:
            raise ValueError("SaisonAusblickModel.predict requires column 'Saison'")
        feats = self.team_features
        if "Team" in X.columns and "MarketValue" in X.columns:
            # Caller passed team-feature rows; group by season
            feats = X
            seasons = sorted(X["Saison"].astype(int).unique())
        else:
            seasons = [int(s) for s in X["Saison"].tolist()]

        import json as _json

        return [
            _json.dumps(
                {
                    k: self.predict_saison(season, team_features=feats)[k]
                    for k in (
                        "Saison",
                        "champion",
                        "herbstmeister",
                        "bottom3",
                        "top_scorer_team",
                        "expected_saison_score",
                    )
                },
                ensure_ascii=False,
            )
            for season in seasons
        ]


def load(bundle_dir: Path | str) -> SaisonAusblickModel:
    return SaisonAusblickModel(bundle_dir)
