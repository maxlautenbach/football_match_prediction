"""Train time-decayed Dixon-Coles ratings into a bundle directory.

Ratings are refitted at every matchday checkpoint of the recent seasons using
only matches played strictly before that matchday. The bundle therefore holds a
causal rating table: predicting matchday N uses in-season evidence up to N-1,
exactly what is available in production, without leaking the match itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from models.common.teams import normalize_team_name
from models.contract import write_bundle_json
from models.dixon_coles.ratings import DEFAULT_RHO_BOUNDS, fit_dixon_coles

MODEL_TYPE = "dixon_coles"

BUNDLE_FILES = ["dixon_coles.json", "ratings.joblib", "checkpoints.joblib"]

MIN_FIT_MATCHES = 200


def _prepare_history(train_df_all: pd.DataFrame, holdout_df: pd.DataFrame) -> pd.DataFrame:
    history = pd.concat([train_df_all, holdout_df], ignore_index=True)
    history = history.sort_values(["Saison", "Spieltag"]).reset_index(drop=True)

    goals = history["Ergebnis"].astype(str).str.split(":", expand=True)
    history["home_goals"] = goals[0].astype(int)
    history["away_goals"] = goals[1].astype(int)
    history["team_home_norm"] = history["Team Home"].map(normalize_team_name)
    history["team_away_norm"] = history["Team Away"].map(normalize_team_name)

    matchdays_per_season = history.groupby("Saison")["Spieltag"].transform("max")
    history["match_time"] = history["Saison"] + (history["Spieltag"] - 1) / matchdays_per_season
    return history


def _checkpoint_grid(history: pd.DataFrame, seasons: list[int]) -> list[tuple[int, int]]:
    """(Saison, Spieltag) checkpoints, incl. one past the last played matchday."""
    grid: list[tuple[int, int]] = []
    for season in seasons:
        season_rows = history[history["Saison"] == season]
        if season_rows.empty:
            continue
        last_md = int(season_rows["Spieltag"].max())
        grid.extend((int(season), md) for md in range(1, last_md + 2))
    return grid


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
    del mv_df_raw  # unused — market values are not part of this model

    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    liga = str(params.get("liga", "bl1")).lower()
    fit_leagues = [str(x).lower() for x in params.get("fit_leagues", ["bl1", "bl2"])]
    goal_cap = int(params.get("goal_cap", 7))
    half_life_seasons = float(params.get("half_life_seasons", 1.5))
    l2 = float(params.get("l2", 0.02))
    rho_bounds = (
        float(params.get("rho_min", DEFAULT_RHO_BOUNDS[0])),
        float(params.get("rho_max", DEFAULT_RHO_BOUNDS[1])),
    )
    max_iter = int(params.get("max_iter", 500))
    checkpoint_seasons_back = int(params.get("checkpoint_seasons_back", 2))
    newcomer_quantile = float(params.get("newcomer_quantile", 0.2))
    min_team_evidence = float(params.get("min_team_evidence", 10.0))

    if half_life_seasons <= 0:
        raise ValueError(f"half_life_seasons must be > 0, got {half_life_seasons}")

    history = _prepare_history(train_df_all, holdout_df)

    if "Liga" in history.columns:
        fit_mask = history["Liga"].astype(str).str.lower().isin(fit_leagues)
        fit_pool = history[fit_mask].reset_index(drop=True)
    else:
        fit_pool = history
    if fit_pool.empty:
        raise ValueError(f"No matches left after filtering leagues {fit_leagues}")
    print(
        f"[train] Fitting pool: {len(fit_pool)}/{len(history)} matches "
        f"from leagues {fit_leagues}"
    )

    teams = sorted(set(fit_pool["team_home_norm"]) | set(fit_pool["team_away_norm"]))
    team_index = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)

    home_idx = fit_pool["team_home_norm"].map(team_index).to_numpy(dtype=np.int64)
    away_idx = fit_pool["team_away_norm"].map(team_index).to_numpy(dtype=np.int64)
    home_goals = fit_pool["home_goals"].to_numpy(dtype=np.float64)
    away_goals = fit_pool["away_goals"].to_numpy(dtype=np.float64)
    match_time = fit_pool["match_time"].to_numpy(dtype=np.float64)

    seasons = sorted(int(s) for s in history["Saison"].unique())
    checkpoint_seasons = [s for s in seasons if s > holdout_season - checkpoint_seasons_back]
    checkpoints = _checkpoint_grid(history, checkpoint_seasons)
    if not checkpoints:
        raise ValueError(f"No checkpoints for seasons {checkpoint_seasons}")
    print(
        f"[train] {len(checkpoints)} matchday checkpoints over seasons "
        f"{checkpoint_seasons} (half-life {half_life_seasons} seasons)"
    )

    matchdays_per_season = history.groupby("Saison")["Spieltag"].max().to_dict()

    rating_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    warm_start: np.ndarray | None = None
    n_converged = 0

    for season, matchday in checkpoints:
        checkpoint_time = season + (matchday - 1) / float(matchdays_per_season[season])
        past = match_time < checkpoint_time
        n_past = int(past.sum())
        if n_past < MIN_FIT_MATCHES:
            continue

        weights = 0.5 ** ((checkpoint_time - match_time[past]) / half_life_seasons)

        fit = fit_dixon_coles(
            home_idx[past],
            away_idx[past],
            home_goals[past],
            away_goals[past],
            weights,
            n_teams,
            l2=l2,
            rho_bounds=rho_bounds,
            max_iter=max_iter,
            x0=warm_start,
        )
        warm_start = fit.theta
        n_converged += int(fit.converged)

        evidence = np.bincount(home_idx[past], weights, n_teams) + np.bincount(
            away_idx[past], weights, n_teams
        )
        known = evidence >= min_team_evidence
        if not known.any():
            raise ValueError(
                f"No team reaches min_team_evidence={min_team_evidence} at "
                f"checkpoint {season}/{matchday}"
            )

        for i in np.flatnonzero(known):
            rating_rows.append(
                {
                    "Saison": int(season),
                    "Spieltag": int(matchday),
                    "team_norm": teams[i],
                    "attack": float(fit.attack[i]),
                    "defence": float(fit.defence[i]),
                    "evidence": float(evidence[i]),
                }
            )

        checkpoint_rows.append(
            {
                "Saison": int(season),
                "Spieltag": int(matchday),
                "intercept": fit.intercept,
                "home_advantage": fit.home_advantage,
                "rho": fit.rho,
                "newcomer_attack": float(np.quantile(fit.attack[known], newcomer_quantile)),
                "newcomer_defence": float(np.quantile(fit.defence[known], newcomer_quantile)),
                "n_fit_matches": n_past,
                "n_known_teams": int(known.sum()),
                "log_likelihood": fit.log_likelihood,
                "converged": bool(fit.converged),
            }
        )

    if not checkpoint_rows:
        raise ValueError("No checkpoint could be fitted — not enough historical matches")

    ratings_df = pd.DataFrame(rating_rows)
    checkpoints_df = pd.DataFrame(checkpoint_rows).sort_values(["Saison", "Spieltag"])
    checkpoints_df = checkpoints_df.reset_index(drop=True)

    last = checkpoints_df.iloc[-1]
    print(
        f"[train] Fitted {len(checkpoints_df)} checkpoints "
        f"({n_converged} converged), latest: home_adv={last['home_advantage']:.3f} "
        f"rho={last['rho']:.3f} teams={int(last['n_known_teams'])}"
    )

    joblib.dump(ratings_df, bundle_dir / "ratings.joblib")
    joblib.dump(checkpoints_df, bundle_dir / "checkpoints.joblib")

    train_seasons = f"{int(train_df_all['Saison'].min())}-{int(train_df_all['Saison'].max())}"
    config = {
        "goal_cap": goal_cap,
        "half_life_seasons": half_life_seasons,
        "l2": l2,
        "rho_bounds": list(rho_bounds),
        "fit_leagues": fit_leagues,
        "liga": liga,
        "newcomer_quantile": newcomer_quantile,
        "min_team_evidence": min_team_evidence,
        "checkpoint_seasons": checkpoint_seasons,
        "holdout_season": holdout_season,
        "train_seasons": train_seasons,
    }
    (bundle_dir / "dixon_coles.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    write_bundle_json(
        bundle_dir,
        model_type=MODEL_TYPE,
        recipe_name=recipe_name,
        holdout_season=holdout_season,
        train_seasons=train_seasons,
        files=BUNDLE_FILES,
    )

    print(f"[train] Bundle written to {bundle_dir}")

    n_train_liga = int((train_df_all["Liga"].astype(str).str.lower() == liga).sum())
    return {
        "model_type": MODEL_TYPE,
        "n_train": n_train_liga,
        "n_fit_matches": int(len(fit_pool)),
        "n_teams": n_teams,
        "n_checkpoints": int(len(checkpoints_df)),
        "n_converged": n_converged,
        "final_home_advantage": float(last["home_advantage"]),
        "final_rho": float(last["rho"]),
        "final_intercept": float(last["intercept"]),
        "train_seasons": train_seasons,
        "liga": liga,
    }
