"""Train a market-value adjusted Dixon-Coles model bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from eval.metrics import kicktipp_raw_points
from models.catboost_poisson.features import build_mv_alias_map
from models.common.teams import normalize_team_name
from models.contract import write_bundle_json
from models.dixon_coles import model as dc_model
from models.dixon_coles import train as dc_train
from models.market_dixon_coles import model as market_model

MODEL_TYPE = "market_dixon_coles"

BUNDLE_FILES = [
    "market_dixon_coles.json",
    "mv_alias_map.json",
    "market_values.joblib",
    "dixon_coles",
]


def _prepare_market_values(
    mv_df_raw: pd.DataFrame,
    match_teams: pd.Series | np.ndarray,
) -> tuple[pd.DataFrame, dict[str, str], int]:
    required = {"Team", "Saison", "MarketValue"}
    missing_columns = sorted(required - set(mv_df_raw.columns))
    if missing_columns:
        raise KeyError(f"Market-value data missing columns: {missing_columns}")

    market_values = mv_df_raw[list(required)].copy()
    market_values["MarketValue"] = pd.to_numeric(
        market_values["MarketValue"], errors="coerce"
    )
    market_values = market_values[
        market_values["MarketValue"].notna() & (market_values["MarketValue"] > 0)
    ].copy()
    if market_values.empty:
        raise ValueError("No positive market values available")

    market_values["Saison"] = market_values["Saison"].astype(int)
    market_values["team_norm"] = market_values["Team"].map(normalize_team_name)
    market_values = (
        market_values.groupby(["team_norm", "Saison"], as_index=False)["MarketValue"]
        .median()
        .sort_values(["Saison", "team_norm"])
        .reset_index(drop=True)
    )

    log_values = np.log(market_values["MarketValue"].astype(float))
    market_values["log_market_value"] = log_values
    season_median = market_values.groupby("Saison")["log_market_value"].transform(
        "median"
    )
    season_std = market_values.groupby("Saison")["log_market_value"].transform(
        lambda values: float(values.std(ddof=0))
    )
    season_std = season_std.where(season_std > 1e-8, 1.0)
    market_values["market_value_z"] = (
        market_values["log_market_value"] - season_median
    ) / season_std

    mv_teams = pd.unique(market_values["team_norm"])
    alias_map = build_mv_alias_map(match_teams, mv_teams)
    mapped = {alias_map.get(str(team), str(team)) for team in match_teams}
    coverage = sum(team in set(mv_teams) for team in mapped)
    return (
        market_values[["team_norm", "Saison", "MarketValue", "market_value_z"]],
        alias_map,
        int(coverage),
    )


def _filter_liga(df: pd.DataFrame, liga: str) -> pd.DataFrame:
    if "Liga" not in df.columns:
        return df.copy().reset_index(drop=True)
    return (
        df[df["Liga"].astype(str).str.lower() == liga]
        .copy()
        .reset_index(drop=True)
    )


def _causal_backtest(
    bundle_dir: Path,
    train_df_all: pd.DataFrame,
    *,
    liga: str,
    n_backtest_seasons: int,
) -> dict[int, float]:
    candidate = market_model.load(bundle_dir)
    baseline = dc_model.load(bundle_dir / "dixon_coles")

    available = sorted(int(season) for season in train_df_all["Saison"].unique())
    seasons = available[-n_backtest_seasons:]
    candidate_scores: dict[int, float] = {}
    baseline_scores: dict[int, float] = {}

    for season in seasons:
        season_df = _filter_liga(
            train_df_all[train_df_all["Saison"].astype(int) == season],
            liga,
        )
        if season_df.empty:
            continue
        X = season_df.drop(columns=["Ergebnis"])
        candidate_scores[season] = float(
            kicktipp_raw_points(
                season_df["Ergebnis"],
                pd.Series(candidate.predict(X)),
            )
        )
        baseline_scores[season] = float(
            kicktipp_raw_points(
                season_df["Ergebnis"],
                pd.Series(baseline.predict(X)),
            )
        )

    if not candidate_scores:
        raise ValueError("No seasons available for causal backtest")

    deltas = {
        season: candidate_scores[season] - baseline_scores[season]
        for season in candidate_scores
    }
    values = np.asarray(list(candidate_scores.values()), dtype=float)
    wins = sum(delta > 0 for delta in deltas.values())
    worst_delta = min(deltas.values())
    delta_pooled = sum(deltas.values())

    score_text = ", ".join(
        f"{season}={candidate_scores[season]:.0f} ({deltas[season]:+.0f})"
        for season in candidate_scores
    )
    print(f"[train] Causal backtest score (delta vs Dixon-Coles): {score_text}")
    print(
        "[train] Causal backtest pooled: "
        f"{values.sum():.0f}, delta={delta_pooled:+.0f}, "
        f"wins={wins}/{len(candidate_scores)}, worst_delta={worst_delta:+.0f}"
    )
    # Scores feed the holdout z-score; keep MLflow to the standard metric set.
    return candidate_scores


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
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    liga = str(params.get("liga", "bl1")).lower()
    goal_cap = int(params.get("goal_cap", 7))
    market_attack_weight = float(params.get("market_attack_weight", 0.1))
    market_defence_weight = float(params.get("market_defence_weight", 0.2))
    market_decay_matchdays = float(params.get("market_decay_matchdays", 22.5))
    probability_temperature = float(params.get("probability_temperature", 1.35))
    n_backtest_seasons = int(params.get("n_backtest_seasons", 12))

    if market_attack_weight < 0 or market_defence_weight < 0:
        raise ValueError("market-value weights must be non-negative")
    if market_decay_matchdays <= 0:
        raise ValueError("market_decay_matchdays must be positive")
    if probability_temperature <= 0:
        raise ValueError("probability_temperature must be positive")
    if n_backtest_seasons < 1:
        raise ValueError("n_backtest_seasons must be positive")

    feature_history = pd.concat([train_df_all, holdout_df], ignore_index=True)
    match_teams = pd.unique(
        pd.concat(
            [
                feature_history["Team Home"].map(normalize_team_name),
                feature_history["Team Away"].map(normalize_team_name),
            ]
        )
    )
    market_values, alias_map, coverage = _prepare_market_values(
        mv_df_raw,
        match_teams,
    )
    print(f"[train] Market-value alias coverage: {coverage}/{len(match_teams)} teams")

    dc_params = dict(params.get("dixon_coles") or {})
    dc_params.setdefault("liga", liga)
    dc_params.setdefault("goal_cap", goal_cap)
    dc_params["checkpoint_seasons_back"] = max(
        int(dc_params.get("checkpoint_seasons_back", 2)),
        n_backtest_seasons + 1,
    )
    # The wrapper owns calibration and temperature; keeping the base neutral
    # avoids applying either transformation twice.
    dc_params["home_lambda_scale"] = 1.0
    dc_params["away_lambda_scale"] = 1.0
    dc_params["probability_temperature"] = 1.0

    base_info = dc_train.train(
        train_df_all.copy(),
        holdout_df.copy(),
        mv_df_raw,
        bundle_dir / "dixon_coles",
        params=dc_params,
        recipe_name=recipe_name,
        holdout_season=holdout_season,
    )

    train_seasons = (
        f"{int(train_df_all['Saison'].min())}-{int(train_df_all['Saison'].max())}"
    )
    config = {
        "goal_cap": goal_cap,
        "liga": liga,
        "market_attack_weight": market_attack_weight,
        "market_defence_weight": market_defence_weight,
        "market_decay_matchdays": market_decay_matchdays,
        "probability_temperature": probability_temperature,
        "n_backtest_seasons": n_backtest_seasons,
        "holdout_season": holdout_season,
        "train_seasons": train_seasons,
    }
    (bundle_dir / "market_dixon_coles.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (bundle_dir / "mv_alias_map.json").write_text(
        json.dumps(alias_map, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    joblib.dump(market_values, bundle_dir / "market_values.joblib")

    write_bundle_json(
        bundle_dir,
        model_type=MODEL_TYPE,
        recipe_name=recipe_name,
        holdout_season=holdout_season,
        train_seasons=train_seasons,
        files=BUNDLE_FILES,
    )

    backtest_scores = _causal_backtest(
        bundle_dir,
        train_df_all,
        liga=liga,
        n_backtest_seasons=n_backtest_seasons,
    )
    print(f"[train] Bundle written to {bundle_dir}")

    n_train_liga = len(_filter_liga(train_df_all, liga))
    return {
        "model_type": MODEL_TYPE,
        "n_train": n_train_liga,
        "n_fit_matches": base_info.get("n_fit_matches"),
        "n_teams": base_info.get("n_teams"),
        "n_checkpoints": base_info.get("n_checkpoints"),
        "market_attack_weight": market_attack_weight,
        "market_defence_weight": market_defence_weight,
        "market_decay_matchdays": market_decay_matchdays,
        "probability_temperature": probability_temperature,
        "market_value_team_coverage": coverage,
        "backtest_seasons": (
            f"{min(backtest_scores)}-{max(backtest_scores)}"
            if backtest_scores
            else ""
        ),
        "train_seasons": train_seasons,
        "liga": liga,
        "_backtest_season_scores": backtest_scores,
    }
