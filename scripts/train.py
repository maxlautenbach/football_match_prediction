from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

BASE_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from data_loader import update_match_data_delta, update_next_matchday_df
from dataset_utils import DEFAULT_HOLDOUT_SEASON, generate_datasets_from_pickle
from eval.baselines import predict_majority_from_train
from eval.metrics import evaluate_predictions, print_metrics
from eval.mlflow_model import log_kicktipp_pyfunc
from eval.mlflow_utils import (
    REGISTERED_MODEL_NAME,
    log_metrics,
    log_params,
    log_pandas_dataset,
    set_model_alias,
    setup_mlflow,
)
from model import Model

DATASETS_DIR = BASE_DIR / "datasets"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"

TRAIN_CSV = DATASETS_DIR / "train.csv"
MV_CSV = DATASETS_DIR / "TeamMarketValues.csv"


def normalize_team_name(name: str) -> str:
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return ""
    s = str(name).strip()
    s = " ".join(s.split())

    # Normalize dot spacing: "1. FC" / "1.FC" -> "1.FC"
    s = s.replace(". ", ".")

    # Common punctuation normalizations
    s = s.replace("'", "'")
    s = s.replace("–", "-")
    s = s.replace("—", "-")

    return s


def parse_result(result_str: str) -> Tuple[int, int]:
    home, away = str(result_str).split(":")
    return int(home), int(away)


def _best_fuzzy_match(needle: str, haystack: Iterable[str]) -> Tuple[str | None, float]:
    # Lightweight fuzzy matcher using SequenceMatcher ratio.
    # Avoids extra dependencies.
    from difflib import SequenceMatcher

    best_name = None
    best_score = 0.0
    for candidate in haystack:
        score = SequenceMatcher(None, needle, candidate).ratio()
        if score > best_score:
            best_name = candidate
            best_score = score
    return best_name, best_score


def build_mv_alias_map(
    match_teams: Iterable[str],
    mv_teams: Iterable[str],
    min_score: float = 0.92,
) -> Dict[str, str]:
    mv_set = set(mv_teams)
    alias_map: Dict[str, str] = {}

    # Manual overrides for known common mismatches
    manual = {
        normalize_team_name("Erzgebirge Aue"): normalize_team_name("FC Erzgebirge Aue"),
    }
    for k, v in manual.items():
        if v in mv_set:
            alias_map[k] = v

    for t in match_teams:
        if t in mv_set:
            alias_map.setdefault(t, t)
            continue
        if t in alias_map:
            continue

        best, score = _best_fuzzy_match(t, mv_set)
        if best is not None and score >= min_score:
            alias_map[t] = best

    return alias_map


@dataclass(frozen=True)
class FeatureConfig:
    feature_columns: Tuple[str, ...]
    cat_feature_names: Tuple[str, ...]
    goal_cap: int


def compute_team_aggregates(train_df: pd.DataFrame) -> pd.DataFrame:
    # Aggregates are computed from training data only.
    # They are used as global priors at inference time.
    df = train_df.copy()

    # Home context
    home_stats = (
        df.groupby("team_home_norm")
        .agg(
            home_gf_mean=("home_goals", "mean"),
            home_ga_mean=("away_goals", "mean"),
            home_matches=("home_goals", "size"),
        )
        .reset_index()
    )

    # Away context
    away_stats = (
        df.groupby("team_away_norm")
        .agg(
            away_gf_mean=("away_goals", "mean"),
            away_ga_mean=("home_goals", "mean"),
            away_matches=("away_goals", "size"),
        )
        .reset_index()
        .rename(columns={"team_away_norm": "team_home_norm"})
    )

    # Merge into single team table keyed by team_home_norm (normalized team name)
    team_stats = home_stats.merge(away_stats, on="team_home_norm", how="outer")
    team_stats = team_stats.rename(columns={"team_home_norm": "team_norm"})

    # Fill missing means with global means
    global_home_gf = df["home_goals"].mean()
    global_home_ga = df["away_goals"].mean()
    global_away_gf = df["away_goals"].mean()
    global_away_ga = df["home_goals"].mean()

    team_stats["home_gf_mean"] = team_stats["home_gf_mean"].fillna(global_home_gf)
    team_stats["home_ga_mean"] = team_stats["home_ga_mean"].fillna(global_home_ga)
    team_stats["away_gf_mean"] = team_stats["away_gf_mean"].fillna(global_away_gf)
    team_stats["away_ga_mean"] = team_stats["away_ga_mean"].fillna(global_away_ga)
    team_stats["home_matches"] = team_stats["home_matches"].fillna(0).astype(int)
    team_stats["away_matches"] = team_stats["away_matches"].fillna(0).astype(int)

    return team_stats


def compute_team_matchday_form_table(train_df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Compute a snapshot table of team form before each matchday.

    Keyed by (team_norm, Saison, Spieltag), where Spieltag refers to the matchday
    whose matches have NOT been played yet.
    """

    df = train_df.sort_values(["Saison", "Spieltag"]).reset_index(drop=True)

    def empty_state() -> dict:
        return {
            "played": 0,
            "points": 0,
            "gf": 0,
            "ga": 0,
            "recent_points": [],
            "recent_gf": [],
            "recent_ga": [],
        }

    rows = []

    for saison, df_s in df.groupby("Saison", sort=True):
        state: Dict[str, dict] = {}
        matchdays = sorted(df_s["Spieltag"].unique())

        for spieltag in matchdays:
            # Snapshot BEFORE processing this matchday
            for team, st in state.items():
                played = st["played"]
                gf = st["gf"]
                ga = st["ga"]
                pts = st["points"]

                recent_pts = st["recent_points"][-window:]
                recent_gf = st["recent_gf"][-window:]
                recent_ga = st["recent_ga"][-window:]

                rows.append(
                    {
                        "team_norm": team,
                        "Saison": int(saison),
                        "Spieltag": int(spieltag),
                        "season_played": played,
                        "season_points_per_game": (pts / played) if played > 0 else np.nan,
                        "season_gf_per_game": (gf / played) if played > 0 else np.nan,
                        "season_ga_per_game": (ga / played) if played > 0 else np.nan,
                        "recent_played": len(recent_pts),
                        "recent_points_per_game": (sum(recent_pts) / len(recent_pts)) if recent_pts else np.nan,
                        "recent_gf_per_game": (sum(recent_gf) / len(recent_gf)) if recent_gf else np.nan,
                        "recent_ga_per_game": (sum(recent_ga) / len(recent_ga)) if recent_ga else np.nan,
                    }
                )

            # Update using matches of this matchday
            md_matches = df_s[df_s["Spieltag"] == spieltag]
            for _, r in md_matches.iterrows():
                home = r["team_home_norm"]
                away = r["team_away_norm"]
                hg = int(r["home_goals"])
                ag = int(r["away_goals"])

                state.setdefault(home, empty_state())
                state.setdefault(away, empty_state())

                if hg > ag:
                    home_pts, away_pts = 3, 0
                elif hg < ag:
                    home_pts, away_pts = 0, 3
                else:
                    home_pts, away_pts = 1, 1

                # Home update
                st_h = state[home]
                st_h["played"] += 1
                st_h["points"] += home_pts
                st_h["gf"] += hg
                st_h["ga"] += ag
                st_h["recent_points"].append(home_pts)
                st_h["recent_gf"].append(hg)
                st_h["recent_ga"].append(ag)

                # Away update
                st_a = state[away]
                st_a["played"] += 1
                st_a["points"] += away_pts
                st_a["gf"] += ag
                st_a["ga"] += hg
                st_a["recent_points"].append(away_pts)
                st_a["recent_gf"].append(ag)
                st_a["recent_ga"].append(hg)

        # Also snapshot for matchday after the last one (useful for future matches)
        next_spieltag = int(max(matchdays)) + 1
        for team, st in state.items():
            played = st["played"]
            gf = st["gf"]
            ga = st["ga"]
            pts = st["points"]

            recent_pts = st["recent_points"][-window:]
            recent_gf = st["recent_gf"][-window:]
            recent_ga = st["recent_ga"][-window:]

            rows.append(
                {
                    "team_norm": team,
                    "Saison": int(saison),
                    "Spieltag": next_spieltag,
                    "season_played": played,
                    "season_points_per_game": (pts / played) if played > 0 else np.nan,
                    "season_gf_per_game": (gf / played) if played > 0 else np.nan,
                    "season_ga_per_game": (ga / played) if played > 0 else np.nan,
                    "recent_played": len(recent_pts),
                    "recent_points_per_game": (sum(recent_pts) / len(recent_pts)) if recent_pts else np.nan,
                    "recent_gf_per_game": (sum(recent_gf) / len(recent_gf)) if recent_gf else np.nan,
                    "recent_ga_per_game": (sum(recent_ga) / len(recent_ga)) if recent_ga else np.nan,
                }
            )

    return pd.DataFrame(rows)


def compute_team_matchday_elo_table(
    train_df: pd.DataFrame,
    k_factor: float = 20.0,
    home_advantage: float = 50.0,
    base_rating: float = 1500.0,
) -> pd.DataFrame:
    """Compute Elo ratings snapshot before each matchday.

    Keyed by (team_norm, Saison, Spieltag) where Spieltag is the matchday about to be played.
    """

    df = train_df.sort_values(["Saison", "Spieltag"]).reset_index(drop=True)
    rows = []

    def expected_score(r_a: float, r_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

    for saison, df_s in df.groupby("Saison", sort=True):
        ratings: Dict[str, float] = {}
        matchdays = sorted(df_s["Spieltag"].unique())

        for spieltag in matchdays:
            # Snapshot BEFORE processing this matchday
            for team, r in ratings.items():
                rows.append(
                    {
                        "team_norm": team,
                        "Saison": int(saison),
                        "Spieltag": int(spieltag),
                        "elo": float(r),
                    }
                )

            md_matches = df_s[df_s["Spieltag"] == spieltag]
            for _, r in md_matches.iterrows():
                home = r["team_home_norm"]
                away = r["team_away_norm"]
                hg = int(r["home_goals"])
                ag = int(r["away_goals"])

                ratings.setdefault(home, base_rating)
                ratings.setdefault(away, base_rating)

                r_home = ratings[home] + home_advantage
                r_away = ratings[away]

                exp_home = expected_score(r_home, r_away)
                exp_away = 1.0 - exp_home

                if hg > ag:
                    act_home, act_away = 1.0, 0.0
                elif hg < ag:
                    act_home, act_away = 0.0, 1.0
                else:
                    act_home, act_away = 0.5, 0.5

                ratings[home] = ratings[home] + k_factor * (act_home - exp_home)
                ratings[away] = ratings[away] + k_factor * (act_away - exp_away)

        # Snapshot for matchday after the last one
        next_spieltag = int(max(matchdays)) + 1
        for team, r in ratings.items():
            rows.append(
                {
                    "team_norm": team,
                    "Saison": int(saison),
                    "Spieltag": next_spieltag,
                    "elo": float(r),
                }
            )

    return pd.DataFrame(rows)


def build_features(
    X: pd.DataFrame,
    mv_df: pd.DataFrame,
    mv_alias_map: Dict[str, str],
    team_aggs: pd.DataFrame,
    team_form: pd.DataFrame,
    team_elo: pd.DataFrame,
) -> pd.DataFrame:
    df = X.copy()

    df["team_home_norm"] = df["Team Home"].map(normalize_team_name)
    df["team_away_norm"] = df["Team Away"].map(normalize_team_name)

    # Apply alias mapping to improve MV joins
    df["team_home_mv"] = df["team_home_norm"].map(lambda x: mv_alias_map.get(x, x))
    df["team_away_mv"] = df["team_away_norm"].map(lambda x: mv_alias_map.get(x, x))

    mv = mv_df.copy()
    mv = mv.rename(columns={"MarketValue": "market_value"})

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

    # Market value derived features
    df["home_market_value"] = df["home_market_value"].astype(float)
    df["away_market_value"] = df["away_market_value"].astype(float)
    df["mv_diff"] = df["home_market_value"] - df["away_market_value"]

    eps = 1e-6
    df["mv_ratio_log"] = np.log((df["home_market_value"] + eps) / (df["away_market_value"] + eps))

    # Team aggregate priors
    aggs = team_aggs.copy()
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

    # Fill any remaining missing aggregate priors with league averages
    for col in [
        "home_home_gf_mean",
        "home_home_ga_mean",
        "home_away_gf_mean",
        "home_away_ga_mean",
        "away_home_gf_mean",
        "away_home_ga_mean",
        "away_away_gf_mean",
        "away_away_ga_mean",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    # Elo ratings per matchday
    te = team_elo.copy()
    home_te = te.rename(columns={"team_norm": "team_home_norm", "elo": "home_elo"})
    away_te = te.rename(columns={"team_norm": "team_away_norm", "elo": "away_elo"})
    df = df.merge(home_te, on=["team_home_norm", "Saison", "Spieltag"], how="left")
    df = df.merge(away_te, on=["team_away_norm", "Saison", "Spieltag"], how="left")
    df["home_elo"] = df["home_elo"].fillna(1500.0)
    df["away_elo"] = df["away_elo"].fillna(1500.0)
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    # Team form (per-season / matchday snapshot)
    tf = team_form.copy()
    key_cols = {"team_norm", "Saison", "Spieltag"}

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

    # Final feature frame
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

    return features


def _prepare_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["team_home_norm"] = out["Team Home"].map(normalize_team_name)
    out["team_away_norm"] = out["Team Away"].map(normalize_team_name)
    goals = out["Ergebnis"].map(parse_result)
    out["home_goals"] = [g[0] for g in goals]
    out["away_goals"] = [g[1] for g in goals]
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train CatBoost models with season holdout")
    parser.add_argument("--holdout-season", type=int, default=DEFAULT_HOLDOUT_SEASON)
    parser.add_argument("--skip-delta", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--register-model",
        type=str,
        default=REGISTERED_MODEL_NAME,
        help="Registered model name (empty string to skip registry)",
    )
    parser.add_argument(
        "--alias",
        type=str,
        default="candidate",
        help="Registry alias to set on the new version (default: candidate)",
    )
    args = parser.parse_args(argv)

    holdout_season = args.holdout_season
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_delta:
        print("[train] Running delta update...")
        update_match_data_delta(data_dir=DATA_DIR, verbose=True)
        update_next_matchday_df(data_dir=DATA_DIR, verbose=True)
    else:
        print("[train] Skipping delta update")

    print(f"[train] Generating datasets (holdout season={holdout_season})...")
    train_df_all, holdout_df, mv_df_raw = generate_datasets_from_pickle(
        DATA_DIR, holdout_season=holdout_season, datasets_dir=DATASETS_DIR
    )

    feature_history = pd.concat([train_df_all, holdout_df], ignore_index=True)
    feature_history = feature_history.sort_values(["Saison", "Spieltag"]).reset_index(drop=True)

    train_df_all.to_csv(TRAIN_CSV, index=False)
    holdout_df.to_csv(DATASETS_DIR / "test.csv", index=False)
    mv_df_raw.to_csv(MV_CSV, index=False)
    print(f"[train] Saved datasets to {DATASETS_DIR}")

    mv_df = mv_df_raw.copy()
    mv_df["team_norm"] = mv_df["Team"].map(normalize_team_name)

    train_df_all = _prepare_labels(train_df_all)
    feature_history = _prepare_labels(feature_history)

    match_teams = pd.unique(
        pd.concat([feature_history["team_home_norm"], feature_history["team_away_norm"]])
    )
    mv_teams = pd.unique(mv_df["team_norm"])
    mv_alias_map = build_mv_alias_map(match_teams, mv_teams)

    tmp = pd.DataFrame({
        "team": match_teams,
        "team_mv": [mv_alias_map.get(t, t) for t in match_teams],
    })
    missing = (~tmp["team_mv"].isin(set(mv_teams))).sum()
    print(f"[train] MV alias coverage: {len(match_teams) - missing}/{len(match_teams)} teams mapped")

    team_aggs = compute_team_aggregates(train_df_all)
    team_form = compute_team_matchday_form_table(feature_history, window=5)
    elo_k = float(os.getenv("ELO_K", "20"))
    elo_home_adv = float(os.getenv("ELO_HOME_ADV", "50"))
    team_elo = compute_team_matchday_elo_table(
        feature_history, k_factor=elo_k, home_advantage=elo_home_adv
    )
    print(
        f"[train] Feature tables: aggs={len(team_aggs)} teams, "
        f"form={len(team_form)} rows, elo={len(team_elo)} rows"
    )

    if "Liga" not in train_df_all.columns:
        raise KeyError("Expected column 'Liga' in train data")
    train_df = train_df_all[train_df_all["Liga"].astype(str).str.lower() == "bl1"].copy().reset_index(drop=True)
    print(f"[train] Training filter Liga=bl1: {len(train_df)}/{len(train_df_all)} matches")

    X_all = train_df.drop(columns=["Ergebnis", "home_goals", "away_goals"], errors="ignore")
    features = build_features(X_all, mv_df, mv_alias_map, team_aggs, team_form, team_elo)
    y_home = train_df["home_goals"].astype(int)
    y_away = train_df["away_goals"].astype(int)

    sort_idx = train_df[["Saison", "Spieltag"]].sort_values(["Saison", "Spieltag"]).index
    features = features.loc[sort_idx].reset_index(drop=True)
    y_home = y_home.loc[sort_idx].reset_index(drop=True)
    y_away = y_away.loc[sort_idx].reset_index(drop=True)

    n = len(features)
    n_val = max(int(n * 0.15), 500)
    n_train = n - n_val

    X_train, X_val = features.iloc[:n_train], features.iloc[n_train:]
    y_home_train, y_home_val = y_home.iloc[:n_train], y_home.iloc[n_train:]
    y_away_train, y_away_val = y_away.iloc[:n_train], y_away.iloc[n_train:]

    cat_feature_names = ("team_home", "team_away", "wochentag")
    cat_feature_indices = [X_train.columns.get_loc(c) for c in cat_feature_names]

    train_pool_home = Pool(X_train, y_home_train, cat_features=cat_feature_indices)
    val_pool_home = Pool(X_val, y_home_val, cat_features=cat_feature_indices)
    train_pool_away = Pool(X_train, y_away_train, cat_features=cat_feature_indices)
    val_pool_away = Pool(X_val, y_away_val, cat_features=cat_feature_indices)

    cb_depth = int(os.getenv("CB_DEPTH", "6"))
    cb_lr = float(os.getenv("CB_LR", "0.03"))
    cb_l2 = float(os.getenv("CB_L2", "10.0"))

    common_params = dict(
        loss_function="Poisson",
        eval_metric="Poisson",
        depth=cb_depth,
        learning_rate=cb_lr,
        l2_leaf_reg=cb_l2,
        iterations=4000,
        random_seed=42,
        verbose=200,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=100,
    )

    home_model = CatBoostRegressor(**common_params)
    away_model = CatBoostRegressor(**common_params)

    print("[train] Fitting home-goals model...")
    home_model.fit(train_pool_home, eval_set=val_pool_home, use_best_model=True)

    print("[train] Fitting away-goals model...")
    away_model.fit(train_pool_away, eval_set=val_pool_away, use_best_model=True)

    home_model_path = ARTIFACTS_DIR / "home_goals.cbm"
    away_model_path = ARTIFACTS_DIR / "away_goals.cbm"
    home_model.save_model(home_model_path)
    away_model.save_model(away_model_path)

    lam_home_val = np.asarray(home_model.predict(X_val), dtype=float)
    lam_away_val = np.asarray(away_model.predict(X_val), dtype=float)

    def safe_scale(y: pd.Series, lam: np.ndarray) -> float:
        denom = float(np.mean(lam))
        if denom <= 1e-9:
            return 1.0
        return float(y.mean() / denom)

    home_lambda_scale = float(np.clip(safe_scale(y_home_val, lam_home_val), 0.7, 1.3))
    away_lambda_scale = float(np.clip(safe_scale(y_away_val, lam_away_val), 0.7, 1.3))

    cfg = FeatureConfig(
        feature_columns=tuple(features.columns),
        cat_feature_names=tuple(cat_feature_names),
        goal_cap=7,
    )
    meta = {
        "feature_columns": list(cfg.feature_columns),
        "cat_feature_names": list(cfg.cat_feature_names),
        "goal_cap": cfg.goal_cap,
        "home_lambda_scale": home_lambda_scale,
        "away_lambda_scale": away_lambda_scale,
        "holdout_season": holdout_season,
        "train_seasons": f"{int(train_df_all['Saison'].min())}-{int(train_df_all['Saison'].max())}",
    }

    (ARTIFACTS_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (ARTIFACTS_DIR / "mv_alias_map.json").write_text(
        json.dumps(mv_alias_map, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    joblib.dump(team_aggs, ARTIFACTS_DIR / "team_aggs.joblib")
    joblib.dump(mv_df[["team_norm", "Saison", "MarketValue"]], ARTIFACTS_DIR / "market_values.joblib")
    joblib.dump(team_form, ARTIFACTS_DIR / "team_form.joblib")
    joblib.dump(team_elo, ARTIFACTS_DIR / "team_elo.joblib")

    print(f"[train] Saved: {home_model_path}")
    print(f"[train] Saved: {away_model_path}")
    print(f"[train] Saved: {ARTIFACTS_DIR / 'meta.json'}")

    holdout_bl1 = holdout_df[holdout_df["Liga"].astype(str).str.lower() == "bl1"].copy().reset_index(drop=True)
    model = Model(artifacts_dir=ARTIFACTS_DIR)
    y_true = holdout_bl1["Ergebnis"]
    y_pred = pd.Series(model.predict(holdout_bl1.drop(columns=["Ergebnis"])))
    holdout_metrics = evaluate_predictions(y_true, y_pred)
    print_metrics(f"Holdout season {holdout_season} (new model)", holdout_metrics)

    maj, maj_preds = predict_majority_from_train(train_df_all, n=len(holdout_bl1), liga="bl1")
    baseline_metrics = evaluate_predictions(y_true, pd.Series(maj_preds))
    print_metrics(f"Holdout season {holdout_season} (majority {maj})", baseline_metrics)

    if not args.no_mlflow:
        setup_mlflow()
        run_name = args.run_name or f"train-holdout-{holdout_season}"
        train_bl1 = train_df[
            ["Team Home", "Team Away", "Ergebnis", "Saison", "Spieltag", "Wochentag", "Liga"]
        ].copy()
        register_name = args.register_model.strip() or None

        with mlflow.start_run(run_name=run_name) as run:
            log_params({
                "model_type": "catboost_poisson",
                "holdout_season": holdout_season,
                "cb_depth": cb_depth,
                "cb_lr": cb_lr,
                "cb_l2": cb_l2,
                "elo_k": elo_k,
                "elo_home_adv": elo_home_adv,
                "home_lambda_scale": home_lambda_scale,
                "away_lambda_scale": away_lambda_scale,
                "n_train_bl1": len(train_df),
                "n_holdout_bl1": len(holdout_bl1),
                "registered_model_name": register_name or "",
            })
            log_metrics(holdout_metrics)

            # Datasets appear under the Datasets / Inputs section of the run
            log_pandas_dataset(
                train_bl1,
                name=f"bl1-train-lt{holdout_season}",
                context="training",
                source=TRAIN_CSV,
                targets="Ergebnis",
            )
            log_pandas_dataset(
                holdout_bl1,
                name=f"bl1-holdout-{holdout_season}",
                context="validation",
                source=DATASETS_DIR / "test.csv",
                targets="Ergebnis",
            )

            model_info = log_kicktipp_pyfunc(
                ARTIFACTS_DIR,
                input_example=holdout_bl1.drop(columns=["Ergebnis"]),
                registered_model_name=register_name,
                artifact_path="model",
                code_dir=BASE_DIR,
            )
            print(f"[train] Logged pyfunc model: {model_info.model_uri}")

            if register_name and args.alias:
                version = model_info.registered_model_version
                if version is not None:
                    set_model_alias(register_name, args.alias, version)
                    print(
                        f"[train] Registered {register_name} v{version} "
                        f"with alias @{args.alias}"
                    )
                    print(f"[train] Load via: models:/{register_name}@{args.alias}")

            print(f"[train] Run id: {run.info.run_id}")

        with mlflow.start_run(run_name=f"baseline-majority-{holdout_season}"):
            log_params({
                "model_type": "majority",
                "majority_class": maj,
                "holdout_season": holdout_season,
            })
            log_metrics(baseline_metrics)
            log_pandas_dataset(
                holdout_bl1,
                name=f"bl1-holdout-{holdout_season}",
                context="validation",
                source=DATASETS_DIR / "test.csv",
                targets="Ergebnis",
            )

        print("[train] Logged runs to MLflow (sqlite:///mlflow.db)")


if __name__ == "__main__":
    main()
