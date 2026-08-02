"""Train CatBoost Poisson home/away goal models into a bundle directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from models.catboost_poisson.features import (
    DEFAULT_CAT_FEATURES,
    FeatureConfig,
    build_features,
    build_mv_alias_map,
    compute_team_aggregates,
    compute_team_matchday_elo_table,
    compute_team_matchday_form_table,
    normalize_team_name,
    prepare_labels,
)
from models.contract import write_bundle_json

MODEL_TYPE = "catboost_poisson"

BUNDLE_FILES = [
    "bundle.json",
    "meta.json",
    "home_goals.cbm",
    "away_goals.cbm",
    "mv_alias_map.json",
    "team_aggs.joblib",
    "market_values.joblib",
    "team_form.joblib",
    "team_elo.joblib",
]


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
    """Fit models and write a complete bundle. Does not touch production artifacts/."""
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    liga = str(params.get("liga", "bl1")).lower()
    goal_cap = int(params.get("goal_cap", 7))
    cb_depth = int(params.get("cb_depth", 6))
    cb_lr = float(params.get("cb_lr", 0.03))
    cb_l2 = float(params.get("cb_l2", 10.0))
    elo_k = float(params.get("elo_k", 20))
    elo_home_adv = float(params.get("elo_home_adv", 50))
    form_window = int(params.get("form_window", 5))
    iterations = int(params.get("iterations", 4000))
    val_fraction = float(params.get("val_fraction", 0.15))
    min_val = int(params.get("min_val", 500))

    feature_history = pd.concat([train_df_all, holdout_df], ignore_index=True)
    feature_history = feature_history.sort_values(["Saison", "Spieltag"]).reset_index(drop=True)

    mv_df = mv_df_raw.copy()
    mv_df["team_norm"] = mv_df["Team"].map(normalize_team_name)

    train_df_all = prepare_labels(train_df_all)
    feature_history = prepare_labels(feature_history)

    match_teams = pd.unique(
        pd.concat([feature_history["team_home_norm"], feature_history["team_away_norm"]])
    )
    mv_teams = pd.unique(mv_df["team_norm"])
    mv_alias_map = build_mv_alias_map(match_teams, mv_teams)

    tmp = pd.DataFrame(
        {
            "team": match_teams,
            "team_mv": [mv_alias_map.get(t, t) for t in match_teams],
        }
    )
    missing = (~tmp["team_mv"].isin(set(mv_teams))).sum()
    print(f"[train] MV alias coverage: {len(match_teams) - missing}/{len(match_teams)} teams mapped")

    team_aggs = compute_team_aggregates(train_df_all)
    team_form = compute_team_matchday_form_table(feature_history, window=form_window)
    team_elo = compute_team_matchday_elo_table(
        feature_history, k_factor=elo_k, home_advantage=elo_home_adv
    )
    print(
        f"[train] Feature tables: aggs={len(team_aggs)} teams, "
        f"form={len(team_form)} rows, elo={len(team_elo)} rows"
    )

    if "Liga" not in train_df_all.columns:
        raise KeyError("Expected column 'Liga' in train data")
    train_df = (
        train_df_all[train_df_all["Liga"].astype(str).str.lower() == liga]
        .copy()
        .reset_index(drop=True)
    )
    print(f"[train] Training filter Liga={liga}: {len(train_df)}/{len(train_df_all)} matches")

    X_all = train_df.drop(columns=["Ergebnis", "home_goals", "away_goals"], errors="ignore")
    features = build_features(X_all, mv_df, mv_alias_map, team_aggs, team_form, team_elo)
    y_home = train_df["home_goals"].astype(int)
    y_away = train_df["away_goals"].astype(int)

    sort_idx = train_df[["Saison", "Spieltag"]].sort_values(["Saison", "Spieltag"]).index
    features = features.loc[sort_idx].reset_index(drop=True)
    y_home = y_home.loc[sort_idx].reset_index(drop=True)
    y_away = y_away.loc[sort_idx].reset_index(drop=True)

    n = len(features)
    n_val = max(int(n * val_fraction), min_val)
    n_train = n - n_val

    X_train, X_val = features.iloc[:n_train], features.iloc[n_train:]
    y_home_train, y_home_val = y_home.iloc[:n_train], y_home.iloc[n_train:]
    y_away_train, y_away_val = y_away.iloc[:n_train], y_away.iloc[n_train:]

    cat_feature_names = DEFAULT_CAT_FEATURES
    cat_feature_indices = [X_train.columns.get_loc(c) for c in cat_feature_names]

    train_pool_home = Pool(X_train, y_home_train, cat_features=cat_feature_indices)
    val_pool_home = Pool(X_val, y_home_val, cat_features=cat_feature_indices)
    train_pool_away = Pool(X_train, y_away_train, cat_features=cat_feature_indices)
    val_pool_away = Pool(X_val, y_away_val, cat_features=cat_feature_indices)

    common_params = dict(
        loss_function="Poisson",
        eval_metric="Poisson",
        depth=cb_depth,
        learning_rate=cb_lr,
        l2_leaf_reg=cb_l2,
        iterations=iterations,
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

    home_model_path = bundle_dir / "home_goals.cbm"
    away_model_path = bundle_dir / "away_goals.cbm"
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
        goal_cap=goal_cap,
    )
    train_seasons = f"{int(train_df_all['Saison'].min())}-{int(train_df_all['Saison'].max())}"
    meta = {
        "feature_columns": list(cfg.feature_columns),
        "cat_feature_names": list(cfg.cat_feature_names),
        "goal_cap": cfg.goal_cap,
        "home_lambda_scale": home_lambda_scale,
        "away_lambda_scale": away_lambda_scale,
        "holdout_season": holdout_season,
        "train_seasons": train_seasons,
    }

    (bundle_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (bundle_dir / "mv_alias_map.json").write_text(
        json.dumps(mv_alias_map, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    joblib.dump(team_aggs, bundle_dir / "team_aggs.joblib")
    joblib.dump(mv_df[["team_norm", "Saison", "MarketValue"]], bundle_dir / "market_values.joblib")
    joblib.dump(team_form, bundle_dir / "team_form.joblib")
    joblib.dump(team_elo, bundle_dir / "team_elo.joblib")

    write_bundle_json(
        bundle_dir,
        model_type=MODEL_TYPE,
        recipe_name=recipe_name,
        holdout_season=holdout_season,
        train_seasons=train_seasons,
        files=[f for f in BUNDLE_FILES if f != "bundle.json"],
    )

    print(f"[train] Bundle written to {bundle_dir}")

    return {
        "model_type": MODEL_TYPE,
        "n_train": len(train_df),
        "home_lambda_scale": home_lambda_scale,
        "away_lambda_scale": away_lambda_scale,
        "cb_depth": cb_depth,
        "cb_lr": cb_lr,
        "cb_l2": cb_l2,
        "elo_k": elo_k,
        "elo_home_adv": elo_home_adv,
        "train_seasons": train_seasons,
        "liga": liga,
    }
