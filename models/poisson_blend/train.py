"""Train the CatBoost x Dixon-Coles lambda blend into a bundle directory.

The blend weight is chosen WITHOUT looking at the holdout season: the last
``n_calibration_seasons`` training seasons are replayed as inner holdouts
(sub-models retrained on strictly earlier data), the weight grid is scored on
pooled Kicktipp points there, and only then are both sub-models retrained on
the full training data with the frozen weight. The real holdout therefore
stays untouched during weight selection, keeping its z-score honest.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from models.catboost_poisson import model as cb_model
from models.catboost_poisson import train as cb_train
from models.common.kicktipp import build_kicktipp_points_matrix
from models.contract import write_bundle_json
from models.dixon_coles import model as dc_model
from models.dixon_coles import train as dc_train
from models.poisson_blend.model import decode_blended

MODEL_TYPE = "poisson_blend"

BUNDLE_FILES = ["blend.json", "catboost", "dixon_coles"]


def _filter_liga(df: pd.DataFrame, liga: str) -> pd.DataFrame:
    return (
        df[df["Liga"].astype(str).str.lower() == liga].copy().reset_index(drop=True)
    )


def _train_submodels(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    mv_df_raw: pd.DataFrame,
    dst: Path,
    *,
    cb_params: Mapping[str, Any],
    dc_params: Mapping[str, Any],
    recipe_name: str,
    holdout_season: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cb_info = cb_train.train(
        train_df.copy(),
        holdout_df.copy(),
        mv_df_raw,
        dst / "catboost",
        params=cb_params,
        recipe_name=recipe_name,
        holdout_season=holdout_season,
    )
    dc_info = dc_train.train(
        train_df.copy(),
        holdout_df.copy(),
        mv_df_raw,
        dst / "dixon_coles",
        params=dc_params,
        recipe_name=recipe_name,
        holdout_season=holdout_season,
    )
    return cb_info, dc_info


def _collect_lambdas(bundle_root: Path, eval_df: pd.DataFrame) -> dict[str, np.ndarray]:
    X = eval_df.drop(columns=["Ergebnis"])
    cb = cb_model.load(bundle_root / "catboost")
    dc = dc_model.load(bundle_root / "dixon_coles")
    cb_home, cb_away = cb.predict_lambdas(X)
    dc_home, dc_away, rho = dc.predict_lambdas(X)
    return {
        "cb_home": cb_home,
        "cb_away": cb_away,
        "dc_home": dc_home,
        "dc_away": dc_away,
        "rho": rho,
    }


def _calibrate_weight(
    train_df_all: pd.DataFrame,
    mv_df_raw: pd.DataFrame,
    *,
    liga: str,
    goal_cap: int,
    n_calibration_seasons: int,
    weight_grid_step: float,
    cb_params: Mapping[str, Any],
    dc_params: Mapping[str, Any],
    recipe_name: str,
) -> dict[str, Any]:
    from eval.metrics import kicktipp_raw_points

    seasons = sorted(int(s) for s in train_df_all["Saison"].unique())
    calib_seasons = seasons[-n_calibration_seasons:]
    if len(seasons) <= n_calibration_seasons:
        raise ValueError(
            f"Need more than {n_calibration_seasons} train seasons for calibration, "
            f"got {len(seasons)}"
        )

    rows: list[dict[str, np.ndarray]] = []
    y_true_parts: list[pd.Series] = []
    for season in calib_seasons:
        inner_train = train_df_all[train_df_all["Saison"] < season].reset_index(drop=True)
        inner_hold = train_df_all[train_df_all["Saison"] == season].reset_index(drop=True)
        print(
            f"[train] Calibration fold {season}: "
            f"{len(inner_train)} train rows, {len(inner_hold)} eval rows"
        )

        tmp = Path(tempfile.mkdtemp(prefix=f"blend_calib_{season}_"))
        try:
            _train_submodels(
                inner_train,
                inner_hold,
                mv_df_raw,
                tmp,
                cb_params=cb_params,
                dc_params=dc_params,
                recipe_name=f"{recipe_name}-calib-{season}",
                holdout_season=season,
            )
            eval_df = _filter_liga(inner_hold, liga)
            rows.append(_collect_lambdas(tmp, eval_df))
            y_true_parts.append(eval_df["Ergebnis"].reset_index(drop=True))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    pooled = {
        key: np.concatenate([r[key] for r in rows]) for key in rows[0]
    }
    y_true = pd.concat(y_true_parts, ignore_index=True)

    points_matrix = build_kicktipp_points_matrix(goal_cap)
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, weight_grid_step), 4)
    curve: dict[float, int] = {}
    for w in grid:
        preds = decode_blended(
            pooled["cb_home"],
            pooled["cb_away"],
            pooled["dc_home"],
            pooled["dc_away"],
            pooled["rho"],
            float(w),
            goal_cap,
            points_matrix,
        )
        curve[float(w)] = int(kicktipp_raw_points(y_true, pd.Series(preds)))

    best_points = max(curve.values())
    # Break ties toward the middle: an even blend is the least presumptuous.
    best_w = min(
        (w for w, pts in curve.items() if pts == best_points),
        key=lambda w: abs(w - 0.5),
    )
    print("[train] Calibration curve (weight → Kicktipp raw points):")
    for w in sorted(curve):
        marker = "  <-- chosen" if w == best_w else ""
        print(f"    w={w:.2f}: {curve[w]}{marker}")

    return {
        "blend_weight": float(best_w),
        "calibration_seasons": calib_seasons,
        "calibration_curve": {f"{w:.2f}": pts for w, pts in sorted(curve.items())},
        "calib_kt_raw_catboost": curve[0.0],
        "calib_kt_raw_dixon_coles": curve[1.0],
        "calib_kt_raw_best": best_points,
        "n_calibration_matches": int(len(y_true)),
    }


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
    blend_weight = params.get("blend_weight", "auto")
    n_calibration_seasons = int(params.get("n_calibration_seasons", 2))
    weight_grid_step = float(params.get("weight_grid_step", 0.1))

    cb_params = dict(params.get("catboost") or {})
    cb_params.setdefault("liga", liga)
    cb_params.setdefault("goal_cap", goal_cap)
    dc_params = dict(params.get("dixon_coles") or {})
    dc_params.setdefault("liga", liga)
    dc_params.setdefault("goal_cap", goal_cap)

    calibration: dict[str, Any]
    if isinstance(blend_weight, str):
        if blend_weight != "auto":
            raise ValueError(f"blend_weight must be a float or 'auto', got {blend_weight!r}")
        calibration = _calibrate_weight(
            train_df_all,
            mv_df_raw,
            liga=liga,
            goal_cap=goal_cap,
            n_calibration_seasons=n_calibration_seasons,
            weight_grid_step=weight_grid_step,
            cb_params=cb_params,
            dc_params=dc_params,
            recipe_name=recipe_name,
        )
    else:
        w = float(blend_weight)
        if not 0.0 <= w <= 1.0:
            raise ValueError(f"blend_weight must be in [0, 1], got {w}")
        calibration = {"blend_weight": w, "calibration_seasons": []}

    weight = float(calibration["blend_weight"])
    print(f"[train] Blend weight (CatBoost→Dixon-Coles): {weight:.2f}")

    print("[train] Training final sub-models on full training data...")
    cb_info, dc_info = _train_submodels(
        train_df_all,
        holdout_df,
        mv_df_raw,
        bundle_dir,
        cb_params=cb_params,
        dc_params=dc_params,
        recipe_name=recipe_name,
        holdout_season=holdout_season,
    )

    train_seasons = f"{int(train_df_all['Saison'].min())}-{int(train_df_all['Saison'].max())}"
    blend_config = {
        "blend_weight": weight,
        "goal_cap": goal_cap,
        "liga": liga,
        "holdout_season": holdout_season,
        "train_seasons": train_seasons,
        **{k: v for k, v in calibration.items() if k != "blend_weight"},
    }
    (bundle_dir / "blend.json").write_text(
        json.dumps(blend_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
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
    info: dict[str, Any] = {
        "model_type": MODEL_TYPE,
        "n_train": n_train_liga,
        "blend_weight": weight,
        "weight_grid_step": weight_grid_step,
        "calibration_seasons": calibration.get("calibration_seasons", []),
        "cb_home_lambda_scale": cb_info.get("home_lambda_scale"),
        "cb_away_lambda_scale": cb_info.get("away_lambda_scale"),
        "dc_final_rho": dc_info.get("final_rho"),
        "dc_final_home_advantage": dc_info.get("final_home_advantage"),
        "train_seasons": train_seasons,
        "liga": liga,
    }
    for key in (
        "calib_kt_raw_catboost",
        "calib_kt_raw_dixon_coles",
        "calib_kt_raw_best",
        "n_calibration_matches",
    ):
        if key in calibration:
            info[key] = calibration[key]
    return info
