"""Generic recipe-driven training orchestrator.

Writes a temporary bundle, evaluates holdout, logs to MLflow.
Sets @candidate (main models) or @baseline (majority). Does NOT overwrite
production artifacts/ — use scripts/promote_run.py for that.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from data_loader import update_match_data_delta, update_next_matchday_df
from dataset_utils import DEFAULT_HOLDOUT_SEASON, generate_datasets_from_pickle
from eval.metrics import (
    evaluate_predictions,
    holdout_kicktipp_z_score,
    kicktipp_scores_by_season,
    print_metrics,
)
from eval.mlflow_model import log_kicktipp_pyfunc
from eval.mlflow_utils import (
    ALIAS_BASELINE,
    ALIAS_CANDIDATE,
    BASELINE_REGISTERED_MODEL_NAME,
    log_metrics,
    log_params,
    log_pandas_dataset,
    set_model_alias,
    setup_mlflow,
)
from model import Model
from models.contract import BUNDLE_SCHEMA_VERSION, REQUIRED_COLUMNS
from models.registry import default_registered_model_name, train_model

DATASETS_DIR = BASE_DIR / "datasets"
DATA_DIR = BASE_DIR / "data"
TRAIN_CSV = DATASETS_DIR / "train.csv"
MV_CSV = DATASETS_DIR / "TeamMarketValues.csv"
DEFAULT_RECIPE = BASE_DIR / "recipes" / "catboost_poisson.toml"


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=BASE_DIR,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except Exception:
        return None


def load_recipe(path: Path) -> dict[str, Any]:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    if "model_type" not in raw:
        raise ValueError(f"Recipe {path} missing model_type")
    raw.setdefault("params", {})
    raw["_recipe_path"] = str(path)
    raw["_recipe_name"] = path.stem
    return raw


def _filter_liga(df: pd.DataFrame, liga: str) -> pd.DataFrame:
    if "Liga" not in df.columns:
        return df.copy()
    return df[df["Liga"].astype(str).str.lower() == liga.lower()].copy().reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a model from a recipe (temp bundle → MLflow)")
    parser.add_argument(
        "--recipe",
        type=Path,
        default=DEFAULT_RECIPE,
        help="Path to recipes/*.toml",
    )
    parser.add_argument("--holdout-season", type=int, default=DEFAULT_HOLDOUT_SEASON)
    parser.add_argument("--skip-delta", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--register-model",
        type=str,
        default=None,
        help="Override registered model name (empty string skips registry)",
    )
    parser.add_argument(
        "--alias",
        type=str,
        default=None,
        help="Registry alias (default: baseline for majority, else candidate)",
    )
    parser.add_argument(
        "--keep-bundle-dir",
        type=Path,
        default=None,
        help="If set, copy the trained bundle here after training (still does not touch artifacts/)",
    )
    args = parser.parse_args(argv)

    recipe_path = args.recipe if args.recipe.is_absolute() else BASE_DIR / args.recipe
    recipe = load_recipe(recipe_path)
    model_type = str(recipe["model_type"])
    recipe_name = str(recipe["_recipe_name"])
    params = dict(recipe.get("params") or {})
    liga = str(params.get("liga", "bl1"))
    holdout_season = args.holdout_season

    if args.register_model is not None:
        register_name = args.register_model.strip() or None
    else:
        recipe_reg = str(recipe.get("registered_model_name") or "").strip()
        register_name = recipe_reg or default_registered_model_name(model_type)

    if args.alias is not None:
        alias = args.alias.strip() or None
    elif register_name == BASELINE_REGISTERED_MODEL_NAME or model_type == "majority_baseline":
        alias = ALIAS_BASELINE
    else:
        alias = ALIAS_CANDIDATE

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

    train_df_all.to_csv(TRAIN_CSV, index=False)
    holdout_df.to_csv(DATASETS_DIR / "test.csv", index=False)
    mv_df_raw.to_csv(MV_CSV, index=False)
    print(f"[train] Saved datasets to {DATASETS_DIR}")

    tmp_root = Path(tempfile.mkdtemp(prefix="kicktipp_bundle_"))
    bundle_dir = tmp_root / "bundle"
    try:
        print(f"[train] Training {model_type} (recipe={recipe_name}) → {bundle_dir}")
        train_info = train_model(
            model_type,
            train_df_all,
            holdout_df,
            mv_df_raw,
            bundle_dir,
            params=params,
            recipe_name=recipe_name,
            holdout_season=holdout_season,
        )
        causal_backtest_scores = train_info.pop("_backtest_season_scores", None)

        holdout_bl1 = _filter_liga(holdout_df, liga)
        train_bl1 = _filter_liga(train_df_all, liga)
        model = Model(artifacts_dir=bundle_dir)
        y_true = holdout_bl1["Ergebnis"]
        y_pred = pd.Series(model.predict(holdout_bl1.drop(columns=["Ergebnis"])))
        holdout_metrics = evaluate_predictions(y_true, y_pred)

        # Prefer genuine rolling-origin scores supplied by a trainer. Legacy
        # model types fall back to their historical in-sample diagnostic.
        if causal_backtest_scores:
            train_season_scores = {
                int(season): float(score)
                for season, score in causal_backtest_scores.items()
            }
        else:
            train_y_true = train_bl1["Ergebnis"].reset_index(drop=True)
            train_y_pred = pd.Series(
                model.predict(train_bl1.drop(columns=["Ergebnis"]).reset_index(drop=True))
            )
            train_season_scores = kicktipp_scores_by_season(
                train_y_true,
                train_y_pred,
                train_bl1["Saison"].reset_index(drop=True),
            )
        holdout_metrics.update(
            holdout_kicktipp_z_score(holdout_metrics["kicktipp_score"], train_season_scores)
        )
        print_metrics(f"Holdout season {holdout_season} ({model_type})", holdout_metrics)

        if args.keep_bundle_dir is not None:
            dst = args.keep_bundle_dir
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(bundle_dir, dst)
            print(f"[train] Copied bundle to {dst}")

        if not args.no_mlflow:
            setup_mlflow()
            run_name = args.run_name or f"{model_type}-{recipe_name}-holdout-{holdout_season}"
            train_cols = [
                c
                for c in ["Team Home", "Team Away", "Ergebnis", "Saison", "Spieltag", "Wochentag", "Liga"]
                if c in train_bl1.columns
            ]

            with mlflow.start_run(run_name=run_name) as run:
                log_params(
                    {
                        "model_type": model_type,
                        "recipe_name": recipe_name,
                        "recipe_path": recipe["_recipe_path"],
                        "holdout_season": holdout_season,
                        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
                        "n_train_bl1": train_info.get("n_train", len(train_bl1)),
                        "n_holdout_bl1": len(holdout_bl1),
                        "registered_model_name": register_name or "",
                        "git_commit": _git_commit() or "",
                        **{k: v for k, v in params.items()},
                        **{
                            k: v
                            for k, v in train_info.items()
                            if k not in {"model_type", "n_train"}
                            and not k.startswith("_")
                            and k not in params
                        },
                    }
                )
                log_metrics(holdout_metrics)

                log_pandas_dataset(
                    train_bl1[train_cols],
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
                    bundle_dir,
                    input_example=holdout_bl1[list(REQUIRED_COLUMNS)],
                    registered_model_name=register_name,
                    artifact_path="model",
                    code_dir=BASE_DIR,
                )
                print(f"[train] Logged pyfunc model: {model_info.model_uri}")

                if register_name and alias:
                    version = model_info.registered_model_version
                    if version is not None:
                        set_model_alias(register_name, alias, version)
                        print(
                            f"[train] Registered {register_name} v{version} "
                            f"with alias @{alias}"
                        )
                        print(f"[train] Load via: models:/{register_name}@{alias}")

                print(f"[train] Run id: {run.info.run_id}")
                print("[train] artifacts/ was NOT modified — promote with scripts/promote_run.py")

            print("[train] Logged run to MLflow (sqlite:///mlflow.db)")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
