"""Train saison outlook model → temp bundle → MLflow experiment kicktipp-saison.

Does NOT touch match artifacts/. Sets @candidate on kicktipp-saison-ausblick.
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

from eval.mlflow_model import log_saison_pyfunc
from eval.mlflow_utils import (
    ALIAS_CANDIDATE,
    SAISON_EXPERIMENT_NAME,
    SAISON_REGISTERED_MODEL_NAME,
    log_artifact_dir,
    log_metrics,
    log_pandas_dataset,
    log_params,
    set_model_alias,
    setup_mlflow,
)
from eval.saison_metrics import (
    evaluate_saison_predictions,
    holdout_saison_z_score,
    print_saison_metrics,
    saison_scores_by_season,
)
from models.contract import BUNDLE_SCHEMA_VERSION
from models.registry import default_registered_model_name
from models.saison_ausblick.model import SaisonAusblickModel
from models.saison_ausblick.train import train as train_saison_ausblick
from saison_dataset_utils import build_saison_dataset

DATASETS_DIR = BASE_DIR / "datasets"
DATA_DIR = BASE_DIR / "data"
DEFAULT_RECIPE = BASE_DIR / "recipes" / "saison_ausblick.toml"
DEFAULT_HOLDOUT_SEASON = 2025


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train saison outlook model (MLflow kicktipp-saison)")
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--holdout-season", type=int, default=DEFAULT_HOLDOUT_SEASON)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--register-model", type=str, default=None)
    parser.add_argument("--alias", type=str, default=None)
    parser.add_argument("--keep-bundle-dir", type=Path, default=None)
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Rebuild saison CSVs from pickles before training",
    )
    args = parser.parse_args(argv)

    recipe_path = args.recipe if args.recipe.is_absolute() else BASE_DIR / args.recipe
    recipe = load_recipe(recipe_path)
    model_type = str(recipe["model_type"])
    if model_type != "saison_ausblick":
        raise ValueError(f"train_saison.py expects model_type=saison_ausblick, got {model_type}")
    recipe_name = str(recipe["_recipe_name"])
    params = dict(recipe.get("params") or {})
    holdout_season = args.holdout_season

    if args.register_model is not None:
        register_name = args.register_model.strip() or None
    else:
        recipe_reg = str(recipe.get("registered_model_name") or "").strip()
        register_name = recipe_reg or default_registered_model_name(model_type)

    alias = ALIAS_CANDIDATE if args.alias is None else (args.alias.strip() or None)

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    train_csv = DATASETS_DIR / "saison_train.csv"
    holdout_csv = DATASETS_DIR / "saison_holdout.csv"
    feats_csv = DATASETS_DIR / "saison_team_features.csv"

    if args.rebuild_dataset or not train_csv.exists() or not feats_csv.exists():
        print(f"[train_saison] Building datasets (holdout={holdout_season})...")
        train_labels, holdout_labels, team_features = build_saison_dataset(
            DATA_DIR,
            holdout_season=holdout_season,
            datasets_dir=DATASETS_DIR,
        )
        train_labels.to_csv(train_csv, index=False)
        holdout_labels.to_csv(holdout_csv, index=False)
        team_features.to_csv(feats_csv, index=False)
    else:
        print("[train_saison] Loading existing saison CSVs...")
        train_labels = pd.read_csv(train_csv)
        holdout_labels = pd.read_csv(holdout_csv)
        team_features = pd.read_csv(feats_csv)

    print(
        f"[train_saison] train seasons={len(train_labels)}, "
        f"holdout={len(holdout_labels)}, team_feature_rows={len(team_features)}"
    )

    tmp_root = Path(tempfile.mkdtemp(prefix="kicktipp_saison_bundle_"))
    bundle_dir = tmp_root / "bundle"
    try:
        print(f"[train_saison] Training {model_type} (recipe={recipe_name}) → {bundle_dir}")
        train_info = train_saison_ausblick(
            train_labels,
            holdout_labels,
            team_features,
            bundle_dir,
            params=params,
            recipe_name=recipe_name,
            holdout_season=holdout_season,
        )
        train_expected = train_info.pop("_train_expected_by_season", {}) or {}

        model = SaisonAusblickModel(bundle_dir)

        # Leave-one-season scores on train (parametric model → same params)
        train_preds = [
            model.predict_saison(int(s), team_features=team_features)
            for s in train_labels["Saison"].tolist()
        ]
        train_season_scores = saison_scores_by_season(train_labels, train_preds)

        holdout_preds = [
            model.predict_saison(int(s), team_features=team_features)
            for s in holdout_labels["Saison"].tolist()
        ]
        holdout_metrics = evaluate_saison_predictions(holdout_labels, holdout_preds)
        holdout_metrics.update(
            holdout_saison_z_score(holdout_metrics["saison_score"], train_season_scores)
        )
        print_saison_metrics(f"Holdout season {holdout_season} ({model_type})", holdout_metrics)

        if holdout_preds:
            tip = holdout_preds[0]
            print(
                f"[train_saison] Holdout tips: champion={tip['champion']}, "
                f"herbst={tip['herbstmeister']}, bottom3={tip['bottom3']}, "
                f"scorer={tip['top_scorer_team']}"
            )

        if args.keep_bundle_dir is not None:
            dst = args.keep_bundle_dir
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(bundle_dir, dst)
            print(f"[train_saison] Copied bundle to {dst}")

        if not args.no_mlflow:
            setup_mlflow(experiment_name=SAISON_EXPERIMENT_NAME)
            run_name = args.run_name or f"{model_type}-{recipe_name}-holdout-{holdout_season}"

            with mlflow.start_run(run_name=run_name) as run:
                log_params(
                    {
                        "model_type": model_type,
                        "recipe_name": recipe_name,
                        "recipe_path": recipe["_recipe_path"],
                        "holdout_season": holdout_season,
                        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
                        "n_train_seasons": len(train_labels),
                        "n_holdout_seasons": len(holdout_labels),
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
                if train_expected:
                    log_metrics(
                        {
                            "mean_train_expected_score": float(
                                sum(train_expected.values()) / len(train_expected)
                            )
                        }
                    )

                log_pandas_dataset(
                    train_labels,
                    name=f"saison-train-lt{holdout_season}",
                    context="training",
                    source=train_csv,
                    targets="champion",
                )
                log_pandas_dataset(
                    holdout_labels,
                    name=f"saison-holdout-{holdout_season}",
                    context="validation",
                    source=holdout_csv,
                    targets="champion",
                )
                train_feats = team_features[team_features["Saison"] < holdout_season]
                log_pandas_dataset(
                    train_feats,
                    name=f"saison-team-features-lt{holdout_season}",
                    context="training",
                    source=feats_csv,
                    targets=None,
                )

                log_artifact_dir(bundle_dir, artifact_path="bundle")

                input_example = holdout_labels[["Saison"]].copy()
                model_info = log_saison_pyfunc(
                    bundle_dir,
                    input_example=input_example,
                    registered_model_name=register_name,
                    artifact_path="model",
                    code_dir=BASE_DIR,
                )
                print(f"[train_saison] Logged pyfunc model: {model_info.model_uri}")

                if register_name and alias:
                    version = model_info.registered_model_version
                    if version is not None:
                        set_model_alias(register_name, alias, version)
                        print(
                            f"[train_saison] Registered {register_name} v{version} "
                            f"with alias @{alias}"
                        )
                        print(f"[train_saison] Load via: models:/{register_name}@{alias}")

                print(f"[train_saison] Run id: {run.info.run_id}")
                print("[train_saison] Match artifacts/ was NOT modified")

            print(f"[train_saison] Logged run to MLflow experiment '{SAISON_EXPERIMENT_NAME}'")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
