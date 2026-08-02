"""Compare models and baselines on a holdout season."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "scripts"))

from dataset_utils import DEFAULT_HOLDOUT_SEASON, generate_datasets_from_pickle
from eval.baselines import predict_majority_from_train
from eval.metrics import evaluate_predictions, print_metrics
from eval.mlflow_utils import log_metrics, log_params, setup_mlflow
from model import Model


def _bl1_holdout(holdout_df: pd.DataFrame) -> pd.DataFrame:
    if "Liga" not in holdout_df.columns:
        return holdout_df.copy()
    return holdout_df[holdout_df["Liga"].astype(str).str.lower() == "bl1"].copy()


def evaluate_model_on_holdout(
    model: Model,
    holdout_df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, dict]:
    y_true = holdout_df["Ergebnis"].reset_index(drop=True)
    X = holdout_df.drop(columns=["Ergebnis"]).reset_index(drop=True)
    y_pred = pd.Series(model.predict(X), name="Ergebnis")
    metrics = evaluate_predictions(y_true, y_pred)
    return y_true, y_pred, metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare models on holdout season")
    parser.add_argument("--holdout-season", type=int, default=DEFAULT_HOLDOUT_SEASON)
    parser.add_argument("--artifacts-dir", type=Path, default=BASE_DIR / "artifacts")
    parser.add_argument("--prev-artifacts-dir", type=Path, default=None)
    parser.add_argument("--baseline", choices=["majority", "none"], default="majority")
    parser.add_argument("--log-mlflow", action="store_true", default=True)
    parser.add_argument("--no-log-mlflow", action="store_false", dest="log_mlflow")
    parser.add_argument("--run-name-prefix", type=str, default="compare")
    args = parser.parse_args(argv)

    data_dir = BASE_DIR / "data"
    datasets_dir = BASE_DIR / "datasets"

    print("=" * 60)
    print(f"Holdout compare — season {args.holdout_season}")
    print("=" * 60)

    train_df, holdout_df, _ = generate_datasets_from_pickle(
        data_dir,
        holdout_season=args.holdout_season,
        datasets_dir=datasets_dir,
    )
    holdout_bl1 = _bl1_holdout(holdout_df).reset_index(drop=True)
    print(f"BL1 holdout matches: {len(holdout_bl1)}")

    datasets_dir.mkdir(exist_ok=True)
    train_df.to_csv(datasets_dir / "train.csv", index=False)
    holdout_df.to_csv(datasets_dir / "test.csv", index=False)

    if args.log_mlflow:
        setup_mlflow()

    results: list[tuple[str, dict]] = []

    model = Model(artifacts_dir=args.artifacts_dir)
    _, _, metrics = evaluate_model_on_holdout(model, holdout_bl1)
    name = f"model:{args.artifacts_dir.name}"
    print_metrics(name, metrics)
    results.append((name, metrics))
    if args.log_mlflow:
        with mlflow.start_run(run_name=f"{args.run_name_prefix}-{args.artifacts_dir.name}"):
            log_params(
                {
                    "model_type": "catboost_poisson",
                    "artifacts_dir": str(args.artifacts_dir),
                    "holdout_season": args.holdout_season,
                }
            )
            log_metrics(metrics)

    if args.prev_artifacts_dir is not None and args.prev_artifacts_dir.exists():
        prev = Model(artifacts_dir=args.prev_artifacts_dir)
        _, _, prev_metrics = evaluate_model_on_holdout(prev, holdout_bl1)
        pname = f"model:{args.prev_artifacts_dir.name}"
        print_metrics(pname, prev_metrics)
        results.append((pname, prev_metrics))
        if args.log_mlflow:
            with mlflow.start_run(run_name=f"{args.run_name_prefix}-{args.prev_artifacts_dir.name}"):
                log_params(
                    {
                        "model_type": "catboost_poisson",
                        "artifacts_dir": str(args.prev_artifacts_dir),
                        "holdout_season": args.holdout_season,
                    }
                )
                log_metrics(prev_metrics)

    if args.baseline == "majority":
        maj, preds = predict_majority_from_train(train_df, n=len(holdout_bl1), liga="bl1")
        y_true = holdout_bl1["Ergebnis"].reset_index(drop=True)
        b_metrics = evaluate_predictions(y_true, pd.Series(preds))
        bname = f"baseline:majority:{maj}"
        print_metrics(bname, b_metrics)
        results.append((bname, b_metrics))
        if args.log_mlflow:
            with mlflow.start_run(run_name=f"{args.run_name_prefix}-majority"):
                log_params(
                    {
                        "model_type": "majority",
                        "majority_class": maj,
                        "holdout_season": args.holdout_season,
                    }
                )
                log_metrics(b_metrics)

    print("\n" + "=" * 60)
    print("Summary (Kicktipp norm 306)")
    print("=" * 60)
    for name, m in sorted(results, key=lambda x: -x[1]["kicktipp_score"]):
        print(f"  {m['kicktipp_score']:4.0f}  {name}")


if __name__ == "__main__":
    main()
