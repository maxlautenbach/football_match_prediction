"""Local MLflow helpers for training, datasets, registry, and comparison."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "kicktipp"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
REGISTERED_MODEL_NAME = "kicktipp-catboost-poisson"
BASELINE_REGISTERED_MODEL_NAME = "kicktipp-majority-baseline"
DIXON_COLES_REGISTERED_MODEL_NAME = "kicktipp-dixon-coles"

# Canonical aliases
ALIAS_CANDIDATE = "candidate"
ALIAS_PRODUCTION = "production"
ALIAS_BASELINE = "baseline"


def setup_mlflow(
    tracking_uri: str | None = None,
    experiment_name: str = EXPERIMENT_NAME,
) -> str:
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    # Local SQLite also serves as model registry store
    mlflow.set_registry_uri(uri)
    experiment = mlflow.set_experiment(experiment_name)
    return experiment.experiment_id


def log_metrics(metrics: Mapping[str, Any]) -> None:
    numeric = {
        k: float(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    if numeric:
        mlflow.log_metrics(numeric)


def log_params(params: Mapping[str, Any]) -> None:
    cleaned = {k: str(v) for k, v in params.items() if v is not None}
    if cleaned:
        mlflow.log_params(cleaned)


def log_artifact_dir(artifacts_dir: Path, artifact_path: str = "bundle") -> None:
    if artifacts_dir.exists():
        mlflow.log_artifacts(str(artifacts_dir), artifact_path=artifact_path)


def log_pandas_dataset(
    df: pd.DataFrame,
    *,
    name: str,
    context: str,
    source: str | Path | None = None,
    targets: str | None = "Ergebnis",
) -> None:
    """Log a pandas DataFrame as an MLflow Dataset input on the active run."""
    source_str = str(source) if source is not None else None
    dataset = mlflow.data.from_pandas(
        df,
        source=source_str,
        name=name,
        targets=targets if targets and targets in df.columns else None,
    )
    mlflow.log_input(dataset, context=context)


def set_model_alias(
    model_name: str,
    alias: str,
    version: str | int,
) -> None:
    client = MlflowClient()
    client.set_registered_model_alias(model_name, alias, str(version))


def get_model_version_by_alias(model_name: str, alias: str) -> Any:
    client = MlflowClient()
    return client.get_model_version_by_alias(model_name, alias)


def download_run_artifacts(run_id: str, dst_dir: Path, artifact_path: str = "model") -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    local = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=artifact_path,
        dst_path=str(dst_dir),
    )
    return Path(local)


def download_model_uri(model_uri: str, dst_dir: Path) -> Path:
    """Download a model URI (runs:/... or models:/name@version|@alias) into dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    local = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=str(dst_dir))
    return Path(local)
