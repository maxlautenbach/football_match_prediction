"""MLflow pyfunc wrapper around the Kicktipp Model artifact bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel, PythonModelContext


class KicktippPyFuncModel(PythonModel):
    """Loads `Model` from an artifacts bundle directory."""

    def load_context(self, context: PythonModelContext) -> None:
        from model import Model

        bundle = Path(context.artifacts["bundle"])
        self.model = Model(artifacts_dir=bundle)
        if not self.model.ready:
            raise RuntimeError(f"Failed to load Kicktipp model from {bundle}")

    def predict(self, context: PythonModelContext, model_input: pd.DataFrame, params: dict | None = None) -> pd.Series:
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        preds = self.model.predict(model_input)
        return pd.Series(preds, name="Ergebnis", index=model_input.index)


def log_kicktipp_pyfunc(
    artifacts_dir: Path,
    *,
    input_example: pd.DataFrame,
    registered_model_name: str | None = "bundesliga-kicktipp",
    artifact_path: str = "model",
    code_dir: Path | None = None,
) -> mlflow.models.model.ModelInfo:
    """
    Log the artifact bundle as an MLflow pyfunc and optionally register it.

    Returns ModelInfo from mlflow.pyfunc.log_model.
    """
    from model import Model

    # Ensure example has no label column
    example = input_example.copy()
    if "Ergebnis" in example.columns:
        example = example.drop(columns=["Ergebnis"])

    # Infer signature from a live predict
    live = Model(artifacts_dir=artifacts_dir)
    if not live.ready:
        raise RuntimeError(f"Cannot log pyfunc — artifacts not ready at {artifacts_dir}")
    example_preds = pd.Series(live.predict(example.head(min(5, len(example)))), name="Ergebnis")
    signature = infer_signature(example.head(min(5, len(example))), example_preds)

    code_paths = None
    if code_dir is not None:
        # Ship model.py so the pyfunc can import Model when loaded elsewhere
        model_py = code_dir / "model.py"
        code_paths = [str(model_py)] if model_py.exists() else [str(code_dir)]

    pip_requirements = [
        "pandas>=2.2",
        "numpy>=1.26",
        "catboost>=1.2.7",
        "joblib>=1.4",
        "scikit-learn>=1.5",
    ]

    return mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=KicktippPyFuncModel(),
        artifacts={"bundle": str(artifacts_dir)},
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=example.head(min(3, len(example))),
        code_paths=code_paths,
        pip_requirements=pip_requirements,
        metadata={"bundle_files": sorted(p.name for p in artifacts_dir.iterdir() if p.is_file())},
    )
