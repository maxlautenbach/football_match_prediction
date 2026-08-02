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

    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame,
        params: dict | None = None,
    ) -> pd.Series:
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        preds = self.model.predict(model_input)
        return pd.Series(preds, name="Ergebnis", index=model_input.index)


def _code_paths(code_dir: Path) -> list[str]:
    """Package root model.py + models/ package for pyfunc loading elsewhere."""
    paths: list[str] = []
    model_py = code_dir / "model.py"
    models_pkg = code_dir / "models"
    if model_py.exists():
        paths.append(str(model_py))
    if models_pkg.exists():
        paths.append(str(models_pkg))
    # eval/ needed for majority_baseline trainer imports when loading code only;
    # prediction path uses models/ + model.py. Keep eval out of code_paths to
    # stay lean — majority model.predict does not import eval.
    return paths


def log_kicktipp_pyfunc(
    artifacts_dir: Path,
    *,
    input_example: pd.DataFrame,
    registered_model_name: str | None = None,
    artifact_path: str = "model",
    code_dir: Path | None = None,
) -> Any:
    """
    Log the artifact bundle as an MLflow pyfunc and optionally register it.

    Returns ModelInfo from mlflow.pyfunc.log_model.
    """
    from model import Model
    from eval.mlflow_utils import REGISTERED_MODEL_NAME
    from models.contract import REQUIRED_COLUMNS

    if registered_model_name is None:
        registered_model_name = REGISTERED_MODEL_NAME

    example = input_example.copy()
    if "Ergebnis" in example.columns:
        example = example.drop(columns=["Ergebnis"])
    # Signature must match the public predict contract — drop extras like Liga.
    missing = [c for c in REQUIRED_COLUMNS if c not in example.columns]
    if missing:
        raise ValueError(f"input_example missing required columns: {missing}")
    example = example[list(REQUIRED_COLUMNS)]

    live = Model(artifacts_dir=artifacts_dir)
    if not live.ready:
        raise RuntimeError(f"Cannot log pyfunc — artifacts not ready at {artifacts_dir}")
    example_preds = pd.Series(
        live.predict(example.head(min(5, len(example)))), name="Ergebnis"
    )
    signature = infer_signature(example.head(min(5, len(example))), example_preds)

    code_paths = _code_paths(code_dir) if code_dir is not None else None

    pip_requirements = [
        "pandas>=2.2",
        "numpy>=1.26",
        "scipy>=1.11",
        "catboost>=1.2.7",
        "joblib>=1.4",
        "scikit-learn>=1.5",
    ]

    return mlflow.pyfunc.log_model(
        name=artifact_path,
        python_model=KicktippPyFuncModel(),
        artifacts={"bundle": str(artifacts_dir)},
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=example.head(min(3, len(example))),
        code_paths=code_paths,
        pip_requirements=pip_requirements,
        metadata={
            "bundle_files": sorted(p.name for p in artifacts_dir.iterdir() if p.is_file())
        },
    )
