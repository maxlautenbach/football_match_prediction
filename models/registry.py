"""model_type → train / load dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from models.contract import Predictor

Trainer = Callable[..., dict[str, Any]]
Loader = Callable[[Path | str], Predictor]


def _register() -> dict[str, dict[str, Any]]:
    from models.catboost_poisson import model as cb_model
    from models.catboost_poisson import train as cb_train
    from models.dixon_coles import model as dc_model
    from models.dixon_coles import train as dc_train
    from models.majority_baseline import model as maj_model
    from models.majority_baseline import train as maj_train
    from models.market_dixon_coles import model as market_dc_model
    from models.market_dixon_coles import train as market_dc_train
    from models.poisson_blend import model as blend_model
    from models.poisson_blend import train as blend_train
    from models.saison_ausblick import model as saison_model
    from models.saison_ausblick import train as saison_train

    return {
        "catboost_poisson": {
            "train": cb_train.train,
            "load": cb_model.load,
            "default_registered_name_attr": "REGISTERED_MODEL_NAME",
        },
        "dixon_coles": {
            "train": dc_train.train,
            "load": dc_model.load,
            "default_registered_name_attr": "DIXON_COLES_REGISTERED_MODEL_NAME",
        },
        "majority_baseline": {
            "train": maj_train.train,
            "load": maj_model.load,
            "default_registered_name_attr": "BASELINE_REGISTERED_MODEL_NAME",
        },
        "market_dixon_coles": {
            "train": market_dc_train.train,
            "load": market_dc_model.load,
            "default_registered_name_attr": "MARKET_DIXON_COLES_REGISTERED_MODEL_NAME",
        },
        "poisson_blend": {
            "train": blend_train.train,
            "load": blend_model.load,
            "default_registered_name_attr": "POISSON_BLEND_REGISTERED_MODEL_NAME",
        },
        "saison_ausblick": {
            "train": saison_train.train,
            "load": saison_model.load,
            "default_registered_name_attr": "SAISON_REGISTERED_MODEL_NAME",
        },
    }


_REGISTRY: dict[str, dict[str, Any]] | None = None


def _get_registry() -> dict[str, dict[str, Any]]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _register()
    return _REGISTRY


def list_model_types() -> list[str]:
    return sorted(_get_registry().keys())


def get_trainer(model_type: str) -> Trainer:
    reg = _get_registry()
    if model_type not in reg:
        raise KeyError(f"Unknown model_type {model_type!r}. Known: {list_model_types()}")
    return reg[model_type]["train"]


def get_loader(model_type: str) -> Loader:
    reg = _get_registry()
    if model_type not in reg:
        raise KeyError(f"Unknown model_type {model_type!r}. Known: {list_model_types()}")
    return reg[model_type]["load"]


def default_registered_model_name(model_type: str) -> str:
    """Resolve default MLflow registered model name for a model_type."""
    from eval import mlflow_utils

    reg = _get_registry()
    if model_type not in reg:
        raise KeyError(f"Unknown model_type {model_type!r}")
    attr = reg[model_type]["default_registered_name_attr"]
    return str(getattr(mlflow_utils, attr))


def train_model(
    model_type: str,
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    mv_df: pd.DataFrame,
    bundle_dir: Path,
    *,
    params: Mapping[str, Any],
    recipe_name: str,
    holdout_season: int,
) -> dict[str, Any]:
    trainer = get_trainer(model_type)
    return trainer(
        train_df,
        holdout_df,
        mv_df,
        bundle_dir,
        params=params,
        recipe_name=recipe_name,
        holdout_season=holdout_season,
    )


def load_model(model_type: str, bundle_dir: Path | str) -> Predictor:
    return get_loader(model_type)(bundle_dir)


def detect_model_type(bundle_dir: Path | str) -> str:
    """Detect model_type from bundle.json, with legacy CatBoost fallback."""
    from pathlib import Path as P

    import json

    art = P(bundle_dir)
    bundle_path = art / "bundle.json"
    if bundle_path.exists():
        meta = json.loads(bundle_path.read_text(encoding="utf-8"))
        return str(meta["model_type"])
    # Legacy production bundle (pre-bundle.json)
    if (art / "meta.json").exists() and (art / "home_goals.cbm").exists():
        return "catboost_poisson"
    if (art / "majority.json").exists():
        return "majority_baseline"
    if (art / "dixon_coles.json").exists():
        return "dixon_coles"
    raise FileNotFoundError(
        f"Cannot detect model_type in {art}: missing bundle.json and no legacy markers"
    )
