"""CatBoost Poisson model type."""

from models.catboost_poisson.model import CatBoostPoissonModel
from models.catboost_poisson.train import train as train_catboost_poisson

MODEL_TYPE = "catboost_poisson"

__all__ = ["MODEL_TYPE", "CatBoostPoissonModel", "train_catboost_poisson"]
