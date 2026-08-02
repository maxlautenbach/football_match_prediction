"""Majority-class baseline model type."""

from models.majority_baseline.model import MajorityBaselineModel
from models.majority_baseline.train import train as train_majority_baseline

MODEL_TYPE = "majority_baseline"

__all__ = ["MODEL_TYPE", "MajorityBaselineModel", "train_majority_baseline"]
