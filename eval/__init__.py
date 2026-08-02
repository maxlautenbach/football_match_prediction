"""Evaluation package: metrics, baselines, MLflow helpers, compare CLI."""

from eval.metrics import evaluate_predictions, kicktipp_score, parse_result

__all__ = ["evaluate_predictions", "kicktipp_score", "parse_result"]
