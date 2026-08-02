"""Metrics for Kicktipp saison outlook questions (6 pts each, max 24)."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

POINTS_PER_CORRECT = 6
MAX_SAISON_SCORE = 24


def _norm_bottom3(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = str(value)
    if "|" in s:
        parts = [p for p in s.split("|") if p]
        return "|".join(sorted(parts))
    return s


def score_prediction(y_true: Mapping[str, Any], y_pred: Mapping[str, Any]) -> dict[str, Any]:
    """Score one season's four tips."""
    champ_ok = str(y_true.get("champion")) == str(y_pred.get("champion"))
    herbst_ok = str(y_true.get("herbstmeister")) == str(y_pred.get("herbstmeister"))
    bottom_ok = _norm_bottom3(y_true.get("bottom3")) == _norm_bottom3(y_pred.get("bottom3"))
    scorer_ok = str(y_true.get("top_scorer_team")) == str(y_pred.get("top_scorer_team"))

    score = POINTS_PER_CORRECT * sum([champ_ok, herbst_ok, bottom_ok, scorer_ok])
    return {
        "acc_champion": float(champ_ok),
        "acc_herbstmeister": float(herbst_ok),
        "acc_bottom3": float(bottom_ok),
        "acc_top_scorer": float(scorer_ok),
        "saison_score": float(score),
        "saison_score_norm": float(score) / MAX_SAISON_SCORE,
        "expected_saison_score": float(y_pred.get("expected_saison_score", 0.0) or 0.0),
    }


def evaluate_saison_predictions(
    labels: pd.DataFrame,
    preds: list[Mapping[str, Any]],
) -> dict[str, float]:
    if len(labels) != len(preds):
        raise ValueError(f"labels ({len(labels)}) vs preds ({len(preds)}) length mismatch")
    if len(labels) == 0:
        return {
            "saison_score": 0.0,
            "saison_score_norm": 0.0,
            "acc_champion": 0.0,
            "acc_herbstmeister": 0.0,
            "acc_bottom3": 0.0,
            "acc_top_scorer": 0.0,
            "expected_saison_score": 0.0,
            "n_seasons": 0.0,
        }

    rows = [
        score_prediction(labels.iloc[i].to_dict(), preds[i]) for i in range(len(labels))
    ]
    df = pd.DataFrame(rows)
    return {
        "saison_score": float(df["saison_score"].mean()),
        "saison_score_norm": float(df["saison_score_norm"].mean()),
        "acc_champion": float(df["acc_champion"].mean()),
        "acc_herbstmeister": float(df["acc_herbstmeister"].mean()),
        "acc_bottom3": float(df["acc_bottom3"].mean()),
        "acc_top_scorer": float(df["acc_top_scorer"].mean()),
        "expected_saison_score": float(df["expected_saison_score"].mean()),
        "n_seasons": float(len(df)),
    }


def saison_scores_by_season(
    labels: pd.DataFrame,
    preds: list[Mapping[str, Any]],
) -> dict[int, float]:
    out: dict[int, float] = {}
    for i in range(len(labels)):
        season = int(labels.iloc[i]["Saison"])
        out[season] = score_prediction(labels.iloc[i].to_dict(), preds[i])["saison_score"]
    return out


def holdout_saison_z_score(
    holdout_score: float,
    train_season_scores: Mapping[int, float],
) -> dict[str, float]:
    vals = np.array(list(train_season_scores.values()), dtype=float)
    n = len(vals)
    if n < 2:
        return {
            "saison_z_score": 0.0,
            "saison_score_loocv_mean": float(vals.mean()) if n else 0.0,
            "saison_score_loocv_std": 0.0,
            "n_train_seasons": float(n),
        }
    mean = float(vals.mean())
    std = float(vals.std(ddof=1))
    z = (float(holdout_score) - mean) / std if std > 1e-9 else 0.0
    return {
        "saison_z_score": z,
        "saison_score_loocv_mean": mean,
        "saison_score_loocv_std": std,
        "n_train_seasons": float(n),
    }


def print_saison_metrics(title: str, metrics: Mapping[str, float]) -> None:
    print(f"\n=== {title} ===")
    for key in (
        "saison_score",
        "saison_score_norm",
        "expected_saison_score",
        "acc_champion",
        "acc_herbstmeister",
        "acc_bottom3",
        "acc_top_scorer",
        "saison_z_score",
        "saison_score_loocv_mean",
        "saison_score_loocv_std",
        "n_train_seasons",
        "n_seasons",
    ):
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")
