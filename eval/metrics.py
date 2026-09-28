"""Kicktipp evaluation metrics."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


def parse_result(result_str: str) -> tuple[int | None, int | None]:
    try:
        home, away = str(result_str).split(":")
        return int(home), int(away)
    except Exception:
        return None, None


def calculate_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    if len(y_true) == 0:
        return 0.0
    return float((y_true == y_pred).sum() / len(y_true) * 100)


def calculate_goal_difference_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    correct = 0
    total = 0
    for true, pred in zip(y_true, y_pred):
        true_home, true_away = parse_result(true)
        pred_home, pred_away = parse_result(pred)
        if true_home is None or pred_home is None:
            continue
        total += 1
        if (true_home - true_away) == (pred_home - pred_away):
            correct += 1
    return (correct / total * 100) if total > 0 else 0.0


def kicktipp_points_one(true_str: str, pred_str: str) -> int | None:
    """Kicktipp points for one match: 5 / 3 / 1 / 0, or None if unparsable."""
    true_home, true_away = parse_result(true_str)
    pred_home, pred_away = parse_result(pred_str)
    if true_home is None or pred_home is None:
        return None
    if true_home == pred_home and true_away == pred_away:
        return 5
    if (true_home - true_away) == (pred_home - pred_away):
        return 3
    if (
        (true_home > true_away and pred_home > pred_away)
        or (true_home < true_away and pred_home < pred_away)
        or (true_home == true_away and pred_home == pred_away)
    ):
        return 1
    return 0


def kicktipp_raw_points(y_true: pd.Series, y_pred: pd.Series) -> int:
    score_value = 0
    for true_str, pred_str in zip(y_true, y_pred):
        pts = kicktipp_points_one(str(true_str), str(pred_str))
        if pts is not None:
            score_value += pts
    return int(score_value)


def kicktipp_score(y_true: pd.Series, y_pred: pd.Series, season_matches: int = 306) -> float:
    if len(y_true) == 0:
        return 0.0
    raw = kicktipp_raw_points(y_true, y_pred)
    return round(raw / (len(y_true) / season_matches))


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    outcome_matches = 0
    for true, pred in zip(y_true, y_pred):
        true_home, true_away = parse_result(true)
        pred_home, pred_away = parse_result(pred)
        if true_home is None or pred_home is None:
            continue
        true_outcome = "W" if true_home > true_away else ("D" if true_home == true_away else "L")
        pred_outcome = "W" if pred_home > pred_away else ("D" if pred_home == pred_away else "L")
        if true_outcome == pred_outcome:
            outcome_matches += 1

    raw = kicktipp_raw_points(y_true, y_pred)
    return {
        "n_matches": int(len(y_true)),
        "exact_accuracy": calculate_accuracy(y_true, y_pred),
        "goal_difference_accuracy": calculate_goal_difference_accuracy(y_true, y_pred),
        "outcome_accuracy": (outcome_matches / len(y_true) * 100) if len(y_true) > 0 else 0.0,
        "kicktipp_raw": raw,
        "kicktipp_score": kicktipp_score(y_true, y_pred),
    }


def kicktipp_scores_by_season(
    y_true: pd.Series,
    y_pred: pd.Series,
    seasons: pd.Series,
    *,
    season_matches: int = 306,
) -> dict[int, float]:
    """Per-season Kicktipp scores (norm ``season_matches``) aligned by index."""
    frame = pd.DataFrame(
        {
            "y_true": y_true.reset_index(drop=True),
            "y_pred": y_pred.reset_index(drop=True),
            "saison": seasons.reset_index(drop=True).astype(int),
        }
    )
    out: dict[int, float] = {}
    for saison, group in frame.groupby("saison", sort=True):
        out[int(saison)] = float(
            kicktipp_score(group["y_true"], group["y_pred"], season_matches=season_matches)
        )
    return out


def holdout_kicktipp_z_score(
    holdout_score: float,
    train_season_scores: Mapping[int, float],
) -> dict[str, float]:
    """How extreme the holdout Kicktipp score is vs train seasons.

    ``kicktipp_z_score = (holdout - mean(train seasons)) / std(train seasons)``
    with population std (ddof=0). Rough guide: |z| < 1 normal, > 2 unusual.
    """
    scores = np.asarray(list(train_season_scores.values()), dtype=float)
    n = int(scores.size)
    if n == 0:
        return {
            "kicktipp_z_score": float("nan"),
            "kicktipp_train_season_mean": float("nan"),
            "kicktipp_train_season_std": float("nan"),
            "n_train_seasons": 0.0,
        }
    mean = float(scores.mean())
    std = float(scores.std(ddof=0)) if n > 1 else 0.0
    if std <= 1e-12:
        z = 0.0 if abs(float(holdout_score) - mean) <= 1e-12 else float("nan")
    else:
        z = float((float(holdout_score) - mean) / std)
    return {
        "kicktipp_z_score": z,
        "kicktipp_train_season_mean": mean,
        "kicktipp_train_season_std": std,
        "n_train_seasons": float(n),
    }


def print_metrics(name: str, metrics: dict[str, Any]) -> None:
    print(f"\n{name}")
    print(f"  Matches: {metrics['n_matches']}")
    print(f"  Exact Match Accuracy (5): {metrics['exact_accuracy']:.2f}%")
    print(f"  Tordifferenz (3): {metrics['goal_difference_accuracy']:.2f}%")
    print(f"  Tendenz (1): {metrics['outcome_accuracy']:.2f}%")
    print(f"  Kicktipp raw: {metrics['kicktipp_raw']}")
    print(f"  Kicktipp (norm 306): {metrics['kicktipp_score']}")
    if "kicktipp_z_score" in metrics and metrics["kicktipp_z_score"] == metrics["kicktipp_z_score"]:
        print(
            f"  Kicktipp z-score vs train seasons: {metrics['kicktipp_z_score']:+.2f} "
            f"(mean={metrics.get('kicktipp_train_season_mean', float('nan')):.1f}, "
            f"std={metrics.get('kicktipp_train_season_std', float('nan')):.1f}, "
            f"n={int(metrics.get('n_train_seasons', 0))})"
        )
