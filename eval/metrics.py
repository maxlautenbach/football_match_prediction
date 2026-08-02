"""Kicktipp evaluation metrics."""

from __future__ import annotations

from typing import Any

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


def kicktipp_raw_points(y_true: pd.Series, y_pred: pd.Series) -> int:
    score_value = 0
    for true_str, pred_str in zip(y_true, y_pred):
        true_home, true_away = parse_result(true_str)
        pred_home, pred_away = parse_result(pred_str)
        if true_home is None or pred_home is None:
            continue
        if true_home == pred_home and true_away == pred_away:
            score_value += 5
        elif (true_home - true_away) == (pred_home - pred_away):
            score_value += 3
        elif (
            (true_home > true_away and pred_home > pred_away)
            or (true_home < true_away and pred_home < pred_away)
            or (true_home == true_away and pred_home == pred_away)
        ):
            score_value += 1
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


def print_metrics(name: str, metrics: dict[str, Any]) -> None:
    print(f"\n{name}")
    print(f"  Matches: {metrics['n_matches']}")
    print(f"  Exact Match Accuracy (5): {metrics['exact_accuracy']:.2f}%")
    print(f"  Tordifferenz (3): {metrics['goal_difference_accuracy']:.2f}%")
    print(f"  Tendenz (1): {metrics['outcome_accuracy']:.2f}%")
    print(f"  Kicktipp raw: {metrics['kicktipp_raw']}")
    print(f"  Kicktipp (norm 306): {metrics['kicktipp_score']}")
