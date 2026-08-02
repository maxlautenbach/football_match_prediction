"""Prediction baselines for model comparison."""

from __future__ import annotations

from typing import List

import pandas as pd


def majority_class(train_results: pd.Series) -> str:
    if len(train_results) == 0:
        return "1:1"
    return str(train_results.value_counts().idxmax())


def majority_predictions(n: int, majority: str) -> List[str]:
    return [majority] * n


def predict_majority_from_train(
    train_df: pd.DataFrame,
    n: int,
    liga: str = "bl1",
) -> tuple[str, List[str]]:
    results = train_df["Ergebnis"]
    if "Liga" in train_df.columns:
        mask = train_df["Liga"].astype(str).str.lower() == liga.lower()
        filtered = train_df.loc[mask, "Ergebnis"]
        if len(filtered) > 0:
            results = filtered
    maj = majority_class(results)
    return maj, majority_predictions(n, maj)
