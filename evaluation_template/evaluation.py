"""
Evaluation Template for Football Match Prediction

This script loads test data, removes the actual results,
makes predictions using a model, and evaluates the predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from model import Model


def parse_result(result_str: str) -> tuple:
    """
    Parse a result string "X:Y" into (home_goals, away_goals).
    
    Args:
        result_str (str): Result string in format "X:Y"
    
    Returns:
        tuple: (home_goals, away_goals)
    """
    try:
        home, away = result_str.split(":")
        return int(home), int(away)
    except:
        return None, None


def calculate_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate exact match accuracy (percentage of exact result matches).
    
    Args:
        y_true (pd.Series): True results
        y_pred (pd.Series): Predicted results
    
    Returns:
        float: Accuracy as percentage
    """
    exact_matches = (y_true == y_pred).sum()
    return (exact_matches / len(y_true)) * 100


def calculate_goal_difference_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate goal difference accuracy (percentage of matches with correct goal difference).
    This corresponds to 3 points in Kicktipp scoring.
    
    Args:
        y_true (pd.Series): True results
        y_pred (pd.Series): Predicted results
    
    Returns:
        float: Goal difference accuracy as percentage
    """
    correct_differences = 0
    total = 0
    
    for true, pred in zip(y_true, y_pred):
        true_home, true_away = parse_result(true)
        pred_home, pred_away = parse_result(pred)
        
        if true_home is not None and pred_home is not None:
            total += 1
            true_diff = int(true_home) - int(true_away)
            pred_diff = int(pred_home) - int(pred_away)
            if true_diff == pred_diff:
                correct_differences += 1
    
    return (correct_differences / total * 100) if total > 0 else 0


def kicktipp_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate Kicktipp score for predictions.
    
    Scoring system:
    - Exact match: 5 points
    - Goal difference correct: 3 points
    - Winner correct: 1 point
    
    Args:
        y_true (pd.Series): True results in format "X:Y"
        y_pred (pd.Series): Predicted results in format "X:Y"
    
    Returns:
        float: Kicktipp score (normalized to Kicktipp scale)
    """
    score_value = 0
    
    for true_str, pred_str in zip(y_true, y_pred):
        # Parse results
        true_home, true_away = parse_result(true_str)
        pred_home, pred_away = parse_result(pred_str)
        
        if true_home is None or pred_home is None:
            continue
        
        # Convert to integers for comparison
        true_home = int(true_home)
        true_away = int(true_away)
        pred_home = int(pred_home)
        pred_away = int(pred_away)
        
        # Exact match: 5 points
        if true_home == pred_home and true_away == pred_away:
            score_value += 5
        # Goal difference correct: 3 points
        elif (true_home - true_away) == (pred_home - pred_away):
            score_value += 3
        # Winner correct: 1 point
        elif ((true_home > true_away) and (pred_home > pred_away)) or \
             ((true_home < true_away) and (pred_home < pred_away)) or \
             ((true_home == true_away) and (pred_home == pred_away)):
            score_value += 1
    
    # Normalize to Kicktipp scale (306 matches per season)
    return round(score_value / (len(y_true) / 306))


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Comprehensive evaluation of predictions.
    
    Args:
        y_true (pd.Series): True results
        y_pred (pd.Series): Predicted results
    
    Returns:
        dict: Dictionary with evaluation metrics
    """
    metrics = {}
    
    # Exact match accuracy (5 points in Kicktipp)
    metrics['exact_accuracy'] = calculate_accuracy(y_true, y_pred)
    
    # Goal difference accuracy (3 points in Kicktipp)
    metrics['goal_difference_accuracy'] = calculate_goal_difference_accuracy(y_true, y_pred)
    
    # Outcome accuracy (Win/Draw/Loss for home team) - Tendenz (1 point in Kicktipp)
    outcome_matches = 0
    for true, pred in zip(y_true, y_pred):
        true_home, true_away = parse_result(true)
        pred_home, pred_away = parse_result(pred)
        
        if true_home is not None and pred_home is not None:
            true_outcome = "W" if true_home > true_away else ("D" if true_home == true_away else "L")
            pred_outcome = "W" if pred_home > pred_away else ("D" if pred_home == pred_away else "L")
            if true_outcome == pred_outcome:
                outcome_matches += 1
    
    metrics['outcome_accuracy'] = (outcome_matches / len(y_true)) * 100 if len(y_true) > 0 else 0
    
    # Add Kicktipp score to metrics
    metrics['kicktipp_score'] = kicktipp_score(y_true, y_pred)
    
    return metrics


def main():
    """
    Main evaluation function.
    """
    # Define paths
    base_dir = Path(__file__).parent
    test_csv_path = base_dir / "datasets" / "test.csv"
    
    print("=" * 60)
    print("Football Match Prediction - Evaluation")
    print("=" * 60)
    
    # Load test data
    print(f"\nLoading test data from: {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)
    print(f"Loaded {len(test_df)} test samples")
    
    # Filter for 1. Bundesliga (bl1) only
    if "Liga" in test_df.columns:
        original_count = len(test_df)
        test_df = test_df[test_df["Liga"] == "bl1"].copy()
        filtered_count = len(test_df)
        print(f"Filtered for 1. Bundesliga (bl1): {original_count} -> {filtered_count} matches")
    else:
        print("Warning: 'Liga' column not found, evaluating on all matches")
    
    # Separate features and target
    print("\nPreparing data...")
    # Store the actual results before removing them
    y_true = test_df["Ergebnis"].copy()
    
    # Remove Ergebnis column for prediction
    X_test = test_df.drop(columns=["Ergebnis"])
    
    print(f"Features shape: {X_test.shape}")
    print(f"Features columns: {list(X_test.columns)}")
    
    # Initialize model
    print("\nInitializing model...")
    model = Model()
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    y_pred = pd.Series(y_pred, name="Ergebnis", index=y_true.index)
    
    print(f"Predicted {len(y_pred)} results")
    
    # Evaluate predictions
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    metrics = evaluate_predictions(y_true, y_pred)
    
    print(f"\nExact Match Accuracy (5 Punkte): {metrics['exact_accuracy']:.2f}%")
    print(f"Tordifferenz Genauigkeit (3 Punkte): {metrics['goal_difference_accuracy']:.2f}%")
    print(f"Tendenz Genauigkeit (1 Punkt): {metrics['outcome_accuracy']:.2f}%")
    print(f"Kicktipp Score: {metrics['kicktipp_score']} Punkte")
    
    # Show some example predictions
    print("\n" + "=" * 60)
    print("Sample Predictions")
    print("=" * 60)
    print("\nFirst 10 predictions:")
    comparison_df = pd.DataFrame({
        'Team Home': X_test['Team Home'].head(10),
        'Team Away': X_test['Team Away'].head(10),
        'True Result': y_true.head(10),
        'Predicted Result': y_pred.head(10)
    })
    print(comparison_df.to_string(index=False))
    
    # Save predictions to file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = base_dir / f"prediction_{timestamp}.csv"
    predictions_df = X_test.copy()
    predictions_df['True_Ergebnis'] = y_true
    predictions_df['Predicted_Ergebnis'] = y_pred
    predictions_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

