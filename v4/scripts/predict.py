"""
Prediction script for next matchday
Uses Delta-Logic Option A to update data before making predictions
Optionally uploads predictions to betting platform
"""

import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add BASE_DIR to path for imports
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from data_loader import update_match_data_delta, update_next_matchday_df
from model import Model

DATA_DIR = BASE_DIR / "data"


def generate_predictions(save_csv: bool = False, verbose: bool = True) -> pd.DataFrame:
    """
    Generate predictions for next matchday.
    
    Args:
        save_csv: Whether to save predictions to CSV file
        verbose: Whether to print progress messages
        
    Returns:
        DataFrame with predictions
    """
    if verbose:
        print("=" * 60)
        print("Football Match Prediction - Next Matchday")
        print("=" * 60)
    
    # Step 1: Delta-Update (Option A)
    if verbose:
        print("\nStep 1: Running delta update...")
    update_match_data_delta(data_dir=DATA_DIR, verbose=verbose)
    update_next_matchday_df(data_dir=DATA_DIR, verbose=verbose)
    
    # Step 2: Load next matchday data
    if verbose:
        print("\nStep 2: Loading next matchday data...")
    next_matchday_path = DATA_DIR / "next_matchday_df.pck"
    
    if not next_matchday_path.exists():
        raise FileNotFoundError("next_matchday_df.pck not found!")
    
    next_matchday_df = pickle.load(open(next_matchday_path, "rb"))
    
    # Ensure it's a DataFrame
    if not isinstance(next_matchday_df, pd.DataFrame):
        next_matchday_df = pd.DataFrame(next_matchday_df)
    
    if verbose:
        print(f"Loaded {len(next_matchday_df)} matches to predict")
    
    if len(next_matchday_df) == 0:
        raise ValueError("No matches to predict!")
    
    # Step 3: Prepare data for prediction
    if verbose:
        print("\nStep 3: Preparing data for prediction...")
    
    # Create DataFrame with required columns for Model.predict()
    prediction_df = pd.DataFrame({
        "Team Home": next_matchday_df["teamHomeName"],
        "Team Away": next_matchday_df["teamAwayName"],
        "Saison": next_matchday_df["season"],
        "Spieltag": next_matchday_df["matchDay"],
        "Wochentag": pd.to_datetime(next_matchday_df["date"]).dt.day_name(),
    })
    
    # Step 4: Load model and make predictions
    if verbose:
        print("\nStep 4: Making predictions...")
    model = Model()
    
    if not model.ready:
        if verbose:
            print("Warning: Model not ready. Using fallback predictions.")
        predictions = ["0:0"] * len(prediction_df)
        prediction_probabilities = [0.0] * len(prediction_df)
    else:
        predictions = model.predict(prediction_df)
        
        # Get prediction probabilities (simplified - would need model internals for exact probs)
        # For now, we'll use a placeholder
        prediction_probabilities = [0.15] * len(predictions)  # Placeholder
    
    # Step 5: Create results DataFrame
    if verbose:
        print("\nStep 5: Creating results...")
    
    # Get market values if available
    home_values = []
    away_values = []
    
    try:
        market_values_path = DATA_DIR / "market_values_dict.pck"
        market_values_dict = pickle.load(open(market_values_path, "rb"))
        
        for _, row in next_matchday_df.iterrows():
            home_team = row["teamHomeName"]
            away_team = row["teamAwayName"]
            season = row["season"]
            
            # Try to get market values
            home_val = market_values_dict.get(home_team, {}).get(season, 0.0)
            away_val = market_values_dict.get(away_team, {}).get(season, 0.0)
            
            home_values.append(home_val)
            away_values.append(away_val)
    except Exception as e:
        if verbose:
            print(f"Warning: Could not load market values: {e}")
        home_values = [0.0] * len(next_matchday_df)
        away_values = [0.0] * len(next_matchday_df)
    
    results_df = pd.DataFrame({
        "Date": next_matchday_df["date"],
        "Home_Team": next_matchday_df["teamHomeName"],
        "Away_Team": next_matchday_df["teamAwayName"],
        "Matchday": next_matchday_df["matchDay"],
        "Season": next_matchday_df["season"],
        "Prediction": predictions,
        "Prediction_Probability": prediction_probabilities,
        "Home_Value": home_values,
        "Away_Value": away_values,
    })
    
    # Step 6: Display results
    if verbose:
        print("\n" + "=" * 60)
        print("MATCH PREDICTIONS")
        print("=" * 60)
        for idx, row in results_df.iterrows():
            confidence = row["Prediction_Probability"] * 100
            print(f"{row['Home_Team']} vs {row['Away_Team']}")
            print(f"  → {row['Prediction']} (Confidence: {confidence:.1f}%)")
            print()
    
    # Step 7: Save predictions to CSV (optional)
    if save_csv:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = BASE_DIR / f"prediction_{timestamp}.csv"
        results_df.to_csv(output_path, index=False)
        if verbose:
            print(f"Predictions saved to: {output_path}")
    
    if verbose:
        print("\n" + "=" * 60)
        print("Prediction completed!")
        print("=" * 60)
    
    return results_df


def main():
    """
    Main prediction function (does not save CSV by default).
    """
    results_df = generate_predictions(save_csv=False, verbose=True)
    print(f"\nTo upload predictions, run:")
    print(f"  uv run python scripts/upload_predictions.py")


if __name__ == "__main__":
    main()

