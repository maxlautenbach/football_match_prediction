"""
Script to create CSV datasets from pickle files in v4/data
Creates train/test CSV files with match data and a separate CSV for market values
Uses Delta-Logic Option A to update data before generating datasets
"""

import pickle
import sys
from pathlib import Path

import pandas as pd

# Add BASE_DIR to path for imports
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from data_loader import update_match_data_delta, update_next_matchday_df

# Define paths
DATA_DIR = BASE_DIR / "data"
DATASETS_DIR = BASE_DIR / "datasets"

# Ensure datasets directory exists
DATASETS_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("Dataset Creation with Delta-Logic (Option A)")
print("=" * 60)

# Step 1: Delta-Update (Option A)
print("\nStep 1: Running delta update...")
update_match_data_delta(data_dir=DATA_DIR, verbose=True)
update_next_matchday_df(data_dir=DATA_DIR, verbose=True)

# Step 2: Load match data from pickle files
print("\nStep 2: Loading match data from pickle files...")

match_dfs = []
for year in range(2009, 2026):
    file_path = DATA_DIR / f"match_df_{year}.pck"
    if file_path.exists():
        try:
            df = pickle.load(open(file_path, "rb"))
            # Ensure it's a DataFrame
            if isinstance(df, pd.DataFrame):
                match_dfs.append(df)
                print(f"  Loaded {year}: {len(df)} matches")
            else:
                # Convert dict records to DataFrame if needed
                match_dfs.append(pd.DataFrame(df))
                print(f"  Loaded {year}: {len(df)} matches (converted from dict)")
        except Exception as e:
            print(f"  Warning: Could not load {year}: {e}")

if not match_dfs:
    raise ValueError("No match data files found!")

# Combine all match dataframes
all_matches = pd.concat(match_dfs, ignore_index=True)
print(f"\nTotal matches loaded: {len(all_matches)}")

# Filter only finished matches for train/test split
finished_matches = all_matches[all_matches["status"] == "finished"].copy()
print(f"Finished matches: {len(finished_matches)}")

# Extract weekday from date
finished_matches["Wochentag"] = pd.to_datetime(finished_matches["date"]).dt.day_name()

# Create Ergebnis column (goalsHome:goalsAway)
finished_matches["Ergebnis"] = (
    finished_matches["goalsHome"].astype(int).astype(str) + ":" + 
    finished_matches["goalsAway"].astype(int).astype(str)
)

# Select and rename columns for the dataset
dataset_df = finished_matches[[
    "teamHomeName",
    "teamAwayName", 
    "Ergebnis",
    "season",
    "matchDay",
    "Wochentag",
    "league"
]].copy()

# Rename columns to German names
dataset_df.columns = ["Team Home", "Team Away", "Ergebnis", "Saison", "Spieltag", "Wochentag", "Liga"]

# Sort by season and matchday for consistent splitting
dataset_df = dataset_df.sort_values(["Saison", "Spieltag"])

# Split into train and test (80/20)
split_idx = int(len(dataset_df) * 0.8)
train_df = dataset_df.iloc[:split_idx].copy()
test_df = dataset_df.iloc[split_idx:].copy()

print(f"\nTrain set: {len(train_df)} matches")
print(f"Test set: {len(test_df)} matches")
print(f"Train seasons: {train_df['Saison'].min()} - {train_df['Saison'].max()}")
print(f"Test seasons: {test_df['Saison'].min()} - {test_df['Saison'].max()}")

# Save train and test CSV files
train_path = DATASETS_DIR / "train.csv"
test_path = DATASETS_DIR / "test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"\nSaved train.csv: {train_path}")
print(f"Saved test.csv: {test_path}")

# Load market values dictionary
print("\nStep 3: Loading market values...")
market_values_path = DATA_DIR / "market_values_dict.pck"
market_values_dict = pickle.load(open(market_values_path, "rb"))

# Convert market values dict to DataFrame
market_values_rows = []
for team_name, seasons_dict in market_values_dict.items():
    for season, value in seasons_dict.items():
        if value > 0:  # Only include non-zero values
            market_values_rows.append({
                "Team": team_name,
                "Saison": season,
                "MarketValue": value
            })

market_values_df = pd.DataFrame(market_values_rows)
market_values_df = market_values_df.sort_values(["Team", "Saison"])

print(f"Market values: {len(market_values_df)} entries")
print(f"Teams: {market_values_df['Team'].nunique()}")
print(f"Seasons: {market_values_df['Saison'].min()} - {market_values_df['Saison'].max()}")

# Save market values CSV
market_values_path_csv = DATASETS_DIR / "TeamMarketValues.csv"
market_values_df.to_csv(market_values_path_csv, index=False)

print(f"\nSaved TeamMarketValues.csv: {market_values_path_csv}")

print("\n" + "=" * 60)
print("Dataset creation completed!")
print("=" * 60)
print(f"\nSummary:")
print(f"  - Train matches: {len(train_df)}")
print(f"  - Test matches: {len(test_df)}")
print(f"  - Market value entries: {len(market_values_df)}")

