"""Shared dataset loading and season-holdout split."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import pandas as pd

DEFAULT_HOLDOUT_SEASON = 2025
DEFAULT_START_SEASON = 2009


def discover_season_years(data_dir: Path, start_season: int = DEFAULT_START_SEASON) -> list[int]:
    years: list[int] = []
    for path in sorted(data_dir.glob("match_df_*.pck")):
        try:
            year = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        if year >= start_season:
            years.append(year)
    return years


def load_finished_matches(data_dir: Path, start_season: int = DEFAULT_START_SEASON) -> pd.DataFrame:
    """Load all finished matches from match_df_*.pck into the standard CSV schema."""
    match_dfs: list[pd.DataFrame] = []
    for year in discover_season_years(data_dir, start_season=start_season):
        file_path = data_dir / f"match_df_{year}.pck"
        try:
            raw = pickle.load(open(file_path, "rb"))
            df = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
            match_dfs.append(df)
            print(f"  Loaded {year}: {len(df)} matches")
        except Exception as e:
            print(f"  Warning: Could not load {year}: {e}")

    if not match_dfs:
        raise ValueError("No match data files found!")

    all_matches = pd.concat(match_dfs, ignore_index=True)
    print(f"\nTotal matches loaded: {len(all_matches)}")

    finished = all_matches[all_matches["status"] == "finished"].copy()
    print(f"Finished matches: {len(finished)}")

    finished["Wochentag"] = pd.to_datetime(finished["date"]).dt.day_name()
    finished["Ergebnis"] = (
        finished["goalsHome"].astype(int).astype(str)
        + ":"
        + finished["goalsAway"].astype(int).astype(str)
    )

    dataset_df = finished[
        [
            "teamHomeName",
            "teamAwayName",
            "Ergebnis",
            "season",
            "matchDay",
            "Wochentag",
            "league",
        ]
    ].copy()
    dataset_df.columns = [
        "Team Home",
        "Team Away",
        "Ergebnis",
        "Saison",
        "Spieltag",
        "Wochentag",
        "Liga",
    ]
    return dataset_df.sort_values(["Saison", "Spieltag"]).reset_index(drop=True)


def load_market_values(data_dir: Path, datasets_dir: Path | None = None) -> pd.DataFrame:
    """Load market values from pickle, or fall back to TeamMarketValues.csv."""
    market_values_path = data_dir / "market_values_dict.pck"
    if market_values_path.exists():
        market_values_dict = pickle.load(open(market_values_path, "rb"))
        rows = []
        for team_name, seasons_dict in market_values_dict.items():
            for season, value in seasons_dict.items():
                if value > 0:
                    rows.append({"Team": team_name, "Saison": season, "MarketValue": value})
        mv_df = pd.DataFrame(rows)
    else:
        if datasets_dir is None:
            datasets_dir = data_dir.parent / "datasets"
        csv_path = datasets_dir / "TeamMarketValues.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing {market_values_path} and fallback CSV {csv_path}"
            )
        print(f"  market_values_dict.pck missing — loading {csv_path}")
        mv_df = pd.read_csv(csv_path)
        # Rebuild pickle for subsequent runs
        mv_dict: dict = {}
        for _, row in mv_df.iterrows():
            mv_dict.setdefault(row["Team"], {})[int(row["Saison"])] = float(row["MarketValue"])
        data_dir.mkdir(parents=True, exist_ok=True)
        pickle.dump(mv_dict, open(market_values_path, "wb"))
        print(f"  Rebuilt {market_values_path.name}")

    mv_df = mv_df.sort_values(["Team", "Saison"]).reset_index(drop=True)
    print(f"Market values: {len(mv_df)} entries")
    print(f"Teams: {mv_df['Team'].nunique()}")
    print(f"Seasons: {mv_df['Saison'].min()} - {mv_df['Saison'].max()}")
    return mv_df


def split_by_holdout_season(
    dataset_df: pd.DataFrame,
    holdout_season: int = DEFAULT_HOLDOUT_SEASON,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = dataset_df[dataset_df["Saison"] < holdout_season].copy()
    test_df = dataset_df[dataset_df["Saison"] == holdout_season].copy()

    if train_df.empty:
        raise ValueError(f"Train set empty for holdout_season={holdout_season}")
    if test_df.empty:
        raise ValueError(f"Holdout set empty for season={holdout_season}")

    print(f"\nTrain set: {len(train_df)} matches")
    print(f"Holdout set: {len(test_df)} matches (season {holdout_season})")
    print(f"Train seasons: {train_df['Saison'].min()} - {train_df['Saison'].max()}")
    print(f"Holdout seasons: {test_df['Saison'].min()} - {test_df['Saison'].max()}")
    return train_df, test_df


def generate_datasets_from_pickle(
    data_dir: Path,
    holdout_season: int = DEFAULT_HOLDOUT_SEASON,
    start_season: int = DEFAULT_START_SEASON,
    datasets_dir: Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Loading match data from pickle files...")
    dataset_df = load_finished_matches(data_dir, start_season=start_season)
    train_df, holdout_df = split_by_holdout_season(dataset_df, holdout_season=holdout_season)

    print("\nLoading market values...")
    mv_df = load_market_values(data_dir, datasets_dir=datasets_dir)
    return train_df, holdout_df, mv_df
