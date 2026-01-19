"""
Data Loader Module with Delta-Logic (Option A)

This module handles incremental updates of match data by only loading
new or changed matches from the API. Option A: Individual API calls per match.
"""

import datetime
import pickle
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

# Add project root to path for API imports
BASE_DIR = Path(__file__).parent.parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from api import openligadb
except ImportError:
    # Fallback: try importing from parent directory
    import importlib.util
    api_path = PROJECT_ROOT / "api" / "openligadb.py"
    if api_path.exists():
        spec = importlib.util.spec_from_file_location("openligadb", api_path)
        openligadb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(openligadb)
    else:
        raise ImportError("Could not find api/openligadb.py")

DATA_DIR = BASE_DIR / "data"


def extract_match_data(json_data: dict, league: Optional[str] = None) -> dict:
    """
    Extract relevant match data from OpenLigaDB JSON response.
    
    Handles match status including cancelled/postponed matches:
    - "finished": Match has an Endergebnis
    - "future": Match is scheduled and not yet played
    - "cancelled": Match is in the past but has no result (likely cancelled/postponed)
    
    Args:
        json_data (dict): JSON response from OpenLigaDB API
        league (str, optional): League identifier (bl1, bl2)
        
    Returns:
        dict: Processed match data
    """
    match_data = {}
    match_data["id"] = json_data["matchID"]
    match_data["date"] = datetime.datetime.strptime(
        json_data["matchDateTime"], "%Y-%m-%dT%H:%M:%S"
    )
    match_data["teamHomeId"] = json_data["team1"]["teamId"]
    match_data["teamHomeName"] = json_data["team1"]["teamName"]
    match_data["teamAwayId"] = json_data["team2"]["teamId"]
    match_data["teamAwayName"] = json_data["team2"]["teamName"]
    
    # Extract result
    status = "future"
    goals_home = None
    goals_away = None
    
    # Check if match has finished
    has_result = False
    for match_result in json_data["matchResults"]:
        if match_result["resultName"] == "Endergebnis":
            status = "finished"
            goals_home = match_result["pointsTeam1"]
            goals_away = match_result["pointsTeam2"]
            has_result = True
            break
    
    # Handle edge case: Match is in the past but has no result (cancelled/postponed)
    if not has_result:
        match_date = match_data["date"]
        current_time = datetime.datetime.now()
        
        # If match date is more than 24 hours in the past and still no result, 
        # it's likely cancelled or postponed
        if match_date < (current_time - datetime.timedelta(hours=24)):
            status = "cancelled"
    
    match_data["status"] = status
    match_data["goalsHome"] = goals_home
    match_data["goalsAway"] = goals_away
    
    # Extract winner
    if goals_home is not None and goals_away is not None:
        if goals_home > goals_away:
            match_data["winnerTeamId"] = match_data["teamHomeId"]
        elif goals_home < goals_away:
            match_data["winnerTeamId"] = match_data["teamAwayId"]
        else:
            match_data["winnerTeamId"] = None
    else:
        match_data["winnerTeamId"] = None
    
    match_data["matchDay"] = json_data["group"]["groupOrderID"]
    match_data["season"] = json_data["leagueSeason"]
    match_data["league"] = league
    
    return match_data


def get_current_season(current_date: Optional[datetime.datetime] = None) -> int:
    """
    Determine the current season start year.
    
    - If current month is August-December: current season started this year
    - If current month is January-July: current season started last year
    
    Args:
        current_date (datetime, optional): Date to check. Defaults to today.
        
    Returns:
        int: Year when current season started
    """
    if current_date is None:
        current_date = datetime.datetime.now()
    
    current_year = current_date.year
    
    if current_date.month >= 8:  # August to December
        return current_year
    else:  # January to July
        return current_year - 1


def find_outdated_future_matches(
    df: pd.DataFrame, current_date: Optional[datetime.datetime] = None
) -> pd.DataFrame:
    """
    Find matches that are marked as 'future' but are in the past.
    
    Args:
        df (pd.DataFrame): Match DataFrame
        current_date (datetime, optional): Date to check against. Defaults to today.
        
    Returns:
        pd.DataFrame: DataFrame with outdated future matches
    """
    if current_date is None:
        current_date = datetime.datetime.now()
    
    outdated = df[
        (df["status"] == "future") & (df["date"] < current_date)
    ].copy()
    
    return outdated


def update_single_match(match_id: int, league: str) -> dict:
    """
    Load a single match from API and extract data.
    
    Option A: Individual API call per match.
    
    Args:
        match_id (int): Match ID
        league (str): League identifier (bl1, bl2)
        
    Returns:
        dict: Extracted match data
    """
    try:
        match_json = openligadb.get_match_data(match_id)
        return extract_match_data(match_json, league=league)
    except Exception as e:
        print(f"Warning: Could not load match {match_id}: {e}")
        return None


def update_match_data_delta(
    data_dir: Optional[Path] = None,
    current_date: Optional[datetime.datetime] = None,
    verbose: bool = True,
) -> None:
    """
    Update match data incrementally (Delta-Logic Option A).
    
    Only loads new or changed matches from the API.
    Historical seasons are not updated.
    
    Args:
        data_dir (Path, optional): Directory with pickle files. Defaults to DATA_DIR.
        current_date (datetime, optional): Current date. Defaults to today.
        verbose (bool): Whether to print progress messages.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    
    if current_date is None:
        current_date = datetime.datetime.now()
    
    current_season = get_current_season(current_date)
    
    if verbose:
        print(f"Delta-Update: Current season is {current_season}")
        print(f"Checking for outdated matches in season {current_season}...")
    
    # Only update current season
    pickle_file = data_dir / f"match_df_{current_season}.pck"
    
    if not pickle_file.exists():
        if verbose:
            print(f"No pickle file found for season {current_season}, skipping delta update.")
        return
    
    try:
        # Load existing data
        existing_df = pd.DataFrame(pickle.load(open(pickle_file, "rb")))
        
        if verbose:
            print(f"Loaded {len(existing_df)} matches from {pickle_file.name}")
        
        # Find outdated future matches
        outdated_matches = find_outdated_future_matches(existing_df, current_date)
        
        if len(outdated_matches) == 0:
            if verbose:
                print("No outdated matches found. Data is up to date.")
            return
        
        if verbose:
            print(f"Found {len(outdated_matches)} outdated matches. Updating...")
        
        # Option A: Individual API calls per match
        updated_count = 0
        failed_count = 0
        
        for idx, match_row in tqdm(
            outdated_matches.iterrows(),
            total=len(outdated_matches),
            desc="Updating matches",
            disable=not verbose,
        ):
            match_id = match_row["id"]
            league = match_row["league"]
            
            updated_match = update_single_match(match_id, league)
            
            if updated_match is not None:
                # Update the match in the DataFrame
                # Convert dict to Series for proper assignment
                for key, value in updated_match.items():
                    existing_df.loc[existing_df["id"] == match_id, key] = value
                updated_count += 1
            else:
                failed_count += 1
        
        # Save updated data
        if updated_count > 0:
            pickle.dump(existing_df, open(pickle_file, "wb"))
            if verbose:
                print(
                    f"Updated {updated_count} matches. "
                    f"Failed: {failed_count}. Saved to {pickle_file.name}"
                )
        else:
            if verbose:
                print(f"No matches could be updated. Failed: {failed_count}")
    
    except Exception as e:
        if verbose:
            print(f"Error during delta update: {e}")
        raise


def find_next_match_day(match_df: pd.DataFrame) -> pd.DataFrame:
    """
    Find the next match day to predict.
    
    Logic:
    1. Check if there's a matchday with some matches already played but still has future matches
       → Return only the future matches from that matchday (partial matchday)
    2. Otherwise, find the first matchday with all 9 matches still future
       → Return that complete matchday
    
    This handles the edge case when the server restarts during a matchday:
    - Only the remaining (future) matches of the current matchday are predicted
    - We don't skip to the next matchday until the current one is complete
    
    Args:
        match_df (pd.DataFrame): Match DataFrame
        
    Returns:
        pd.DataFrame: DataFrame containing matches to predict
    """
    if len(match_df) == 0:
        return match_df.iloc[0:0].copy()
    
    current_season = match_df["season"].max()
    
    # Group by matchday to analyze each one
    matchday_groups = match_df[match_df["season"] == current_season].groupby("matchDay")
    
    # First pass: Look for a partial matchday (some finished/cancelled, some future)
    for matchday in sorted(matchday_groups.groups.keys()):
        group = matchday_groups.get_group(matchday)
        
        future_count = (group["status"] == "future").sum()
        finished_count = (group["status"] == "finished").sum()
        cancelled_count = (group["status"] == "cancelled").sum()
        
        # If this matchday has both completed and future matches, it's in progress
        if future_count > 0 and (finished_count > 0 or cancelled_count > 0):
            # Return only the future matches from this partial matchday
            return group[group["status"] == "future"].copy()
    
    # Second pass: Look for the first complete matchday with all matches future
    for matchday in sorted(matchday_groups.groups.keys()):
        group = matchday_groups.get_group(matchday)
        
        future_count = (group["status"] == "future").sum()
        
        # A complete future matchday should have at least 9 matches
        # (could have less due to cancelled matches, but typically 9)
        if future_count >= 9:
            return group[group["status"] == "future"].copy()
    
    # Fallback: Return any future matches we can find
    future_matches = match_df[match_df["status"] == "future"]
    if len(future_matches) > 0:
        # Get the lowest matchday number with future matches
        min_matchday = future_matches["matchDay"].min()
        return future_matches[future_matches["matchDay"] == min_matchday].copy()
    
    # No future matches at all
    return match_df.iloc[0:0].copy()


def update_next_matchday_df(
    data_dir: Optional[Path] = None,
    current_date: Optional[datetime.datetime] = None,
    verbose: bool = True,
) -> None:
    """
    Update next_matchday_df.pck after delta update.
    
    Args:
        data_dir (Path, optional): Directory with pickle files. Defaults to DATA_DIR.
        current_date (datetime, optional): Current date. Defaults to today.
        verbose (bool): Whether to print progress messages.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    
    if current_date is None:
        current_date = datetime.datetime.now()
    
    current_season = get_current_season(current_date)
    
    # Load all match data for current season
    pickle_file = data_dir / f"match_df_{current_season}.pck"
    
    if not pickle_file.exists():
        if verbose:
            print(f"No pickle file found for season {current_season}, skipping next_matchday update.")
        return
    
    try:
        match_df = pd.DataFrame(pickle.load(open(pickle_file, "rb")))
        
        # Find next match day
        next_matchday_df = find_next_match_day(match_df)
        
        # Save next_matchday_df
        next_matchday_file = data_dir / "next_matchday_df.pck"
        pickle.dump(next_matchday_df, open(next_matchday_file, "wb"))
        
        if verbose:
            print(
                f"Updated next_matchday_df.pck with {len(next_matchday_df)} matches"
            )
    
    except Exception as e:
        if verbose:
            print(f"Error updating next_matchday_df: {e}")

