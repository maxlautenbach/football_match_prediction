"""
Data Preparation Module for Football Match Prediction

This module contains functions to extract, process, and prepare football match data
for machine learning models. It handles data from OpenLigaDB API and Transfermarkt.

FEATURES:
- Efficient season-based market value fetching using get_season_teams_market_values()
- Levenshtein distance-based team name mapping between OpenLigaDB and Transfermarkt
- Automatic handling of both 1. Bundesliga (bl1) and 2. Bundesliga (bl2)
- Single API call per season instead of individual team lookups for maximum efficiency
- Optional filtering for BL1 only using the bl1_only parameter
- League tracking with automatic removal from final datasets
- Table position calculation (teamHomeRank, teamAwayRank) with chronological accuracy

NEW RANKING FEATURES:
- teamHomeRank: Table position of home team at match time (1-18, 0 if no matches played)
- teamAwayRank: Table position of away team at match time (1-18, 0 if no matches played)
- Rankings respect season boundaries (2024 = season 24/25) and league separation (bl1/bl2)
- Only finished matches before the current match date are considered for ranking calculation
- Standard Bundesliga ranking rules: Points > Goal Difference > Goals Scored

The market value fetching is highly efficient as it fetches all teams for a season
in a single API call instead of individual team lookups.
"""

import datetime
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from collections import defaultdict
# Removed varname import as it's not needed in the refactored version
import os
import sys
import time

# Add parent directory to path for API imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import openligadb
# Import from v3/api for the new improved transfermarkt module
try:
    from v3.api.transfermarkt import get_season_teams_market_values
except ImportError:
    # Fallback to old API if v3 version not available
    get_season_teams_market_values = None

# Define data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def levenshtein_distance(str1, str2):
    """
    Calculate the Levenshtein distance between two strings.
    
    Args:
        str1 (str): First string
        str2 (str): Second string
        
    Returns:
        int: Levenshtein distance
    """
    if len(str1) < len(str2):
        return levenshtein_distance(str2, str1)
    
    if len(str2) == 0:
        return len(str1)
    
    previous_row = list(range(len(str2) + 1))
    for i, c1 in enumerate(str1):
        current_row = [i + 1]
        for j, c2 in enumerate(str2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def find_best_team_match(team_name, transfermarkt_teams, max_distance=5):
    """
    Find the best matching team name from Transfermarkt using Levenshtein distance.
    
    Args:
        team_name (str): Team name from OpenLigaDB
        transfermarkt_teams (list): List of team names from Transfermarkt
        max_distance (int): Maximum allowed Levenshtein distance
        
    Returns:
        str or None: Best matching team name, None if no good match found
    """
    best_match = None
    best_distance = float('inf')
    
    # Normalize team names for comparison
    normalized_team = team_name.lower().strip()
    
    # Handle common variations
    normalized_team = normalized_team.replace("1. fc", "1.fc")
    normalized_team = normalized_team.replace("1. fsv", "1.fsv")
    normalized_team = normalized_team.replace("fc ", "")
    normalized_team = normalized_team.replace("sv ", "")
    
    for tm_team in transfermarkt_teams:
        normalized_tm_team = tm_team.lower().strip()
        
        # Handle common variations in Transfermarkt names
        normalized_tm_team = normalized_tm_team.replace("1. fc", "1.fc")
        normalized_tm_team = normalized_tm_team.replace("1. fsv", "1.fsv")
        normalized_tm_team = normalized_tm_team.replace("fc ", "")
        normalized_tm_team = normalized_tm_team.replace("sv ", "")
        
        distance = levenshtein_distance(normalized_team, normalized_tm_team)
        
        if distance < best_distance:
            best_distance = distance
            best_match = tm_team
    
    # Return match only if distance is within acceptable range
    if best_distance <= max_distance:
        return best_match
    else:
        return None





def extract_result(match_results):
    """
    Extract match result from match results data.
    
    Returns status as "finished" if match has Endergebnis, otherwise "future".
    Note: Status "cancelled" is assigned later in extract_match_data based on date.
    
    Args:
        match_results (list): List of match result dictionaries
        
    Returns:
        tuple: (status, goals_home, goals_away)
    """
    for match_result in match_results:
        if match_result['resultName'] == 'Endergebnis':
            return "finished", match_result['pointsTeam1'], match_result['pointsTeam2']
    return "future", None, None


def extract_winner(match_data):
    """
    Extract winner team ID from match data.
    
    Args:
        match_data (dict): Match data dictionary
        
    Returns:
        int or None: Winner team ID, None for draw, None for future matches
    """
    try:
        if match_data["goalsHome"] > match_data["goalsAway"]:
            return match_data["teamHomeId"]
        elif match_data["goalsHome"] < match_data["goalsAway"]:
            return match_data["teamAwayId"]
        else:
            return None
    except TypeError:
        return None


def extract_match_data(json, league=None):
    """
    Extract relevant match data from OpenLigaDB JSON response.
    
    Handles match status including cancelled/postponed matches:
    - "finished": Match has an Endergebnis
    - "future": Match is scheduled and not yet played
    - "cancelled": Match is in the past but has no result (likely cancelled/postponed)
    
    Args:
        json (dict): JSON response from OpenLigaDB API
        league (str, optional): League identifier (bl1, bl2)
        
    Returns:
        dict: Processed match data
    """
    match_data = {}
    match_data["id"] = json["matchID"]
    match_data["date"] = datetime.datetime.strptime(json["matchDateTime"], "%Y-%m-%dT%H:%M:%S")
    match_data["teamHomeId"] = json["team1"]["teamId"]
    match_data["teamHomeName"] = json["team1"]["teamName"]
    match_data["teamAwayId"] = json["team2"]["teamId"]
    match_data["teamAwayName"] = json["team2"]["teamName"]
    match_data["status"], match_data["goalsHome"], match_data["goalsAway"] = extract_result(json["matchResults"])
    
    # Handle edge case: Match is in the past but has no result (cancelled/postponed)
    if match_data["status"] == "future":
        current_time = datetime.datetime.now()
        # If match date is more than 24 hours in the past and still no result,
        # it's likely cancelled or postponed
        if match_data["date"] < (current_time - datetime.timedelta(hours=24)):
            match_data["status"] = "cancelled"
    
    match_data["winnerTeamId"] = extract_winner(match_data)
    match_data["matchDay"] = json["group"]["groupOrderID"]
    match_data["season"] = json["leagueSeason"]
    match_data["league"] = league  # Add league information
    return match_data


def load_or_create_match_data(start_season=2025, end_season=2025):
    """
    Load match data from pickle file or create it from API if not exists.
    Automatically updates current season data if matches are found that are in the past
    but still marked as 'future'.
    
    The function intelligently determines the current season:
    - If current month is August-December: current season started this year
    - If current month is January-July: current season started last year
    
    Args:
        start_season (int): First season to include (default: 2025)
        end_season (int): Last season to include (default: 2025)
    
    Returns:
        pd.DataFrame: DataFrame containing all match data with league column
    """
    # Validate season parameters
    if start_season > end_season:
        raise ValueError(f"start_season ({start_season}) cannot be greater than end_season ({end_season})")
    
    print(f"Loading match data for seasons {start_season}-{end_season}...")
    match_data = []
    
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Get current date for checking if we need to update current season
    current_date = datetime.datetime.now()
    current_year = current_date.year
    
    # Determine if we're in the current season (August to July of next year)
    # If we're in January-July, the current season started in the previous year
    # If we're in August-December, the current season started in the current year
    if current_date.month >= 8:  # August to December
        current_season_start_year = current_year
    else:  # January to July
        current_season_start_year = current_year - 1
    
    for year in tqdm(range(start_season, end_season + 1), desc="Loading seasons"):
        year_data = []
        year_file = os.path.join(DATA_DIR, f"match_df_{year}.pck")
        
        # Try to load from year-specific pickle file
        try:
            year_df = pickle.load(open(year_file, "rb"))
            year_data = year_df.to_dict('records')
            print(f"Loaded season {year} from pickle file")
            
            # Check if this is the current season and if we need to update it
            if year == current_season_start_year:
                # Check if there are matches in the past that are still marked as 'future'
                year_df_temp = pd.DataFrame(year_data)
                
                # Find matches that are in the past but still marked as 'future'
                outdated_future_matches = year_df_temp[
                    (year_df_temp["status"] == "future") & 
                    (year_df_temp["date"] < current_date)
                ]
                
                if len(outdated_future_matches) > 0:
                    print(f"Found {len(outdated_future_matches)} outdated 'future' matches in season {year}")
                    print("Updating season data from API...")
                    
                    # Re-fetch data from API for current season
                    year_data = []
                    
                    # Bundesliga 1
                    try:
                        json = openligadb.get_all_season_matches("bl1", year)
                        for match_data_json in json:
                            year_data.append(extract_match_data(match_data_json, league="bl1"))
                    except Exception as e:
                        print(f"Warning: Could not load Bundesliga 1 data for season {year}: {e}")
                    
                    # Bundesliga 2 (skip for 2025 as it may not be available)
                    if year != 2025:
                        try:
                            json = openligadb.get_all_season_matches("bl2", year)
                            for match_data_json in json:
                                year_data.append(extract_match_data(match_data_json, league="bl2"))
                        except Exception as e:
                            print(f"Warning: Could not load Bundesliga 2 data for season {year}: {e}")
                    
                    # Save updated year data to pickle
                    if year_data:
                        year_df = pd.DataFrame(year_data)
                        pickle.dump(year_df, open(year_file, "wb"))
                        print(f"Updated and saved season {year} data to pickle file")
                    else:
                        print(f"Warning: No data could be fetched for season {year}")
                        # Keep the old data if API fails
                        year_data = year_df.to_dict('records')
                        
        except (FileNotFoundError, EOFError):
            print(f"Creating data for season {year} from API...")
            # Bundesliga 1
            try:
                json = openligadb.get_all_season_matches("bl1", year)
                for match_data_json in json:
                    year_data.append(extract_match_data(match_data_json, league="bl1"))
            except Exception as e:
                print(f"Warning: Could not load Bundesliga 1 data for season {year}: {e}")
            
            # Bundesliga 2 (skip for 2025 as it may not be available)
            if year != 2025:
                try:
                    json = openligadb.get_all_season_matches("bl2", year)
                    for match_data_json in json:
                        year_data.append(extract_match_data(match_data_json, league="bl2"))
                except Exception as e:
                    print(f"Warning: Could not load Bundesliga 2 data for season {year}: {e}")
            
            # Save year data to pickle
            if year_data:
                year_df = pd.DataFrame(year_data)
                pickle.dump(year_df, open(year_file, "wb"))
                print(f"Saved season {year} data to pickle file")
        
        match_data.extend(year_data)
    
    if not match_data:
        raise ValueError(f"No match data could be loaded for seasons {start_season}-{end_season}")
    
    match_df = pd.DataFrame(match_data)
    print(f"Loaded {len(match_data)} matches across seasons {start_season}-{end_season}")
    
    return match_df


def apply_testing_mode(match_df, testing=False, days_back=14):
    """
    Apply testing mode by converting recent finished matches to future status.
    
    Args:
        match_df (pd.DataFrame): Match DataFrame
        testing (bool): Whether to enable testing mode
        days_back (int): Number of days to look back for testing
        
    Returns:
        pd.DataFrame: Modified DataFrame
    """
    if testing:
        testing_date = datetime.datetime.today() + relativedelta(days=-days_back)
        manip_df = match_df.copy(deep=True)
        manip_df = manip_df.loc[(match_df["status"] == "finished") & (match_df["date"] > testing_date)]
        manip_df["status"] = ["future" for _ in range(len(manip_df.index))]
        manip_df[["goalsHome", "goalsAway", "winnerTeamId"]] = np.NaN
        match_df[(match_df["status"] == "finished") & (match_df["date"] > testing_date)] = manip_df
    
    return match_df


def update_future_matches(match_df):
    """
    Update matches that should be finished but are marked as future.
    
    Args:
        match_df (pd.DataFrame): Match DataFrame
        
    Returns:
        pd.DataFrame: Updated DataFrame
    """
    updatable_df = match_df[(match_df["status"] == "future") & (match_df["date"] < datetime.datetime.today())]
    
    if len(updatable_df) > 0:
        print(f"Updating {len(updatable_df)} matches...")
        for _, row in tqdm(updatable_df.iterrows(), total=len(updatable_df), desc="Updating matches"):
            row_match_data_json = openligadb.get_match_data(row["id"])
            # Extract match data and preserve the league information
            row_match_data = extract_match_data(row_match_data_json, league=row["league"])
            match_df.loc[row.name] = row_match_data
        
        # Save updated data
        match_df_path = os.path.join(DATA_DIR, "match_df.pck")
        pickle.dump(match_df, open(match_df_path, "wb"))
        print("Updated match data saved")
    
    return match_df


def find_next_match_day(match_df):
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


def prepare_match_dataframe(match_df):
    """
    Prepare match DataFrame by filtering and combining finished and future matches.
    
    Args:
        match_df (pd.DataFrame): Raw match DataFrame
        
    Returns:
        tuple: (prepared_match_df, next_match_day_df)
    """
    # Find next match day
    next_match_day_df = find_next_match_day(match_df)
    
    # Filter finished matches and fill NaN values
    finished_matches = match_df[match_df["status"] == "finished"].fillna(0)
    next_match_day_df = next_match_day_df.fillna(0)
    
    # Combine finished matches with next match day
    combined_df = pd.concat([finished_matches, next_match_day_df])
    
    return combined_df, next_match_day_df


def get_team_match_df(match_df, team_id, lookback_periods=None):
    """
    Create team-specific DataFrame with rolling statistics.
    
    Args:
        match_df (pd.DataFrame): Complete match DataFrame
        team_id (int): Team ID
        lookback_periods (list, optional): List of lookback periods for rolling averages. 
                                         Defaults to [5, 10]
        
    Returns:
        pd.DataFrame: Team-specific DataFrame with statistics
    """
    if lookback_periods is None:
        lookback_periods = [5, 10]
    
    team_match_df = match_df[(match_df["teamHomeId"] == team_id) | (match_df["teamAwayId"] == team_id)].copy(deep=True)
    team_match_df = team_match_df.sort_values(by="date")
    
    goals_team = []
    goals_opponent = []
    team_points = []
    
    for _, row in team_match_df.iterrows():
        if row["teamHomeId"] == team_id:
            goals_team.append(row["goalsHome"])
            goals_opponent.append(row["goalsAway"])
        else:
            goals_opponent.append(row["goalsHome"])
            goals_team.append(row["goalsAway"])
        
        if row["winnerTeamId"] == team_id:
            team_points.append(3)
        elif row["winnerTeamId"] == 0:
            team_points.append(1)
        else:
            team_points.append(0)
    
    team_match_df["goalsTeam"] = goals_team
    team_match_df["goalsOpponent"] = goals_opponent
    team_match_df["teamPoints"] = team_points
    
    # Calculate rolling averages for each lookback period
    for period in lookback_periods:
        team_match_df[f"avgScoredGoals{period}"] = team_match_df["goalsTeam"].rolling(window=period).mean().shift(1)
        team_match_df[f"avgGottenGoals{period}"] = team_match_df["goalsOpponent"].rolling(window=period).mean().shift(1)
        team_match_df[f"avgTeamPoints{period}"] = team_match_df["teamPoints"].rolling(window=period).mean().shift(1)
    
    return team_match_df


def create_team_match_dict(match_df, lookback_periods=None):
    """
    Create dictionary of team-specific DataFrames.
    
    Args:
        match_df (pd.DataFrame): Complete match DataFrame
        lookback_periods (list, optional): List of lookback periods for rolling averages.
                                         Defaults to [5, 10]
        
    Returns:
        dict: Dictionary mapping team IDs to their DataFrames
    """
    if lookback_periods is None:
        lookback_periods = [5, 10]
    
    team_match_df_dict = {}
    unique_teams = set(match_df["teamHomeId"])
    
    print(f"Creating team statistics for {len(unique_teams)} teams with lookback periods: {lookback_periods}...")
    for team_id in tqdm(unique_teams, desc="Processing teams"):
        team_match_df_dict[team_id] = get_team_match_df(match_df, team_id, lookback_periods)
    
    return team_match_df_dict


def load_or_create_market_values(match_df, verbose=False):
    """
    Load market values efficiently using season-based API calls.
    
    This function uses the new get_season_teams_market_values function to fetch
    all teams for a season at once, then maps team names using Levenshtein distance.
    
    Args:
        match_df (pd.DataFrame): Match DataFrame
        verbose (bool): Whether to show detailed output
        
    Returns:
        dict: Dictionary containing market values by team and season
    """
    if get_season_teams_market_values is None:
        raise ImportError("get_season_teams_market_values function not available. Please ensure v3.api.transfermarkt is properly imported.")
    
    market_values_path = os.path.join(DATA_DIR, "market_values_dict.pck")
    
    try:
        market_value_dict = pickle.load(open(market_values_path, "rb"))
        if verbose:
            print("Loaded existing market values from pickle file")
    except (FileNotFoundError, EOFError):
        market_value_dict = defaultdict(lambda: defaultdict(float))
        if verbose:
            print("No existing market values found, starting fresh")
    
    # Get unique seasons and leagues from match data
    season_league_data = match_df.groupby(["season"]).agg({
        "teamHomeName": lambda x: list(set(x)),
        "teamAwayName": lambda x: list(set(x))
    }).reset_index()
    
    # Flatten team names and get unique teams per season
    for _, row in season_league_data.iterrows():
        season = row["season"]
        all_teams = list(set(row["teamHomeName"] + row["teamAwayName"]))
        
        # Check which teams are missing market values for this season
        missing_teams = []
        for team in all_teams:
            clean_team = team.replace("1. FC", "1.FC")
            clean_team = clean_team.replace("1. FSV", "1.FSV")
            clean_team = clean_team.replace("FC Kickers Würzburg", "Würzburger Kickers")
            
            if clean_team not in market_value_dict or season not in market_value_dict[clean_team] or market_value_dict[clean_team][season] == 0.0:
                missing_teams.append(team)
        
        if missing_teams:
            if verbose:
                print(f"Fetching market values for season {season} ({len(missing_teams)} teams missing)...")
            
            # Fetch all teams for this season from both leagues
            season_results = []
            
            # Try 1. Bundesliga (bl1)
            try:
                bl1_results = get_season_teams_market_values(season, league="bl1", verbose=False)
                season_results.extend(bl1_results)
                if verbose:
                    print(f"  Found {len(bl1_results)} teams in 1. Bundesliga")
            except Exception as e:
                if verbose:
                    print(f"  Error fetching 1. Bundesliga data: {e}")
            
            # Try 2. Bundesliga (bl2)
            try:
                bl2_results = get_season_teams_market_values(season, league="bl2", verbose=False)
                season_results.extend(bl2_results)
                if verbose:
                    print(f"  Found {len(bl2_results)} teams in 2. Bundesliga")
            except Exception as e:
                if verbose:
                    print(f"  Error fetching 2. Bundesliga data: {e}")
            
            if season_results:
                # Create mapping from Transfermarkt team names to our team names
                tm_team_names = [result[0] for result in season_results]
                
                # Map missing teams to Transfermarkt teams
                for team in missing_teams:
                    best_match = find_best_team_match(team, tm_team_names)
                    
                    if best_match:
                        # Find the market value for the matched team
                        for tm_team, tm_season, tm_value in season_results:
                            if tm_team == best_match and tm_season == season:
                                # Clean team name for storage
                                clean_team = team.replace("1. FC", "1.FC")
                                clean_team = clean_team.replace("1. FSV", "1.FSV")
                                clean_team = clean_team.replace("FC Kickers Würzburg", "Würzburger Kickers")
                                
                                market_value_dict[clean_team][season] = tm_value
                                
                                if verbose:
                                    print(f"    Mapped '{team}' -> '{best_match}' (distance: {levenshtein_distance(team.lower(), best_match.lower())}) = {tm_value} Mio. €")
                                break
                    else:
                        if verbose:
                            print(f"    No match found for '{team}'")
        
        # Add small delay between seasons to be polite
        if verbose and season != season_league_data.iloc[-1]["season"]:
            time.sleep(1)
    
    # Convert back to regular dict and save
    market_value_dict = dict(market_value_dict)
    os.makedirs(DATA_DIR, exist_ok=True)
    pickle.dump(market_value_dict, open(market_values_path, "wb"))
    
    if verbose:
        print(f"Market values saved to pickle file")
    
    return market_value_dict


def get_market_value(team, season, market_value_dict):
    """
    Get market value for a team in a specific season.
    
    Args:
        team (str): Team name
        season (int): Season year
        market_value_dict (dict): Market values dictionary
        
    Returns:
        float: Market value
    """
    # Handle team name variations
    team = team.replace("1. FC", "1.FC")
    team = team.replace("1. FSV", "1.FSV")
    team = team.replace("FC Kickers Würzburg", "Würzburger Kickers")
    return market_value_dict[team][season]


def calculate_team_standings(match_df, current_date, season, league):
    """
    Calculate team standings (table positions) up to a specific date.
    
    Args:
        match_df (pd.DataFrame): Match DataFrame
        current_date (datetime): Calculate standings up to this date
        season (int): Season to calculate standings for
        league (str): League identifier (bl1, bl2)
        
    Returns:
        dict: Dictionary mapping team IDs to their table positions
    """
    # Filter matches for the specific season and league up to current_date
    relevant_matches = match_df[
        (match_df["season"] == season) & 
        (match_df["league"] == league) & 
        (match_df["status"] == "finished") & 
        (match_df["date"] < current_date)
    ].copy()
    
    if len(relevant_matches) == 0:
        # No matches played yet, return empty dict
        return {}
    
    # Initialize team statistics
    team_stats = defaultdict(lambda: {"points": 0, "goals_for": 0, "goals_against": 0, "matches": 0})
    
    # Calculate points and goal difference for each team
    for _, match in relevant_matches.iterrows():
        home_id = match["teamHomeId"]
        away_id = match["teamAwayId"]
        home_goals = match["goalsHome"]
        away_goals = match["goalsAway"]
        
        # Update match counts
        team_stats[home_id]["matches"] += 1
        team_stats[away_id]["matches"] += 1
        
        # Update goals
        team_stats[home_id]["goals_for"] += home_goals
        team_stats[home_id]["goals_against"] += away_goals
        team_stats[away_id]["goals_for"] += away_goals
        team_stats[away_id]["goals_against"] += home_goals
        
        # Update points based on match result
        if home_goals > away_goals:  # Home win
            team_stats[home_id]["points"] += 3
            team_stats[away_id]["points"] += 0
        elif home_goals < away_goals:  # Away win
            team_stats[home_id]["points"] += 0
            team_stats[away_id]["points"] += 3
        else:  # Draw
            team_stats[home_id]["points"] += 1
            team_stats[away_id]["points"] += 1
    
    # Calculate goal difference and sort teams by standings criteria
    teams_with_stats = []
    for team_id, stats in team_stats.items():
        goal_diff = stats["goals_for"] - stats["goals_against"]
        teams_with_stats.append((
            team_id,
            stats["points"],
            goal_diff,
            stats["goals_for"],
            stats["matches"]
        ))
    
    # Sort by: 1) Points (desc), 2) Goal difference (desc), 3) Goals scored (desc)
    teams_with_stats.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    
    # Create position mapping
    position_dict = {}
    for position, (team_id, _, _, _, _) in enumerate(teams_with_stats, 1):
        position_dict[team_id] = position
    
    return position_dict


def get_team_rank(team_id, current_date, season, league, match_df):
    """
    Get the table position of a team at a specific point in time.
    
    Args:
        team_id (int): Team ID
        current_date (datetime): Date to calculate rank for
        season (int): Season
        league (str): League identifier
        match_df (pd.DataFrame): Complete match DataFrame
        
    Returns:
        int: Table position (1-18 for Bundesliga, 1-18 for 2. Bundesliga), 0 if no matches played
    """
    standings = calculate_team_standings(match_df, current_date, season, league)
    return standings.get(team_id, 0)  # Return 0 if team not found (no matches played)


def add_features_to_match_df(match_df, team_match_df_dict, market_value_dict, lookback_periods=None):
    """
    Add all features to the match DataFrame.
    
    Features added:
    - teamHomeValue, teamAwayValue: Market values
    - teamHomeRank, teamAwayRank: Table positions at match time
    - teamHomeAvgScoredGoals{N}, teamAwayAvgScoredGoals{N}: Rolling average goals scored
    - teamHomeAvgGottenGoals{N}, teamAwayAvgGottenGoals{N}: Rolling average goals conceded
    - teamHomeAvgTeamPoints{N}, teamAwayAvgTeamPoints{N}: Rolling average points per match
    
    Args:
        match_df (pd.DataFrame): Match DataFrame
        team_match_df_dict (dict): Dictionary of team DataFrames
        market_value_dict (dict): Market values dictionary
        lookback_periods (list, optional): List of lookback periods used for features.
                                         Defaults to [5, 10]
        
    Returns:
        pd.DataFrame: Enhanced match DataFrame with all features
    """
    if lookback_periods is None:
        lookback_periods = [5, 10]
    
    print(f"Adding features to match DataFrame with lookback periods: {lookback_periods}...")
    
    # Sort by date to ensure proper chronological order
    match_df = match_df.sort_values(by="date")
    
    # Initialize feature dictionaries
    team_home_value = []
    team_away_value = []
    team_home_rank = []
    team_away_rank = []
    result_class = []
    
    # Dynamic feature lists for each lookback period
    home_features = {}
    away_features = {}
    
    for period in lookback_periods:
        home_features[f"teamHomeAvgScoredGoals{period}"] = []
        home_features[f"teamHomeAvgGottenGoals{period}"] = []
        home_features[f"teamHomeAvgTeamPoints{period}"] = []
        
        away_features[f"teamAwayAvgScoredGoals{period}"] = []
        away_features[f"teamAwayAvgGottenGoals{period}"] = []
        away_features[f"teamAwayAvgTeamPoints{period}"] = []
    
    for _, row in tqdm(match_df.iterrows(), total=len(match_df), desc="Adding features"):
        # Market values
        team_home_value.append(get_market_value(row["teamHomeName"], row["season"], market_value_dict))
        team_away_value.append(get_market_value(row["teamAwayName"], row["season"], market_value_dict))
        
        # Table positions (ranks) - calculated up to the match date
        home_rank = get_team_rank(row["teamHomeId"], row["date"], row["season"], row["league"], match_df)
        away_rank = get_team_rank(row["teamAwayId"], row["date"], row["season"], row["league"], match_df)
        team_home_rank.append(home_rank)
        team_away_rank.append(away_rank)
        
        # Dynamic features for each lookback period
        for period in lookback_periods:
            # Home team statistics
            home_features[f"teamHomeAvgScoredGoals{period}"].append(
                team_match_df_dict[row["teamHomeId"]][f"avgScoredGoals{period}"].loc[row.name]
            )
            home_features[f"teamHomeAvgGottenGoals{period}"].append(
                team_match_df_dict[row["teamHomeId"]][f"avgGottenGoals{period}"].loc[row.name]
            )
            home_features[f"teamHomeAvgTeamPoints{period}"].append(
                team_match_df_dict[row["teamHomeId"]][f"avgTeamPoints{period}"].loc[row.name]
            )
            
            # Away team statistics
            away_features[f"teamAwayAvgScoredGoals{period}"].append(
                team_match_df_dict[row["teamAwayId"]][f"avgScoredGoals{period}"].loc[row.name]
            )
            away_features[f"teamAwayAvgGottenGoals{period}"].append(
                team_match_df_dict[row["teamAwayId"]][f"avgGottenGoals{period}"].loc[row.name]
            )
            away_features[f"teamAwayAvgTeamPoints{period}"].append(
                team_match_df_dict[row["teamAwayId"]][f"avgTeamPoints{period}"].loc[row.name]
            )
        
        # Result class
        result_class.append(str(int(row["goalsHome"])) + ":" + str(int(row["goalsAway"])))
    
    # Add all features to DataFrame and ensure they are numeric
    match_df["teamHomeValue"] = pd.to_numeric(team_home_value, errors='coerce')
    match_df["teamAwayValue"] = pd.to_numeric(team_away_value, errors='coerce')
    match_df["teamHomeRank"] = pd.to_numeric(team_home_rank, errors='coerce')
    match_df["teamAwayRank"] = pd.to_numeric(team_away_rank, errors='coerce')
    
    # Add dynamic features and ensure they are numeric
    for feature_name, feature_values in home_features.items():
        match_df[feature_name] = pd.to_numeric(feature_values, errors='coerce')
    
    for feature_name, feature_values in away_features.items():
        match_df[feature_name] = pd.to_numeric(feature_values, errors='coerce')
    
    match_df["resultClass"] = result_class
    
    return match_df


def save_prepared_data(match_df):
    """
    Save the final prepared datasets.
    
    Args:
        match_df (pd.DataFrame): Complete enhanced match DataFrame
        
    Returns:
        tuple: (training_df, prediction_df)
    """
    print("Saving prepared datasets...")
    
    # Remove NaN values
    match_df = match_df.dropna()
    
    # Separate future matches for prediction
    next_match_day_df = match_df[match_df["status"] == "future"].copy(deep=True)
    next_match_day_df.drop(["id", "winnerTeamId", "goalsHome", "goalsAway", "status", "league"], axis=1, inplace=True)
    
    # Prepare training data (finished matches only)
    training_df = match_df[match_df["status"] == "finished"].copy()
    training_df.drop(["id", "date", "teamHomeName", "teamAwayName", "winnerTeamId", "goalsHome", "goalsAway", "status", "league"], 
                    axis=1, inplace=True)
    
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Save datasets
    pickle.dump(next_match_day_df, open(os.path.join(DATA_DIR, "next_matchday_df.pck"), "wb"))
    pickle.dump(training_df, open(os.path.join(DATA_DIR, "prepped_match_df.pck"), "wb"))
    
    print(f"Training data saved: {len(training_df)} matches")
    print(f"Prediction data saved: {len(next_match_day_df)} matches")
    
    return training_df, next_match_day_df


def run_full_data_preparation(testing=False, lookback_periods=None, verbose=False, 
                             start_season=2025, end_season=2025, bl1_only=False):
    """
    Run the complete data preparation pipeline.
    
    Args:
        testing (bool): Whether to enable testing mode
        lookback_periods (list, optional): List of lookback periods for rolling averages.
                                         Defaults to [5, 10]
        verbose (bool): Whether to show detailed output from market value scraping
        start_season (int): First season to include in dataset (default: 2025)
        end_season (int): Last season to include in dataset (default: 2025)
        bl1_only (bool): Whether to filter for 1. Bundesliga only (default: False)
        
    Returns:
        tuple: (training_df, prediction_df)
    """
    if lookback_periods is None:
        lookback_periods = [5, 10]
    
    print(f"Starting data preparation pipeline:")
    print(f"  - Seasons: {start_season} to {end_season}")
    print(f"  - Lookback periods: {lookback_periods}")
    print(f"  - Testing mode: {testing}")
    print(f"  - BL1 only: {bl1_only}")
    
    # Step 1: Load or create match data
    match_df = load_or_create_match_data(start_season=start_season, end_season=end_season)
    
    # Step 2: Apply testing mode if needed
    match_df = apply_testing_mode(match_df, testing)
    
    # Step 3: Update future matches
    match_df = update_future_matches(match_df)
    
    # Step 4: Prepare match DataFrame
    match_df, _ = prepare_match_dataframe(match_df)
    
    # Step 5: Create team statistics
    team_match_df_dict = create_team_match_dict(match_df, lookback_periods)
    
    # Step 6: Load market values using efficient season-based method
    print("Using efficient season-based market value fetching...")
    market_value_dict = load_or_create_market_values(match_df, verbose=verbose)
    
    # Step 7: Add features
    match_df = add_features_to_match_df(match_df, team_match_df_dict, market_value_dict, lookback_periods)
    
    # Step 7.5: Filter for BL1 only if requested (after all features are added)
    if bl1_only:
        original_count = len(match_df)
        match_df = match_df[match_df["league"] == "bl1"].copy()
        filtered_count = len(match_df)
        print(f"Filtered for BL1 only: {original_count} -> {filtered_count} matches")
    
    # Step 8: Save prepared data
    training_df, prediction_df = save_prepared_data(match_df)
    
    print("Data preparation completed successfully!")
    print(f"Final dataset covers seasons {start_season}-{end_season}")
    return training_df, prediction_df


if __name__ == "__main__":
    # Example: Run with custom lookback periods and season ranges
    custom_lookback_periods = [*range(1,11)]  # Example: 1-10 match lookbacks
    
    # Define season range
    start_season, end_season = 2009, 2025
    
    print(f"Configuration:")
    print(f"  - Season range: {start_season} to {end_season}")
    print(f"  - Lookback periods: {custom_lookback_periods}")
    
    # Run the complete pipeline
    training_data, prediction_data = run_full_data_preparation(
        testing=False, 
        lookback_periods=custom_lookback_periods,
        start_season=start_season,
        end_season=end_season,
        verbose=True,  # Show detailed output for debugging
        bl1_only=True  # Set to True to filter for BL1 only
    )
    
    print(f"\nFinal datasets:")
    print(f"Training data shape: {training_data.shape}")
    print(f"Prediction data shape: {prediction_data.shape}")
    print(f"Features created for lookback periods: {custom_lookback_periods}")
    print(f"Dataset covers seasons: {start_season}-{end_season}")
    
    # Show some example feature columns
    feature_columns = [col for col in training_data.columns if any(f"Avg" in col and str(period) in col for period in custom_lookback_periods)]
    print(f"Example feature columns: {feature_columns[:10]}...")  # Show first 10 feature columns
    
    # Show season distribution in training data
    if 'season' in training_data.columns:
        season_counts = training_data['season'].value_counts().sort_index()
        print(f"\nSeason distribution in training data:")
        for season, count in season_counts.items():
            print(f"  {season}: {count} matches")
