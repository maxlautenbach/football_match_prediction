"""
Transfermarkt API using requests/BeautifulSoup for scraping market values.

This module provides multiple approaches to extract team market values from Transfermarkt:

1. get_teams_market_value() - Individual team lookup via search and kader pages
2. get_season_teams_market_values() - All teams for a season from league overview page (NEW, EFFICIENT)
3. get_multiple_seasons_market_values() - All teams across multiple seasons (NEW)
4. get_teams_market_values_sequential() - Multiple teams with delays
5. get_teams_market_values_threaded2() - Multiple teams with threading

The new season-based functions are more efficient as they extract all team data
from a single page request per season.

League identifiers:
- "bl1" or "L1" for 1. Bundesliga
- "bl2" or "L2" for 2. Bundesliga
"""

import requests
from bs4 import BeautifulSoup
import re
import time
from multiprocessing.pool import ThreadPool
from tqdm import tqdm


def get_teams_market_value(args, verbose=False, max_retries=3):
    """
    Get market value for a single team (compatible with old API).
    
    Args:
        args (tuple): (team_name, year)
        verbose (bool): Whether to show detailed output
        max_retries (int): Maximum number of retries for 503 errors
        
    Returns:
        tuple: (team, year, market_value_in_millions)
    """
    team, year = args
    
    # Clean team name
    team = team.replace("1. FC", "1.FC")
    team = team.replace("1. FSV", "1.FSV")
    team = team.replace("FC Kickers Würzburg", "Würzburger Kickers")

    for attempt in range(max_retries + 1):
        try:
            # Step 1: Search for team
            search_url = f"https://www.transfermarkt.de/schnellsuche/ergebnis/schnellsuche?query={team}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            if verbose:
                print(f"Searching for team: {team} (attempt {attempt + 1})")
            
            response = requests.get(search_url, headers=headers)
            
            # Check for 503 error
            if response.status_code == 503:
                if attempt < max_retries:
                    if verbose:
                        print(f"503 error for {team}, retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                else:
                    if verbose:
                        print(f"503 error for {team}, max retries reached")
                    return team, year, 0.0
            
            response.raise_for_status()  # Raise exception for other HTTP errors
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find team link in search results
            team_link_element = soup.find("td", {"class": "hauptlink"})
            if not team_link_element or not team_link_element.find("a"):
                if verbose:
                    print(f"Could not find team link for {team}")
                return team, year, 0.0
                
            relative_link = team_link_element.find("a")["href"]
            
            # Step 2: Build kader URL with season
            base_link = f"https://www.transfermarkt.de{relative_link}"
            kader_link = base_link.replace("/startseite/", "/kader/")
            team_url = f"{kader_link}?saison_id={year}"
            
            if verbose:
                print(f"Navigating to kader page: {team_url}")
            
            # Step 3: Get team kader page
            response = requests.get(team_url, headers=headers)
            
            # Check for 503 error on team page
            if response.status_code == 503:
                if attempt < max_retries:
                    if verbose:
                        print(f"503 error on team page for {team}, retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                else:
                    if verbose:
                        print(f"503 error on team page for {team}, max retries reached")
                    return team, year, 0.0
            
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Step 4: Find total market value in the "Kaderdetails nach Positionen" table
            # Look for the table that contains "Gesamt:" in the tfoot
            tables = soup.find_all('table')
            for table in tables:
                tfoot = table.find('tfoot')
                if tfoot:
                    # Check if this is the right table by looking for "Gesamt:" 
                    gesamt_cell = tfoot.find('td', string=re.compile(r'Gesamt:'))
                    if gesamt_cell:
                        # This is the "Kaderdetails nach Positionen" table
                        tr = tfoot.find('tr')
                        if tr:
                            tds = tr.find_all('td')
                            # The structure should be: Gesamt: | age | total_value | avg_value
                            # We want the total_value (index 2, 0-based)
                            if len(tds) >= 3:
                                market_value_text = tds[2].get_text(strip=True)
                                if "Mio. €" in market_value_text or "Tsd. €" in market_value_text:
                                    if verbose:
                                        print(f"Found market value: {market_value_text}")
                                    market_value = _parse_market_value(market_value_text)
                                    return team, year, market_value
            
            if verbose:
                print(f"Could not find market value for {team} ({year})")
            return team, year, 0.0
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                if verbose:
                    print(f"Request error for {team}: {e}, retrying in 2 seconds...")
                time.sleep(2)
                continue
            else:
                if verbose:
                    print(f"Request error for {team}: {e}, max retries reached")
                return team, year, 0.0
        except Exception as e:
            if verbose:
                print(f"Error getting market value for {team} ({year}): {e}")
            return team, year, 0.0
    
    # Should never reach here, but just in case
    return team, year, 0.0


def _parse_market_value(value_text):
    """
    Parse market value text to float in millions.
    
    Args:
        value_text (str): Market value text like "596,05 Mio. €"
        
    Returns:
        float: Market value in millions
    """
    if "Mio. €" in value_text:
        # Extract number: "596,05 Mio. €" -> 596.05
        number_part = value_text.replace("Mio. €", "").strip()
        return float(number_part.replace(",", "."))
    elif "Tsd. €" in value_text:
        # Convert thousands to millions: "500 Tsd. €" -> 0.5
        number_part = value_text.replace("Tsd. €", "").strip()
        return float(number_part.replace(",", ".")) / 1000
    else:
        return 0.0


def get_teams_market_values_sequential(teams_on_season, verbose=False, delay=1.0):
    """
    Get market values for multiple teams sequentially (no threading).
    
    This is slower but more polite to the server and helps avoid 503 errors.
    
    Args:
        teams_on_season (pd.DataFrame): DataFrame with 'teamHomeName' and 'season' columns
        verbose (bool): Whether to show detailed output
        delay (float): Delay in seconds between requests (default: 1.0)
        
    Returns:
        list: List of (team, year, market_value) tuples
    """
    results = []
    teams = [(x["teamHomeName"], x["season"]) for _, x in teams_on_season.iterrows()]
    
    if verbose:
        print(f"Processing {len(teams)} teams sequentially with {delay}s delay...")
    
    for i, (team, season) in enumerate(tqdm(teams, desc="Fetching market values")):
        if verbose:
            print(f"\n[{i+1}/{len(teams)}] Processing: {team} ({season})")
        
        result = get_teams_market_value((team, season), verbose=verbose)
        results.append(result)
        
        # Add delay between requests (except for the last one)
        if i < len(teams) - 1:
            if verbose:
                print(f"Waiting {delay}s before next request...")
            time.sleep(delay)
    
    if verbose:
        successful = sum(1 for _, _, value in results if value > 0)
        print(f"\nCompleted: {successful}/{len(results)} successful extractions")
    
    return results


def get_season_teams_market_values(season_id, league="bl1", verbose=False, max_retries=10):
    """
    Get market values for all teams in a specific season directly from the league overview page.
    
    This is more efficient than individual team queries as it gets all data from one page.
    
    Args:
        season_id (int): The season year (e.g., 2025)
        league (str): League identifier (default: "bl1" for 1. Bundesliga, "bl2" for 2. Bundesliga)
        verbose (bool): Whether to show detailed output
        max_retries (int): Maximum number of retries for 503 errors
        
    Returns:
        list: List of (team_name, season_id, market_value_in_millions) tuples
    """
    # Map league identifiers to Transfermarkt league codes
    league_mapping = {
        "bl1": "L1",  # 1. Bundesliga
        "bl2": "L2",  # 2. Bundesliga
        "L1": "L1",   # Keep backward compatibility
        "L2": "L2"    # Keep backward compatibility
    }
    
    transfermarkt_league = league_mapping.get(league, "L1")  # Default to L1 if unknown
    url = f"https://www.transfermarkt.de/bundesliga/startseite/wettbewerb/{transfermarkt_league}/plus/?saison_id={season_id}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    for attempt in range(max_retries + 1):
        try:
            if verbose:
                print(f"Fetching season {season_id} data from: {url} (attempt {attempt + 1})")
            
            response = requests.get(url, headers=headers)
            
            # Check for 503 error
            if response.status_code == 503:
                if attempt < max_retries:
                    if verbose:
                        print(f"503 error, retrying in 5 seconds...")
                    time.sleep(5)
                    continue
                else:
                    if verbose:
                        print(f"503 error, max retries reached")
                    return []
            
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the main table with teams data
            # Try different approaches to find the table
            main_table = None
            tbody = None
            
            # First try: look for table with id="yw1"
            main_table = soup.find('table', {'id': 'yw1'})
            if main_table:
                tbody = main_table.find('tbody')
                if verbose:
                    print("Found table with id 'yw1'")
            
            # Second try: look for any table that contains team data
            if not tbody:
                if verbose:
                    print("Could not find table with id 'yw1', trying alternative approach")
                
                # Look for tables that contain team names or market values
                all_tables = soup.find_all('table')
                for table in all_tables:
                    table_text = table.get_text().lower()
                    # Check for common team names in both leagues
                    has_teams = any(name in table_text for name in ['bayern', 'dortmund', 'hamburg', 'bremen', 'kiel', 'düsseldorf', 'nürnberg'])
                    has_values = 'mio' in table_text or 'tsd' in table_text
                    
                    if has_teams and has_values:
                        tbody = table.find('tbody')
                        if tbody and len(tbody.find_all('tr')) > 10:  # Should have many teams
                            main_table = table
                            if verbose:
                                print(f"Found alternative table with {len(tbody.find_all('tr'))} rows")
                            break
            
            if not tbody:
                if verbose:
                    print("Could not find any suitable table with team data")
                    # Debug: print available table info
                    tables = soup.find_all('table')
                    print(f"Found {len(tables)} tables total")
                    for i, table in enumerate(tables[:5]):  # Show first 5 tables
                        rows = len(table.find_all('tr')) if table.find_all('tr') else 0
                        print(f"Table {i+1}: {rows} rows, id='{table.get('id', 'None')}', class='{table.get('class', 'None')}'")
                return []
            
            results = []
            rows = tbody.find_all('tr')
            
            if verbose:
                print(f"Found {len(rows)} team rows")
            
            for i, row in enumerate(rows):
                try:
                    cells = row.find_all('td')
                    if len(cells) < 7:
                        if verbose:
                            print(f"Row {i+1}: Not enough cells ({len(cells)}), skipping")
                        continue
                    
                    # Extract team name from 2nd column (index 1)
                    team_name_cell = cells[1]
                    team_name_link = team_name_cell.find('a')
                    if team_name_link:
                        team_name = team_name_link.get_text(strip=True)
                    else:
                        team_name = team_name_cell.get_text(strip=True)
                    
                    # Extract market value from 7th column (index 6)
                    market_value_cell = cells[6]
                    market_value_link = market_value_cell.find('a')
                    if market_value_link:
                        market_value_text = market_value_link.get_text(strip=True)
                    else:
                        market_value_text = market_value_cell.get_text(strip=True)
                    
                    # Parse market value
                    market_value = _parse_market_value(market_value_text)
                    
                    if verbose:
                        print(f"Row {i+1}: {team_name} -> {market_value_text} -> {market_value} Mio. €")
                    
                    results.append((team_name, season_id, market_value))
                    
                except Exception as e:
                    if verbose:
                        print(f"Error processing row {i+1}: {e}")
                    continue
            
            if verbose:
                print(f"Successfully extracted {len(results)} teams for season {season_id}")
            
            return results
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                if verbose:
                    print(f"Request error: {e}, retrying in 2 seconds...")
                time.sleep(2)
                continue
            else:
                if verbose:
                    print(f"Request error: {e}, max retries reached")
                return []
        except Exception as e:
            if verbose:
                print(f"Error fetching season data: {e}")
            return []
    
    # Should never reach here, but just in case
    return []


def get_multiple_seasons_market_values(season_ids, league="bl1", verbose=False):
    """
    Get market values for all teams across multiple seasons.
    
    Args:
        season_ids (list): List of season years (e.g., [2023, 2024, 2025])
        league (str): League identifier (default: "bl1" for 1. Bundesliga, "bl2" for 2. Bundesliga)
        verbose (bool): Whether to show detailed output
        
    Returns:
        list: List of (team_name, season_id, market_value_in_millions) tuples
    """
    all_results = []
    
    for season_id in season_ids:
        if verbose:
            print(f"\nFetching data for season {season_id}...")
        
        season_results = get_season_teams_market_values(season_id, league, verbose=False)
        all_results.extend(season_results)
        
        if verbose:
            print(f"Found {len(season_results)} teams for season {season_id}")
        
        # Add small delay between seasons to be polite
        if season_id != season_ids[-1]:  # Don't delay after the last season
            time.sleep(1)
    
    if verbose:
        print(f"\nTotal results: {len(all_results)} team-season combinations")
    
    return all_results


def get_teams_market_values_threaded2(teams_on_season, verbose=False):
    """
    Get market values for multiple teams using threading (compatible with old API).
    
    Args:
        teams_on_season (pd.DataFrame): DataFrame with 'teamHomeName' and 'season' columns
        verbose (bool): Whether to show detailed output
        
    Returns:
        list: List of (team, year, market_value) tuples
    """
    pool = ThreadPool(16)
    teams = [(x["teamHomeName"], x["season"]) for _, x in teams_on_season.iterrows()]
    
    # Create a wrapper function that includes the verbose parameter
    def get_market_value_with_verbose(args):
        return get_teams_market_value(args, verbose=verbose)
    
    results = list(tqdm(pool.imap(get_market_value_with_verbose, teams), total=len(teams)))
    
    pool.close()
    pool.join()
    
    return results


if __name__ == "__main__":
    # Test the new season-based function
    print("Testing new get_season_teams_market_values function for 1. Bundesliga 2025")
    print("=" * 70)
    
    season_results = get_season_teams_market_values(2025, league="bl1", verbose=True)
    
    print(f"\nResults for season 2025:")
    print("-" * 50)
    for team, season, market_value in season_results:
        print(f"{team:<25} | {season} | {market_value:>8.2f} Mio. €")
    
    print(f"\nTotal teams: {len(season_results)}")
    if season_results:
        total_value = sum(mv for _, _, mv in season_results)
        avg_value = total_value / len(season_results)
        print(f"Total market value: {total_value:.2f} Mio. €")
        print(f"Average market value: {avg_value:.2f} Mio. €")
    
    print("\n" + "=" * 70)
    print("Testing individual team function for comparison:")
    
    # Also test individual function for comparison
    bl_teams = [
        "FC Bayern München",
        "Borussia Dortmund"
    ]
    
    print("\nIndividual team results:")
    print("-" * 30)
    for team in bl_teams:
        result = get_teams_market_value((team, 2025), verbose=False)
        print(f"{result[0]:<25} | {result[1]} | {result[2]:>8.2f} Mio. €")
    
    print("\n" + "=" * 70)
    print("Testing multiple seasons function:")
    
    # Test multiple seasons function for 1. Bundesliga
    multi_season_results = get_multiple_seasons_market_values([2024, 2025], league="bl1", verbose=True)
    
    print(f"\nMultiple seasons summary:")
    print("-" * 40)
    seasons_summary = {}
    for team, season, value in multi_season_results:
        if season not in seasons_summary:
            seasons_summary[season] = []
        seasons_summary[season].append(value)
    
    for season in sorted(seasons_summary.keys()):
        values = seasons_summary[season]
        print(f"Season {season}: {len(values)} teams, total: {sum(values):.2f} Mio. €")
    
    print("\n" + "=" * 70)
    print("Testing 2. Bundesliga function:")
    
    # Test 2. Bundesliga
    bl2_results = get_season_teams_market_values(2025, league="bl2", verbose=True)
    
    print(f"\n2. Bundesliga results for season 2025:")
    print("-" * 50)
    for team, season, market_value in bl2_results:
        print(f"{team:<25} | {season} | {market_value:>8.2f} Mio. €")
    
    print(f"\n2. Bundesliga total teams: {len(bl2_results)}")
    if bl2_results:
        total_value = sum(mv for _, _, mv in bl2_results)
        avg_value = total_value / len(bl2_results)
        print(f"2. Bundesliga total market value: {total_value:.2f} Mio. €")
        print(f"2. Bundesliga average market value: {avg_value:.2f} Mio. €")