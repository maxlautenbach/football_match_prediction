import asyncio
import os

import api.openligadb
import api.transfermarkt
import dbconn
from tqdm import tqdm


def extract_result(match_results):
    for match_result in match_results:
        if match_result['resultName'] == 'Endergebnis':
            return match_result['pointsTeam1'], match_result['pointsTeam2']
    return None


def extract_winner(match_data):
    if match_data["goalsHome"] > match_data["goalsAway"]:
        return match_data["teamHomeId"]
    elif match_data["goalsHome"] < match_data["goalsAway"]:
        return match_data["teamAwayId"]
    else:
        return None


def extract_match_data(json):
    match_data = {}
    match_data["id"] = json["matchID"]
    match_data["teamHomeId"] = json["team1"]["teamId"]
    match_data["teamAwayId"] = json["team2"]["teamId"]
    match_data["goalsHome"], match_data["goalsAway"] = extract_result(json["matchResults"])
    match_data["winnerTeamId"] = extract_winner(match_data)
    return match_data

os.system("npm run reset")

for year in [*range(2012, 2022, 1)]:
    season_id = asyncio.run(dbconn.create_season("bl1", year))
    teams = api.openligadb.get_all_season_teams("bl1", year)
    for team in teams:
        team_id = asyncio.run(dbconn.create_team(team["teamId"], team["teamName"]))
        asyncio.run(dbconn.create_team_on_season(team_id, season_id))
    matches = api.openligadb.get_all_season_matches("bl1", year)
    for match in tqdm(matches):
        asyncio.run(dbconn.create_match(extract_match_data(match), season_id))

teams_on_season = asyncio.run(dbconn.get_all_teams_on_season())
market_values = api.transfermarkt.get_teams_market_values_threaded(teams_on_season)
for team_on_season_id, market_value in market_values.items():
    asyncio.run(dbconn.update_market_value(team_on_season_id, market_value))
print("Database filled up")
