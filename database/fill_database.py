import asyncio

import api.openligadb
import dbconn
from tqdm import tqdm

def extract_result(match_results):
    for match_result in match_results:
        if match_result['resultName'] == 'Endergebnis':
            return match_result['pointsTeam1'], match_result['pointsTeam2']
    return None

def extract_winner(match_data):
    if match_data["homeGoals"] > match_data["awayGoals"]:
        return match_data["homeTeamId"]
    elif match_data["homeGoals"] < match_data["awayGoals"]:
        return match_data["awayTeamId"]
    else:
        return None

def extract_match_data(json):
    match_data = {}
    match_data["id"] = json["matchID"]
    match_data["homeTeamId"] = json["team1"]["teamId"]
    match_data["awayTeamId"] = json["team2"]["teamId"]
    match_data["homeGoals"], match_data["awayGoals"] = extract_result(json["matchResults"])
    print(json)
    return match_data

for year in tqdm([*range(2015, 2022, 1)]):
    #season_id = asyncio.run(dbconn.create_season("bl1", year))
    #teams = api.openligadb.get_all_season_teams("bl1", year)
    #for team in teams:
        #team_id = asyncio.run(dbconn.create_team(team["teamId"], team["teamName"]))
        #asyncio.run(dbconn.create_team_on_season(team_id, season_id))
    matches = api.openligadb.get_all_season_matches("bl1", year)
    for match in matches:
        extract_match_data(match)