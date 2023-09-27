import asyncio
import pickle

from database import dbconn
import pandas as pd

def get_market_values_by_match(match_data, home_bool):
    if home_bool:
        return next(item.marketValue for item in match_data.teamHome.seasons if item.seasonId == match_data.seasonId)
    return next(item.marketValue for item in match_data.teamAway.seasons if item.seasonId == match_data.seasonId)

def create_initial_dataframe():
    matches_list = asyncio.run(dbconn.get_all_matches())
    matches_list = [[x.id, x.teamHomeId, x.teamAwayId, get_market_values_by_match(x, True), get_market_values_by_match(x, False), x.goalsHome, x.goalsAway, x.winnerTeamId] for x in matches_list]
    columns = ["Match ID", "Team Home ID", "Team Away ID", "Market Value Home", "Market Value Away", "Goals Home", "Goals Away", "Winner Team ID"]
    matches_df = pd.DataFrame(matches_list, columns=columns)
    matches_df = matches_df.fillna(0)
    matches_df["Winner Team ID"] = matches_df["Winner Team ID"].astype('int32')
    pickle.dump(matches_df, open("database/matches_df.pck", "wb"))

def get_matches_by_team(team_id):
    team = asyncio.run(dbconn.get_matches_by_team(team_id))
    return team
print("END")