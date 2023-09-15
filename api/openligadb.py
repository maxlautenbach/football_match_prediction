import requests


def get_all_season_matches(league, season):
    res = requests.get("https://api.openligadb.de/getmatchdata/" + str(league) + "/" + str(season))
    body = res.json()
    return body


def get_all_season_teams(league, season):
    res = requests.get("https://api.openligadb.de/getavailableteams/" + str(league) + "/" + str(season))
    body = res.json()
    return body


#print(get_all_season_teams("bl1", 2022))
