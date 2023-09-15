from multiprocessing.pool import ThreadPool
import re
import requests as requests
from bs4 import BeautifulSoup


def get_teams_market_value(args):
    team, year = args
    team = team.replace("1. FC", "1.FC")
    team = team.replace("1. FSV", "1.FSV")
    team = team.replace("FC Kickers Würzburg", "Würzburger Kickers")

    try:
        url = "https://www.transfermarkt.de/schnellsuche/ergebnis/schnellsuche?query=" + team

        headers = {'User-Agent':
                       'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}

        # Senden des GET-Requests
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        link = "https://www.transfermarkt.de" + soup.findAll("td", {"class": "hauptlink"})[0].next_element[
            "href"] + "?saison_id=" + str(year)
        response = requests.get(link, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')

        market_values = soup.findAll("td", {"class": "rechts hauptlink"})
        total_value = 0
        for market_value in market_values:
            try:
                market_value = market_value.next_element.text
                if "Tsd" in market_value:
                    market_value = float(re.sub("[^\d]", "", market_value)) * 1000
                elif "Mio" in market_value:
                    market_value = float(re.sub("[^\d]", "", market_value)) * 1000000
                total_value += market_value
            except TypeError:
                pass
        return total_value
    except IndexError:
        print(args)
        return 10000000


def get_teams_market_values_threaded(teams, year):
    pool = ThreadPool(16)
    results = pool.map(get_teams_market_value, [(x, year) for x in teams])
    results_dict = dict(zip(teams, results))
    return results_dict