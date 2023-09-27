import asyncio

import prisma.errors
from prisma import Prisma


async def create_team(id, name) -> int:
    db = Prisma()
    await db.connect()

    try:
        await db.team.create(
            {
                "id": id,
                "name": name
            }
        )
        return id
    except prisma.errors.UniqueViolationError:
        return id
    finally:
        await db.disconnect()


async def create_season(shortcut, year) -> int:
    db = Prisma()
    await db.connect()

    title = "Fussballliga"
    if shortcut == "bl1":
        title = "1. Bundesliga"

    try:
        season = await db.season.create(
            {
                "title": title,
                "year": year,
                "shortcut": shortcut
            }
        )
        return season.id
    except prisma.errors.UniqueViolationError:
        return 0
    finally:
        await db.disconnect()


async def create_team_on_season(team_id, season_id) -> str:
    db = Prisma()
    await db.connect()

    try:
        teamsonseason = await db.teamsonseasons.create(
            {
                "teamId": team_id,
                "seasonId": season_id
            }
        )
        return ""
    except prisma.errors.UniqueViolationError:
        return "None"
    except prisma.errors.ForeignKeyViolationError:
        return "None"
    finally:
        await db.disconnect()


async def get_all_teams_on_season() -> list:
    db = Prisma()
    await db.connect()

    teams_on_season = await db.teamsonseasons.find_many(
        include={
            'team': True,
            'season': True
        },
    )
    teams_on_season = [[x.id, x.team.name, x.season.year] for x in teams_on_season]

    await db.disconnect()

    return teams_on_season


async def update_market_value(teamonseasons_id, market_value) -> None:
    db = Prisma()
    await db.connect()

    await db.teamsonseasons.update(
        where={
            'id': teamonseasons_id
        },
        data={
            'marketValue': market_value
        }
    )

    await db.disconnect()


async def create_match(match_data, season_id) -> None:
    db = Prisma()
    await db.connect()

    try:
        await db.match.create(
            {
                "id": match_data["id"],
                "teamHomeId": match_data["teamHomeId"],
                "teamAwayId": match_data["teamAwayId"],
                "winnerTeamId": match_data["winnerTeamId"],
                "goalsHome": match_data["goalsHome"],
                "goalsAway": match_data["goalsAway"],
                "seasonId": season_id
            }
        )
    except prisma.errors.UniqueViolationError:
        pass
    finally:
        await db.disconnect()


async def get_all_matches() -> list:
    db = Prisma()
    await db.connect()

    matches_list = await db.match.find_many(
        include={
            'teamHome': {
                "include": {"seasons": True}
            },
            'teamAway': {
                "include": {"seasons": True}
            },
        }
    )

    await db.disconnect()

    return matches_list


async def get_matches_by_team(team_id) -> tuple:
    db = Prisma()
    await db.connect()

    team_data = await db.team.find_first(
        where={
            'id': team_id
        },
        include={
            'awayMatches': True,
            'homeMatches': True
        }
    )

    await db.disconnect()

    return team_data.homeMatches, team_data.awayMatches
