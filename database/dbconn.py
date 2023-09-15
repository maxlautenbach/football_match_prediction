import asyncio
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
    except Exception:
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
    except Exception:
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
    except Exception:
        return "None"
    finally:
        await db.disconnect()
