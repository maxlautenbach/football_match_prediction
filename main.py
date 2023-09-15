import asyncio
from database import dbconn

asyncio.run(dbconn.create(2, "FC RWE"))