import importlib.metadata
import sys

CLIENT_USER_AGENT = (
    "WikiDataPeopleGatorBot/1.0 "
    f"Python/{sys.version_info[0]}.{sys.version_info[1]} "
    f"(https://wiki-data-people-gator-bot.capturemyhand.com/bot-ip-ranges.json; ivasko@fit.vut.cz) "
    f"requests/{importlib.metadata.version('requests')}"
)

CLIENT_HEADERS = {"User-Agent": CLIENT_USER_AGENT}
