"""Global commands the user can type at any prompt."""
from __future__ import annotations

from enum import Enum
from typing import Dict


class Command(Enum):
    """Global commands the user can type at the prompts."""

    HELP = "help"
    BACK = "back"
    UNITS = "units"
    RECENT = "recent"
    QUIT = "quit"


COMMAND_ALIASES: Dict[str, Command] = {
    "help": Command.HELP, "h": Command.HELP, "?": Command.HELP,
    "back": Command.BACK, "b": Command.BACK,
    "units": Command.UNITS,
    "recent": Command.RECENT,
    "quit": Command.QUIT, "q": Command.QUIT, "exit": Command.QUIT,
}
