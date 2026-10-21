"""Pure validation and parsing of raw user input (no I/O)."""
from __future__ import annotations

from typing import List, Optional

from .commands import COMMAND_ALIASES, Command
from .config import METRICS


class InputParser:
    """Validates and parses raw text typed by the user.

    All methods are pure: they never print or read, they only return a parsed
    value or raise ``ValueError`` with a message that can be shown to the user.
    """

    BACK = "0"
    ALL = "A"
    MIN_CITY_LENGTH = 2
    MAX_CITY_LENGTH = 100

    @staticmethod
    def parse_command(raw: str) -> Optional[Command]:
        """Recognizes a global command.

        Args:
            raw: Raw user input.

        Returns:
            The command (aliases are case-insensitive), or ``None`` for other text.
        """
        return COMMAND_ALIASES.get(raw.strip().lower())

    @classmethod
    def parse_city_name(cls, raw: str) -> str:
        """Validates a city name.

        Args:
            raw: Raw user input.

        Returns:
            The trimmed city name.

        Raises:
            TypeError: If ``raw`` is not a string.
            ValueError: If the name is empty, too short/long, or has no letters.
        """
        if not isinstance(raw, str):
            raise TypeError("City name must be a string.")
        name = " ".join(raw.split())
        if not name:
            raise ValueError("City name cannot be empty.")
        if not cls.MIN_CITY_LENGTH <= len(name) <= cls.MAX_CITY_LENGTH:
            raise ValueError(
                f"City name must be {cls.MIN_CITY_LENGTH}-{cls.MAX_CITY_LENGTH} characters long."
            )
        if not any(char.isalpha() for char in name):
            raise ValueError("City name must contain at least one letter.")
        return name

    @staticmethod
    def parse_index(raw: str, upper: int) -> int:
        """Parses a menu choice in the range ``0..upper``.

        Args:
            raw: Raw user input.
            upper: Largest valid choice (must be >= 1).

        Returns:
            The chosen number (0 means "back").

        Raises:
            ValueError: If ``upper`` < 1 or the input is not a number in range.
        """
        if upper < 1:
            raise ValueError("Menu must have at least one option.")
        text = raw.strip()
        if not text.isdecimal() or not 0 <= int(text) <= upper:
            raise ValueError(f"Please enter a number between 0 and {upper}.")
        return int(text)

    @classmethod
    def parse_metric_selection(cls, raw: str) -> Optional[List[str]]:
        """Parses a comma-separated metric selection such as ``"1, 3"`` or ``"A"``.

        Args:
            raw: Raw user input.

        Returns:
            ``None`` if the user chose "back" (``0``), otherwise the selected
            metric keys in the order typed, without duplicates.

        Raises:
            ValueError: If the selection is empty or contains an unknown option.
        """
        text = raw.strip().upper()
        if text == cls.BACK:
            return None
        if text == cls.ALL:
            return list(METRICS)
        tokens = [token.strip() for token in text.split(",") if token.strip()]
        if not tokens or any(token not in METRICS for token in tokens):
            raise ValueError(
                f"Choose option numbers 1-{len(METRICS)} separated by commas, "
                f"'{cls.ALL}' for all, or '{cls.BACK}' to go back."
            )
        return list(dict.fromkeys(tokens))

    @staticmethod
    def parse_yes_no(raw: str) -> bool:
        """Parses a yes/no answer.

        Args:
            raw: Raw user input.

        Returns:
            True for yes, False for no.

        Raises:
            ValueError: If the answer is neither yes nor no.
        """
        answer = raw.strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        raise ValueError("Please answer 'y' or 'n'.")
