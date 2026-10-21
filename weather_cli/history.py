"""In-memory history of recently viewed locations."""
from __future__ import annotations

from typing import List, Tuple

from .models import Location


class RecentLocations:
    """Most-recently-viewed locations for the current session (newest first)."""

    def __init__(self, max_size: int = 5) -> None:
        """Creates an empty history.

        Args:
            max_size: Maximum number of locations to keep (>= 1).

        Raises:
            ValueError: If ``max_size`` is not a positive integer.
        """
        if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size < 1:
            raise ValueError("max_size must be a positive integer.")
        self._max_size = max_size
        self._items: List[Location] = []

    @property
    def items(self) -> Tuple[Location, ...]:
        """Returns the stored locations, newest first."""
        return tuple(self._items)

    def add(self, location: Location) -> None:
        """Adds a location to the front, moving it there if already present.

        Args:
            location: The location to remember.

        Raises:
            TypeError: If ``location`` is not a ``Location``.
        """
        if not isinstance(location, Location):
            raise TypeError("location must be a Location instance.")
        if location in self._items:
            self._items.remove(location)
        self._items.insert(0, location)
        del self._items[self._max_size:]
