"""Immutable data models: locations, daily forecasts, unit systems and metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple


def _validate_coordinate(name: str, value: Any, limit: float) -> None:
    """Validates that a coordinate is a number within ``[-limit, limit]``.

    Args:
        name: Coordinate name, used in error messages.
        value: The value to validate.
        limit: Maximum absolute value allowed.

    Raises:
        TypeError: If ``value`` is not an int or float (bool is rejected).
        ValueError: If ``value`` is outside the allowed range.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
    if not -limit <= value <= limit:
        raise ValueError(f"{name} must be between {-limit} and {limit}, got {value}.")


@dataclass(frozen=True)
class Location:
    """Immutable geographical location returned by the geocoding API.

    Attributes:
        name: City name.
        latitude: Latitude in degrees, between -90 and 90.
        longitude: Longitude in degrees, between -180 and 180.
        country: Country name, if known.
        region: First-level administrative region (state/district), if known.

    Raises:
        TypeError: If a coordinate is not numeric.
        ValueError: If the name is empty or a coordinate is out of range.
    """

    name: str
    latitude: float
    longitude: float
    country: Optional[str] = None
    region: Optional[str] = None

    def __post_init__(self) -> None:
        """Validates the fields right after construction."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Location name must be a non-empty string.")
        _validate_coordinate("latitude", self.latitude, 90.0)
        _validate_coordinate("longitude", self.longitude, 180.0)

    @property
    def display_name(self) -> str:
        """Returns a human-readable name such as ``"Paris, Texas, United States"``."""
        extras = [part for part in (self.region, self.country) if part and part != self.name]
        return ", ".join([self.name, *extras])


@dataclass(frozen=True)
class DailyForecast:
    """Immutable weather values for a single day.

    Attributes:
        date: ISO date string (``YYYY-MM-DD``).
        values: Mapping of API field name to its value (``None`` when missing).
    """

    date: str
    values: Mapping[str, Optional[float]]

    def value(self, field_name: str) -> Optional[float]:
        """Returns the value of one API field.

        Args:
            field_name: Open-Meteo daily field name.

        Returns:
            The numeric value, or ``None`` if the field is missing.
        """
        return self.values.get(field_name)


@dataclass(frozen=True)
class UnitSystem:
    """A set of display units and the matching Open-Meteo request parameters.

    Attributes:
        name: System name (``metric`` or ``imperial``).
        temperature: Temperature display unit.
        wind_speed: Wind speed display unit.
        precipitation: Precipitation display unit.
        api_params: Query parameters that make the API return these units.
    """

    name: str
    temperature: str
    wind_speed: str
    precipitation: str
    api_params: Mapping[str, str]

    def unit_for(self, kind: str) -> str:
        """Returns the display unit for a kind of measurement.

        Args:
            kind: One of ``temperature``, ``wind_speed``, ``precipitation``, ``percent``.

        Returns:
            The display unit.

        Raises:
            ValueError: If ``kind`` is unknown.
        """
        units = {
            "temperature": self.temperature,
            "wind_speed": self.wind_speed,
            "precipitation": self.precipitation,
            "percent": "%",
        }
        if kind not in units:
            raise ValueError(f"Unknown measurement kind: {kind!r}.")
        return units[kind]

    @property
    def summary(self) -> str:
        """Returns a short description such as ``"metric (°C, km/h, mm)"``."""
        return f"{self.name} ({self.temperature}, {self.wind_speed}, {self.precipitation})"


METRIC_UNITS = UnitSystem(
    "metric", "°C", "km/h", "mm",
    {"temperature_unit": "celsius", "wind_speed_unit": "kmh", "precipitation_unit": "mm"},
)
IMPERIAL_UNITS = UnitSystem(
    "imperial", "°F", "mph", "inch",
    {"temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "precipitation_unit": "inch"},
)
UNIT_SYSTEMS: Dict[str, UnitSystem] = {units.name: units for units in (METRIC_UNITS, IMPERIAL_UNITS)}


@dataclass(frozen=True)
class Metric:
    """A user-selectable weather metric backed by one or more API fields.

    Attributes:
        label: Short display name.
        kind: Measurement kind, used to pick the display unit.
        fields: Open-Meteo daily field names (a min/max pair is shown as a range).
    """

    label: str
    kind: str
    fields: Tuple[str, ...]

    def menu_label(self, units: UnitSystem) -> str:
        """Returns the label shown in the metrics menu.

        Args:
            units: The active unit system.

        Returns:
            The label with its unit, e.g. ``"Temperature (°C)"``.
        """
        return f"{self.label} ({units.unit_for(self.kind)})"
