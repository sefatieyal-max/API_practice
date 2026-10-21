"""Weather CLI: an interactive command-line client for the Open-Meteo REST API.

The user searches for a city, picks a forecast range (today or 7 days) and the
metrics to show. Today's forecast is printed as a detailed summary and the
weekly forecast as ASCII bar charts. Global commands (``help``, ``back``,
``units``, ``recent``, ``quit``) work at the prompts, and a one-shot
non-interactive mode is available through command-line flags.

Layers (each one has a single responsibility):
    * Exceptions and data models.
    * ``InputParser``: pure validation and parsing of raw user text.
    * ``WeatherAPIClient``: HTTP communication and response parsing.
    * ``ForecastFormatter``: turns data into printable lines (no I/O).
    * ``RecentLocations``: in-memory history of viewed locations.
    * ``WeatherCLI``: the interactive loop (all terminal I/O lives here).

Example:
    $ python weather.py
    $ python weather.py --units imperial
    $ python weather.py --city London --days 7 --metrics 1,5
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ==========================================
# 1. Custom Exceptions
# ==========================================
class WeatherAppError(Exception):
    """Base exception for all application errors."""


class APICommunicationError(WeatherAppError):
    """Raised when the API cannot be reached or answers with an HTTP error."""


class InvalidResponseError(WeatherAppError):
    """Raised when the API answers with a payload in an unexpected shape."""


class QuitRequested(Exception):
    """Raised internally when the user asks to leave the application."""


# ==========================================
# 2. Data Models
# ==========================================
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


# ==========================================
# 3. Menus, Commands and Metric Mapping
# ==========================================
METRICS: Dict[str, Metric] = {
    "1": Metric("Temperature", "temperature", ("temperature_2m_min", "temperature_2m_max")),
    "2": Metric("Feels Like", "temperature", ("apparent_temperature_min", "apparent_temperature_max")),
    "3": Metric("Max Wind Speed", "wind_speed", ("wind_speed_10m_max",)),
    "4": Metric("Precipitation", "precipitation", ("precipitation_sum",)),
    "5": Metric("Rain Chance", "percent", ("precipitation_probability_max",)),
}

FIELD_TITLES: Dict[str, str] = {
    "temperature_2m_min": "Min Temperature",
    "temperature_2m_max": "Max Temperature",
    "apparent_temperature_min": "Min Feels Like",
    "apparent_temperature_max": "Max Feels Like",
    "wind_speed_10m_max": "Max Wind Speed",
    "precipitation_sum": "Total Precipitation",
    "precipitation_probability_max": "Rain Probability",
}

DEFAULT_TIMEOUT: float = 10.0  # seconds
MAX_FORECAST_DAYS: int = 16  # Open-Meteo limit

# (menu label, number of forecast days)
FORECAST_MODES: Tuple[Tuple[str, int], ...] = (
    ("Today's Forecast (1 Day)", 1),
    ("Weekly Forecast (7 Days)", 7),
)


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

# Context help shown under the command list, per prompt.
CITY_HELP = "Type a city name (e.g. 'London'), or 'recent' to pick a recently viewed city."
LOCATION_HELP = "Several places share this name. Pick the number of the one you meant."
MODE_HELP = "1 shows today's detailed forecast, 2 shows 7-day charts."
METRICS_HELP = "Type option numbers separated by commas (e.g. '1,3') or 'A' for all."
YES_NO_HELP = "Answer 'y' to use the suggested location or 'n' to search again."
RECENT_HELP = "Pick the number of a city you viewed earlier in this session."


# ==========================================
# 4. Input Parsing (pure, no I/O)
# ==========================================
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


# ==========================================
# 5. API Client
# ==========================================
class WeatherAPIClient:
    """Client responsible for communication with the Open-Meteo REST APIs.

    Uses a shared ``requests.Session`` with automatic retries for transient
    failures (HTTP 429/5xx and connection errors). Can be used as a context
    manager so the session is always closed.

    Attributes:
        timeout: Per-request timeout in seconds.
    """

    GEOCODE_URL: str = "https://geocoding-api.open-meteo.com/v1/search"
    FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"
    DEFAULT_TIMEOUT: float = DEFAULT_TIMEOUT
    MAX_RESULTS: int = 100
    MAX_FORECAST_DAYS: int = MAX_FORECAST_DAYS
    RETRY_STATUSES: Tuple[int, ...] = (429, 500, 502, 503, 504)

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        session: Optional[requests.Session] = None,
        retries: int = 2,
    ) -> None:
        """Creates the client.

        Args:
            timeout: Per-request timeout in seconds (must be positive).
            session: Optional pre-built session (useful for tests).
            retries: Number of retries for transient failures.

        Raises:
            TypeError: If ``timeout`` is not a number.
            ValueError: If ``timeout`` is not positive or ``retries`` is negative.
        """
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("timeout must be a number.")
        if timeout <= 0:
            raise ValueError("timeout must be positive.")
        if retries < 0:
            raise ValueError("retries cannot be negative.")
        self.timeout = timeout
        self._session = session if session is not None else self._build_session(retries)

    @classmethod
    def _build_session(cls, retries: int) -> requests.Session:
        """Builds a session that retries idempotent GET requests with backoff.

        Args:
            retries: Number of retries.

        Returns:
            A configured ``requests.Session``.
        """
        retry = Retry(
            total=retries,
            backoff_factor=0.5,
            status_forcelist=cls.RETRY_STATUSES,
            allowed_methods=("GET",),
            raise_on_status=False,
        )
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=retry))
        return session

    def close(self) -> None:
        """Closes the underlying HTTP session."""
        self._session.close()

    def __enter__(self) -> "WeatherAPIClient":
        """Returns the client itself for use in a ``with`` block."""
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Closes the session when leaving a ``with`` block."""
        self.close()

    def search_locations(self, city_name: str, count: int = 5) -> List[Location]:
        """Searches for locations matching a city name.

        Args:
            city_name: Name of the city to search for.
            count: Maximum number of results (1-100).

        Returns:
            Matching locations; empty if nothing matched. Malformed records in
            the response are skipped.

        Raises:
            ValueError: If ``city_name`` is empty or ``count`` is out of range.
            APICommunicationError: On network or HTTP errors.
            InvalidResponseError: If the response is not in the expected shape.
        """
        if not isinstance(city_name, str) or not city_name.strip():
            raise ValueError("city_name must be a non-empty string.")
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= self.MAX_RESULTS:
            raise ValueError(f"count must be an integer between 1 and {self.MAX_RESULTS}.")

        params = {"name": city_name.strip(), "count": count, "format": "json"}
        payload = self._get_json(self.GEOCODE_URL, params)
        results = payload.get("results") or []
        if not isinstance(results, list):
            raise InvalidResponseError("Geocoding response has an invalid 'results' field.")
        locations = (self._parse_location(item) for item in results)
        return [location for location in locations if location is not None]

    def get_forecast(
        self,
        location: Location,
        fields: Sequence[str],
        days: int,
        units: UnitSystem = METRIC_UNITS,
    ) -> List[DailyForecast]:
        """Fetches the daily forecast for the requested fields.

        Args:
            location: Target location.
            fields: Open-Meteo daily field names (keys of ``FIELD_TITLES``).
            days: Number of forecast days (1-16).
            units: Unit system the API should return values in.

        Returns:
            One ``DailyForecast`` per day, in chronological order.

        Raises:
            TypeError: If ``location`` or ``units`` has the wrong type.
            ValueError: If ``fields`` is empty/unknown or ``days`` is out of range.
            APICommunicationError: On network or HTTP errors.
            InvalidResponseError: If the response is not in the expected shape.
        """
        self._validate_forecast_request(location, fields, days, units)
        unique_fields = list(dict.fromkeys(fields))
        params = {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "daily": ",".join(unique_fields),
            "forecast_days": days,
            "timezone": "auto",
            **units.api_params,
        }
        payload = self._get_json(self.FORECAST_URL, params)
        return self._parse_daily(payload, unique_fields)

    def _validate_forecast_request(
        self, location: Location, fields: Sequence[str], days: int, units: UnitSystem
    ) -> None:
        """Validates the arguments of ``get_forecast``.

        Args:
            location: Target location.
            fields: Requested field names.
            days: Number of forecast days.
            units: Requested unit system.

        Raises:
            TypeError: If ``location`` or ``units`` has the wrong type.
            ValueError: If ``fields`` is empty/unknown or ``days`` is out of range.
        """
        if not isinstance(location, Location):
            raise TypeError("location must be a Location instance.")
        if not isinstance(units, UnitSystem):
            raise TypeError("units must be a UnitSystem instance.")
        if isinstance(fields, str) or not fields:
            raise ValueError("fields must be a non-empty sequence of field names.")
        unknown = [name for name in fields if name not in FIELD_TITLES]
        if unknown:
            raise ValueError(f"Unknown forecast fields: {', '.join(map(str, unknown))}.")
        if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= self.MAX_FORECAST_DAYS:
            raise ValueError(f"days must be an integer between 1 and {self.MAX_FORECAST_DAYS}.")

    def _get_json(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Performs a GET request and returns the decoded JSON object.

        Args:
            url: Endpoint URL.
            params: Query string parameters.

        Returns:
            The decoded JSON object.

        Raises:
            APICommunicationError: On timeout, connection or HTTP errors.
            InvalidResponseError: If the body is not a JSON object.
        """
        try:
            response = self._session.get(url, params=params, timeout=self.timeout)
        except requests.exceptions.Timeout as exc:
            raise APICommunicationError(
                f"The weather service did not respond within {self.timeout:g} seconds."
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise APICommunicationError(
                "Could not connect to the weather service. Check your internet connection."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise APICommunicationError(f"Request to the weather service failed: {exc}") from exc

        self._raise_for_status(response)
        return self._decode_json(response)

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        """Raises a readable error if the response has an HTTP error status.

        Open-Meteo returns ``{"error": true, "reason": "..."}`` on bad requests,
        so the reason is included in the message when present.

        Args:
            response: The HTTP response.

        Raises:
            APICommunicationError: If the status code is 4xx/5xx.
        """
        if response.ok:
            return
        reason = response.reason or "unknown error"
        try:
            body = response.json()
            if isinstance(body, dict) and body.get("reason"):
                reason = str(body["reason"])
        except ValueError:
            pass  # Body is not JSON; keep the HTTP reason phrase.
        raise APICommunicationError(
            f"Weather service returned HTTP {response.status_code}: {reason}"
        )

    @staticmethod
    def _decode_json(response: requests.Response) -> Dict[str, Any]:
        """Decodes a response body that must be a JSON object.

        Args:
            response: The HTTP response.

        Returns:
            The decoded JSON object.

        Raises:
            InvalidResponseError: If the body is not valid JSON or not an object.
        """
        try:
            data = response.json()
        except ValueError as exc:
            raise InvalidResponseError("Weather service returned invalid JSON.") from exc
        if not isinstance(data, dict):
            raise InvalidResponseError("Weather service returned an unexpected JSON shape.")
        return data

    @staticmethod
    def _parse_location(raw: Any) -> Optional[Location]:
        """Converts one geocoding record into a ``Location``.

        Args:
            raw: One element of the ``results`` list.

        Returns:
            A ``Location``, or ``None`` if the record is malformed.
        """
        if not isinstance(raw, dict):
            return None
        try:
            return Location(
                name=str(raw["name"]),
                latitude=raw["latitude"],
                longitude=raw["longitude"],
                country=raw.get("country"),
                region=raw.get("admin1"),
            )
        except (KeyError, TypeError, ValueError):
            return None

    @classmethod
    def _parse_daily(cls, payload: Dict[str, Any], fields: Sequence[str]) -> List[DailyForecast]:
        """Converts the ``daily`` block of a forecast response into day objects.

        The API returns column-oriented data (one list per field); this turns
        it into one object per day.

        Args:
            payload: Decoded forecast response.
            fields: Field names that were requested.

        Returns:
            One ``DailyForecast`` per date.

        Raises:
            InvalidResponseError: If the ``daily`` block or its dates are missing.
        """
        daily = payload.get("daily")
        if not isinstance(daily, dict) or not isinstance(daily.get("time"), list) or not daily["time"]:
            raise InvalidResponseError("Weather response is missing the expected daily data.")
        return [
            DailyForecast(
                date=str(day),
                values={name: cls._value_at(daily.get(name), index) for name in fields},
            )
            for index, day in enumerate(daily["time"])
        ]

    @staticmethod
    def _value_at(series: Any, index: int) -> Optional[float]:
        """Safely reads one numeric value from an API data column.

        Args:
            series: The column (expected to be a list).
            index: Position to read.

        Returns:
            The value as float, or ``None`` if missing or not numeric.
        """
        if not isinstance(series, list) or index >= len(series):
            return None
        value = series[index]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)


# ==========================================
# 6. Output Formatting (pure, no I/O)
# ==========================================
class ForecastFormatter:
    """Builds the printable lines for a forecast (detailed view or ASCII charts)."""

    BAR_WIDTH = 25
    SEPARATOR_WIDTH = 55
    ATTACHED_UNITS = frozenset({"°C", "°F", "%"})

    def format_forecast(
        self,
        location: Location,
        forecast: Sequence[DailyForecast],
        metrics: Sequence[Metric],
        units: UnitSystem = METRIC_UNITS,
    ) -> List[str]:
        """Formats a full forecast report.

        A single day is shown as a detailed summary; several days as charts.

        Args:
            location: The forecast's location.
            forecast: Forecast days.
            metrics: Metrics the user selected.
            units: Unit system the values are in.

        Returns:
            Lines ready to be printed.

        Raises:
            ValueError: If ``forecast`` or ``metrics`` is empty.
        """
        if not forecast:
            raise ValueError("forecast cannot be empty.")
        if not metrics:
            raise ValueError("metrics cannot be empty.")
        if len(forecast) == 1:
            body = self.format_day_details(forecast[0], metrics, units)
        else:
            body = self.format_charts(forecast, metrics, units)
        return self._format_header(location) + body + ["=" * self.SEPARATOR_WIDTH, ""]

    def _format_header(self, location: Location) -> List[str]:
        """Formats the report title.

        Args:
            location: The forecast's location.

        Returns:
            Header lines.
        """
        separator = "=" * self.SEPARATOR_WIDTH
        return ["", separator, f" Weather Forecast for {location.display_name}", separator]

    def format_day_details(
        self, day: DailyForecast, metrics: Sequence[Metric], units: UnitSystem = METRIC_UNITS
    ) -> List[str]:
        """Formats a detailed one-day summary.

        Args:
            day: The day to show.
            metrics: Metrics to include.
            units: Unit system the values are in.

        Returns:
            Summary lines.
        """
        lines = ["", f" Date: {self.format_date(day.date)}", "-" * 35]
        for metric in metrics:
            unit = units.unit_for(metric.kind)
            values = [self.format_value(day.value(name), unit) for name in metric.fields]
            lines.append(f"  {metric.label + ':':<16}{' to '.join(values)}")
        return lines

    def format_charts(
        self,
        forecast: Sequence[DailyForecast],
        metrics: Sequence[Metric],
        units: UnitSystem = METRIC_UNITS,
    ) -> List[str]:
        """Formats one bar chart per selected API field.

        Args:
            forecast: Forecast days.
            metrics: Metrics to chart.
            units: Unit system the values are in.

        Returns:
            Chart lines.
        """
        labels = [self.format_date(day.date) for day in forecast]
        lines: List[str] = []
        for metric in metrics:
            unit = units.unit_for(metric.kind)
            for name in metric.fields:
                values = [day.value(name) for day in forecast]
                lines += self.format_bar_chart(FIELD_TITLES[name], labels, values, unit)
        return lines

    def format_bar_chart(
        self, title: str, labels: Sequence[str], values: Sequence[Optional[float]], unit: str
    ) -> List[str]:
        """Formats a horizontal ASCII bar chart.

        Bars are scaled by the largest absolute value so negative values
        (e.g. sub-zero temperatures) are drawn too, with a lighter character.

        Args:
            title: Chart title.
            labels: Row labels (dates).
            values: Row values; ``None`` is shown as N/A.
            unit: Value unit.

        Returns:
            Chart lines.

        Raises:
            ValueError: If ``labels`` and ``values`` differ in length.
        """
        if len(labels) != len(values):
            raise ValueError("labels and values must have the same length.")
        lines = ["", f"--- {title} ---"]
        known = [abs(value) for value in values if value is not None]
        if not known:
            return lines + ["  No data available."]
        # Percentages have a natural maximum; otherwise 25% would fill the whole bar.
        max_abs = max(max(known), 100.0) if unit == "%" else max(known)
        for label, value in zip(labels, values):
            bar = self._bar(value, max_abs)
            lines.append(f"{label} | {bar}{' ' if bar else ''}{self.format_value(value, unit)}")
        return lines

    def _bar(self, value: Optional[float], max_abs: float) -> str:
        """Builds the bar for one value.

        Args:
            value: The value (``None`` gives an empty bar).
            max_abs: Largest absolute value in the chart.

        Returns:
            The bar string.
        """
        if value is None:
            return ""
        if value == 0 or max_abs == 0:
            return "▏"
        # Round half up (built-in round() uses banker's rounding: round(12.5) == 12).
        length = max(1, int(abs(value) / max_abs * self.BAR_WIDTH + 0.5))
        return ("█" if value > 0 else "░") * length

    @classmethod
    def format_value(cls, value: Optional[float], unit: str) -> str:
        """Formats a number with its unit, e.g. ``21.5°C`` or ``12 km/h``.

        Args:
            value: The number, or ``None``.
            unit: The unit (degree and percent signs are attached without a space).

        Returns:
            The formatted value, or ``"N/A"`` for ``None``.
        """
        if value is None:
            return "N/A"
        separator = "" if unit in cls.ATTACHED_UNITS else " "
        return f"{value:g}{separator}{unit}"

    @staticmethod
    def format_date(iso_date: str) -> str:
        """Formats an ISO date with its weekday, e.g. ``Wed 2026-10-21``.

        Args:
            iso_date: Date as ``YYYY-MM-DD``.

        Returns:
            The formatted date, or the original text if it cannot be parsed.
        """
        try:
            return date.fromisoformat(iso_date).strftime("%a %Y-%m-%d")
        except (TypeError, ValueError):
            return str(iso_date)


# ==========================================
# 7. Session History
# ==========================================
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


# ==========================================
# 8. CLI Presentation Layer
# ==========================================
class WeatherCLI:
    """Interactive terminal flow: city -> location -> forecast mode -> metrics.

    Global commands work at the prompts: ``help``, ``back`` (or ``0``),
    ``units``, ``recent`` (at the city prompt) and ``quit``. Input and output
    functions are injectable so the whole flow can be tested without a terminal.
    """

    def __init__(
        self,
        client: WeatherAPIClient,
        formatter: Optional[ForecastFormatter] = None,
        input_func: Callable[[str], str] = input,
        output_func: Callable[[str], None] = print,
        units: UnitSystem = METRIC_UNITS,
    ) -> None:
        """Creates the CLI.

        Args:
            client: API client used to fetch data.
            formatter: Output formatter (a default one is created if omitted).
            input_func: Function that reads one line of user input.
            output_func: Function that prints one line.
            units: Initial unit system.
        """
        self._client = client
        self._formatter = formatter or ForecastFormatter()
        self._input = input_func
        self._out = output_func
        self._units = units
        self._recent = RecentLocations()

    def run(self) -> int:
        """Runs the interactive loop until the user quits.

        Returns:
            Process exit code (0).
        """
        self._out("=== Weather CLI (powered by Open-Meteo) ===")
        self._out("Type 'help' for commands, 'back' to go back, 'q' to exit.")
        try:
            while True:
                self._handle_city_search()
        except (QuitRequested, KeyboardInterrupt):
            pass
        self._out("\nExiting application. Goodbye!")
        return 0

    def run_once(self, city: str, days: int, metric_keys: Sequence[str]) -> int:
        """Prints one forecast without prompting (non-interactive mode).

        The best (first) geocoding match is used.

        Args:
            city: City name.
            days: Number of forecast days.
            metric_keys: Keys of ``METRICS`` to show.

        Returns:
            0 on success, 1 if the city is invalid/not found or the API failed.
        """
        try:
            city = InputParser.parse_city_name(city)
            locations = self._client.search_locations(city)
        except (ValueError, WeatherAppError) as err:
            self._out(f"[Error] {err}")
            return 1
        if not locations:
            self._out(f"[Error] City '{city}' was not found.")
            return 1
        metrics = [METRICS[key] for key in metric_keys]
        return 0 if self._show_forecast(locations[0], days, metrics) else 1

    def _ask(self, prompt: str, context_help: str) -> str:
        """Reads one trimmed line and handles the global commands.

        ``help`` and ``units`` are handled here and the prompt is repeated;
        ``back`` is returned as ``"0"``; ``recent`` is returned as typed so the
        city prompt can handle it.

        Args:
            prompt: Prompt text.
            context_help: Help text for this prompt, shown by ``help``.

        Returns:
            The trimmed input.

        Raises:
            QuitRequested: If the user typed a quit word or input ended (EOF).
        """
        while True:
            try:
                raw = self._input(prompt).strip()
            except EOFError:
                raise QuitRequested from None
            command = InputParser.parse_command(raw)
            if command is Command.QUIT:
                raise QuitRequested
            if command is Command.HELP:
                self._print_help(context_help)
            elif command is Command.UNITS:
                self._toggle_units()
            elif command is Command.BACK:
                return InputParser.BACK
            else:
                return raw

    def _print_help(self, context_help: str) -> None:
        """Prints the global commands and the help for the current prompt.

        Args:
            context_help: Help text for the current prompt.
        """
        self._out("\nCommands (available at any prompt):")
        self._out("  help, h, ?    Show this help")
        self._out("  back, b, 0    Go back one step")
        self._out(f"  units         Switch metric/imperial (now: {self._units.summary})")
        self._out("  recent        Pick a recently viewed city (at the city prompt)")
        self._out("  quit, q       Exit the application")
        self._out(f"Here: {context_help}\n")

    def _toggle_units(self) -> None:
        """Switches between metric and imperial units."""
        self._units = IMPERIAL_UNITS if self._units is METRIC_UNITS else METRIC_UNITS
        self._out(f"Units switched to {self._units.summary}.")

    def _ask_valid(self, prompt: str, parser: Callable[[str], Any], context_help: str) -> Any:
        """Asks repeatedly until ``parser`` accepts the input.

        Args:
            prompt: Prompt text.
            parser: Function that parses the text or raises ``ValueError``.
            context_help: Help text for this prompt.

        Returns:
            The parsed value.
        """
        while True:
            try:
                return parser(self._ask(prompt, context_help))
            except ValueError as err:
                self._out(f"[Invalid] {err}")

    def _handle_city_search(self) -> None:
        """Runs one full search: city -> location -> forecasts for that location."""
        location = self._ask_location()
        if location is not None:
            self._recent.add(location)
            self._forecast_loop(location)

    def _ask_location(self) -> Optional[Location]:
        """Asks for a city (or ``recent``) and resolves it to a location.

        Returns:
            The chosen location, or ``None`` to ask again.
        """
        raw = self._ask("\nEnter city name: ", CITY_HELP)
        if raw == InputParser.BACK:
            self._out("[Notice] This is the first step. Type 'q' to exit.")
            return None
        if InputParser.parse_command(raw) is Command.RECENT:
            return self._choose_recent()
        try:
            city = InputParser.parse_city_name(raw)
        except ValueError as err:
            self._out(f"[Invalid] {err}")
            return None
        try:
            return self._resolve_location(city)
        except WeatherAppError as err:
            self._out(f"[Error] {err}")
            return None

    def _choose_recent(self) -> Optional[Location]:
        """Lets the user pick a location viewed earlier in this session.

        Returns:
            The chosen location, or ``None`` if there is none or the user went back.
        """
        items = self._recent.items
        if not items:
            self._out("[Notice] No recent cities yet.")
            return None
        self._out("\nRecently viewed cities:")
        for index, location in enumerate(items, start=1):
            self._out(f"[{index}] {location.display_name}")
        self._out("[0] Back to City Search")
        choice = self._ask_menu_choice(f"Select city (0-{len(items)}): ", len(items), RECENT_HELP)
        return None if choice == 0 else items[choice - 1]

    def _resolve_location(self, city: str) -> Optional[Location]:
        """Searches the city and lets the user pick or confirm a match.

        Args:
            city: Validated city name.

        Returns:
            The chosen location, or ``None`` to go back to the city prompt.

        Raises:
            WeatherAppError: If the geocoding request fails.
        """
        locations = self._client.search_locations(city)
        if not locations:
            self._out(f"[Notice] City '{city}' was not found.")
            return None
        if len(locations) == 1:
            return self._confirm_single_match(city, locations[0])
        return self._choose_from_matches(city, locations)

    def _confirm_single_match(self, city: str, location: Location) -> Optional[Location]:
        """Asks for confirmation when the only match differs from the query.

        Args:
            city: The searched name.
            location: The single match.

        Returns:
            The location if accepted, otherwise ``None``.
        """
        if city.casefold() in location.name.casefold():
            return location
        self._out(f"[Warning] Exact match for '{city}' not found.")
        self._out(f"Closest match: {location.display_name}")
        if self._ask_valid("Continue with this location? (y/n): ", InputParser.parse_yes_no, YES_NO_HELP):
            return location
        self._out("Search cancelled.")
        return None

    def _choose_from_matches(self, city: str, locations: Sequence[Location]) -> Optional[Location]:
        """Lets the user pick one of several matches.

        Args:
            city: The searched name.
            locations: Matching locations.

        Returns:
            The chosen location, or ``None`` to go back.
        """
        self._out(f"\nMultiple matches for '{city}'. Please select one:")
        for index, location in enumerate(locations, start=1):
            self._out(f"[{index}] {location.display_name}")
        self._out("[0] Back to City Search")
        prompt = f"Select location (0-{len(locations)}): "
        choice = self._ask_menu_choice(prompt, len(locations), LOCATION_HELP)
        return None if choice == 0 else locations[choice - 1]

    def _ask_menu_choice(self, prompt: str, upper: int, context_help: str) -> int:
        """Asks for a menu number between 0 and ``upper``.

        Args:
            prompt: Prompt text.
            upper: Largest valid option.
            context_help: Help text for this prompt.

        Returns:
            The chosen number.
        """
        return self._ask_valid(prompt, lambda raw: InputParser.parse_index(raw, upper), context_help)

    def _forecast_loop(self, location: Location) -> None:
        """Shows forecasts for one location until the user goes back.

        Args:
            location: The chosen location.
        """
        while True:
            days = self._select_forecast_days()
            if days is None:
                return
            metrics = self._select_metrics()
            if metrics is not None:
                self._show_forecast(location, days, metrics)

    def _select_forecast_days(self) -> Optional[int]:
        """Asks for the forecast range.

        Returns:
            Number of days, or ``None`` to go back to the city prompt.
        """
        self._out("\nSelect Forecast Mode:")
        for index, (label, _) in enumerate(FORECAST_MODES, start=1):
            self._out(f"[{index}] {label}")
        self._out("[0] Back to City Search")
        prompt = f"Enter choice (0-{len(FORECAST_MODES)}): "
        choice = self._ask_menu_choice(prompt, len(FORECAST_MODES), MODE_HELP)
        return None if choice == 0 else FORECAST_MODES[choice - 1][1]

    def _select_metrics(self) -> Optional[List[Metric]]:
        """Asks which metrics to show.

        Returns:
            Selected metrics, or ``None`` to go back to the forecast mode menu.
        """
        self._out("\nSelect the weather metrics you want to receive (e.g. '1,3' or 'A'):")
        for key, metric in METRICS.items():
            self._out(f"[{key}] {metric.menu_label(self._units)}")
        self._out(f"[{InputParser.ALL}] All metrics")
        self._out(f"[{InputParser.BACK}] Back to Forecast Mode Selection")
        keys = self._ask_valid("Enter choices: ", InputParser.parse_metric_selection, METRICS_HELP)
        return None if keys is None else [METRICS[key] for key in keys]

    def _show_forecast(self, location: Location, days: int, metrics: Sequence[Metric]) -> bool:
        """Fetches and prints a forecast; API errors are reported, not raised.

        Args:
            location: Target location.
            days: Number of forecast days.
            metrics: Metrics to show.

        Returns:
            True if the forecast was shown, False if the API call failed.
        """
        fields = [name for metric in metrics for name in metric.fields]
        self._out(f"\nFetching forecast for {location.display_name}...")
        try:
            forecast = self._client.get_forecast(location, fields, days, self._units)
        except WeatherAppError as err:
            self._out(f"[Error] {err}")
            return False
        for line in self._formatter.format_forecast(location, forecast, metrics, self._units):
            self._out(line)
        return True


# ==========================================
# 9. Entrypoint
# ==========================================
def _positive_float(raw: str) -> float:
    """``argparse`` type that accepts only positive, finite numbers.

    Args:
        raw: Command-line text.

    Returns:
        The parsed number.

    Raises:
        argparse.ArgumentTypeError: If the text is not a positive number.
    """
    try:
        value = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{raw}' is not a number.") from None
    if not value > 0 or value == float("inf"):
        raise argparse.ArgumentTypeError("must be a positive number.")
    return value


def _forecast_days(raw: str) -> int:
    """``argparse`` type for the number of forecast days.

    Args:
        raw: Command-line text.

    Returns:
        The number of days.

    Raises:
        argparse.ArgumentTypeError: If the text is not an integer in range.
    """
    if not raw.strip().isdecimal() or not 1 <= int(raw) <= MAX_FORECAST_DAYS:
        raise argparse.ArgumentTypeError(f"must be an integer between 1 and {MAX_FORECAST_DAYS}.")
    return int(raw)


def _metric_keys(raw: str) -> List[str]:
    """``argparse`` type for a metric selection such as ``1,5`` or ``A``.

    Args:
        raw: Command-line text.

    Returns:
        The selected metric keys.

    Raises:
        argparse.ArgumentTypeError: If the selection is invalid.
    """
    try:
        keys = InputParser.parse_metric_selection(raw)
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err)) from None
    if keys is None:
        raise argparse.ArgumentTypeError(f"choose 1-{len(METRICS)} or 'A'.")
    return keys


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the command-line argument parser.

    Returns:
        The configured parser.
    """
    metric_list = ", ".join(f"{key}={metric.label}" for key, metric in METRICS.items())
    parser = argparse.ArgumentParser(
        description="Weather forecast CLI (Open-Meteo). Runs interactively unless --city is given.",
        epilog="Example: python weather.py --city London --days 7 --metrics 1,5",
    )
    parser.add_argument("--city", help="Print one forecast for this city and exit (best match is used).")
    parser.add_argument("--days", type=_forecast_days,
                        help=f"Forecast days, 1-{MAX_FORECAST_DAYS} (default: 1). Needs --city.")
    parser.add_argument("--metrics", type=_metric_keys,
                        help=f"Comma-separated metrics or A for all (default: A). {metric_list}. Needs --city.")
    parser.add_argument("--units", choices=sorted(UNIT_SYSTEMS), default=METRIC_UNITS.name,
                        help="Unit system (default: %(default)s).")
    parser.add_argument("--timeout", type=_positive_float, default=DEFAULT_TIMEOUT,
                        help="HTTP timeout in seconds (default: %(default)s).")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parses arguments and runs the CLI (interactive or one-shot).

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.city is None and (args.days is not None or args.metrics is not None):
        parser.error("--days and --metrics require --city.")

    with WeatherAPIClient(timeout=args.timeout) as client:
        cli = WeatherCLI(client, units=UNIT_SYSTEMS[args.units])
        if args.city is not None:
            return cli.run_once(args.city, args.days or 1, args.metrics or list(METRICS))
        return cli.run()


if __name__ == "__main__":
    sys.exit(main())
