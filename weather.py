"""Weather CLI: an interactive command-line client for the Open-Meteo REST API.

The user searches for a city, picks a forecast range (today or 7 days) and the
metrics to show. Today's forecast is printed as a detailed summary and the
weekly forecast as ASCII bar charts.

Layers (each one has a single responsibility):
    * Exceptions and data models.
    * ``InputParser``: pure validation and parsing of raw user text.
    * ``WeatherAPIClient``: HTTP communication and response parsing.
    * ``ForecastFormatter``: turns data into printable lines (no I/O).
    * ``WeatherCLI``: the interactive loop (all terminal I/O lives here).

Example:
    $ python weather.py
    $ python weather.py --timeout 5
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date
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
class Metric:
    """A user-selectable weather metric backed by one or more API fields.

    Attributes:
        label: Short display name.
        unit: Display unit.
        fields: Open-Meteo daily field names (a min/max pair is shown as a range).
    """

    label: str
    unit: str
    fields: Tuple[str, ...]

    @property
    def menu_label(self) -> str:
        """Returns the label shown in the metrics menu."""
        return f"{self.label} ({self.unit})"


# ==========================================
# 3. Menus and Metric Mapping
# ==========================================
METRICS: Dict[str, Metric] = {
    "1": Metric("Temperature", "°C", ("temperature_2m_min", "temperature_2m_max")),
    "2": Metric("Feels Like", "°C", ("apparent_temperature_min", "apparent_temperature_max")),
    "3": Metric("Max Wind Speed", "km/h", ("wind_speed_10m_max",)),
    "4": Metric("Precipitation", "mm", ("precipitation_sum",)),
    "5": Metric("Rain Chance", "%", ("precipitation_probability_max",)),
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

# (menu label, number of forecast days)
FORECAST_MODES: Tuple[Tuple[str, int], ...] = (
    ("Today's Forecast (1 Day)", 1),
    ("Weekly Forecast (7 Days)", 7),
)


# ==========================================
# 4. Input Parsing (pure, no I/O)
# ==========================================
class InputParser:
    """Validates and parses raw text typed by the user.

    All methods are pure: they never print or read, they only return a parsed
    value or raise ``ValueError`` with a message that can be shown to the user.
    """

    QUIT_WORDS = frozenset({"q", "quit", "exit"})
    BACK = "0"
    ALL = "A"
    MIN_CITY_LENGTH = 2
    MAX_CITY_LENGTH = 100

    @classmethod
    def is_quit(cls, raw: str) -> bool:
        """Checks whether the text is a request to exit.

        Args:
            raw: Raw user input.

        Returns:
            True if the input is one of the quit words (case-insensitive).
        """
        return raw.strip().lower() in cls.QUIT_WORDS

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
    DEFAULT_TIMEOUT: float = 10.0
    MAX_RESULTS: int = 100
    MAX_FORECAST_DAYS: int = 16
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
        self, location: Location, fields: Sequence[str], days: int
    ) -> List[DailyForecast]:
        """Fetches the daily forecast for the requested fields.

        Args:
            location: Target location.
            fields: Open-Meteo daily field names (keys of ``FIELD_TITLES``).
            days: Number of forecast days (1-16).

        Returns:
            One ``DailyForecast`` per day, in chronological order.

        Raises:
            TypeError: If ``location`` is not a ``Location``.
            ValueError: If ``fields`` is empty/unknown or ``days`` is out of range.
            APICommunicationError: On network or HTTP errors.
            InvalidResponseError: If the response is not in the expected shape.
        """
        self._validate_forecast_request(location, fields, days)
        unique_fields = list(dict.fromkeys(fields))
        params = {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "daily": ",".join(unique_fields),
            "forecast_days": days,
            "timezone": "auto",
        }
        payload = self._get_json(self.FORECAST_URL, params)
        return self._parse_daily(payload, unique_fields)

    def _validate_forecast_request(
        self, location: Location, fields: Sequence[str], days: int
    ) -> None:
        """Validates the arguments of ``get_forecast``.

        Args:
            location: Target location.
            fields: Requested field names.
            days: Number of forecast days.

        Raises:
            TypeError: If ``location`` is not a ``Location``.
            ValueError: If ``fields`` is empty/unknown or ``days`` is out of range.
        """
        if not isinstance(location, Location):
            raise TypeError("location must be a Location instance.")
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

    def format_forecast(
        self, location: Location, forecast: Sequence[DailyForecast], metrics: Sequence[Metric]
    ) -> List[str]:
        """Formats a full forecast report.

        A single day is shown as a detailed summary; several days as charts.

        Args:
            location: The forecast's location.
            forecast: Forecast days.
            metrics: Metrics the user selected.

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
            body = self.format_day_details(forecast[0], metrics)
        else:
            body = self.format_charts(forecast, metrics)
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

    def format_day_details(self, day: DailyForecast, metrics: Sequence[Metric]) -> List[str]:
        """Formats a detailed one-day summary.

        Args:
            day: The day to show.
            metrics: Metrics to include.

        Returns:
            Summary lines.
        """
        lines = ["", f" Date: {self.format_date(day.date)}", "-" * 35]
        for metric in metrics:
            values = [self.format_value(day.value(name), metric.unit) for name in metric.fields]
            lines.append(f"  {metric.label + ':':<16}{' to '.join(values)}")
        return lines

    def format_charts(self, forecast: Sequence[DailyForecast], metrics: Sequence[Metric]) -> List[str]:
        """Formats one bar chart per selected API field.

        Args:
            forecast: Forecast days.
            metrics: Metrics to chart.

        Returns:
            Chart lines.
        """
        labels = [self.format_date(day.date) for day in forecast]
        lines: List[str] = []
        for metric in metrics:
            for name in metric.fields:
                values = [day.value(name) for day in forecast]
                lines += self.format_bar_chart(FIELD_TITLES[name], labels, values, metric.unit)
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

    @staticmethod
    def format_value(value: Optional[float], unit: str) -> str:
        """Formats a number with its unit, e.g. ``21.5°C`` or ``12 km/h``.

        Args:
            value: The number, or ``None``.
            unit: The unit (degree and percent signs are attached without a space).

        Returns:
            The formatted value, or ``"N/A"`` for ``None``.
        """
        if value is None:
            return "N/A"
        separator = "" if unit in ("°C", "%") else " "
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
# 7. CLI Presentation Layer
# ==========================================
class WeatherCLI:
    """Interactive terminal flow: city -> location -> forecast mode -> metrics.

    ``0`` goes back one step and ``q`` exits from any prompt. Input and output
    functions are injectable so the whole flow can be tested without a terminal.
    """

    def __init__(
        self,
        client: WeatherAPIClient,
        formatter: Optional[ForecastFormatter] = None,
        input_func: Callable[[str], str] = input,
        output_func: Callable[[str], None] = print,
    ) -> None:
        """Creates the CLI.

        Args:
            client: API client used to fetch data.
            formatter: Output formatter (a default one is created if omitted).
            input_func: Function that reads one line of user input.
            output_func: Function that prints one line.
        """
        self._client = client
        self._formatter = formatter or ForecastFormatter()
        self._input = input_func
        self._out = output_func

    def run(self) -> int:
        """Runs the interactive loop until the user quits.

        Returns:
            Process exit code (0).
        """
        self._out("=== Weather CLI (powered by Open-Meteo) ===")
        self._out("Type 'q' at any prompt to exit, '0' to go back.\n")
        try:
            while True:
                self._handle_city_search()
        except (QuitRequested, KeyboardInterrupt):
            pass
        self._out("\nExiting application. Goodbye!")
        return 0

    def _ask(self, prompt: str) -> str:
        """Reads one trimmed line, turning quit words and EOF into ``QuitRequested``.

        Args:
            prompt: Prompt text.

        Returns:
            The trimmed input.

        Raises:
            QuitRequested: If the user typed a quit word or input ended (EOF).
        """
        try:
            raw = self._input(prompt)
        except EOFError:
            raise QuitRequested from None
        if InputParser.is_quit(raw):
            raise QuitRequested
        return raw.strip()

    def _ask_valid(self, prompt: str, parser: Callable[[str], Any]) -> Any:
        """Asks repeatedly until ``parser`` accepts the input.

        Args:
            prompt: Prompt text.
            parser: Function that parses the text or raises ``ValueError``.

        Returns:
            The parsed value.
        """
        while True:
            try:
                return parser(self._ask(prompt))
            except ValueError as err:
                self._out(f"[Invalid] {err}")

    def _handle_city_search(self) -> None:
        """Runs one full search: city -> location -> forecasts for that location."""
        city = self._ask_valid("\nEnter city name: ", InputParser.parse_city_name)
        try:
            location = self._resolve_location(city)
        except WeatherAppError as err:
            self._out(f"[Error] {err}")
            return
        if location is not None:
            self._forecast_loop(location)

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
        if self._ask_valid("Continue with this location? (y/n): ", InputParser.parse_yes_no):
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
        choice = self._ask_menu_choice(f"Select location (0-{len(locations)}): ", len(locations))
        return None if choice == 0 else locations[choice - 1]

    def _ask_menu_choice(self, prompt: str, upper: int) -> int:
        """Asks for a menu number between 0 and ``upper``.

        Args:
            prompt: Prompt text.
            upper: Largest valid option.

        Returns:
            The chosen number.
        """
        return self._ask_valid(prompt, lambda raw: InputParser.parse_index(raw, upper))

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
        choice = self._ask_menu_choice(f"Enter choice (0-{len(FORECAST_MODES)}): ", len(FORECAST_MODES))
        return None if choice == 0 else FORECAST_MODES[choice - 1][1]

    def _select_metrics(self) -> Optional[List[Metric]]:
        """Asks which metrics to show.

        Returns:
            Selected metrics, or ``None`` to go back to the forecast mode menu.
        """
        self._out("\nSelect the weather metrics you want to receive (e.g. '1,3' or 'A'):")
        for key, metric in METRICS.items():
            self._out(f"[{key}] {metric.menu_label}")
        self._out(f"[{InputParser.ALL}] All metrics")
        self._out(f"[{InputParser.BACK}] Back to Forecast Mode Selection")
        keys = self._ask_valid("Enter choices: ", InputParser.parse_metric_selection)
        return None if keys is None else [METRICS[key] for key in keys]

    def _show_forecast(self, location: Location, days: int, metrics: Sequence[Metric]) -> None:
        """Fetches and prints a forecast; API errors are reported, not raised.

        Args:
            location: Target location.
            days: Number of forecast days.
            metrics: Metrics to show.
        """
        fields = [name for metric in metrics for name in metric.fields]
        self._out(f"\nFetching forecast for {location.display_name}...")
        try:
            forecast = self._client.get_forecast(location, fields, days)
        except WeatherAppError as err:
            self._out(f"[Error] {err}")
            return
        for line in self._formatter.format_forecast(location, forecast, metrics):
            self._out(line)


# ==========================================
# 8. Entrypoint
# ==========================================
def _positive_float(raw: str) -> float:
    """``argparse`` type that accepts only positive numbers.

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


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the command-line argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(description="Interactive weather forecast CLI (Open-Meteo).")
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=WeatherAPIClient.DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds (default: %(default)s).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parses arguments and runs the interactive CLI.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    args = build_arg_parser().parse_args(argv)
    with WeatherAPIClient(timeout=args.timeout) as client:
        return WeatherCLI(client).run()


if __name__ == "__main__":
    sys.exit(main())
