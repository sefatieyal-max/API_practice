"""HTTP client for the Open-Meteo geocoding and forecast REST APIs."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import DEFAULT_TIMEOUT, FIELD_TITLES, MAX_FORECAST_DAYS
from .exceptions import APICommunicationError, InvalidResponseError
from .models import METRIC_UNITS, DailyForecast, Location, UnitSystem


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
