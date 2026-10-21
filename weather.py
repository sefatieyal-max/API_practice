from dataclasses import dataclass
import sys
from typing import Dict, List, Optional, Any
import requests


# ==========================================
# 1. Custom Exceptions
# ==========================================
class WeatherAppError(Exception):
    """Base exception for all application errors."""
    pass


class APICommunicationError(WeatherAppError):
    """Raised when a network or HTTP error occurs."""
    pass


# ==========================================
# 2. Data Models
# ==========================================
@dataclass(frozen=True)
class Location:
    """Immutable data model representing geographical coordinates."""
    name: str
    latitude: float
    longitude: float
    country: Optional[str] = None


@dataclass(frozen=True)
class DailyForecast:
    """Immutable data model representing a single day's weather forecast."""
    date: str
    values: Dict[str, Any]


# ==========================================
# 3. Available Metrics Mapping
# ==========================================
METRIC_OPTIONS: Dict[str, Dict[str, str]] = {
    "1": {"label": "Temperature Max/Min (°C)", "daily_param": "temperature_2m_max,temperature_2m_min"},
    "2": {"label": "Apparent Temperature / Feels Like (°C)", "daily_param": "apparent_temperature_max,apparent_temperature_min"},
    "3": {"label": "Max Wind Speed (km/h)", "daily_param": "wind_speed_10m_max"},
    "4": {"label": "Total Precipitation (mm)", "daily_param": "precipitation_sum"},
    "5": {"label": "Max Rain Probability (%)", "daily_param": "precipitation_probability_max"},
}


# ==========================================
# 4. API Client
# ==========================================
class WeatherAPIClient:
    """Client responsible for raw communication with Open-Meteo APIs."""

    GEOCODE_URL: str = "https://geocoding-api.open-meteo.com/v1/search"
    FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"
    DEFAULT_TIMEOUT: int = 10  # seconds

    def __init__(self, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.session = requests.Session()
        self.timeout = timeout

    def search_locations(self, city_name: str, count: int = 5) -> List[Location]:
        """
        Searches for coordinates matching a given city name.

        :param city_name: Name of the city to query.
        :param count: Maximum number of results to return.
        :return: A list of Location instances.
        :raises APICommunicationError: On network or protocol errors.
        """
        params = {"name": city_name, "count": count, "format": "json"}

        try:
            response = self.session.get(self.GEOCODE_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            results = data.get("results")
            if not results:
                return []

            return [
                Location(
                    name=res.get("name", city_name),
                    latitude=res["latitude"],
                    longitude=res["longitude"],
                    country=res.get("country"),
                )
                for res in results
            ]

        except requests.exceptions.RequestException as exc:
            raise APICommunicationError(f"Geocoding service error: {exc}") from exc

    def get_forecast(
            self, location: Location, daily_params: List[str], days: int
    ) -> List[DailyForecast]:
        """
        Fetches daily forecast data for specific requested parameters.

        :param location: Target Location instance.
        :param daily_params: List of daily API parameter names.
        :param days: Number of forecast days (e.g., 1 or 7).
        :return: List of DailyForecast instances.
        :raises APICommunicationError: On network or parsing errors.
        """
        daily_query = ",".join(daily_params)
        params = {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "daily": daily_query,
            "forecast_days": days,
            "timezone": "auto",
        }

        try:
            response = self.session.get(self.FORECAST_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            daily_data = data.get("daily")
            if not daily_data or "time" not in daily_data:
                raise APICommunicationError("Weather response was missing expected daily payload.")

            forecast_list: List[DailyForecast] = []
            dates = daily_data["time"]

            for idx, date_str in enumerate(dates):
                day_values: Dict[str, Any] = {}
                for key, val_list in daily_data.items():
                    if key != "time" and isinstance(val_list, list) and idx < len(val_list):
                        day_values[key] = val_list[idx]

                forecast_list.append(DailyForecast(date=date_str, values=day_values))

            return forecast_list

        except requests.exceptions.RequestException as exc:
            raise APICommunicationError(f"Weather service error: {exc}") from exc


# ==========================================
# 5. CLI Presentation Layer
# ==========================================
class WeatherCLI:
    """Manages CLI interaction, menu prompts, and output formatting."""

    def __init__(self, client: WeatherAPIClient) -> None:
        self.client = client

    @staticmethod
    def _choose_location(query: str, locations: List[Location]) -> Optional[Location]:
        """Prompts user to confirm or choose a location from search results."""
        if not locations:
            print(f"[Notice] City '{query}' was not found.\n")
            return None

        if len(locations) == 1:
            loc = locations[0]
            if query.lower() not in loc.name.lower():
                print(f"[Warning] Exact match for '{query}' not found.")
                print(f"Closest match: {loc.name}, {loc.country or 'N/A'}")
                ans = input("Continue with this location? (y/n): ").strip().lower()
                if ans != 'y':
                    print("Search cancelled.\n")
                    return None
            return loc

        print(f"\nMultiple matches for '{query}'. Please select one:")
        for idx, loc in enumerate(locations, start=1):
            country = f", {loc.country}" if loc.country else ""
            print(f"[{idx}] {loc.name}{country}")
        print("[0] Cancel search")

        while True:
            choice = input(f"Select location (0-{len(locations)}): ").strip()
            if choice == '0':
                print("Search cancelled.\n")
                return None
            if choice.isdigit() and 1 <= int(choice) <= len(locations):
                return locations[int(choice) - 1]
            print("Invalid selection. Try again.")

    @staticmethod
    def _select_forecast_type() -> Optional[int]:
        """Prompts user to select between today's forecast or a 7-day forecast."""
        print("\nSelect Forecast Mode:")
        print("[1] Today's Forecast (1 Day)")
        print("[2] Weekly Forecast (7 Days)")
        print("[0] Back / Cancel")

        while True:
            choice = input("Enter choice (0-2): ").strip()
            if choice == "1":
                return 1
            if choice == "2":
                return 7
            if choice == "0":
                print("Action cancelled.\n")
                return None
            print("Invalid choice. Please enter 0, 1, or 2.")

    @staticmethod
    def _select_metrics() -> Optional[List[str]]:
        """Prompts user to select which weather metrics they wish to view."""
        print("\nSelect the weather metrics you want to receive (e.g., '1,3' or 'A'):")
        for key, info in METRIC_OPTIONS.items():
            print(f"[{key}] {info['label']}")
        print("[A] All metrics")
        print("[0] Cancel")

        while True:
            raw_input = input("Enter choices: ").strip().upper()
            if raw_input == "0":
                print("Action cancelled.\n")
                return None

            if raw_input == "A":
                selected_params = []
                for info in METRIC_OPTIONS.values():
                    selected_params.extend(info["daily_param"].split(","))
                return selected_params

            chosen_keys = [k.strip() for k in raw_input.split(",") if k.strip() in METRIC_OPTIONS]
            if chosen_keys:
                selected_params = []
                for key in chosen_keys:
                    selected_params.extend(METRIC_OPTIONS[key]["daily_param"].split(","))
                return list(dict.fromkeys(selected_params))

            print("Invalid selection. Please choose valid option numbers or 'A'.")

    @staticmethod
    def display_forecast(location: Location, forecast_data: List[DailyForecast]) -> None:
        """Formats and displays the forecast output."""
        country_suffix = f", {location.country}" if location.country else ""
        print("\n" + "=" * 50)
        print(f" Weather Forecast for {location.name}{country_suffix}")
        print("=" * 50)

        for day in forecast_data:
            print(f"\n Date: {day.date}")
            print("-" * 30)

            vals = day.values
            if "temperature_2m_max" in vals and "temperature_2m_min" in vals:
                print(f"  Temperature:    {vals['temperature_2m_min']}°C to {vals['temperature_2m_max']}°C")
            if "apparent_temperature_max" in vals and "apparent_temperature_min" in vals:
                print(f"  Feels Like:     {vals['apparent_temperature_min']}°C to {vals['apparent_temperature_max']}°C")
            if "wind_speed_10m_max" in vals:
                print(f"  Max Wind Speed: {vals['wind_speed_10m_max']} km/h")
            if "precipitation_sum" in vals:
                print(f"  Precipitation:  {vals['precipitation_sum']} mm")
            if "precipitation_probability_max" in vals:
                print(f"  Rain Chance:    {vals['precipitation_probability_max']}%")

        print("=" * 50 + "\n")

    def run(self) -> None:
        """Main interaction loop."""
        print("=== Advanced Weather CLI ===")
        print("Type 'quit' or 'q' to exit.\n")

        while True:
            city_input = input("Enter city name: ").strip()

            if city_input.lower() in ["quit", "q"]:
                print("Exiting application. Goodbye!")
                sys.exit(0)

            if not city_input:
                print("Input cannot be empty. Try again.\n")
                continue

            try:
                # Step 1: Search location
                locations = self.client.search_locations(city_input, count=5)
                selected_location = self._choose_location(city_input, locations)
                if not selected_location:
                    continue

                # Step 2: Select forecast timeframe (Today vs Weekly)
                forecast_days = self._select_forecast_type()
                if not forecast_days:
                    continue

                # Step 3: Select metrics
                daily_params = self._select_metrics()
                if not daily_params:
                    continue

                # Step 4: Fetch and display forecast
                forecast = self.client.get_forecast(selected_location, daily_params, forecast_days)
                self.display_forecast(selected_location, forecast)

            except APICommunicationError as err:
                print(f"[Error] Network/API failure: {err}\n")


# ==========================================
# 6. Entrypoint
# ==========================================
if __name__ == "__main__":
    api_client = WeatherAPIClient()
    app = WeatherCLI(client=api_client)
    app.run()