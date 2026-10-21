"""Interactive terminal flow: all user input and output happens here."""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence

from .api_client import WeatherAPIClient
from .commands import Command
from .config import FORECAST_MODES, METRICS
from .exceptions import QuitRequested, WeatherAppError
from .formatter import ForecastFormatter
from .history import RecentLocations
from .input_parser import InputParser
from .models import IMPERIAL_UNITS, METRIC_UNITS, Location, Metric, UnitSystem


# Context help shown under the command list, per prompt.
CITY_HELP = "Type a city name (e.g. 'London'), or 'recent' to pick a recently viewed city."
LOCATION_HELP = "Several places share this name. Pick the number of the one you meant."
MODE_HELP = "1 shows today's detailed forecast, 2 shows 7-day charts."
METRICS_HELP = "Type option numbers separated by commas (e.g. '1,3') or 'A' for all."
YES_NO_HELP = "Answer 'y' to use the suggested location or 'n' to search again."
RECENT_HELP = "Pick the number of a city you viewed earlier in this session."


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
