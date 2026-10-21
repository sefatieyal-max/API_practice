"""Unit tests for weather.py. Run with: python -m unittest -v"""
import io
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any, List, Optional
from unittest import mock

import requests

from weather import (
    IMPERIAL_UNITS,
    METRIC_UNITS,
    METRICS,
    APICommunicationError,
    Command,
    DailyForecast,
    ForecastFormatter,
    InputParser,
    InvalidResponseError,
    Location,
    RecentLocations,
    WeatherAPIClient,
    WeatherCLI,
    build_arg_parser,
    main,
)

LONDON = Location("London", 51.5, -0.12, "United Kingdom", "England")


def make_response(status: int = 200, payload: Any = None, invalid_json: bool = False,
                  reason: str = "OK") -> mock.Mock:
    """Builds a fake ``requests.Response``."""
    response = mock.Mock()
    response.status_code = status
    response.ok = status < 400
    response.reason = reason
    if invalid_json:
        response.json.side_effect = ValueError("not json")
    else:
        response.json.return_value = payload
    return response


def make_client(*responses: Any) -> WeatherAPIClient:
    """Builds a client whose session returns (or raises) the given values in order."""
    session = mock.Mock()
    session.get.side_effect = list(responses)
    return WeatherAPIClient(session=session)


class TestLocation(unittest.TestCase):
    def test_valid_location(self) -> None:
        self.assertEqual(LONDON.latitude, 51.5)

    def test_display_name_includes_region_and_country(self) -> None:
        self.assertEqual(LONDON.display_name, "London, England, United Kingdom")

    def test_display_name_skips_missing_and_duplicate_parts(self) -> None:
        self.assertEqual(Location("Singapore", 1.3, 103.8, "Singapore").display_name, "Singapore")

    def test_boundary_coordinates_are_valid(self) -> None:
        Location("Pole", 90, 180)
        Location("Pole", -90, -180)

    def test_out_of_range_latitude_raises(self) -> None:
        with self.assertRaises(ValueError):
            Location("X", 90.1, 0)

    def test_out_of_range_longitude_raises(self) -> None:
        with self.assertRaises(ValueError):
            Location("X", 0, -180.5)

    def test_non_numeric_coordinate_raises(self) -> None:
        with self.assertRaises(TypeError):
            Location("X", "51", 0)

    def test_bool_coordinate_raises(self) -> None:
        with self.assertRaises(TypeError):
            Location("X", True, 0)

    def test_empty_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            Location("  ", 0, 0)


class TestInputParser(unittest.TestCase):
    def test_command_aliases(self) -> None:
        expected = {"q": Command.QUIT, " QUIT ": Command.QUIT, "exit": Command.QUIT,
                    "help": Command.HELP, "H": Command.HELP, "?": Command.HELP,
                    "back": Command.BACK, "b": Command.BACK,
                    "units": Command.UNITS, "Recent": Command.RECENT}
        for raw, command in expected.items():
            with self.subTest(raw=raw):
                self.assertIs(InputParser.parse_command(raw), command)

    def test_non_commands(self) -> None:
        for raw in ("Quito", "0", "", "London", "helper"):
            with self.subTest(raw=raw):
                self.assertIsNone(InputParser.parse_command(raw))

    def test_city_name_is_trimmed_and_collapsed(self) -> None:
        self.assertEqual(InputParser.parse_city_name("  New   York "), "New York")

    def test_city_name_unicode(self) -> None:
        self.assertEqual(InputParser.parse_city_name("תל אביב"), "תל אביב")

    def test_city_name_invalid_values(self) -> None:
        for raw in ("", "   ", "a", "x" * 101, "12345", "!!"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                InputParser.parse_city_name(raw)

    def test_city_name_boundary_lengths(self) -> None:
        self.assertEqual(InputParser.parse_city_name("Ab"), "Ab")
        self.assertEqual(len(InputParser.parse_city_name("x" * 100)), 100)

    def test_city_name_non_string_raises(self) -> None:
        with self.assertRaises(TypeError):
            InputParser.parse_city_name(None)  # type: ignore[arg-type]

    def test_parse_index_valid(self) -> None:
        self.assertEqual(InputParser.parse_index(" 2 ", 3), 2)
        self.assertEqual(InputParser.parse_index("0", 3), 0)

    def test_parse_index_invalid(self) -> None:
        for raw in ("4", "-1", "", "abc", "1.5", "²"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                InputParser.parse_index(raw, 3)

    def test_parse_index_empty_menu_raises(self) -> None:
        with self.assertRaises(ValueError):
            InputParser.parse_index("0", 0)

    def test_metric_selection_list(self) -> None:
        self.assertEqual(InputParser.parse_metric_selection("1, 3"), ["1", "3"])

    def test_metric_selection_removes_duplicates_and_empty_tokens(self) -> None:
        self.assertEqual(InputParser.parse_metric_selection("3,,1,3"), ["3", "1"])

    def test_metric_selection_all_is_case_insensitive(self) -> None:
        self.assertEqual(InputParser.parse_metric_selection(" a "), list(METRICS))

    def test_metric_selection_back(self) -> None:
        self.assertIsNone(InputParser.parse_metric_selection("0"))

    def test_metric_selection_invalid(self) -> None:
        for raw in ("", ",", "6", "1,6", "0,1", "1-3", "A,1"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                InputParser.parse_metric_selection(raw)

    def test_yes_no(self) -> None:
        self.assertTrue(InputParser.parse_yes_no(" Y "))
        self.assertTrue(InputParser.parse_yes_no("yes"))
        self.assertFalse(InputParser.parse_yes_no("n"))
        with self.assertRaises(ValueError):
            InputParser.parse_yes_no("maybe")


class TestClientConstruction(unittest.TestCase):
    def test_invalid_timeout(self) -> None:
        with self.assertRaises(ValueError):
            WeatherAPIClient(timeout=0, session=mock.Mock())
        with self.assertRaises(TypeError):
            WeatherAPIClient(timeout="10", session=mock.Mock())  # type: ignore[arg-type]

    def test_negative_retries(self) -> None:
        with self.assertRaises(ValueError):
            WeatherAPIClient(retries=-1)

    def test_context_manager_closes_session(self) -> None:
        session = mock.Mock()
        with WeatherAPIClient(session=session):
            pass
        session.close.assert_called_once()


class TestSearchLocations(unittest.TestCase):
    def test_parses_results(self) -> None:
        payload = {"results": [
            {"name": "London", "latitude": 51.5, "longitude": -0.12,
             "country": "United Kingdom", "admin1": "England"},
            {"name": "London", "latitude": 42.98, "longitude": -81.23, "country": "Canada"},
        ]}
        client = make_client(make_response(payload=payload))
        locations = client.search_locations("London")
        self.assertEqual(len(locations), 2)
        self.assertEqual(locations[0], LONDON)
        self.assertIsNone(locations[1].region)

    def test_sends_expected_params(self) -> None:
        client = make_client(make_response(payload={}))
        client.search_locations("  Paris ", count=3)
        _, kwargs = client._session.get.call_args
        self.assertEqual(kwargs["params"], {"name": "Paris", "count": 3, "format": "json"})
        self.assertEqual(kwargs["timeout"], client.timeout)

    def test_no_results_returns_empty_list(self) -> None:
        client = make_client(make_response(payload={"generationtime_ms": 0.1}))
        self.assertEqual(client.search_locations("Nowhereville"), [])

    def test_malformed_records_are_skipped(self) -> None:
        payload = {"results": [
            "garbage",
            {"name": "NoCoords"},
            {"name": "Bad", "latitude": 999, "longitude": 0},
            {"name": "Good", "latitude": 1, "longitude": 2},
        ]}
        client = make_client(make_response(payload=payload))
        self.assertEqual([loc.name for loc in client.search_locations("x y")], ["Good"])

    def test_results_not_a_list_raises(self) -> None:
        client = make_client(make_response(payload={"results": "oops"}))
        with self.assertRaises(InvalidResponseError):
            client.search_locations("London")

    def test_invalid_arguments(self) -> None:
        client = make_client()
        for city, count in (("", 5), ("   ", 5), ("London", 0), ("London", 101), ("London", True)):
            with self.subTest(city=city, count=count), self.assertRaises(ValueError):
                client.search_locations(city, count)

    def test_timeout_raises_communication_error(self) -> None:
        client = make_client(requests.exceptions.Timeout())
        with self.assertRaisesRegex(APICommunicationError, "did not respond"):
            client.search_locations("London")

    def test_connection_error_raises_communication_error(self) -> None:
        client = make_client(requests.exceptions.ConnectionError())
        with self.assertRaisesRegex(APICommunicationError, "Could not connect"):
            client.search_locations("London")

    def test_other_request_error_raises_communication_error(self) -> None:
        client = make_client(requests.exceptions.TooManyRedirects("loop"))
        with self.assertRaises(APICommunicationError):
            client.search_locations("London")

    def test_http_error_includes_api_reason(self) -> None:
        response = make_response(400, {"error": True, "reason": "Parameter count invalid"},
                                 reason="Bad Request")
        client = make_client(response)
        with self.assertRaisesRegex(APICommunicationError, "HTTP 400: Parameter count invalid"):
            client.search_locations("London")

    def test_http_error_without_json_body(self) -> None:
        client = make_client(make_response(503, invalid_json=True, reason="Service Unavailable"))
        with self.assertRaisesRegex(APICommunicationError, "HTTP 503: Service Unavailable"):
            client.search_locations("London")

    def test_invalid_json_raises(self) -> None:
        client = make_client(make_response(invalid_json=True))
        with self.assertRaises(InvalidResponseError):
            client.search_locations("London")

    def test_non_object_json_raises(self) -> None:
        client = make_client(make_response(payload=[1, 2]))
        with self.assertRaises(InvalidResponseError):
            client.search_locations("London")


class TestGetForecast(unittest.TestCase):
    FIELDS = ["temperature_2m_min", "temperature_2m_max"]

    def test_parses_daily_columns_into_days(self) -> None:
        payload = {"daily": {
            "time": ["2026-10-21", "2026-10-22"],
            "temperature_2m_min": [10.1, 11.0],
            "temperature_2m_max": [20.5, 21],
        }}
        client = make_client(make_response(payload=payload))
        days = client.get_forecast(LONDON, self.FIELDS, 7)
        self.assertEqual([d.date for d in days], ["2026-10-21", "2026-10-22"])
        self.assertEqual(days[1].value("temperature_2m_max"), 21.0)

    def test_sends_expected_params(self) -> None:
        payload = {"daily": {"time": ["2026-10-21"]}}
        client = make_client(make_response(payload=payload))
        client.get_forecast(LONDON, self.FIELDS + ["temperature_2m_min"], 1)
        _, kwargs = client._session.get.call_args
        self.assertEqual(kwargs["params"]["daily"], "temperature_2m_min,temperature_2m_max")
        self.assertEqual(kwargs["params"]["forecast_days"], 1)
        self.assertEqual(kwargs["params"]["timezone"], "auto")

    def test_missing_or_bad_values_become_none(self) -> None:
        payload = {"daily": {
            "time": ["2026-10-21", "2026-10-22"],
            "temperature_2m_min": [None, "cold"],
            "temperature_2m_max": [20],
        }}
        client = make_client(make_response(payload=payload))
        days = client.get_forecast(LONDON, self.FIELDS, 7)
        self.assertIsNone(days[0].value("temperature_2m_min"))
        self.assertIsNone(days[1].value("temperature_2m_min"))
        self.assertIsNone(days[1].value("temperature_2m_max"))

    def test_missing_daily_block_raises(self) -> None:
        for payload in ({}, {"daily": None}, {"daily": {"time": []}}, {"daily": {"time": "x"}}):
            client = make_client(make_response(payload=payload))
            with self.subTest(payload=payload), self.assertRaises(InvalidResponseError):
                client.get_forecast(LONDON, self.FIELDS, 1)

    def test_invalid_arguments(self) -> None:
        client = make_client()
        for fields, days in (([], 1), ("temperature_2m_max", 1), (["unknown"], 1),
                             (self.FIELDS, 0), (self.FIELDS, 17), (self.FIELDS, True)):
            with self.subTest(fields=fields, days=days), self.assertRaises(ValueError):
                client.get_forecast(LONDON, fields, days)
        client._session.get.assert_not_called()

    def test_invalid_location_type(self) -> None:
        with self.assertRaises(TypeError):
            make_client().get_forecast("London", self.FIELDS, 1)  # type: ignore[arg-type]


class TestForecastFormatter(unittest.TestCase):
    def setUp(self) -> None:
        self.formatter = ForecastFormatter()

    def test_format_value(self) -> None:
        self.assertEqual(ForecastFormatter.format_value(21.5, "°C"), "21.5°C")
        self.assertEqual(ForecastFormatter.format_value(12.0, "km/h"), "12 km/h")
        self.assertEqual(ForecastFormatter.format_value(80.0, "%"), "80%")
        self.assertEqual(ForecastFormatter.format_value(None, "mm"), "N/A")

    def test_format_date(self) -> None:
        self.assertEqual(ForecastFormatter.format_date("2026-10-21"), "Wed 2026-10-21")
        self.assertEqual(ForecastFormatter.format_date("not-a-date"), "not-a-date")

    def test_bar_chart_handles_negative_zero_and_missing(self) -> None:
        lines = self.formatter.format_bar_chart("T", ["a", "b", "c", "d"], [10.0, -5.0, 0.0, None], "°C")
        self.assertEqual(lines[2], "a | " + "█" * 25 + " 10°C")
        self.assertEqual(lines[3], "b | " + "░" * 13 + " -5°C")
        self.assertEqual(lines[4], "c | ▏ 0°C")
        self.assertEqual(lines[5], "d | N/A")

    def test_bar_chart_small_value_gets_at_least_one_block(self) -> None:
        lines = self.formatter.format_bar_chart("T", ["a", "b"], [100.0, 0.1], "mm")
        self.assertEqual(lines[3], "b | █ 0.1 mm")

    def test_bar_chart_percent_is_scaled_to_100(self) -> None:
        lines = self.formatter.format_bar_chart("P", ["a"], [20.0], "%")
        self.assertEqual(lines[2], "a | " + "█" * 5 + " 20%")

    def test_bar_chart_all_missing(self) -> None:
        lines = self.formatter.format_bar_chart("T", ["a"], [None], "mm")
        self.assertIn("  No data available.", lines)

    def test_bar_chart_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.formatter.format_bar_chart("T", ["a"], [1.0, 2.0], "mm")

    def test_single_day_shows_detailed_ranges(self) -> None:
        day = DailyForecast("2026-10-21", {"temperature_2m_min": 10.0, "temperature_2m_max": 20.0})
        lines = self.formatter.format_forecast(LONDON, [day], [METRICS["1"]])
        self.assertIn("  Temperature:    10°C to 20°C", lines)
        self.assertIn(" Weather Forecast for London, England, United Kingdom", lines)

    def test_multi_day_shows_chart_per_field(self) -> None:
        days = [DailyForecast(f"2026-10-2{i}", {"temperature_2m_min": 5.0 + i,
                                                 "temperature_2m_max": 15.0 + i}) for i in range(1, 3)]
        lines = self.formatter.format_forecast(LONDON, days, [METRICS["1"]])
        self.assertIn("--- Min Temperature ---", lines)
        self.assertIn("--- Max Temperature ---", lines)

    def test_empty_inputs_raise(self) -> None:
        day = DailyForecast("2026-10-21", {})
        with self.assertRaises(ValueError):
            self.formatter.format_forecast(LONDON, [], [METRICS["1"]])
        with self.assertRaises(ValueError):
            self.formatter.format_forecast(LONDON, [day], [])


class ScriptedCLIMixin:
    """Helpers that run WeatherCLI with scripted input and capture its output."""

    def run_cli(self, inputs: List[Any], client: Optional[mock.Mock] = None) -> List[str]:
        """Runs the CLI with scripted inputs and returns everything it printed."""
        self.client = client or mock.Mock(spec=WeatherAPIClient)
        output: List[str] = []
        script = iter(inputs)

        def fake_input(prompt: str) -> str:
            output.append(prompt)
            item = next(script)
            if isinstance(item, BaseException):
                raise item
            return item

        exit_code = WeatherCLI(self.client, input_func=fake_input, output_func=output.append).run()
        self.assertEqual(exit_code, 0)
        return output

    def single_day(self) -> List[DailyForecast]:
        return [DailyForecast("2026-10-21", {"temperature_2m_min": 9.0, "temperature_2m_max": 18.0})]


class TestWeatherCLI(ScriptedCLIMixin, unittest.TestCase):
    def test_full_flow_with_multiple_matches(self) -> None:
        client = mock.Mock(spec=WeatherAPIClient)
        client.search_locations.return_value = [LONDON, Location("London", 42.98, -81.23, "Canada")]
        client.get_forecast.return_value = self.single_day()
        output = self.run_cli(["London", "1", "1", "1", "0", "q"], client)
        client.get_forecast.assert_called_once_with(
            LONDON, ["temperature_2m_min", "temperature_2m_max"], 1, METRIC_UNITS)
        self.assertIn("  Temperature:    9°C to 18°C", output)
        self.assertIn("[2] London, Canada", output)
        self.assertEqual(output[-1], "\nExiting application. Goodbye!")

    def test_stays_on_same_city_after_forecast(self) -> None:
        client = mock.Mock(spec=WeatherAPIClient)
        client.search_locations.return_value = [LONDON]
        client.get_forecast.return_value = self.single_day()
        self.run_cli(["London", "1", "1", "1", "4", "q"], client)
        self.assertEqual(client.get_forecast.call_count, 2)
        client.search_locations.assert_called_once()

    def test_back_from_metrics_returns_to_mode_menu(self) -> None:
        client = mock.Mock(spec=WeatherAPIClient)
        client.search_locations.return_value = [LONDON]
        output = self.run_cli(["London", "2", "0", "0", "q"], client)
        client.get_forecast.assert_not_called()
        self.assertEqual(output.count("\nSelect Forecast Mode:"), 2)

    def test_invalid_inputs_reprompt(self) -> None:
        client = mock.Mock(spec=WeatherAPIClient)
        client.search_locations.return_value = [LONDON]
        output = self.run_cli(["", "a", "London", "9", "abc", "0", "q"], client)
        invalid = [line for line in output if line.startswith("[Invalid]")]
        self.assertEqual(len(invalid), 4)

    def test_city_not_found(self) -> None:
        client = mock.Mock(spec=WeatherAPIClient)
        client.search_locations.return_value = []
        output = self.run_cli(["Atlantis", "q"], client)
        self.assertIn("[Notice] City 'Atlantis' was not found.", output)

    def test_fuzzy_single_match_can_be_declined(self) -> None:
        client = mock.Mock(spec=WeatherAPIClient)
        client.search_locations.return_value = [Location("Londrina", -23.3, -51.2, "Brazil")]
        output = self.run_cli(["Londn", "maybe", "n", "q"], client)
        self.assertIn("Search cancelled.", output)

    def test_search_error_is_reported_and_loop_continues(self) -> None:
        client = mock.Mock(spec=WeatherAPIClient)
        client.search_locations.side_effect = APICommunicationError("offline")
        output = self.run_cli(["London", "Paris", "q"], client)
        self.assertEqual(output.count("[Error] offline"), 2)

    def test_forecast_error_keeps_user_on_same_city(self) -> None:
        client = mock.Mock(spec=WeatherAPIClient)
        client.search_locations.return_value = [LONDON]
        client.get_forecast.side_effect = InvalidResponseError("bad data")
        output = self.run_cli(["London", "1", "a", "0", "q"], client)
        self.assertIn("[Error] bad data", output)

    def test_quit_from_nested_prompt(self) -> None:
        client = mock.Mock(spec=WeatherAPIClient)
        client.search_locations.return_value = [LONDON]
        output = self.run_cli(["London", "1", "QUIT"], client)
        self.assertEqual(output[-1], "\nExiting application. Goodbye!")

    def test_eof_and_ctrl_c_exit_cleanly(self) -> None:
        for interrupt in (EOFError(), KeyboardInterrupt()):
            with self.subTest(interrupt=type(interrupt).__name__):
                output = self.run_cli([interrupt])
                self.assertEqual(output[-1], "\nExiting application. Goodbye!")


class TestArgParser(unittest.TestCase):
    def test_default_and_custom_timeout(self) -> None:
        self.assertEqual(build_arg_parser().parse_args([]).timeout, 10.0)
        self.assertEqual(build_arg_parser().parse_args(["--timeout", "2.5"]).timeout, 2.5)

    def test_invalid_timeout_exits(self) -> None:
        for raw in ("0", "-3", "abc", "nan", "inf"):
            with self.subTest(raw=raw), mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                build_arg_parser().parse_args(["--timeout", raw])



class TestUnitsAndMetrics(unittest.TestCase):
    def test_unit_for_each_kind(self) -> None:
        self.assertEqual(METRIC_UNITS.unit_for("temperature"), "°C")
        self.assertEqual(IMPERIAL_UNITS.unit_for("temperature"), "°F")
        self.assertEqual(IMPERIAL_UNITS.unit_for("wind_speed"), "mph")
        self.assertEqual(IMPERIAL_UNITS.unit_for("precipitation"), "inch")
        self.assertEqual(IMPERIAL_UNITS.unit_for("percent"), "%")

    def test_unknown_kind_raises(self) -> None:
        with self.assertRaises(ValueError):
            METRIC_UNITS.unit_for("pressure")

    def test_summary(self) -> None:
        self.assertEqual(METRIC_UNITS.summary, "metric (°C, km/h, mm)")

    def test_menu_label_follows_units(self) -> None:
        self.assertEqual(METRICS["1"].menu_label(METRIC_UNITS), "Temperature (°C)")
        self.assertEqual(METRICS["3"].menu_label(IMPERIAL_UNITS), "Max Wind Speed (mph)")

    def test_get_forecast_sends_unit_params(self) -> None:
        client = make_client(make_response(payload={"daily": {"time": ["2026-10-21"]}}))
        client.get_forecast(LONDON, ["temperature_2m_max"], 1, IMPERIAL_UNITS)
        _, kwargs = client._session.get.call_args
        self.assertEqual(kwargs["params"]["temperature_unit"], "fahrenheit")
        self.assertEqual(kwargs["params"]["wind_speed_unit"], "mph")
        self.assertEqual(kwargs["params"]["precipitation_unit"], "inch")

    def test_get_forecast_rejects_bad_units(self) -> None:
        with self.assertRaises(TypeError):
            make_client().get_forecast(LONDON, ["temperature_2m_max"], 1, "imperial")  # type: ignore[arg-type]

    def test_imperial_formatting(self) -> None:
        self.assertEqual(ForecastFormatter.format_value(70.0, "°F"), "70°F")
        self.assertEqual(ForecastFormatter.format_value(0.02, "inch"), "0.02 inch")
        day = DailyForecast("2026-10-21", {"wind_speed_10m_max": 8.0})
        lines = ForecastFormatter().format_day_details(day, [METRICS["3"]], IMPERIAL_UNITS)
        self.assertIn("  Max Wind Speed: 8 mph", lines)


class TestRecentLocations(unittest.TestCase):
    PARIS = Location("Paris", 48.85, 2.35, "France")
    ROME = Location("Rome", 41.9, 12.5, "Italy")

    def test_newest_first_and_moves_duplicates_to_front(self) -> None:
        recent = RecentLocations()
        for location in (LONDON, self.PARIS, LONDON):
            recent.add(location)
        self.assertEqual(recent.items, (LONDON, self.PARIS))

    def test_trims_to_max_size(self) -> None:
        recent = RecentLocations(max_size=2)
        for location in (LONDON, self.PARIS, self.ROME):
            recent.add(location)
        self.assertEqual(recent.items, (self.ROME, self.PARIS))

    def test_items_is_read_only_copy(self) -> None:
        recent = RecentLocations()
        recent.add(LONDON)
        self.assertIsInstance(recent.items, tuple)

    def test_invalid_arguments(self) -> None:
        for size in (0, -1, True, 2.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                RecentLocations(size)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            RecentLocations().add("London")  # type: ignore[arg-type]


class TestCLICommands(ScriptedCLIMixin, unittest.TestCase):
    """Global commands; uses the scripted-run helpers."""

    def london_client(self) -> mock.Mock:
        client = mock.Mock(spec=WeatherAPIClient)
        client.search_locations.return_value = [LONDON]
        client.get_forecast.return_value = self.single_day()
        return client

    def test_help_shows_commands_and_context_then_reprompts(self) -> None:
        output = self.run_cli(["help", "?", "q"])
        self.assertEqual(output.count("\nCommands (available at any prompt):"), 2)
        self.assertIn("Here: Type a city name (e.g. 'London'), or 'recent' to pick a "
                      "recently viewed city.\n", output)
        self.assertEqual(output.count("\nEnter city name: "), 3)

    def test_help_context_changes_per_prompt(self) -> None:
        output = self.run_cli(["London", "h", "q"], self.london_client())
        self.assertIn("Here: 1 shows today's detailed forecast, 2 shows 7-day charts.\n", output)

    def test_units_toggle_changes_request_and_menu(self) -> None:
        client = self.london_client()
        output = self.run_cli(["units", "London", "1", "1", "q"], client)
        self.assertIn("Units switched to imperial (°F, mph, inch).", output)
        self.assertIn("[1] Temperature (°F)", output)
        self.assertIs(client.get_forecast.call_args.args[3], IMPERIAL_UNITS)
        self.assertIn("  Temperature:    9°F to 18°F", output)

    def test_units_toggle_twice_returns_to_metric(self) -> None:
        output = self.run_cli(["units", "units", "q"])
        self.assertIn("Units switched to metric (°C, km/h, mm).", output)

    def test_back_alias_works_like_zero(self) -> None:
        client = self.london_client()
        output = self.run_cli(["London", "1", "b", "back", "q"], client)
        client.get_forecast.assert_not_called()
        self.assertEqual(output.count("\nSelect Forecast Mode:"), 2)
        self.assertEqual(output.count("\nEnter city name: "), 2)

    def test_back_at_city_prompt_shows_notice(self) -> None:
        output = self.run_cli(["back", "q"])
        self.assertIn("[Notice] This is the first step. Type 'q' to exit.", output)

    def test_recent_when_empty(self) -> None:
        output = self.run_cli(["recent", "q"])
        self.assertIn("[Notice] No recent cities yet.", output)

    def test_recent_reuses_location_without_new_search(self) -> None:
        client = self.london_client()
        output = self.run_cli(["London", "0", "recent", "1", "1", "1", "q"], client)
        self.assertIn("[1] London, England, United Kingdom", output)
        client.search_locations.assert_called_once()
        client.get_forecast.assert_called_once()

    def test_recent_back_returns_to_city_prompt(self) -> None:
        client = self.london_client()
        self.run_cli(["London", "0", "recent", "0", "q"], client)
        client.get_forecast.assert_not_called()


class TestRunOnce(unittest.TestCase):
    def setUp(self) -> None:
        self.client = mock.Mock(spec=WeatherAPIClient)
        self.client.search_locations.return_value = [LONDON, Location("London", 42.98, -81.23, "Canada")]
        self.client.get_forecast.return_value = [
            DailyForecast("2026-10-21", {"temperature_2m_min": 9.0, "temperature_2m_max": 18.0})]
        self.output: List[str] = []
        self.cli = WeatherCLI(self.client, output_func=self.output.append,
                              input_func=lambda _: self.fail("must not prompt"))

    def test_success_uses_best_match(self) -> None:
        self.assertEqual(self.cli.run_once("London", 1, ["1"]), 0)
        self.client.get_forecast.assert_called_once_with(
            LONDON, ["temperature_2m_min", "temperature_2m_max"], 1, METRIC_UNITS)
        self.assertIn("  Temperature:    9°C to 18°C", self.output)

    def test_city_not_found(self) -> None:
        self.client.search_locations.return_value = []
        self.assertEqual(self.cli.run_once("Atlantis", 1, ["1"]), 1)
        self.assertIn("[Error] City 'Atlantis' was not found.", self.output)

    def test_invalid_city(self) -> None:
        self.assertEqual(self.cli.run_once("1", 1, ["1"]), 1)
        self.client.search_locations.assert_not_called()

    def test_search_error(self) -> None:
        self.client.search_locations.side_effect = APICommunicationError("offline")
        self.assertEqual(self.cli.run_once("London", 1, ["1"]), 1)

    def test_forecast_error(self) -> None:
        self.client.get_forecast.side_effect = InvalidResponseError("bad")
        self.assertEqual(self.cli.run_once("London", 1, ["1"]), 1)


class TestMain(unittest.TestCase):
    def test_non_interactive_mode_end_to_end(self) -> None:
        client = mock.Mock(spec=WeatherAPIClient)
        client.search_locations.return_value = [LONDON]
        client.get_forecast.return_value = [
            DailyForecast(f"2026-10-2{i}", {"temperature_2m_min": 9.0}) for i in range(1, 4)]
        with (mock.patch("weather.WeatherAPIClient") as client_cls,
              mock.patch("sys.stdout", new_callable=io.StringIO) as stdout):
            client_cls.return_value.__enter__.return_value = client
            code = main(["--city", "London", "--days", "3", "--metrics", "1,5", "--units", "imperial"])
        self.assertEqual(code, 0)
        self.assertIn("--- Min Temperature ---", stdout.getvalue())
        client.get_forecast.assert_called_once_with(
            LONDON,
            ["temperature_2m_min", "temperature_2m_max", "precipitation_probability_max"],
            3,
            IMPERIAL_UNITS,
        )

    def test_days_or_metrics_without_city_is_an_error(self) -> None:
        for argv in (["--days", "3"], ["--metrics", "1"]):
            with self.subTest(argv=argv), mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                main(argv)

    def test_invalid_flag_values(self) -> None:
        bad = (["--days", "0"], ["--days", "17"], ["--days", "abc"], ["--metrics", "0"],
               ["--metrics", "9"], ["--units", "kelvin"])
        for argv in bad:
            with self.subTest(argv=argv), mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                build_arg_parser().parse_args(["--city", "London", *argv])

    def test_valid_flag_values(self) -> None:
        args = build_arg_parser().parse_args(["--city", "Rome", "--days", "16", "--metrics", "a"])
        self.assertEqual(args.days, 16)
        self.assertEqual(args.metrics, list(METRICS))
        self.assertEqual(args.units, "metric")

class TestEntryPoint(unittest.TestCase):
    def test_main_py_runs_help(self) -> None:
        script = Path(__file__).with_name("main.py")
        result = subprocess.run([sys.executable, str(script), "--help"],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0)
        self.assertIn("--city", result.stdout)

if __name__ == "__main__":
    unittest.main()
