"""Command-line arguments and the program entry point."""
from __future__ import annotations

import argparse
from typing import List, Optional, Sequence

from .api_client import WeatherAPIClient
from .cli import WeatherCLI
from .config import DEFAULT_TIMEOUT, MAX_FORECAST_DAYS, METRICS
from .input_parser import InputParser
from .models import METRIC_UNITS, UNIT_SYSTEMS


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
