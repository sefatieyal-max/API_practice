"""Weather CLI: an interactive command-line client for the Open-Meteo REST API.

The user searches for a city, picks a forecast range (today or 7 days) and the
metrics to show. Today's forecast is printed as a detailed summary and the
weekly forecast as ASCII bar charts. Global commands (``help``, ``back``,
``units``, ``recent``, ``quit``) work at the prompts, and a one-shot
non-interactive mode is available through command-line flags.

Modules (each one has a single responsibility):
    * ``exceptions``: the application's exception classes.
    * ``models``: immutable data models (Location, DailyForecast, UnitSystem, Metric).
    * ``config``: constants such as the selectable metrics and API limits.
    * ``commands``: the global commands and their aliases.
    * ``input_parser``: pure validation and parsing of raw user text.
    * ``api_client``: HTTP communication with Open-Meteo and response parsing.
    * ``formatter``: turns data into printable lines (no I/O).
    * ``history``: in-memory history of viewed locations.
    * ``cli``: the interactive loop (all terminal I/O lives here).
    * ``app``: command-line arguments and the ``main()`` entry point.

Example:
    $ python main.py
    $ python -m weather_cli --units imperial
    $ python main.py --city London --days 7 --metrics 1,5
"""
from .app import main

__all__ = ["main"]
