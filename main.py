"""Entry point for the Weather CLI.

Example:
    $ python main.py
    $ python main.py --city London --days 7 --metrics 1,5
    $ python main.py --help
"""
import sys

from weather_cli import main

if __name__ == "__main__":
    sys.exit(main())
