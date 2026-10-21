"""Application constants: selectable metrics, API field titles, limits and forecast modes."""
from __future__ import annotations

from typing import Dict, Tuple

from .models import Metric


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
