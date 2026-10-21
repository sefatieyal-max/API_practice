"""Turns forecast data into printable lines: a detailed day view or ASCII charts (no I/O)."""
from __future__ import annotations

from datetime import date
from typing import List, Optional, Sequence

from .config import FIELD_TITLES
from .models import METRIC_UNITS, DailyForecast, Location, Metric, UnitSystem


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
