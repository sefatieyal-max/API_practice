"""Exceptions used across the Weather CLI."""


class WeatherAppError(Exception):
    """Base exception for all application errors."""


class APICommunicationError(WeatherAppError):
    """Raised when the API cannot be reached or answers with an HTTP error."""


class InvalidResponseError(WeatherAppError):
    """Raised when the API answers with a payload in an unexpected shape."""


class QuitRequested(Exception):
    """Raised internally when the user asks to leave the application."""
