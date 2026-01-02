"""Shared utilities for MCP tools.

Provides common helpers used across multiple tool modules.
"""

from datetime import datetime
from typing import Optional, Union

from graphbrain.mcp.errors import invalid_input_error


def to_isoformat(value: Union[datetime, str, None]) -> Optional[str]:
    """Convert a datetime or string to ISO format string.

    Handles the case where the value might already be a string (e.g., from SQLite)
    or a datetime object (e.g., from PostgreSQL).

    Args:
        value: A datetime object, ISO format string, or None

    Returns:
        ISO format string or None
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    # Fallback for any other type
    return str(value)


def validate_threshold(threshold: float, name: str = "threshold") -> Optional[dict]:
    """Validate a threshold parameter is between 0.0 and 1.0.

    Args:
        threshold: The threshold value to validate
        name: Parameter name for error messages

    Returns:
        None if valid, error response dict if invalid
    """
    if not isinstance(threshold, (int, float)):
        return invalid_input_error(
            f"{name} must be a number",
            {name: threshold}
        )
    if not 0.0 <= threshold <= 1.0:
        return invalid_input_error(
            f"{name} must be between 0.0 and 1.0",
            {name: threshold}
        )
    return None


def validate_positive_int(value: int, name: str = "value", allow_zero: bool = False) -> Optional[dict]:
    """Validate a parameter is a positive integer.

    Args:
        value: The value to validate
        name: Parameter name for error messages
        allow_zero: If True, zero is allowed

    Returns:
        None if valid, error response dict if invalid
    """
    if not isinstance(value, int):
        return invalid_input_error(
            f"{name} must be an integer",
            {name: value}
        )
    if allow_zero:
        if value < 0:
            return invalid_input_error(
                f"{name} must be non-negative",
                {name: value}
            )
    else:
        if value <= 0:
            return invalid_input_error(
                f"{name} must be positive",
                {name: value}
            )
    return None


def validate_limit(limit: int, max_limit: int = 10000) -> Optional[dict]:
    """Validate a limit parameter.

    Args:
        limit: The limit value to validate
        max_limit: Maximum allowed limit

    Returns:
        None if valid, error response dict if invalid
    """
    error = validate_positive_int(limit, "limit")
    if error:
        return error
    if limit > max_limit:
        return invalid_input_error(
            f"limit cannot exceed {max_limit}",
            {"limit": limit, "max_limit": max_limit}
        )
    return None


def get_lifespan_context(server):
    """Get lifespan context data from MCP server.

    Args:
        server: FastMCP server instance

    Returns:
        Dict with hg, repo, searcher, current_session, etc.
    """
    ctx = server.get_context()
    return ctx.request_context.lifespan_context


def calculate_confidence(entry) -> float:
    """Calculate confidence score for a predicate bank entry.

    Centralizes the confidence calculation logic used across tools.

    Args:
        entry: PredicateBankEntry with similarity_score and is_seed fields

    Returns:
        Confidence score between 0.0 and 1.0
    """
    if entry.similarity_score is not None:
        return entry.similarity_score
    elif entry.is_seed:
        return 1.0
    else:
        return 0.7
