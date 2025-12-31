"""Standardized error responses for MCP tools.

Provides consistent error formatting across all graphbrain MCP tools.
"""

from enum import Enum
from typing import Optional, Any


class ErrorCode(str, Enum):
    """Standard error codes for MCP tools."""
    # Input errors
    INVALID_EDGE = "invalid_edge"
    INVALID_PATTERN = "invalid_pattern"
    INVALID_INPUT = "invalid_input"
    MISSING_PARAMETER = "missing_parameter"

    # Resource errors
    NOT_FOUND = "not_found"
    ALREADY_EXISTS = "already_exists"

    # Service errors
    SERVICE_UNAVAILABLE = "service_unavailable"
    DATABASE_ERROR = "database_error"

    # Internal errors
    INTERNAL_ERROR = "internal_error"


def error_response(
    code: ErrorCode,
    message: str,
    details: Optional[dict[str, Any]] = None,
) -> dict:
    """Create a standardized error response.

    Args:
        code: Error code from ErrorCode enum
        message: Human-readable error message
        details: Optional additional context

    Returns:
        Standardized error dict with status, code, message, and optional details
    """
    response = {
        "status": "error",
        "code": code.value,
        "message": message,
    }
    if details:
        response["details"] = details
    return response


def success_response(data: dict) -> dict:
    """Wrap successful response with status field.

    Args:
        data: The response data

    Returns:
        Response dict with status="success" and data fields merged
    """
    return {"status": "success", **data}


# Convenience functions for common errors

def invalid_edge_error(edge: str, exception: Exception) -> dict:
    """Error for invalid edge syntax."""
    return error_response(
        ErrorCode.INVALID_EDGE,
        f"Invalid edge syntax: {exception}",
        {"edge": edge},
    )


def invalid_pattern_error(pattern: str, exception: Exception) -> dict:
    """Error for invalid pattern syntax."""
    return error_response(
        ErrorCode.INVALID_PATTERN,
        f"Invalid pattern syntax: {exception}",
        {"pattern": pattern},
    )


def invalid_input_error(message: str, details: Optional[dict] = None) -> dict:
    """Error for invalid input."""
    return error_response(
        ErrorCode.INVALID_INPUT,
        message,
        details,
    )


def not_found_error(resource_type: str, identifier: str) -> dict:
    """Error for resource not found."""
    return error_response(
        ErrorCode.NOT_FOUND,
        f"{resource_type} '{identifier}' not found",
        {"resource_type": resource_type, "identifier": identifier},
    )


def already_exists_error(resource_type: str, identifier: str) -> dict:
    """Error for resource already exists."""
    return error_response(
        ErrorCode.ALREADY_EXISTS,
        f"{resource_type} '{identifier}' already exists",
        {"resource_type": resource_type, "identifier": identifier},
    )


def service_unavailable_error(service: str, reason: Optional[str] = None) -> dict:
    """Error for unavailable service."""
    message = f"{service} not available"
    if reason:
        message += f": {reason}"
    return error_response(
        ErrorCode.SERVICE_UNAVAILABLE,
        message,
        {"service": service},
    )


def database_error(operation: str, exception: Exception) -> dict:
    """Error for database operations."""
    return error_response(
        ErrorCode.DATABASE_ERROR,
        f"Database error during {operation}: {exception}",
        {"operation": operation},
    )
