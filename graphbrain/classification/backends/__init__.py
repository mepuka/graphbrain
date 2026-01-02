"""Classification storage backends.

Provides abstract base class and implementations for PostgreSQL and SQLite.
"""

import logging
from enum import Enum
from typing import Union

from graphbrain.classification.backends.base import ClassificationBackend

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Classification backend types."""
    POSTGRES = "postgres"
    SQLITE = "sqlite"
    UNKNOWN = "unknown"


def detect_backend_type(connection_string: str) -> BackendType:
    """Detect backend type from connection string.

    Centralizes backend detection logic to avoid fragile string matching
    scattered throughout the codebase.

    Args:
        connection_string: Database connection string or file path

    Returns:
        BackendType enum value
    """
    if not connection_string:
        return BackendType.UNKNOWN

    # PostgreSQL connection strings
    if connection_string.startswith(('postgresql://', 'postgres://')):
        return BackendType.POSTGRES

    # SQLite file extensions
    sqlite_extensions = ('.db', '.sqlite', '.sqlite3')
    if any(connection_string.endswith(ext) for ext in sqlite_extensions):
        return BackendType.SQLITE

    # Path-like structure suggests file-based (SQLite)
    if '/' in connection_string or '\\' in connection_string:
        return BackendType.SQLITE

    return BackendType.UNKNOWN


def is_postgres(connection_string: str) -> bool:
    """Check if connection string is for PostgreSQL."""
    return detect_backend_type(connection_string) == BackendType.POSTGRES


def is_sqlite(connection_string: str) -> bool:
    """Check if connection string is for SQLite."""
    return detect_backend_type(connection_string) == BackendType.SQLITE


def get_classification_backend(connection_string: str) -> ClassificationBackend:
    """Factory function to create appropriate classification backend.

    Uses detect_backend_type() for consistent backend detection.

    Args:
        connection_string: Database connection string or file path

    Returns:
        Appropriate ClassificationBackend implementation

    Raises:
        ValueError: If backend type cannot be determined
        ImportError: If required dependencies are not available
    """
    backend_type = detect_backend_type(connection_string)

    if backend_type == BackendType.POSTGRES:
        from graphbrain.classification.backends.postgres import PostgresBackend
        logger.info("Using PostgreSQL classification backend")
        return PostgresBackend(connection_string)

    if backend_type == BackendType.SQLITE:
        from graphbrain.classification.backends.sqlite import SqliteBackend
        logger.info(f"Using SQLite classification backend: {connection_string}")
        return SqliteBackend(connection_string)

    raise ValueError(
        f"Cannot determine backend type from connection string: {connection_string}. "
        "Use 'postgresql://...' for PostgreSQL or a file path ending in .db/.sqlite for SQLite."
    )


__all__ = [
    'BackendType',
    'ClassificationBackend',
    'detect_backend_type',
    'get_classification_backend',
    'is_postgres',
    'is_sqlite',
]
