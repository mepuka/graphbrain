"""Classification storage backends.

Provides abstract base class and implementations for PostgreSQL and SQLite.
"""

import logging
from typing import Union

from graphbrain.classification.backends.base import ClassificationBackend

logger = logging.getLogger(__name__)


def get_classification_backend(connection_string: str) -> ClassificationBackend:
    """Factory function to create appropriate classification backend.

    Detects backend type from connection string:
    - postgresql:// or postgres:// -> PostgresBackend
    - File path with .db, .sqlite, .sqlite3 -> SqliteBackend

    Args:
        connection_string: Database connection string or file path

    Returns:
        Appropriate ClassificationBackend implementation

    Raises:
        ValueError: If backend type cannot be determined
        ImportError: If required dependencies are not available
    """
    # Check for PostgreSQL
    if connection_string.startswith(('postgresql://', 'postgres://')):
        from graphbrain.classification.backends.postgres import PostgresBackend
        logger.info(f"Using PostgreSQL classification backend")
        return PostgresBackend(connection_string)

    # Check for SQLite file extensions
    sqlite_extensions = ('.db', '.sqlite', '.sqlite3')
    if any(connection_string.endswith(ext) for ext in sqlite_extensions):
        from graphbrain.classification.backends.sqlite import SqliteBackend
        logger.info(f"Using SQLite classification backend: {connection_string}")
        return SqliteBackend(connection_string)

    # Try to detect based on path-like structure
    if '/' in connection_string or '\\' in connection_string:
        # Looks like a file path, assume SQLite
        from graphbrain.classification.backends.sqlite import SqliteBackend
        logger.info(f"Using SQLite classification backend (path detected): {connection_string}")
        return SqliteBackend(connection_string)

    raise ValueError(
        f"Cannot determine backend type from connection string: {connection_string}. "
        "Use 'postgresql://...' for PostgreSQL or a file path ending in .db/.sqlite for SQLite."
    )


__all__ = [
    'ClassificationBackend',
    'get_classification_backend',
]
