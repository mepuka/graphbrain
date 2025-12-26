"""Search backends for hybrid BM25 + semantic search.

Provides abstract base class and implementations for PostgreSQL and SQLite.
"""

import logging
from typing import Optional

from graphbrain.classification.search.base import SearchBackend, SearchResult

logger = logging.getLogger(__name__)


def get_search_backend(
    connection_string: str,
    embedding_model: Optional[str] = None,
    default_weights: Optional[dict] = None,
) -> SearchBackend:
    """Factory function to create appropriate search backend.

    Detects backend type from connection string:
    - postgresql:// or postgres:// -> PostgresSearchBackend
    - File path with .db, .sqlite, .sqlite3 -> SqliteSearchBackend

    Args:
        connection_string: Database connection string or file path
        embedding_model: Optional sentence transformer model name
        default_weights: Optional default fusion weights

    Returns:
        Appropriate SearchBackend implementation

    Raises:
        ValueError: If backend type cannot be determined
        ImportError: If required dependencies are not available
    """
    # Check for PostgreSQL
    if connection_string.startswith(('postgresql://', 'postgres://')):
        from graphbrain.classification.search.postgres import PostgresSearchBackend
        logger.info("Using PostgreSQL search backend")
        return PostgresSearchBackend(
            connection_string,
            embedding_model=embedding_model,
            default_weights=default_weights,
        )

    # Check for SQLite file extensions
    sqlite_extensions = ('.db', '.sqlite', '.sqlite3')
    if any(connection_string.endswith(ext) for ext in sqlite_extensions):
        from graphbrain.classification.search.sqlite import SqliteSearchBackend
        logger.info(f"Using SQLite search backend: {connection_string}")
        return SqliteSearchBackend(
            connection_string,
            embedding_model=embedding_model,
            default_weights=default_weights,
        )

    # Try to detect based on path-like structure
    if '/' in connection_string or '\\' in connection_string:
        from graphbrain.classification.search.sqlite import SqliteSearchBackend
        logger.info(f"Using SQLite search backend (path detected): {connection_string}")
        return SqliteSearchBackend(
            connection_string,
            embedding_model=embedding_model,
            default_weights=default_weights,
        )

    raise ValueError(
        f"Cannot determine backend type from connection string: {connection_string}. "
        "Use 'postgresql://...' for PostgreSQL or a file path ending in .db/.sqlite for SQLite."
    )


__all__ = [
    'SearchBackend',
    'SearchResult',
    'get_search_backend',
]
