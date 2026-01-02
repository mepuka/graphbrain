#!/usr/bin/env python3
"""Launch the graphbrain MCP server.

Usage:
    # With SQLite (default: urbanist.db)
    python scripts/run_mcp_server.py

    # With specific database
    python scripts/run_mcp_server.py /path/to/database.db

    # With PostgreSQL
    python scripts/run_mcp_server.py postgresql://user:pass@host/db

Environment variables:
    GRAPHBRAIN_DB: Default database path/connection string
    GRAPHBRAIN_EMBEDDING_MODEL: Sentence transformer model (default: intfloat/e5-base-v2)
"""

import os
import sys
import logging

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphbrain.mcp.server import create_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr,  # MCP uses stdout for communication
)
logger = logging.getLogger(__name__)


def main():
    # Get database connection from args or environment
    if len(sys.argv) > 1:
        connection_string = sys.argv[1]
    else:
        connection_string = os.environ.get(
            'GRAPHBRAIN_DB',
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'urbanist.db')
        )

    # Get embedding model from environment
    embedding_model = os.environ.get('GRAPHBRAIN_EMBEDDING_MODEL', 'intfloat/e5-base-v2')

    logger.info(f"Starting graphbrain MCP server")
    logger.info(f"  Database: {connection_string}")
    logger.info(f"  Embedding model: {embedding_model}")

    # Create and run server
    server = create_server(
        connection_string=connection_string,
        name="graphbrain",
        embedding_model=embedding_model,
    )

    # Run with stdio transport (for MCP)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
