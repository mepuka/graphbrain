"""CLI entry point for the graphbrain MCP server.

Usage:
    python -m graphbrain.mcp --pg postgresql://localhost/graphbrain
    python -m graphbrain.mcp --pg postgresql://localhost/graphbrain --transport http --port 8000
"""

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Graphbrain MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with stdio transport (default, for Claude Code integration)
    python -m graphbrain.mcp --pg postgresql://localhost/graphbrain

    # Run as HTTP server
    python -m graphbrain.mcp --pg postgresql://localhost/graphbrain --transport http --port 8000

    # Run as SSE server
    python -m graphbrain.mcp --pg postgresql://localhost/graphbrain --transport sse --port 8000

Environment variables:
    GRAPHBRAIN_PG_URI: PostgreSQL connection string (alternative to --pg)
"""
    )

    parser.add_argument(
        "--pg", "--postgres",
        dest="pg_connection",
        help="PostgreSQL connection string (e.g., postgresql://localhost/graphbrain)",
        default=None,
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "sse"],
        default="stdio",
        help="Transport type (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP/SSE transport (default: 8000)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for HTTP/SSE transport (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--name",
        default="graphbrain",
        help="Server name (default: graphbrain)",
    )
    parser.add_argument(
        "--embedding-model",
        dest="embedding_model",
        default=None,
        help="Sentence transformer model for semantic search",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Get PostgreSQL connection
    pg_connection = args.pg_connection
    if not pg_connection:
        import os
        pg_connection = os.environ.get("GRAPHBRAIN_PG_URI")

    if not pg_connection:
        logger.error("PostgreSQL connection required. Use --pg or set GRAPHBRAIN_PG_URI")
        sys.exit(1)

    # Create and run server
    from graphbrain.mcp import create_server

    logger.info(f"Starting graphbrain MCP server...")
    logger.info(f"  Transport: {args.transport}")
    logger.info(f"  PostgreSQL: {pg_connection.split('@')[-1] if '@' in pg_connection else pg_connection}")

    server = create_server(
        pg_connection=pg_connection,
        name=args.name,
        embedding_model=args.embedding_model,
    )

    if args.transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(
            transport=args.transport,
            host=args.host,
            port=args.port,
        )


if __name__ == "__main__":
    main()
