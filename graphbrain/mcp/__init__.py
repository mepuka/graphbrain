"""MCP (Model Context Protocol) server for graphbrain.

This module provides an MCP server that exposes graphbrain operations
as tools for use with Claude and other MCP-compatible clients.

Usage:
    from graphbrain.mcp import create_server

    server = create_server(
        pg_connection="postgresql://localhost/graphbrain",
        name="graphbrain"
    )

    # Run with stdio transport
    server.run()

    # Or run as HTTP server
    server.run(transport="http", port=8000)
"""

from graphbrain.mcp.server import create_server, GraphbrainMCP

__all__ = [
    'create_server',
    'GraphbrainMCP',
]
