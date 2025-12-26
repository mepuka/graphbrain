"""Main MCP server for graphbrain.

Provides a FastMCP-based server that exposes graphbrain operations
as tools for use with Claude and other MCP-compatible clients.

Supports both PostgreSQL and SQLite backends.
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional, Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


class GraphbrainMCP:
    """Graphbrain MCP server wrapper.

    Manages database connections and provides the FastMCP server instance
    with all graphbrain tools registered.

    Supports both PostgreSQL and SQLite backends, automatically detecting
    the backend type from the connection string.
    """

    def __init__(
        self,
        connection_string: str,
        name: str = "graphbrain",
        embedding_model: Optional[str] = None,
    ):
        """Initialize the graphbrain MCP server.

        Args:
            connection_string: Database connection string or file path.
                - PostgreSQL: 'postgresql://user:pass@host/db'
                - SQLite: '/path/to/database.db'
            name: Server name for MCP identification
            embedding_model: Optional sentence transformer model for semantic search
        """
        self.connection_string = connection_string
        self.embedding_model = embedding_model or "intfloat/e5-base-v2"

        # Initialize FastMCP with lifespan for database connections
        self.server = FastMCP(
            name=name,
            instructions="""
Graphbrain knowledge graph tools for semantic analysis.

Available capabilities:
- Search and query hypergraph edges
- Pattern matching with semantic hypergraph notation
- Predicate classification with hybrid BM25 + semantic search
- Semantic class management for domain-adaptive classification
- Human-in-the-loop feedback for iterative improvement

Use these tools to explore knowledge graphs, classify predicates,
discover patterns, and build domain-specific semantic models.
""",
            lifespan=self._lifespan,
        )

        # Register all tools
        self._register_tools()

    @asynccontextmanager
    async def _lifespan(self, server: FastMCP):
        """Manage database connections during server lifetime."""
        import graphbrain
        from graphbrain.classification.backends import get_classification_backend
        from graphbrain.classification.search import get_search_backend

        logger.info(f"Connecting to: {self.connection_string}")

        # Initialize hypergraph connection (works with both SQLite and PostgreSQL)
        hg = graphbrain.hgraph(self.connection_string)

        # Initialize classification backend (auto-detects type)
        repo = None
        try:
            repo = get_classification_backend(self.connection_string)
            logger.info(f"Classification backend initialized: {type(repo).__name__}")
        except Exception as e:
            logger.warning(f"Classification backend initialization failed: {e}")

        # Initialize search backend (auto-detects type, lazy loads embedding model)
        searcher = None
        try:
            searcher = get_search_backend(
                self.connection_string,
                embedding_model=self.embedding_model
            )
            logger.info(f"Search backend initialized: {type(searcher).__name__}")
        except Exception as e:
            logger.warning(f"Search backend initialization failed: {e}")

        # Store in context for tools to access
        yield {
            "hg": hg,
            "repo": repo,
            "searcher": searcher,
        }

        # Cleanup
        logger.info("Closing database connections")
        hg.close()
        if repo:
            repo.close()
        if searcher:
            searcher.close()

    def _register_tools(self):
        """Register all graphbrain tools with the server."""
        # Import tool registration functions
        from graphbrain.mcp.tools.hypergraph import register_hypergraph_tools
        from graphbrain.mcp.tools.classification import register_classification_tools
        from graphbrain.mcp.tools.semantic_classes import register_semantic_class_tools
        from graphbrain.mcp.tools.predicates import register_predicate_tools
        from graphbrain.mcp.tools.feedback import register_feedback_tools
        from graphbrain.mcp.tools.agents import register_agent_tools

        register_hypergraph_tools(self.server)
        register_classification_tools(self.server)
        register_semantic_class_tools(self.server)
        register_predicate_tools(self.server)
        register_feedback_tools(self.server)
        register_agent_tools(self.server)

    def run(self, transport: str = "stdio", **kwargs):
        """Run the MCP server.

        Args:
            transport: Transport type ("stdio", "sse", "http")
            **kwargs: Additional arguments for the transport
        """
        self.server.run(transport=transport, **kwargs)


def create_server(
    connection_string: str,
    name: str = "graphbrain",
    embedding_model: Optional[str] = None,
) -> GraphbrainMCP:
    """Create a graphbrain MCP server.

    Args:
        connection_string: Database connection string or file path.
            - PostgreSQL: 'postgresql://user:pass@host/db'
            - SQLite: '/path/to/database.db'
        name: Server name for MCP identification
        embedding_model: Optional sentence transformer model for semantic search

    Returns:
        Configured GraphbrainMCP instance

    Example:
        # PostgreSQL
        server = create_server("postgresql://localhost/graphbrain")
        server.run()  # stdio transport (default)

        # SQLite
        server = create_server("/path/to/knowledge.db")
        server.run()

        # Or with HTTP transport
        server.run(transport="http", port=8000)
    """
    return GraphbrainMCP(
        connection_string=connection_string,
        name=name,
        embedding_model=embedding_model,
    )
