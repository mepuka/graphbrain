"""Main MCP server for graphbrain.

Provides a FastMCP-based server that exposes graphbrain operations
as tools for use with Claude and other MCP-compatible clients.
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
    """

    def __init__(
        self,
        pg_connection: str,
        name: str = "graphbrain",
        embedding_model: Optional[str] = None,
    ):
        """Initialize the graphbrain MCP server.

        Args:
            pg_connection: PostgreSQL connection string
            name: Server name for MCP identification
            embedding_model: Optional sentence transformer model for semantic search
        """
        self.pg_connection = pg_connection
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
        from graphbrain.classification import ClassificationRepository, HybridSearcher

        logger.info(f"Connecting to PostgreSQL: {self.pg_connection}")

        # Initialize hypergraph connection
        hg = graphbrain.hgraph(self.pg_connection)

        # Initialize classification repository
        repo = ClassificationRepository(self.pg_connection)

        # Initialize hybrid searcher (lazy loads embedding model)
        searcher = None
        try:
            searcher = HybridSearcher(self.pg_connection, embedding_model=self.embedding_model)
        except Exception as e:
            logger.warning(f"HybridSearcher initialization failed: {e}")

        # Store in context for tools to access
        yield {
            "hg": hg,
            "repo": repo,
            "searcher": searcher,
        }

        # Cleanup
        logger.info("Closing database connections")
        hg.close()
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

        register_hypergraph_tools(self.server)
        register_classification_tools(self.server)
        register_semantic_class_tools(self.server)
        register_predicate_tools(self.server)
        register_feedback_tools(self.server)

    def run(self, transport: str = "stdio", **kwargs):
        """Run the MCP server.

        Args:
            transport: Transport type ("stdio", "sse", "http")
            **kwargs: Additional arguments for the transport
        """
        self.server.run(transport=transport, **kwargs)


def create_server(
    pg_connection: str,
    name: str = "graphbrain",
    embedding_model: Optional[str] = None,
) -> GraphbrainMCP:
    """Create a graphbrain MCP server.

    Args:
        pg_connection: PostgreSQL connection string
        name: Server name for MCP identification
        embedding_model: Optional sentence transformer model for semantic search

    Returns:
        Configured GraphbrainMCP instance

    Example:
        server = create_server("postgresql://localhost/graphbrain")
        server.run()  # stdio transport (default)

        # Or with HTTP transport
        server.run(transport="http", port=8000)
    """
    return GraphbrainMCP(
        pg_connection=pg_connection,
        name=name,
        embedding_model=embedding_model,
    )
