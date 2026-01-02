"""Tests for the MCP server."""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock

# Check if psycopg2 is available
try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

# Check if mcp is available
try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


@pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP not available")
class TestMCPToolRegistration:
    """Test that tools can be registered."""

    def test_all_tools_import(self):
        """Test that all tool modules import correctly."""
        from graphbrain.mcp.tools import (
            register_hypergraph_tools,
            register_classification_tools,
            register_semantic_class_tools,
            register_predicate_tools,
            register_feedback_tools,
        )
        assert callable(register_hypergraph_tools)
        assert callable(register_classification_tools)
        assert callable(register_semantic_class_tools)
        assert callable(register_predicate_tools)
        assert callable(register_feedback_tools)

    def test_tool_registration(self):
        """Test that tools can be registered with FastMCP."""
        from mcp.server.fastmcp import FastMCP
        from graphbrain.mcp.tools import (
            register_hypergraph_tools,
            register_classification_tools,
            register_semantic_class_tools,
            register_predicate_tools,
            register_feedback_tools,
        )

        server = FastMCP('test')
        register_hypergraph_tools(server)
        register_classification_tools(server)
        register_semantic_class_tools(server)
        register_predicate_tools(server)
        register_feedback_tools(server)

        async def count_tools():
            tools = await server.list_tools()
            return len(tools)

        count = asyncio.run(count_tools())
        # 5 hypergraph + 4 classification + 7 semantic_class + 7 predicate + 5 feedback = 28 tools
        assert count == 28

    def test_server_creation(self):
        """Test server module structure."""
        from graphbrain.mcp import create_server, GraphbrainMCP
        assert callable(create_server)
        assert GraphbrainMCP is not None


@pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP not available")
class TestMCPErrors:
    """Test standardized error responses."""

    def test_error_codes(self):
        """Test error code enum."""
        from graphbrain.mcp.errors import ErrorCode

        assert ErrorCode.INVALID_EDGE.value == "invalid_edge"
        assert ErrorCode.NOT_FOUND.value == "not_found"
        assert ErrorCode.DATABASE_ERROR.value == "database_error"

    def test_error_response(self):
        """Test error response helper."""
        from graphbrain.mcp.errors import error_response, ErrorCode

        response = error_response(
            ErrorCode.INVALID_EDGE,
            "Invalid edge syntax",
            {"edge": "(broken/syntax"}
        )
        assert response["status"] == "error"
        assert response["code"] == "invalid_edge"
        assert response["message"] == "Invalid edge syntax"
        assert response["details"]["edge"] == "(broken/syntax"

    def test_convenience_error_functions(self):
        """Test convenience error functions."""
        from graphbrain.mcp.errors import (
            invalid_edge_error,
            not_found_error,
            service_unavailable_error,
        )

        # Test invalid_edge_error
        result = invalid_edge_error("(bad", ValueError("test"))
        assert result["status"] == "error"
        assert result["code"] == "invalid_edge"
        assert result["details"]["edge"] == "(bad"

        # Test not_found_error
        result = not_found_error("semantic_class", "abc123")
        assert result["status"] == "error"
        assert result["code"] == "not_found"
        assert "abc123" in result["message"]

        # Test service_unavailable_error
        result = service_unavailable_error("searcher", "not initialized")
        assert result["status"] == "error"
        assert result["code"] == "service_unavailable"


@pytest.mark.skipif(
    not (MCP_AVAILABLE and PSYCOPG2_AVAILABLE),
    reason="MCP or psycopg2 not available"
)
class TestMCPToolsIntegration:
    """Integration tests with real database (requires PostgreSQL)."""

    @pytest.fixture
    def pg_connection(self):
        """Get PostgreSQL connection string."""
        import os
        return os.environ.get(
            "TEST_POSTGRES_URI",
            "postgresql://localhost/graphbrain_test"
        )

    @pytest.fixture
    def can_connect_to_db(self, pg_connection):
        """Check if we can connect to the database."""
        try:
            conn = psycopg2.connect(pg_connection)
            conn.close()
            return True
        except Exception:
            return False

    def test_server_with_database(self, pg_connection, can_connect_to_db):
        """Test server with real database connection."""
        if not can_connect_to_db:
            pytest.skip("Cannot connect to PostgreSQL")

        from graphbrain.mcp import create_server

        server = create_server(pg_connection)
        assert server is not None
        assert server.pg_connection == pg_connection


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
