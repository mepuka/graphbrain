"""Tests for PostgreSQL hypergraph backend.

Requires a running PostgreSQL instance with pgvector extension.
Tests are automatically skipped if psycopg2 is not installed or
PostgreSQL is not available.

Environment variables:
    GRAPHBRAIN_TEST_PG_URI: PostgreSQL connection string
        (default: postgresql://workflow:workflow@localhost:5432/workflow)
"""
import os
import unittest

import pytest

# Check if psycopg2 is available
try:
    import psycopg2
    _PSYCOPG2_AVAILABLE = True
except ImportError:
    _PSYCOPG2_AVAILABLE = False

# Get PostgreSQL connection string from environment or use default
PG_URI = os.environ.get(
    'GRAPHBRAIN_TEST_PG_URI',
    'postgresql://workflow:workflow@localhost:5432/workflow'
)


def can_connect_to_postgres():
    """Check if we can connect to PostgreSQL."""
    if not _PSYCOPG2_AVAILABLE:
        return False
    try:
        conn = psycopg2.connect(PG_URI)
        conn.close()
        return True
    except Exception:
        return False


# Skip all tests if PostgreSQL is not available
pytestmark = pytest.mark.skipif(
    not can_connect_to_postgres(),
    reason="PostgreSQL not available or psycopg2 not installed"
)


from graphbrain.tests.hypergraph import Hypergraph


class TestPostgreSQL(Hypergraph, unittest.TestCase):
    """Test PostgreSQL backend using the common Hypergraph test suite."""

    def setUp(self):
        self.hg_str = PG_URI
        super().setUp()
        # Clean database before each test
        self.hg.destroy()

    def tearDown(self):
        # Clean up after each test
        if hasattr(self, 'hg') and self.hg:
            try:
                self.hg.destroy()
            except Exception:
                pass
            super().tearDown()


class TestPostgreSQLSpecific(unittest.TestCase):
    """Test PostgreSQL-specific features."""

    def setUp(self):
        from graphbrain import hgraph
        self.hg = hgraph(PG_URI)
        self.hg.destroy()

    def tearDown(self):
        if hasattr(self, 'hg') and self.hg:
            self.hg.destroy()
            self.hg.close()

    def test_text_content(self):
        """Test setting and retrieving text content."""
        edge = self.hg.add('(is/Pd graphbrain/C great/C)')
        self.hg.set_text_content(edge, 'Graphbrain is a great knowledge graph tool')

        # Verify it was set (check directly in database)
        with self.hg._conn.cursor() as cur:
            cur.execute('SELECT text_content FROM edges WHERE edge_key = %s', (str(edge),))
            result = cur.fetchone()
            self.assertEqual(result[0], 'Graphbrain is a great knowledge graph tool')

    def test_full_text_search(self):
        """Test PostgreSQL full-text search."""
        edge1 = self.hg.add('(is/Pd graphbrain/C great/C)')
        edge2 = self.hg.add('(is/Pd python/C awesome/C)')

        self.hg.set_text_content(edge1, 'Graphbrain is a great knowledge graph tool')
        self.hg.set_text_content(edge2, 'Python is an awesome programming language')

        # Search for knowledge graph
        results = list(self.hg.full_text_search('knowledge graph'))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], edge1)

        # Search for programming
        results = list(self.hg.full_text_search('programming'))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], edge2)

    def test_count_edges(self):
        """Test edge counting."""
        self.assertEqual(self.hg.count_edges(), 0)

        self.hg.add('(is/Pd graphbrain/C great/C)')
        # This adds the main edge plus child edges
        count = self.hg.count_edges()
        self.assertGreater(count, 0)

    def test_connection_pooling(self):
        """Test that connection pooling works."""
        from graphbrain import hgraph
        from graphbrain.memory.postgres import PostgreSQL

        # The pool is already created from setUp, just verify it exists
        self.assertIn(PG_URI, PostgreSQL._pools)

        # Create another connection - should reuse existing pool
        pool_count_before = len(PostgreSQL._pools)
        hg2 = hgraph(PG_URI)

        # Pool count should be the same (reusing existing pool)
        self.assertEqual(len(PostgreSQL._pools), pool_count_before)

        hg2.close()


if __name__ == '__main__':
    unittest.main()
