"""PostgreSQL hypergraph storage backend.

This module provides a PostgreSQL-based storage backend for graphbrain,
replacing SQLite/LevelDB with a unified PostgreSQL database that can also
store embeddings (via pgvector) and support full-text search (via tsvector).

Usage:
    from graphbrain import hgraph

    # Connect to PostgreSQL
    hg = hgraph('postgresql://user:pass@localhost/graphbrain')

    # Or with explicit connection string
    hg = hgraph('postgres://localhost/graphbrain')

    # With custom embedding configuration
    from graphbrain.embeddings.config import EmbeddingConfig
    config = EmbeddingConfig(index_type="hnsw", dimensions=768)
    hg = hgraph('postgresql://localhost/graphbrain', embedding_config=config)
"""

import json
import logging
from typing import Iterator, Optional, TYPE_CHECKING

from graphbrain.exceptions import EdgeParseError, TransactionError, StorageError
from graphbrain.hyperedge import hedge
from graphbrain.memory.keyvalue import KeyValue
from graphbrain.memory.permutations import str_plus_1

if TYPE_CHECKING:
    from graphbrain.embeddings.config import EmbeddingConfig

try:
    import psycopg2
    import psycopg2.extras
    from psycopg2.pool import SimpleConnectionPool
    _PSYCOPG2_AVAILABLE = True
except ImportError:
    _PSYCOPG2_AVAILABLE = False
    # Define stub for type checking when psycopg2 not available
    SimpleConnectionPool = None

logger = logging.getLogger(__name__)


# SQL statements for schema creation
_CREATE_EXTENSIONS = """
-- Enable pgvector if available (optional)
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS vector;
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'pgvector extension not available, embeddings disabled';
END $$;

-- Enable pg_trgm for fuzzy text search (optional)
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS pg_trgm;
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'pg_trgm extension not available';
END $$;
"""

_CREATE_EDGES_TABLE = """
CREATE TABLE IF NOT EXISTS edges (
    edge_key TEXT PRIMARY KEY,
    attributes JSONB NOT NULL DEFAULT '{}',
    text_content TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for prefix-based range queries on edge keys
CREATE INDEX IF NOT EXISTS idx_edges_key_prefix
    ON edges (edge_key text_pattern_ops);

-- Index for full-text search on text_content (if populated)
CREATE INDEX IF NOT EXISTS idx_edges_text_tsv
    ON edges USING gin(to_tsvector('english', COALESCE(text_content, '')));

-- Index on JSONB attributes for common queries
CREATE INDEX IF NOT EXISTS idx_edges_attrs_p
    ON edges ((attributes->>'p'));
"""

_CREATE_PERMUTATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS permutations (
    perm_key TEXT PRIMARY KEY,
    edge_key TEXT NOT NULL
);

-- Index for prefix-based range queries on permutation keys
CREATE INDEX IF NOT EXISTS idx_perms_key_prefix
    ON permutations (perm_key text_pattern_ops);

-- Index for finding permutations by edge
CREATE INDEX IF NOT EXISTS idx_perms_edge
    ON permutations (edge_key);
"""

# Template for adding embedding column - dimensions filled in at runtime
_ADD_EMBEDDING_COLUMN_TEMPLATE = """
DO $$
BEGIN
    ALTER TABLE edges ADD COLUMN IF NOT EXISTS embedding vector({dimensions});
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Could not add embedding column (pgvector may not be available)';
END $$;
"""


def _get_pgvector_version(conn) -> Optional[tuple]:
    """Get the installed pgvector version as a tuple.

    Returns:
        Version tuple (major, minor, patch) or None if not available.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT extversion FROM pg_extension WHERE extname = 'vector'
            """)
            row = cur.fetchone()
            if row:
                version_str = row[0]
                parts = version_str.split('.')
                return tuple(int(p) for p in parts[:3])
    except Exception:
        pass
    return None


def _get_embedding_index_sql(
    dimensions: int,
    index_type: str,
    distance_metric: str,
    hnsw_m: int = 16,
    hnsw_ef_construction: int = 64,
    ivfflat_lists: int = 100,
) -> str:
    """Generate SQL for creating a vector index.

    Args:
        dimensions: Vector dimensions
        index_type: 'hnsw', 'ivfflat', or 'none'
        distance_metric: 'cosine', 'l2', or 'inner_product'
        hnsw_m: HNSW max connections per layer
        hnsw_ef_construction: HNSW construction exploration factor
        ivfflat_lists: Number of IVFFlat clusters

    Returns:
        SQL CREATE INDEX statement, or empty string if index_type is 'none'.
    """
    if index_type == "none":
        return ""

    ops_map = {
        "cosine": "vector_cosine_ops",
        "l2": "vector_l2_ops",
        "inner_product": "vector_ip_ops",
    }
    ops_class = ops_map.get(distance_metric, "vector_cosine_ops")

    if index_type == "hnsw":
        return f"""
DO $$
BEGIN
    CREATE INDEX IF NOT EXISTS idx_edges_embedding_hnsw
        ON edges USING hnsw (embedding {ops_class})
        WITH (m = {hnsw_m}, ef_construction = {hnsw_ef_construction});
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Could not create HNSW index: %', SQLERRM;
END $$;
"""
    elif index_type == "ivfflat":
        return f"""
DO $$
BEGIN
    CREATE INDEX IF NOT EXISTS idx_edges_embedding_ivfflat
        ON edges USING ivfflat (embedding {ops_class})
        WITH (lists = {ivfflat_lists});
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Could not create IVFFlat index: %', SQLERRM;
END $$;
"""
    else:
        return ""


class PostgreSQL(KeyValue):
    """PostgreSQL hypergraph storage backend.

    Implements the KeyValue interface using PostgreSQL for storage.
    Supports transactions, connection pooling, and optionally embeddings
    via pgvector and full-text search via tsvector.

    Connection string formats:
        postgresql://user:pass@host:port/database
        postgres://user:pass@host:port/database
        postgresql://localhost/graphbrain

    Embedding configuration:
        Pass an EmbeddingConfig to customize vector dimensions and index type.
        HNSW indexes (default) require pgvector >= 0.5.0; falls back to IVFFlat.
    """

    # Class-level connection pool (shared across instances for same connection)
    # Type annotation uses string to avoid NameError when psycopg2 unavailable
    _pools: dict = {}

    def __init__(
        self,
        locator_string: str,
        pool_size: int = 5,
        embedding_config: Optional["EmbeddingConfig"] = None,
    ):
        """Initialize PostgreSQL backend.

        Args:
            locator_string: PostgreSQL connection string (postgresql://... or postgres://...)
            pool_size: Size of connection pool (default 5)
            embedding_config: Optional embedding configuration for pgvector setup.
                             If not provided, uses default 768-dim with HNSW index.
        """
        if not _PSYCOPG2_AVAILABLE:
            raise StorageError(
                "psycopg2 not available. Install with: pip install psycopg2-binary",
                operation="connect"
            )

        super().__init__(locator_string)

        self.pool_size = pool_size
        self._conn: Optional[psycopg2.extensions.connection] = None
        self._cur: Optional[psycopg2.extensions.cursor] = None
        self._transaction_depth = 0
        self._savepoint_counter = 0

        # Store embedding config (create default if not provided)
        if embedding_config is None:
            from graphbrain.embeddings.config import EmbeddingConfig
            embedding_config = EmbeddingConfig()
        self.embedding_config = embedding_config

        # Track pgvector version for index type fallback
        self._pgvector_version: Optional[tuple] = None

        # Initialize connection and schema
        self._init_connection()
        self._init_schema()

    def _init_connection(self):
        """Initialize connection pool or get connection from pool."""
        # Normalize connection string
        conn_str = self.locator_string
        if conn_str.startswith('postgres://'):
            conn_str = 'postgresql://' + conn_str[len('postgres://'):]

        # Create or get pool
        if conn_str not in PostgreSQL._pools:
            try:
                PostgreSQL._pools[conn_str] = SimpleConnectionPool(
                    1, self.pool_size, conn_str
                )
                logger.info(f"Created PostgreSQL connection pool for {conn_str}")
            except psycopg2.Error as e:
                raise StorageError(
                    f"Failed to connect to PostgreSQL: {e}",
                    operation="connect"
                )

        # Get connection from pool
        self._conn = PostgreSQL._pools[conn_str].getconn()
        self._conn.autocommit = True  # We manage transactions explicitly

    def _init_schema(self):
        """Initialize database schema if not exists."""
        try:
            with self._conn.cursor() as cur:
                # Create extensions (optional, won't fail if not available)
                cur.execute(_CREATE_EXTENSIONS)

                # Create core tables
                cur.execute(_CREATE_EDGES_TABLE)
                cur.execute(_CREATE_PERMUTATIONS_TABLE)

                # Check pgvector version for index type selection
                self._pgvector_version = _get_pgvector_version(self._conn)

                # Determine index type (fall back to ivfflat if HNSW not supported)
                index_type = self.embedding_config.index_type
                if index_type == "hnsw" and self._pgvector_version:
                    # HNSW requires pgvector >= 0.5.0
                    if self._pgvector_version < (0, 5, 0):
                        logger.warning(
                            f"pgvector {'.'.join(map(str, self._pgvector_version))} "
                            "doesn't support HNSW; falling back to IVFFlat"
                        )
                        index_type = "ivfflat"

                # Try to add embedding column (optional)
                add_column_sql = _ADD_EMBEDDING_COLUMN_TEMPLATE.format(
                    dimensions=self.embedding_config.dimensions
                )
                cur.execute(add_column_sql)

                # Create vector index
                index_sql = _get_embedding_index_sql(
                    dimensions=self.embedding_config.dimensions,
                    index_type=index_type,
                    distance_metric=self.embedding_config.distance_metric,
                    hnsw_m=self.embedding_config.hnsw_m,
                    hnsw_ef_construction=self.embedding_config.hnsw_ef_construction,
                    ivfflat_lists=self.embedding_config.ivfflat_lists,
                )
                if index_sql:
                    cur.execute(index_sql)

                self._conn.commit()
                logger.debug(
                    f"PostgreSQL schema initialized (pgvector: "
                    f"{'.'.join(map(str, self._pgvector_version)) if self._pgvector_version else 'not available'}, "
                    f"index: {index_type})"
                )
        except psycopg2.Error as e:
            self._conn.rollback()
            raise StorageError(
                f"Failed to initialize schema: {e}",
                operation="schema_init"
            )

    # ===================================
    # Implementation of interface methods
    # ===================================

    def close(self):
        """Close connection and return to pool."""
        if self._cur:
            self._cur.close()
            self._cur = None
        if self._conn:
            # Normalize connection string for pool lookup
            conn_str = self.locator_string
            if conn_str.startswith('postgres://'):
                conn_str = 'postgresql://' + conn_str[len('postgres://'):]

            if conn_str in PostgreSQL._pools:
                PostgreSQL._pools[conn_str].putconn(self._conn)
            else:
                self._conn.close()
            self._conn = None

    def destroy(self):
        """Erase all data from the hypergraph."""
        with self._conn.cursor() as cur:
            cur.execute('TRUNCATE TABLE permutations')
            cur.execute('TRUNCATE TABLE edges CASCADE')
            self._conn.commit()

    def all(self) -> Iterator:
        """Yield all edges in the hypergraph."""
        # Use regular cursor (not server-side) to avoid transaction requirement
        with self._conn.cursor() as cur:
            cur.execute('SELECT edge_key FROM edges')
            for row in cur:
                try:
                    yield hedge(row[0])
                except EdgeParseError:
                    logger.warning(f"Skipping invalid edge key: {row[0]!r}")

    def all_attributes(self) -> Iterator:
        """Yield (edge, attributes) tuples for all edges."""
        # Use regular cursor (not server-side) to avoid transaction requirement
        with self._conn.cursor() as cur:
            cur.execute('SELECT edge_key, attributes FROM edges')
            for row in cur:
                try:
                    edge = hedge(row[0])
                    attrs = row[1] if isinstance(row[1], dict) else json.loads(row[1])
                    yield edge, attrs
                except EdgeParseError:
                    logger.warning(f"Skipping invalid edge key: {row[0]!r}")

    def begin_transaction(self):
        """Begin a transaction (supports nesting via savepoints)."""
        if self.batch_mode:
            return

        if self._transaction_depth == 0:
            # Start actual transaction
            self._conn.autocommit = False
            self._cur = self._conn.cursor()
            self._savepoint_counter = 0
        else:
            # Create savepoint for nested transaction
            self._savepoint_counter += 1
            savepoint_name = f"sp_{self._savepoint_counter}"
            self._cur.execute(f"SAVEPOINT {savepoint_name}")

        self._transaction_depth += 1

    def end_transaction(self):
        """End (commit) the current transaction."""
        if self.batch_mode:
            return

        if self._transaction_depth == 0:
            raise TransactionError(
                "Cannot end transaction: no transaction is in progress",
                operation="commit"
            )

        self._transaction_depth -= 1

        if self._transaction_depth == 0:
            # Commit actual transaction
            self._conn.commit()
            self._conn.autocommit = True
            if self._cur:
                self._cur.close()
                self._cur = None
        else:
            # Release savepoint
            savepoint_name = f"sp_{self._savepoint_counter}"
            self._cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
            self._savepoint_counter -= 1

    def rollback(self):
        """Rollback the current transaction."""
        if self._transaction_depth == 0:
            raise TransactionError(
                "Cannot rollback: no transaction is in progress",
                operation="rollback"
            )

        if self._transaction_depth == 1:
            # Rollback entire transaction
            self._conn.rollback()
            self._conn.autocommit = True
            if self._cur:
                self._cur.close()
                self._cur = None
        else:
            # Rollback to savepoint
            savepoint_name = f"sp_{self._savepoint_counter}"
            self._cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
            self._savepoint_counter -= 1

        self._transaction_depth = 0 if self._transaction_depth == 1 else self._transaction_depth - 1

    def in_transaction(self) -> bool:
        """Check if a transaction is currently in progress."""
        return self._transaction_depth > 0

    # ==========================================
    # Implementation of private abstract methods
    # ==========================================

    def _edge2key(self, edge) -> str:
        """Convert edge to database key."""
        return edge.to_str()

    def _exists_key(self, key: str) -> bool:
        """Check if key exists in edges table."""
        cur = self._cur if self._cur else self._conn.cursor()
        cur.execute('SELECT 1 FROM edges WHERE edge_key = %s', (key,))
        result = cur.fetchone() is not None
        if not self._cur:
            cur.close()
        return result

    def _add_key(self, key: str, attributes: dict):
        """Add or update edge with attributes."""
        if not self._cur:
            raise TransactionError(
                "Cannot add key outside of transaction",
                operation="add"
            )

        # Use JSONB directly - PostgreSQL handles dict conversion
        self._cur.execute(
            """
            INSERT INTO edges (edge_key, attributes, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (edge_key) DO UPDATE SET
                attributes = EXCLUDED.attributes,
                updated_at = NOW()
            """,
            (key, json.dumps(attributes))
        )

    def _attribute_key(self, key: str) -> Optional[dict]:
        """Get attributes for an edge key."""
        cur = self._cur if self._cur else self._conn.cursor()
        cur.execute('SELECT attributes FROM edges WHERE edge_key = %s', (key,))
        row = cur.fetchone()
        if not self._cur:
            cur.close()

        if row:
            return row[0] if isinstance(row[0], dict) else json.loads(row[0])
        return None

    def _write_edge_permutation(self, perm: str):
        """Write a permutation to the permutations table."""
        if not self._cur:
            raise TransactionError(
                "Cannot write permutation outside of transaction",
                operation="write_permutation"
            )

        # Extract edge_key from permutation (everything before the last space + number)
        parts = perm.rsplit(' |', 1)
        edge_key = parts[0] if len(parts) > 1 else perm

        self._cur.execute(
            """
            INSERT INTO permutations (perm_key, edge_key)
            VALUES (%s, %s)
            ON CONFLICT (perm_key) DO NOTHING
            """,
            (perm, edge_key)
        )

    def _remove_edge_permutation(self, perm: str):
        """Remove a permutation from the permutations table."""
        if not self._cur:
            raise TransactionError(
                "Cannot remove permutation outside of transaction",
                operation="remove_permutation"
            )

        self._cur.execute('DELETE FROM permutations WHERE perm_key = %s', (perm,))

    def _remove_key(self, key: str):
        """Remove an edge from the edges table."""
        if not self._cur:
            raise TransactionError(
                "Cannot remove key outside of transaction",
                operation="remove"
            )

        self._cur.execute('DELETE FROM edges WHERE edge_key = %s', (key,))

    def _permutations_with_prefix(self, prefix: str) -> Iterator[str]:
        """Find all permutations with given prefix (range query)."""
        end_str = str_plus_1(prefix)
        # Use regular cursor to avoid transaction requirement
        # Use COLLATE "C" for binary ordering (same as SQLite)
        with self._conn.cursor() as cur:
            cur.execute(
                '''SELECT perm_key FROM permutations
                   WHERE perm_key COLLATE "C" >= %s AND perm_key COLLATE "C" < %s''',
                (prefix, end_str)
            )
            for row in cur:
                yield row[0]

    def _edges_with_prefix(self, prefix: str) -> Iterator:
        """Find all edges with given prefix (range query)."""
        end_str = str_plus_1(prefix)
        # Use regular cursor to avoid transaction requirement
        # Use COLLATE "C" for binary ordering (same as SQLite)
        with self._conn.cursor() as cur:
            cur.execute(
                '''SELECT edge_key FROM edges
                   WHERE edge_key COLLATE "C" >= %s AND edge_key COLLATE "C" < %s''',
                (prefix, end_str)
            )
            for row in cur:
                yield hedge(row[0])

    # ===================================
    # PostgreSQL-specific methods
    # ===================================

    def set_text_content(self, edge, text: str):
        """Set the text content for an edge (enables full-text search)."""
        key = self._edge2key(hedge(edge))
        with self._conn.cursor() as cur:
            cur.execute(
                'UPDATE edges SET text_content = %s, updated_at = NOW() WHERE edge_key = %s',
                (text, key)
            )
            self._conn.commit()

    def get_text_content(self, edge) -> Optional[str]:
        """Get text content for an edge."""
        key = self._edge2key(hedge(edge))
        with self._conn.cursor() as cur:
            cur.execute('SELECT text_content FROM edges WHERE edge_key = %s', (key,))
            row = cur.fetchone()
            return row[0] if row else None

    def full_text_search(
        self,
        query: str,
        limit: int = 100,
        query_type: str = 'websearch',
        return_snippets: bool = False,
        headline_options: str = 'StartSel=<mark>, StopSel=</mark>, MaxWords=35, MinWords=15',
    ) -> Iterator:
        """Search edges by text content using PostgreSQL full-text search.

        Supports multiple query types:
        - 'websearch': Web-style search with AND/OR/- operators (default)
        - 'plain': Simple word matching
        - 'phrase': Exact phrase matching

        Args:
            query: Search query
            limit: Maximum results
            query_type: Query parser type ('websearch', 'plain', 'phrase')
            return_snippets: If True, return highlighted snippets
            headline_options: PostgreSQL ts_headline options for snippets
        """
        # Select query function based on type
        if query_type == 'phrase':
            query_func = 'phraseto_tsquery'
        elif query_type == 'websearch':
            query_func = 'websearch_to_tsquery'
        else:  # 'plain' or default
            query_func = 'plainto_tsquery'

        with self._conn.cursor() as cur:
            try:
                if return_snippets:
                    cur.execute(
                        f"""
                        SELECT edge_key,
                               ts_rank_cd(to_tsvector('english', COALESCE(text_content, '')),
                                          {query_func}('english', %s)) as rank,
                               ts_headline('english', COALESCE(text_content, ''),
                                          {query_func}('english', %s), %s) as snippet
                        FROM edges
                        WHERE to_tsvector('english', COALESCE(text_content, '')) @@ {query_func}('english', %s)
                        ORDER BY rank DESC
                        LIMIT %s
                        """,
                        (query, query, headline_options, query, limit)
                    )
                    for row in cur:
                        try:
                            yield hedge(row[0]), row[1], row[2]
                        except EdgeParseError:
                            logger.warning(f"Skipping invalid edge key: {row[0]!r}")
                else:
                    cur.execute(
                        f"""
                        SELECT edge_key, ts_rank_cd(to_tsvector('english', COALESCE(text_content, '')),
                                                    {query_func}('english', %s)) as rank
                        FROM edges
                        WHERE to_tsvector('english', COALESCE(text_content, '')) @@ {query_func}('english', %s)
                        ORDER BY rank DESC
                        LIMIT %s
                        """,
                        (query, query, limit)
                    )
                    for row in cur:
                        try:
                            yield hedge(row[0]), row[1]
                        except EdgeParseError:
                            logger.warning(f"Skipping invalid edge key: {row[0]!r}")
            except psycopg2.Error as e:
                # Fallback to plain query if websearch not supported (pg < 11)
                if query_type == 'websearch':
                    logger.warning(f"websearch_to_tsquery not available, falling back to plain: {e}")
                    yield from self.full_text_search(query, limit, query_type='plain', return_snippets=return_snippets)
                else:
                    raise

    def phrase_search(self, phrase: str, limit: int = 100, return_snippets: bool = True) -> Iterator:
        """Search for an exact phrase."""
        yield from self.full_text_search(phrase, limit, query_type='phrase', return_snippets=return_snippets)

    def fts_stats(self) -> dict:
        """Get full-text search statistics."""
        stats = {"fts_available": True, "indexed_edges": 0}
        with self._conn.cursor() as cur:
            try:
                cur.execute("SELECT COUNT(*) FROM edges WHERE text_content IS NOT NULL")
                stats["indexed_edges"] = cur.fetchone()[0]
            except Exception:
                pass
        return stats

    def set_embedding(self, edge, embedding: list[float]):
        """Set the embedding vector for an edge (requires pgvector)."""
        key = self._edge2key(hedge(edge))
        with self._conn.cursor() as cur:
            try:
                cur.execute(
                    'UPDATE edges SET embedding = %s, updated_at = NOW() WHERE edge_key = %s',
                    (embedding, key)
                )
                self._conn.commit()
            except psycopg2.Error as e:
                self._conn.rollback()
                logger.warning(f"Could not set embedding (pgvector may not be available): {e}")

    def similar_edges(self, embedding: list[float], limit: int = 10) -> Iterator:
        """Find edges similar to given embedding using configured distance metric."""
        distance_op = self.embedding_config.get_distance_operator()
        metric = self.embedding_config.distance_metric

        with self._conn.cursor() as cur:
            try:
                # Build query based on distance metric
                if metric == "cosine":
                    # Cosine distance: convert to similarity (1 - distance)
                    cur.execute(
                        f"""
                        SELECT edge_key, 1 - (embedding {distance_op} %s::vector) as similarity
                        FROM edges
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding {distance_op} %s::vector
                        LIMIT %s
                        """,
                        (embedding, embedding, limit)
                    )
                elif metric == "l2":
                    # L2 distance: use negative distance for ranking
                    cur.execute(
                        f"""
                        SELECT edge_key, -(embedding {distance_op} %s::vector) as similarity
                        FROM edges
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding {distance_op} %s::vector
                        LIMIT %s
                        """,
                        (embedding, embedding, limit)
                    )
                elif metric == "inner_product":
                    # Inner product: negative inner product operator, negate for similarity
                    cur.execute(
                        f"""
                        SELECT edge_key, -(embedding {distance_op} %s::vector) as similarity
                        FROM edges
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding {distance_op} %s::vector
                        LIMIT %s
                        """,
                        (embedding, embedding, limit)
                    )
                else:
                    # Default to cosine
                    cur.execute(
                        """
                        SELECT edge_key, 1 - (embedding <=> %s::vector) as similarity
                        FROM edges
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (embedding, embedding, limit)
                    )

                for row in cur:
                    try:
                        yield hedge(row[0]), row[1]
                    except EdgeParseError:
                        logger.warning(f"Skipping invalid edge key: {row[0]!r}")
            except psycopg2.Error as e:
                logger.warning(f"Similarity search failed (pgvector may not be available): {e}")

    def count_edges(self) -> int:
        """Return total number of edges in the hypergraph."""
        with self._conn.cursor() as cur:
            cur.execute('SELECT COUNT(*) FROM edges')
            return cur.fetchone()[0]

    def vacuum(self):
        """Run VACUUM ANALYZE on tables for performance optimization."""
        # VACUUM cannot run inside a transaction
        old_isolation = self._conn.isolation_level
        self._conn.set_isolation_level(0)
        with self._conn.cursor() as cur:
            cur.execute('VACUUM ANALYZE edges')
            cur.execute('VACUUM ANALYZE permutations')
        self._conn.set_isolation_level(old_isolation)
        logger.info("VACUUM ANALYZE completed")
