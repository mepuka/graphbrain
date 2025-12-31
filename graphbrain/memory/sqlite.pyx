import base64
import json
import logging
from typing import Iterator, Optional, Tuple

import msgpack
from sqlite3 import connect

from graphbrain.exceptions import EdgeParseError, TransactionError
from graphbrain.hyperedge import hedge
from graphbrain.memory.keyvalue import KeyValue
from graphbrain.memory.permutations import str_plus_1


logger = logging.getLogger(__name__)

# Prefix to identify msgpack-encoded data in SQLite TEXT column
_MSGPACK_PREFIX = 'mp:'


def _encode_attributes(attributes):
    """Encode attributes using msgpack with base64 for SQLite TEXT storage."""
    packed = msgpack.packb(attributes, use_bin_type=True)
    return _MSGPACK_PREFIX + base64.b64encode(packed).decode('ascii')


def _decode_attributes(value):
    """Decode attributes with backward compatibility for legacy JSON data."""
    if value.startswith(_MSGPACK_PREFIX):
        # Modern msgpack format (base64-encoded)
        packed = base64.b64decode(value[len(_MSGPACK_PREFIX):])
        return msgpack.unpackb(packed, raw=False)
    # Legacy JSON format
    return json.loads(value)


# FTS5 schema for integrated full-text search
# Using contentless FTS5 for simplicity - stores text directly in FTS table
_FTS5_SCHEMA = """
-- FTS5 virtual table for full-text search (contentless mode)
CREATE VIRTUAL TABLE IF NOT EXISTS v_fts USING fts5(
    key,
    text_content,
    tokenize='porter unicode61'
);
"""


class SQLite(KeyValue):
    """Implements SQLite hypergraph storage with integrated FTS5 search."""

    def __init__(self, locator_string):
        super().__init__(locator_string)

        self.conn = connect(self.locator_string, isolation_level=None)
        self.cur = None
        self._transaction_depth = 0
        self._fts_available = False

        # Performance pragmas - WAL mode is 10-100x faster for writes
        # and allows concurrent readers during writes
        self.conn.execute('PRAGMA journal_mode = WAL')
        self.conn.execute('PRAGMA synchronous = NORMAL')  # Safe with WAL
        self.conn.execute('PRAGMA cache_size = -64000')  # 64MB cache
        self.conn.execute('PRAGMA temp_store = MEMORY')
        self.conn.execute('PRAGMA mmap_size = 268435456')  # 256MB memory-mapped I/O

        # Create main tables with text_content column for FTS
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS v (
                key TEXT PRIMARY KEY,
                value TEXT,
                text_content TEXT
            )
        ''')
        self.conn.execute('CREATE TABLE IF NOT EXISTS p (key TEXT PRIMARY KEY)')

        # Add text_content column if missing (migration for existing DBs)
        self._migrate_add_text_content_column()

        # Add indexes for prefix queries (range scans)
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_v_key ON v(key)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_p_key ON p(key)')

        # Initialize FTS5 virtual table
        self._init_fts5()

    def _migrate_add_text_content_column(self):
        """Add text_content column to existing databases."""
        try:
            cur = self.conn.execute("PRAGMA table_info(v)")
            columns = {row[1] for row in cur.fetchall()}
            if 'text_content' not in columns:
                self.conn.execute('ALTER TABLE v ADD COLUMN text_content TEXT')
                logger.info("Migrated v table: added text_content column")
        except Exception as e:
            logger.warning(f"Could not check/add text_content column: {e}")

    def _init_fts5(self):
        """Initialize FTS5 virtual table for full-text search."""
        try:
            self.conn.executescript(_FTS5_SCHEMA)
            self._fts_available = True
            logger.debug("FTS5 virtual table initialized")
        except Exception as e:
            self._fts_available = False
            logger.warning(f"FTS5 not available: {e}")

    # ===================================
    # Full-text search methods
    # ===================================

    @property
    def fts_available(self) -> bool:
        """Whether FTS5 is available for full-text search."""
        return self._fts_available

    def set_text_content(self, edge, text: str):
        """Set text content for an edge, enabling full-text search.

        Args:
            edge: The hyperedge
            text: Text content to associate with the edge
        """
        key = self._edge2key(edge)
        cur = self.conn.cursor()

        # Update the main table
        cur.execute('UPDATE v SET text_content = ? WHERE key = ?', (text, key))
        if cur.rowcount == 0:
            logger.warning(f"set_text_content: edge not found: {key}")
            return

        if self._fts_available:
            # Sync with FTS5 table (delete old entry, insert new)
            cur.execute('DELETE FROM v_fts WHERE key = ?', (key,))
            cur.execute('INSERT INTO v_fts (key, text_content) VALUES (?, ?)', (key, text))

    def get_text_content(self, edge) -> Optional[str]:
        """Get text content for an edge.

        Args:
            edge: The hyperedge

        Returns:
            Text content or None if not set
        """
        key = self._edge2key(edge)
        cur = self.conn.cursor()
        cur.execute('SELECT text_content FROM v WHERE key = ?', (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def full_text_search(
        self,
        query: str,
        limit: int = 100,
        return_snippets: bool = False,
        snippet_size: int = 64,
        highlight_start: str = '<b>',
        highlight_end: str = '</b>',
    ) -> Iterator[Tuple]:
        """Search edges by text content using FTS5 BM25 ranking.

        Supports FTS5 query syntax:
        - Simple words: "hello world" matches edges containing both words
        - Phrase search: '"hello world"' matches exact phrase
        - Prefix search: "hel*" matches words starting with "hel"
        - Boolean: "hello OR world", "hello NOT world"
        - Column filter: "text_content:hello"

        Args:
            query: FTS5 query string
            limit: Maximum results to return
            return_snippets: If True, return highlighted snippets
            snippet_size: Approximate snippet size in tokens
            highlight_start: HTML/marker for highlight start
            highlight_end: HTML/marker for highlight end

        Yields:
            Tuples of (edge, score, text_or_snippet)
        """
        if not self._fts_available:
            logger.warning("full_text_search: FTS5 not available")
            return

        cur = self.conn.cursor()

        try:
            if return_snippets:
                # Use snippet() function for highlighted excerpts
                cur.execute(
                    '''
                    SELECT key, bm25(v_fts) as score,
                           snippet(v_fts, 1, ?, ?, '...', ?) as snippet
                    FROM v_fts
                    WHERE v_fts MATCH ?
                    ORDER BY score
                    LIMIT ?
                    ''',
                    (highlight_start, highlight_end, snippet_size, query, limit)
                )
            else:
                cur.execute(
                    '''
                    SELECT key, bm25(v_fts) as score, text_content
                    FROM v_fts
                    WHERE v_fts MATCH ?
                    ORDER BY score
                    LIMIT ?
                    ''',
                    (query, limit)
                )

            for row in cur:
                try:
                    edge = hedge(row[0])
                    # BM25 returns negative scores (lower is better), convert to positive
                    score = -row[1] if row[1] else 0.0
                    text = row[2]
                    yield (edge, score, text)
                except EdgeParseError:
                    logger.warning(f"Skipping invalid edge key: {row[0]!r}")

        except Exception as e:
            logger.error(f"full_text_search error: {e}")

    def highlight_search(
        self,
        query: str,
        limit: int = 100,
        highlight_start: str = '<b>',
        highlight_end: str = '</b>',
    ) -> Iterator[Tuple]:
        """Search with full text highlighted (not just snippets).

        Args:
            query: FTS5 query string
            limit: Maximum results
            highlight_start: Marker for match start
            highlight_end: Marker for match end

        Yields:
            Tuples of (edge, score, highlighted_text)
        """
        if not self._fts_available:
            return

        cur = self.conn.cursor()

        try:
            cur.execute(
                '''
                SELECT key, bm25(v_fts) as score,
                       highlight(v_fts, 1, ?, ?) as highlighted
                FROM v_fts
                WHERE v_fts MATCH ?
                ORDER BY score
                LIMIT ?
                ''',
                (highlight_start, highlight_end, query, limit)
            )

            for row in cur:
                try:
                    edge = hedge(row[0])
                    score = -row[1] if row[1] else 0.0
                    highlighted = row[2]
                    yield (edge, score, highlighted)
                except EdgeParseError:
                    pass

        except Exception as e:
            logger.error(f"highlight_search error: {e}")

    def rebuild_fts_index(self):
        """Rebuild the FTS5 index from the main table.

        Call this after bulk updates or to fix index corruption.
        """
        if not self._fts_available:
            logger.warning("rebuild_fts_index: FTS5 not available")
            return

        try:
            # Delete and repopulate the FTS index
            self.conn.execute("DELETE FROM v_fts")
            self.conn.execute('''
                INSERT INTO v_fts(key, text_content)
                SELECT key, text_content FROM v WHERE text_content IS NOT NULL
            ''')
            self.conn.commit()
            logger.info("FTS5 index rebuilt successfully")
        except Exception as e:
            logger.error(f"rebuild_fts_index error: {e}")

    def fts_stats(self) -> dict:
        """Get FTS5 index statistics.

        Returns:
            Dict with index stats including document count
        """
        stats = {
            "fts_available": self._fts_available,
            "indexed_edges": 0,
        }

        if not self._fts_available:
            return stats

        try:
            cur = self.conn.execute("SELECT COUNT(*) FROM v WHERE text_content IS NOT NULL")
            stats["indexed_edges"] = cur.fetchone()[0]
        except Exception:
            pass

        return stats

    # ===================================
    # Implementation of interface methods
    # ===================================

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def destroy(self):
        cur = self.conn.cursor()
        cur.execute('DELETE FROM v')
        cur.execute('DELETE FROM p')

    def all(self):
        cur = self.conn.cursor()
        for row in cur.execute('SELECT key FROM v'):
            try:
                yield hedge(row[0])
            except EdgeParseError:
                logger.warning(f"Skipping invalid edge key in database: {row[0]!r}")

    def all_attributes(self):
        cur = self.conn.cursor()
        for row in cur.execute('SELECT key, value FROM v'):
            try:
                edge = hedge(row[0])
                attributes = _decode_attributes(row[1])
                yield edge, attributes
            except EdgeParseError:
                logger.warning(f"Skipping invalid edge key in database: {row[0]!r}")

    def begin_transaction(self):
        """Begin a transaction.

        Supports nested calls - only the outermost begin/end pair actually
        starts and commits the transaction. This allows internal methods
        to wrap their operations in transactions without conflicting with
        user-level transactions.
        """
        if self.batch_mode:
            return
        if self._transaction_depth == 0:
            self.cur = self.conn.cursor()
            self.cur.execute('BEGIN TRANSACTION')
        self._transaction_depth += 1

    def end_transaction(self):
        """End (commit) the current transaction.

        Raises:
            TransactionError: If no transaction is in progress.
        """
        if self.batch_mode:
            return
        if self._transaction_depth == 0:
            raise TransactionError(
                "Cannot end transaction: no transaction is in progress",
                operation="commit"
            )
        self._transaction_depth -= 1
        if self._transaction_depth == 0:
            self.conn.commit()
            self.cur = None

    def rollback(self):
        """Rollback the current transaction, discarding all changes.

        Raises:
            TransactionError: If no transaction is in progress.
        """
        if self._transaction_depth == 0:
            raise TransactionError(
                "Cannot rollback: no transaction is in progress",
                operation="rollback"
            )
        self.conn.rollback()
        self.cur = None
        self._transaction_depth = 0

    def in_transaction(self):
        """Check if a transaction is currently in progress.

        Returns:
            bool: True if a transaction is active, False otherwise.
        """
        return self._transaction_depth > 0

    # ==========================================
    # Implementation of private abstract methods
    # ==========================================

    def _edge2key(self, edge):
        return edge.to_str()

    def _exists_key(self, key):
        """Checks if the given key exists."""
        cur = self.conn.cursor()
        cur.execute('SELECT 1 FROM v WHERE key = ?', (key,))
        return cur.fetchone() is not None

    def _add_key(self, key, attributes):
        """Adds the given edge, given its key."""
        value = _encode_attributes(attributes)
        self.cur.execute('INSERT OR REPLACE INTO v (key, value) VALUES(?, ?)', (key, value))

    def _attribute_key(self, key):
        cur = self.conn.cursor()
        cur.execute('SELECT value FROM v WHERE key = ?', (key,))
        row = cur.fetchone()
        if row:
            return _decode_attributes(row[0])
        return None

    def _write_edge_permutation(self, perm):
        """Writes a given permutation."""
        self.cur.execute('INSERT OR IGNORE INTO p (key) VALUES(?)', (perm,))

    def _remove_edge_permutation(self, perm):
        """Removes a given permutation."""
        self.cur.execute('DELETE FROM p WHERE key = ?', (perm,))

    def _remove_key(self, key):
        """Removes an edge, given its key."""
        self.cur.execute('DELETE FROM v WHERE key = ?', (key,))

    def _permutations_with_prefix(self, prefix):
        end_str = str_plus_1(prefix)
        cur = self.conn.cursor()
        for row in cur.execute('SELECT * FROM p WHERE key >= ? AND key < ?', (prefix, end_str)):
            yield row[0]

    def _edges_with_prefix(self, prefix):
        end_str = str_plus_1(prefix)
        cur = self.conn.cursor()
        for row in cur.execute('SELECT key FROM v WHERE key >= ? AND key < ?', (prefix, end_str)):
            yield hedge(row[0])
