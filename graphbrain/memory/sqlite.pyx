import base64
import json
import logging

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


class SQLite(KeyValue):
    """Implements SQLite hypergraph storage."""

    def __init__(self, locator_string):
        super().__init__(locator_string)

        self.conn = connect(self.locator_string, isolation_level=None)
        self.cur = None
        self._transaction_depth = 0

        # self.conn.execute('PRAGMA synchronous = OFF')
        # self.conn.execute('PRAGMA journal_mode = MEMORY')

        self.conn.execute('CREATE TABLE IF NOT EXISTS v (key TEXT PRIMARY KEY, value TEXT)')
        self.conn.execute('CREATE TABLE IF NOT EXISTS p (key TEXT PRIMARY KEY)')
        # Add indexes for prefix queries (range scans)
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_v_key ON v(key)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_p_key ON p(key)')

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
        for key, _ in cur.execute('SELECT * FROM v'):
            try:
                yield hedge(key)
            except EdgeParseError:
                logger.warning(f"Skipping invalid edge key in database: {key!r}")

    def all_attributes(self):
        cur = self.conn.cursor()
        for key, value in cur.execute('SELECT * FROM v'):
            try:
                edge = hedge(key)
                attributes = _decode_attributes(value)
                yield edge, attributes
            except EdgeParseError:
                logger.warning(f"Skipping invalid edge key in database: {key!r}")

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
        for key, _ in cur.execute('SELECT * FROM v WHERE key = ?', (key, )):
            return True
        return False

    def _add_key(self, key, attributes):
        """Adds the given edge, given its key."""
        value = _encode_attributes(attributes)
        self.cur.execute('INSERT OR REPLACE INTO v (key, value) VALUES(?, ?)', (key, value))

    def _attribute_key(self, key):
        cur = self.conn.cursor()
        for key, value in cur.execute('SELECT * FROM v WHERE key = ?', (key,)):
            return _decode_attributes(value)
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
        for key, _ in cur.execute('SELECT * FROM v WHERE key >= ? AND key < ?', (prefix, end_str)):
            yield hedge(key)
