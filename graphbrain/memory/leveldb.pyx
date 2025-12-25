import json
import logging

import msgpack
import plyvel

from graphbrain.exceptions import EdgeParseError, TransactionError
from graphbrain.hyperedge import hedge
from graphbrain.memory.keyvalue import KeyValue
from graphbrain.memory.permutations import str_plus_1


logger = logging.getLogger(__name__)


def _encode_attributes(attributes):
    """Encode attributes using msgpack for better performance."""
    return msgpack.packb(attributes, use_bin_type=True)


def _decode_attributes(value):
    """Decode attributes with backward compatibility for legacy JSON data."""
    # Msgpack binary data typically starts with specific bytes
    # JSON data starts with '{' (0x7b) for objects
    if value and value[0:1] == b'{':
        # Legacy JSON format - decode for backward compatibility
        return json.loads(value.decode('utf-8'))
    # Modern msgpack format
    return msgpack.unpackb(value, raw=False)


class LevelDB(KeyValue):
    """Implements LevelDB hypergraph storage."""

    def __init__(self, locator_string):
        super().__init__(locator_string)
        self.db = plyvel.DB(self.locator_string, create_if_missing=True)
        self._write_batch = None
        self._transaction_depth = 0

    # ===================================
    # Implementation of interface methods
    # ===================================

    def close(self):
        self.db.close()

    def destroy(self):
        self.db.close()
        plyvel.destroy_db(self.locator_string)
        self.db = plyvel.DB(self.locator_string, create_if_missing=True)

    def all(self):
        start_str = 'v'
        end_str = str_plus_1(start_str)
        start_key = start_str.encode('utf-8')
        end_key = end_str.encode('utf-8')

        for key, value in self.db.iterator(start=start_key, stop=end_key):
            try:
                yield hedge(key.decode('utf-8')[1:])
            except EdgeParseError:
                logger.warning(f"Skipping invalid edge key in database: {key!r}")

    def all_attributes(self):
        start_str = 'v'
        end_str = str_plus_1(start_str)
        start_key = start_str.encode('utf-8')
        end_key = end_str.encode('utf-8')

        for key, value in self.db.iterator(start=start_key, stop=end_key):
            try:
                edge = hedge(key.decode('utf-8')[1:])
                attributes = _decode_attributes(value)
                yield edge, attributes
            except EdgeParseError:
                logger.warning(f"Skipping invalid edge key in database: {key!r}")

    def begin_transaction(self):
        """Begin a transaction using LevelDB WriteBatch.

        Supports nested calls - only the outermost begin/end pair actually
        creates and writes the batch.
        """
        if self.batch_mode:
            return
        if self._transaction_depth == 0:
            self._write_batch = self.db.write_batch()
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
            self._write_batch.write()
            self._write_batch = None

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
        # Simply discard the write batch - changes were never written
        self._write_batch = None
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
        return (''.join(('v', edge.to_str()))).encode('utf-8')

    def _exists_key(self, key):
        """Checks if the given key exists."""
        return self.db.get(key) is not None

    def _add_key(self, key, attributes):
        """Adds the given edge, given its key."""
        value = _encode_attributes(attributes)
        if self._write_batch is not None:
            self._write_batch.put(key, value)
        else:
            self.db.put(key, value)

    def _attribute_key(self, key):
        value = self.db.get(key)
        return _decode_attributes(value)

    def _write_edge_permutation(self, perm):
        """Writes a given permutation."""
        perm_key = (''.join(('p', perm))).encode('utf-8')
        if self._write_batch is not None:
            self._write_batch.put(perm_key, b'x')
        else:
            self.db.put(perm_key, b'x')

    def _remove_edge_permutation(self, perm):
        """Removes a given permutation."""
        perm_key = (''.join(('p', perm))).encode('utf-8')
        if self._write_batch is not None:
            self._write_batch.delete(perm_key)
        else:
            self.db.delete(perm_key)

    def _remove_key(self, key):
        """Removes an edge, given its key."""
        if self._write_batch is not None:
            self._write_batch.delete(key)
        else:
            self.db.delete(key)

    def _permutations_with_prefix(self, prefix):
        end_str = str_plus_1(prefix)
        start_key = (''.join(('p', prefix))).encode('utf-8')
        end_key = (''.join(('p', end_str))).encode('utf-8')
        for key, _ in self.db.iterator(start=start_key, stop=end_key):
            perm_str = key.decode('utf-8')
            yield perm_str[1:]

    def _edges_with_prefix(self, prefix):
        end_str = str_plus_1(prefix)
        start_key = (''.join(('v', prefix))).encode('utf-8')
        end_key = (''.join(('v', end_str))).encode('utf-8')
        for key, _ in self.db.iterator(start=start_key, stop=end_key):
            yield hedge(key.decode('utf-8')[1:])
