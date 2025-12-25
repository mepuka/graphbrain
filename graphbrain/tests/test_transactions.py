"""Tests for transaction management in hypergraph storage backends."""
import unittest
import tempfile
import os

from graphbrain import hgraph
from graphbrain.hyperedge import hedge
from graphbrain.exceptions import TransactionError


class TestSQLiteTransactions(unittest.TestCase):
    """Test SQLite transaction management."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.hg = hgraph(self.db_path)

    def tearDown(self):
        self.hg.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.tmpdir)

    def test_transaction_state_tracking(self):
        """Transaction state is tracked correctly."""
        self.assertFalse(self.hg.in_transaction())
        self.hg.begin_transaction()
        self.assertTrue(self.hg.in_transaction())
        self.hg.end_transaction()
        self.assertFalse(self.hg.in_transaction())

    def test_nested_transactions_supported(self):
        """Nested transactions are supported via reference counting."""
        self.hg.begin_transaction()
        self.assertTrue(self.hg.in_transaction())
        self.hg.begin_transaction()  # Nested, should not raise
        self.assertTrue(self.hg.in_transaction())
        self.hg.end_transaction()  # Decrements counter, still in transaction
        self.assertTrue(self.hg.in_transaction())
        self.hg.end_transaction()  # Outermost, actually commits
        self.assertFalse(self.hg.in_transaction())

    def test_end_without_begin_raises_error(self):
        """Ending transaction without active one raises TransactionError."""
        with self.assertRaises(TransactionError) as ctx:
            self.hg.end_transaction()
        self.assertIn("no", str(ctx.exception).lower())

    def test_rollback_transaction(self):
        """Rollback reverts uncommitted changes."""
        edge = hedge("(test/P a/C b/C)")
        self.hg.begin_transaction()
        self.hg.add(edge)
        self.hg.rollback()
        self.assertFalse(self.hg.in_transaction())
        # Edge should not exist after rollback
        self.assertFalse(self.hg.exists(edge))

    def test_rollback_without_transaction_raises_error(self):
        """Rollback without active transaction raises TransactionError."""
        with self.assertRaises(TransactionError):
            self.hg.rollback()

    def test_commit_persists_changes(self):
        """Changes are persisted after end_transaction."""
        edge = hedge("(persist/P data/C here/C)")
        self.hg.begin_transaction()
        self.hg.add(edge)
        self.hg.end_transaction()
        # Reopen database to verify persistence
        self.hg.close()
        self.hg = hgraph(self.db_path)
        self.assertTrue(self.hg.exists(edge))

    def test_batch_mode_skips_transaction_checks(self):
        """Batch mode bypasses transaction start/end."""
        self.hg.batch_mode = True
        # These should not raise even without explicit transaction
        self.hg.begin_transaction()
        self.hg.begin_transaction()  # Would normally fail
        self.hg.end_transaction()
        self.hg.end_transaction()
        self.hg.batch_mode = False

    def test_add_with_transaction(self):
        """Add operation works within transaction."""
        edge = hedge("(works/P inside/C transaction/C)")
        self.hg.begin_transaction()
        self.hg.add(edge)
        self.hg.end_transaction()
        self.assertTrue(self.hg.exists(edge))

    def test_remove_with_transaction(self):
        """Remove operation works within transaction."""
        edge = hedge("(to/P remove/C this/C)")
        self.hg.add(edge)
        self.assertTrue(self.hg.exists(edge))

        self.hg.begin_transaction()
        self.hg.remove(edge)
        self.hg.end_transaction()
        self.assertFalse(self.hg.exists(edge))


class TestTransactionContextManager(unittest.TestCase):
    """Test context manager protocol for transactions."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.hg = hgraph(self.db_path)

    def tearDown(self):
        self.hg.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.tmpdir)

    def test_transaction_context_commits_on_success(self):
        """Transaction context manager commits on successful exit."""
        edge = hedge("(context/P manager/C test/C)")
        with self.hg.transaction():
            self.hg.add(edge)
        self.assertTrue(self.hg.exists(edge))
        self.assertFalse(self.hg.in_transaction())

    def test_transaction_context_rolls_back_on_exception(self):
        """Transaction context manager rolls back on exception."""
        edge = hedge("(should/P not/C exist/C)")
        try:
            with self.hg.transaction():
                self.hg.add(edge)
                raise ValueError("Simulated error")
        except ValueError:
            pass
        self.assertFalse(self.hg.exists(edge))
        self.assertFalse(self.hg.in_transaction())


if __name__ == '__main__':
    unittest.main()
