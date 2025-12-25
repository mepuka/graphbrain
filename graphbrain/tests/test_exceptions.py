import unittest

from graphbrain.exceptions import (
    GraphbrainError,
    EdgeParseError,
    StorageError,
    TransactionError,
    ValidationError
)


class TestExceptionHierarchy(unittest.TestCase):
    """Test exception class hierarchy."""

    def test_base_inherits_from_exception(self):
        """Base class is proper Exception subclass."""
        self.assertTrue(issubclass(GraphbrainError, Exception))

    def test_edge_parse_error_inherits_from_base(self):
        """EdgeParseError inherits from GraphbrainError."""
        self.assertTrue(issubclass(EdgeParseError, GraphbrainError))

    def test_storage_error_inherits_from_base(self):
        """StorageError inherits from GraphbrainError."""
        self.assertTrue(issubclass(StorageError, GraphbrainError))

    def test_transaction_error_inherits_from_storage(self):
        """TransactionError inherits from StorageError."""
        self.assertTrue(issubclass(TransactionError, StorageError))

    def test_validation_error_inherits_from_base(self):
        """ValidationError inherits from GraphbrainError."""
        self.assertTrue(issubclass(ValidationError, GraphbrainError))


class TestEdgeParseError(unittest.TestCase):
    """Test EdgeParseError exception."""

    def test_basic_message(self):
        """EdgeParseError stores basic message."""
        exc = EdgeParseError("Unbalanced parentheses")
        self.assertIn("Unbalanced parentheses", str(exc))

    def test_with_edge_str(self):
        """EdgeParseError stores edge_str context."""
        exc = EdgeParseError("Parse error", edge_str="(a b")
        self.assertEqual(exc.edge_str, "(a b")
        self.assertIn("Parse error", str(exc))

    def test_with_position(self):
        """EdgeParseError stores position context."""
        exc = EdgeParseError("Unexpected character", edge_str="(a b)", position=3)
        self.assertEqual(exc.position, 3)
        self.assertEqual(exc.edge_str, "(a b)")

    def test_can_be_caught_as_base(self):
        """EdgeParseError can be caught as GraphbrainError."""
        with self.assertRaises(GraphbrainError):
            raise EdgeParseError("test")


class TestStorageError(unittest.TestCase):
    """Test StorageError exception."""

    def test_basic_message(self):
        """StorageError stores basic message."""
        exc = StorageError("Database connection failed")
        self.assertIn("Database connection failed", str(exc))

    def test_with_operation(self):
        """StorageError stores operation context."""
        exc = StorageError("Write failed", operation="add")
        self.assertEqual(exc.operation, "add")

    def test_with_key(self):
        """StorageError stores key context."""
        exc = StorageError("Key not found", key="edge_123")
        self.assertEqual(exc.key, "edge_123")


class TestTransactionError(unittest.TestCase):
    """Test TransactionError exception."""

    def test_basic_message(self):
        """TransactionError stores basic message."""
        exc = TransactionError("Transaction already in progress")
        self.assertIn("Transaction already in progress", str(exc))

    def test_can_be_caught_as_storage_error(self):
        """TransactionError can be caught as StorageError."""
        with self.assertRaises(StorageError):
            raise TransactionError("test")

    def test_can_be_caught_as_base(self):
        """TransactionError can be caught as GraphbrainError."""
        with self.assertRaises(GraphbrainError):
            raise TransactionError("test")


class TestValidationError(unittest.TestCase):
    """Test ValidationError exception."""

    def test_basic_message(self):
        """ValidationError stores basic message."""
        exc = ValidationError("Invalid input type")
        self.assertIn("Invalid input type", str(exc))

    def test_with_value(self):
        """ValidationError stores invalid value."""
        exc = ValidationError("Invalid type", value=123)
        self.assertEqual(exc.value, 123)

    def test_with_expected_type(self):
        """ValidationError stores expected type."""
        exc = ValidationError("Invalid type", value=123, expected_type="str")
        self.assertEqual(exc.expected_type, "str")

    def test_can_be_caught_as_base(self):
        """ValidationError can be caught as GraphbrainError."""
        with self.assertRaises(GraphbrainError):
            raise ValidationError("test")


if __name__ == '__main__':
    unittest.main()
