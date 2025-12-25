"""Tests for hgraph() input validation."""
import unittest
import tempfile
import os

from graphbrain import hgraph
from graphbrain.exceptions import ValidationError, StorageError


class TestHgraphValidation(unittest.TestCase):
    """Test hgraph() validates input correctly."""

    def test_non_string_raises_validation_error(self):
        """Non-string input raises ValidationError."""
        with self.assertRaises(ValidationError) as ctx:
            hgraph(12345)
        self.assertIn("must be a string", str(ctx.exception))
        self.assertEqual(ctx.exception.value, 12345)

    def test_none_raises_validation_error(self):
        """None input raises ValidationError."""
        with self.assertRaises(ValidationError) as ctx:
            hgraph(None)
        self.assertIn("must be a string", str(ctx.exception))

    def test_list_raises_validation_error(self):
        """List input raises ValidationError."""
        with self.assertRaises(ValidationError) as ctx:
            hgraph(["test.db"])
        self.assertEqual(ctx.exception.expected_type, "str")

    def test_empty_string_raises_validation_error(self):
        """Empty string raises ValidationError."""
        with self.assertRaises(ValidationError) as ctx:
            hgraph("")
        self.assertIn("cannot be empty", str(ctx.exception))

    def test_whitespace_only_raises_validation_error(self):
        """Whitespace-only string raises ValidationError."""
        with self.assertRaises(ValidationError):
            hgraph("   ")

    def test_no_extension_raises_validation_error(self):
        """Path without extension raises ValidationError."""
        with self.assertRaises(ValidationError) as ctx:
            hgraph("testdb")
        self.assertIn("Unrecognized database extension", str(ctx.exception))
        self.assertIn("Valid extensions", str(ctx.exception))

    def test_unknown_extension_raises_validation_error(self):
        """Unknown extension raises ValidationError."""
        with self.assertRaises(ValidationError) as ctx:
            hgraph("test.txt")
        self.assertIn("Unrecognized database extension", str(ctx.exception))
        self.assertIn("db", str(ctx.exception))  # Lists valid extensions

    def test_valid_sqlite_extension(self):
        """Valid .sqlite extension creates SQLite backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.sqlite")
            hg = hgraph(path)
            try:
                self.assertIsNotNone(hg)
            finally:
                hg.close()

    def test_valid_db_extension(self):
        """Valid .db extension creates SQLite backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.db")
            hg = hgraph(path)
            try:
                self.assertIsNotNone(hg)
            finally:
                hg.close()

    def test_valid_sqlite3_extension(self):
        """Valid .sqlite3 extension creates SQLite backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.sqlite3")
            hg = hgraph(path)
            try:
                self.assertIsNotNone(hg)
            finally:
                hg.close()

    def test_case_insensitive_extension(self):
        """Extension matching is case-insensitive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.DB")
            hg = hgraph(path)
            try:
                self.assertIsNotNone(hg)
            finally:
                hg.close()


if __name__ == '__main__':
    unittest.main()
