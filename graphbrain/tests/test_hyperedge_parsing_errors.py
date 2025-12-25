"""Tests for hyperedge parsing error handling.

These tests verify that parsing failures raise EdgeParseError instead of
silently returning None (which was the previous behavior).
"""
import unittest

from graphbrain.exceptions import EdgeParseError
from graphbrain.hyperedge import hedge, split_edge_str


class TestSplitEdgeStrErrors(unittest.TestCase):
    """Test split_edge_str raises errors for malformed input."""

    def test_unbalanced_close_paren(self):
        """Extra closing parenthesis raises EdgeParseError."""
        with self.assertRaises(EdgeParseError) as ctx:
            split_edge_str("a b)")
        self.assertIn(")", str(ctx.exception))

    def test_unbalanced_open_paren(self):
        """Unclosed parenthesis raises EdgeParseError."""
        with self.assertRaises(EdgeParseError) as ctx:
            split_edge_str("(a b")
        self.assertIn("(", str(ctx.exception))

    def test_multiple_unbalanced_close(self):
        """Multiple extra close parens raises EdgeParseError."""
        with self.assertRaises(EdgeParseError) as ctx:
            split_edge_str("(a b))")
        exc = ctx.exception
        self.assertIsNotNone(exc.edge_str)

    def test_nested_unbalanced(self):
        """Nested unbalanced parens raises EdgeParseError."""
        with self.assertRaises(EdgeParseError) as ctx:
            split_edge_str("(a (b c)")
        exc = ctx.exception
        self.assertEqual(exc.edge_str, "(a (b c)")


class TestHedgeErrors(unittest.TestCase):
    """Test hedge() raises errors for invalid input."""

    def test_unbalanced_parens_in_hedge(self):
        """hedge() raises EdgeParseError for unbalanced parens."""
        with self.assertRaises(EdgeParseError):
            hedge("(a b")

    def test_extra_close_paren_in_hedge(self):
        """hedge() raises EdgeParseError for extra close paren."""
        with self.assertRaises(EdgeParseError):
            hedge("a b)")

    def test_empty_string(self):
        """Empty string raises EdgeParseError."""
        with self.assertRaises(EdgeParseError) as ctx:
            hedge("")
        self.assertIn("empty", str(ctx.exception).lower())

    def test_whitespace_only(self):
        """Whitespace-only string raises EdgeParseError."""
        with self.assertRaises(EdgeParseError):
            hedge("   ")

    def test_empty_parens(self):
        """Empty parentheses raises EdgeParseError."""
        with self.assertRaises(EdgeParseError):
            hedge("()")

    def test_invalid_type_raises_validation_error(self):
        """Invalid input type raises ValidationError."""
        from graphbrain.exceptions import ValidationError
        with self.assertRaises(ValidationError):
            hedge(12345)

    def test_none_raises_validation_error(self):
        """None input raises ValidationError."""
        from graphbrain.exceptions import ValidationError
        with self.assertRaises(ValidationError):
            hedge(None)

    def test_dict_raises_validation_error(self):
        """Dict input raises ValidationError."""
        from graphbrain.exceptions import ValidationError
        with self.assertRaises(ValidationError):
            hedge({"a": "b"})


class TestHedgeValidInput(unittest.TestCase):
    """Verify valid inputs still work correctly after changes."""

    def test_simple_atom(self):
        """Simple atom parses correctly."""
        edge = hedge("word/C")
        self.assertTrue(edge.atom)
        self.assertEqual(str(edge), "word/C")

    def test_simple_edge(self):
        """Simple edge parses correctly."""
        edge = hedge("(is/P mary/C nice/C)")
        self.assertFalse(edge.atom)
        self.assertEqual(len(edge), 3)

    def test_nested_edge(self):
        """Nested edge parses correctly."""
        edge = hedge("(says/P mary/C (is/P john/C nice/C))")
        self.assertFalse(edge.atom)
        self.assertEqual(len(edge), 3)

    def test_list_input(self):
        """List input works correctly."""
        edge = hedge(["is/P", "mary/C", "nice/C"])
        self.assertEqual(len(edge), 3)

    def test_tuple_input(self):
        """Tuple input works correctly."""
        edge = hedge(("is/P", "mary/C", "nice/C"))
        self.assertEqual(len(edge), 3)

    def test_edge_passthrough(self):
        """Existing edge passes through unchanged."""
        original = hedge("(is/P mary/C nice/C)")
        result = hedge(original)
        self.assertIs(result, original)


class TestEdgeParseErrorContext(unittest.TestCase):
    """Test that EdgeParseError includes useful context."""

    def test_error_includes_edge_str(self):
        """EdgeParseError includes the problematic string."""
        try:
            hedge("(unbalanced (parens")
        except EdgeParseError as e:
            self.assertIsNotNone(e.edge_str)
        else:
            self.fail("Expected EdgeParseError")

    def test_error_includes_position_for_close_paren(self):
        """EdgeParseError includes position for extra close paren."""
        try:
            split_edge_str("a b) c")
        except EdgeParseError as e:
            self.assertIsNotNone(e.position)
        else:
            self.fail("Expected EdgeParseError")


if __name__ == '__main__':
    unittest.main()
