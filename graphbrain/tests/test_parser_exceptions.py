"""Tests for parser-specific exceptions."""
import unittest

from graphbrain.exceptions import (
    ParserError, SpacyProcessingError, AtomConstructionError,
    RuleApplicationError, ModelNotFoundError, CoreferenceError
)


class TestParserError(unittest.TestCase):
    """Tests for ParserError base class."""

    def test_basic_creation(self):
        """ParserError can be created with message only."""
        error = ParserError("Test error")
        self.assertEqual(str(error), "Test error")
        self.assertIsNone(error.text)
        self.assertIsNone(error.token)

    def test_with_text(self):
        """ParserError stores text context."""
        error = ParserError("Parse failed", text="The quick brown fox")
        self.assertEqual(error.text, "The quick brown fox")

    def test_is_graphbrain_error(self):
        """ParserError is a GraphbrainError."""
        from graphbrain.exceptions import GraphbrainError
        error = ParserError("Test")
        self.assertIsInstance(error, GraphbrainError)


class TestSpacyProcessingError(unittest.TestCase):
    """Tests for SpacyProcessingError."""

    def test_creation_with_all_attributes(self):
        """SpacyProcessingError stores all context."""
        error = SpacyProcessingError(
            "Tokenization failed",
            text="Test sentence",
            model="en_core_web_lg",
            stage="tokenize"
        )
        self.assertEqual(str(error), "Tokenization failed")
        self.assertEqual(error.text, "Test sentence")
        self.assertEqual(error.model, "en_core_web_lg")
        self.assertEqual(error.stage, "tokenize")

    def test_is_parser_error(self):
        """SpacyProcessingError inherits from ParserError."""
        error = SpacyProcessingError("Test")
        self.assertIsInstance(error, ParserError)


class TestAtomConstructionError(unittest.TestCase):
    """Tests for AtomConstructionError."""

    def test_creation(self):
        """AtomConstructionError stores atom type context."""
        error = AtomConstructionError(
            "Cannot construct atom",
            atom_type="C"
        )
        self.assertEqual(error.atom_type, "C")

    def test_is_parser_error(self):
        """AtomConstructionError inherits from ParserError."""
        error = AtomConstructionError("Test")
        self.assertIsInstance(error, ParserError)


class TestRuleApplicationError(unittest.TestCase):
    """Tests for RuleApplicationError."""

    def test_creation_with_context(self):
        """RuleApplicationError stores rule application context."""
        error = RuleApplicationError(
            "Rule failed to apply",
            rule="P -> C C",
            sentence=["word1", "word2"],
            position=1
        )
        self.assertEqual(error.rule, "P -> C C")
        self.assertEqual(error.sentence, ["word1", "word2"])
        self.assertEqual(error.position, 1)

    def test_is_parser_error(self):
        """RuleApplicationError inherits from ParserError."""
        error = RuleApplicationError("Test")
        self.assertIsInstance(error, ParserError)


class TestModelNotFoundError(unittest.TestCase):
    """Tests for ModelNotFoundError."""

    def test_creation_with_fallbacks(self):
        """ModelNotFoundError stores fallback information."""
        error = ModelNotFoundError(
            "No model found",
            model_name="en_core_web_trf",
            fallbacks_tried=["en_core_web_lg", "en_core_web_md"]
        )
        self.assertEqual(error.model_name, "en_core_web_trf")
        self.assertEqual(error.fallbacks_tried, ["en_core_web_lg", "en_core_web_md"])

    def test_default_fallbacks_empty(self):
        """Default fallbacks_tried is empty list."""
        error = ModelNotFoundError("Test")
        self.assertEqual(error.fallbacks_tried, [])

    def test_is_parser_error(self):
        """ModelNotFoundError inherits from ParserError."""
        error = ModelNotFoundError("Test")
        self.assertIsInstance(error, ParserError)


class TestCoreferenceError(unittest.TestCase):
    """Tests for CoreferenceError."""

    def test_creation_with_cluster(self):
        """CoreferenceError stores cluster context."""
        error = CoreferenceError(
            "Coreference resolution failed",
            text="He said she went there",
            cluster=["He", "John"]
        )
        self.assertEqual(error.text, "He said she went there")
        self.assertEqual(error.cluster, ["He", "John"])

    def test_is_parser_error(self):
        """CoreferenceError inherits from ParserError."""
        error = CoreferenceError("Test")
        self.assertIsInstance(error, ParserError)


class TestExceptionHierarchy(unittest.TestCase):
    """Tests for exception hierarchy relationships."""

    def test_all_parser_errors_catchable(self):
        """All parser errors can be caught with ParserError."""
        errors = [
            SpacyProcessingError("test"),
            AtomConstructionError("test"),
            RuleApplicationError("test"),
            ModelNotFoundError("test"),
            CoreferenceError("test"),
        ]
        for error in errors:
            with self.assertRaises(ParserError):
                raise error


if __name__ == '__main__':
    unittest.main()
