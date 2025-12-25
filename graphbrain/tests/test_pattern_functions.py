"""Tests for new pattern functions (not, all, exists, depth)."""
import unittest

from graphbrain import hedge
from graphbrain.patterns.matcher import Matcher


class TestNotPatternFunction(unittest.TestCase):
    """Tests for the 'not' pattern function."""

    def test_not_matches_when_pattern_fails(self):
        """not matches when sub-pattern doesn't match."""
        edge = hedge('(is/P mary/C happy/C)')
        pattern = hedge('(not sad/C)')
        matcher = Matcher(edge, pattern)
        self.assertEqual(len(matcher.results), 1)

    def test_not_fails_when_pattern_matches(self):
        """not fails when sub-pattern matches."""
        edge = hedge('happy/C')
        pattern = hedge('(not happy/C)')
        matcher = Matcher(edge, pattern)
        self.assertEqual(len(matcher.results), 0)

    def test_not_with_wildcard(self):
        """not with wildcard pattern."""
        edge = hedge('cat/C')
        # Pattern that matches any predicate - cat is not a predicate
        pattern = hedge('(not ./P)')
        matcher = Matcher(edge, pattern)
        self.assertEqual(len(matcher.results), 1)

    def test_not_with_complex_pattern(self):
        """not with complex sub-pattern."""
        edge = hedge('(likes/P mary/C dogs/C)')
        # Should match because edge is not of form (hates ...)
        pattern = hedge('(not (hates/P * *))')
        matcher = Matcher(edge, pattern)
        self.assertEqual(len(matcher.results), 1)


class TestAllPatternFunction(unittest.TestCase):
    """Tests for the 'all' pattern function."""

    def test_all_matches_both_patterns(self):
        """all matches when all sub-patterns match."""
        edge = hedge('happy/Ca')
        pattern = hedge('(all happy/C ./Ca)')
        matcher = Matcher(edge, pattern)
        self.assertEqual(len(matcher.results), 1)

    def test_all_fails_when_one_fails(self):
        """all fails when any sub-pattern fails."""
        edge = hedge('happy/C')
        pattern = hedge('(all happy/C sad/C)')
        matcher = Matcher(edge, pattern)
        self.assertEqual(len(matcher.results), 0)

    def test_all_with_variables(self):
        """all captures variables from all sub-patterns."""
        edge = hedge('(loves/P mary/C john/C)')
        pattern = hedge('(all (loves/P X *) (loves/P * Y))')
        matcher = Matcher(edge, pattern)
        self.assertEqual(len(matcher.results), 1)
        self.assertIn('X', matcher.results[0])
        self.assertIn('Y', matcher.results[0])


class TestExistsPatternFunction(unittest.TestCase):
    """Tests for the 'exists' pattern function."""

    def test_exists_matches_first(self):
        """exists matches when first sub-pattern matches."""
        edge = hedge('cat/C')
        pattern = hedge('(exists cat/C dog/C)')
        matcher = Matcher(edge, pattern)
        self.assertEqual(len(matcher.results), 1)

    def test_exists_matches_second(self):
        """exists matches when second sub-pattern matches."""
        edge = hedge('dog/C')
        pattern = hedge('(exists cat/C dog/C)')
        matcher = Matcher(edge, pattern)
        self.assertEqual(len(matcher.results), 1)

    def test_exists_fails_when_none_match(self):
        """exists fails when no sub-patterns match."""
        edge = hedge('bird/C')
        pattern = hedge('(exists cat/C dog/C)')
        matcher = Matcher(edge, pattern)
        self.assertEqual(len(matcher.results), 0)


class TestDepthPatternFunction(unittest.TestCase):
    """Tests for the 'depth' pattern function."""

    def test_depth_zero_matches_root(self):
        """depth 0 matches at root level."""
        edge = hedge('(is/P (the/M cat/C) happy/C)')
        pattern = hedge('(depth 0 (is/P * *))')
        matcher = Matcher(edge, pattern)
        self.assertEqual(len(matcher.results), 1)

    def test_depth_one_matches_children(self):
        """depth 1 matches at first nesting level."""
        edge = hedge('(is/P (the/M cat/C) happy/C)')
        pattern = hedge('(depth 1 (the/M *))')
        matcher = Matcher(edge, pattern)
        self.assertEqual(len(matcher.results), 1)

    def test_depth_does_not_match_wrong_level(self):
        """depth doesn't match at wrong level."""
        edge = hedge('(is/P (the/M cat/C) happy/C)')
        # the/M cat/C is at depth 1, not 0
        pattern = hedge('(depth 0 (the/M *))')
        matcher = Matcher(edge, pattern)
        self.assertEqual(len(matcher.results), 0)


class TestPatternFunctionCombinations(unittest.TestCase):
    """Tests for combinations of pattern functions."""

    def test_not_with_any(self):
        """Combine not with any."""
        edge = hedge('bird/C')
        # Matches if edge is NOT a cat or dog
        pattern = hedge('(not (any cat/C dog/C))')
        matcher = Matcher(edge, pattern)
        self.assertEqual(len(matcher.results), 1)

    def test_all_with_not(self):
        """Combine all with not."""
        edge = hedge('(loves/P mary/C john/C)')
        # Match if it's a loves relation AND not a hates relation
        pattern = hedge('(all (loves/P * *) (not (hates/P * *)))')
        matcher = Matcher(edge, pattern)
        self.assertEqual(len(matcher.results), 1)


if __name__ == '__main__':
    unittest.main()
