"""Tests for argrole permutation optimization."""
import unittest

from graphbrain import hedge
from graphbrain.patterns.matcher import Matcher
from graphbrain.patterns.argroles import (
    _compute_compatibility_matrix,
    _find_valid_assignments,
    PERMUTATION_THRESHOLD
)


class TestCompatibilityMatrix(unittest.TestCase):
    """Tests for the compatibility matrix computation."""

    def test_simple_atoms_compatible(self):
        """Compatible atoms return True."""
        # Create a mock matcher
        edge = hedge('(is/P mary/C happy/C)')
        pattern = hedge('(is/P ./C ./C)')
        matcher = Matcher(edge, pattern)

        eitems = [hedge('mary/C'), hedge('john/C')]
        pitems = [hedge('./C'), hedge('./C')]

        matrix = _compute_compatibility_matrix(matcher, eitems, pitems, {})

        # Both items should be compatible with both patterns (wildcards)
        self.assertTrue(matrix[0][0])
        self.assertTrue(matrix[0][1])
        self.assertTrue(matrix[1][0])
        self.assertTrue(matrix[1][1])

    def test_incompatible_atoms(self):
        """Incompatible atoms return False."""
        edge = hedge('(is/P mary/C happy/C)')
        pattern = hedge('(is/P ./C ./C)')
        matcher = Matcher(edge, pattern)

        # mary/C is concept, is/P is predicate
        eitems = [hedge('mary/C')]
        pitems = [hedge('./P')]  # Only matches predicates

        matrix = _compute_compatibility_matrix(matcher, eitems, pitems, {})

        # mary/C should not be compatible with ./P
        self.assertFalse(matrix[0][0])

    def test_nonatomic_edge_with_atomic_pattern(self):
        """Non-atomic edge doesn't match atomic pattern."""
        edge = hedge('(is/P mary/C happy/C)')
        pattern = hedge('(is/P ./C ./C)')
        matcher = Matcher(edge, pattern)

        eitems = [hedge('(the/M cat/C)')]  # Non-atomic
        pitems = [hedge('./C')]  # Atomic pattern

        matrix = _compute_compatibility_matrix(matcher, eitems, pitems, {})

        # Non-atomic edge should not match atomic pattern
        self.assertFalse(matrix[0][0])


class TestFindValidAssignments(unittest.TestCase):
    """Tests for the backtracking assignment finder."""

    def test_all_compatible(self):
        """All compatible returns all permutations."""
        # 2x2 matrix, all True
        matrix = [[True, True], [True, True]]
        assignments = _find_valid_assignments(matrix, 2)

        # Should have 2 assignments: (0,1) and (1,0)
        self.assertEqual(len(assignments), 2)
        self.assertIn((0, 1), assignments)
        self.assertIn((1, 0), assignments)

    def test_no_valid_assignments(self):
        """No valid assignments returns empty list."""
        # 2x2 matrix where row 0 can't match anything
        matrix = [[False, False], [True, True]]
        assignments = _find_valid_assignments(matrix, 2)

        # No valid assignment since edge 0 can't match any pattern
        self.assertEqual(len(assignments), 0)

    def test_partial_compatibility(self):
        """Partial compatibility prunes options."""
        # Edge 0 only compatible with pattern 0
        # Edge 1 compatible with both
        matrix = [[True, False], [True, True]]
        assignments = _find_valid_assignments(matrix, 2)

        # Only (0, 1) is valid since edge 0 must go to pattern 0
        self.assertEqual(len(assignments), 1)
        self.assertIn((0, 1), assignments)

    def test_larger_matrix(self):
        """Handles larger matrices correctly."""
        # 3x2 matrix - choosing 2 from 3
        matrix = [
            [True, True],
            [True, False],
            [False, True]
        ]
        assignments = _find_valid_assignments(matrix, 2)

        # Edge 1 can only match pattern 0
        # Edge 2 can only match pattern 1
        # So (1, 2) is valid
        # Edge 0 can match pattern 0 or 1
        # (0, 2) is valid (0->pattern0, 2->pattern1)
        # (1, 0) would need edge 1->pattern0, edge 0->pattern1 - valid
        self.assertIn((1, 2), assignments)
        self.assertIn((0, 2), assignments)


class TestArgroleOptimization(unittest.TestCase):
    """Integration tests for optimized argrole matching."""

    def test_small_set_uses_direct(self):
        """Sets smaller than threshold use direct permutation."""
        # With only 2 args, should use direct method
        edge = hedge('(is/Pd.so mary/C happy/C)')
        pattern = hedge('(is/P.{so} ./C ./C)')
        matcher = Matcher(edge, pattern)

        # Should still work correctly
        self.assertEqual(len(matcher.results), 1)

    def test_pattern_with_unordered_argroles(self):
        """Unordered argrole matching works with optimization."""
        edge = hedge('(give/Pd.sox mary/C book/C john/C)')
        pattern = hedge('(give/P.{sox} X Y Z)')
        matcher = Matcher(edge, pattern)

        self.assertEqual(len(matcher.results), 1)
        self.assertEqual(str(matcher.results[0]['X']), 'mary/C')
        self.assertEqual(str(matcher.results[0]['Y']), 'book/C')
        self.assertEqual(str(matcher.results[0]['Z']), 'john/C')

    def test_complex_unordered_matching(self):
        """Complex unordered matching with variable capture."""
        edge = hedge('(likes/Pd.so john/C mary/C)')
        pattern = hedge('(likes/P.{so} SUBJ OBJ)')
        matcher = Matcher(edge, pattern)

        self.assertEqual(len(matcher.results), 1)
        self.assertEqual(str(matcher.results[0]['SUBJ']), 'john/C')
        self.assertEqual(str(matcher.results[0]['OBJ']), 'mary/C')

    def test_threshold_constant(self):
        """Verify threshold constant is reasonable."""
        # Threshold should be small enough to avoid O(n!) explosion
        self.assertLessEqual(PERMUTATION_THRESHOLD, 6)
        self.assertGreaterEqual(PERMUTATION_THRESHOLD, 3)


class TestArgroleEdgeCases(unittest.TestCase):
    """Edge case tests for argrole matching."""

    def test_empty_role_counts(self):
        """Empty role counts returns current vars."""
        edge = hedge('(is/P mary/C happy/C)')
        pattern = hedge('(is/P ./C ./C)')
        matcher = Matcher(edge, pattern)
        # Should match successfully
        self.assertEqual(len(matcher.results), 1)

    def test_not_enough_items(self):
        """Not enough items for pattern returns empty or partial."""
        edge = hedge('(is/Pd.s mary/C)')  # Only one arg
        pattern = hedge('(is/P.{so} ./C ./C)')  # Expects two
        matcher = Matcher(edge, pattern)
        # Should fail or return partial match
        self.assertEqual(len(matcher.results), 0)


if __name__ == '__main__':
    unittest.main()
