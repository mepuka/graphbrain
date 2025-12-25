"""Tests for semsim availability and feature flag handling."""
import unittest

from graphbrain import hedge
from graphbrain.patterns.properties import SEMSIM_AVAILABLE, FUNS
from graphbrain.patterns.matcher import Matcher


class TestSemsimAvailability(unittest.TestCase):
    """Tests for semsim optional import handling."""

    def test_semsim_available_flag_exists(self):
        """SEMSIM_AVAILABLE flag is defined."""
        self.assertIsInstance(SEMSIM_AVAILABLE, bool)

    def test_base_pattern_functions_always_available(self):
        """Base pattern functions are always available regardless of semsim."""
        base_funs = {'var', 'atoms', 'lemma', 'any', 'not', 'all', 'exists', 'depth'}
        for fun in base_funs:
            self.assertIn(fun, FUNS)

    def test_semsim_functions_in_funs_when_available(self):
        """Semsim functions are in FUNS only when gensim is available."""
        semsim_funs = {'semsim', 'semsim-fix', 'semsim-fix-lemma', 'semsim-ctx'}
        if SEMSIM_AVAILABLE:
            for fun in semsim_funs:
                self.assertIn(fun, FUNS)
        else:
            for fun in semsim_funs:
                self.assertNotIn(fun, FUNS)

    def test_semsim_pattern_without_gensim(self):
        """Semsim patterns are not recognized when gensim is unavailable."""
        if SEMSIM_AVAILABLE:
            self.skipTest("Semsim is available, skipping unavailability test")

        # When gensim is not installed, semsim function names are not in FUNS
        semsim_funs = {'semsim', 'semsim-fix', 'semsim-fix-lemma', 'semsim-ctx'}
        for fun in semsim_funs:
            self.assertNotIn(fun, FUNS)


class TestSemsimFeatureFlag(unittest.TestCase):
    """Tests for the skip_semsim feature flag."""

    def test_skip_semsim_default_is_true(self):
        """Default skip_semsim is True for backward compatibility."""
        from graphbrain.patterns.entrypoints import match_pattern
        import inspect

        sig = inspect.signature(match_pattern)
        skip_semsim_param = sig.parameters['skip_semsim']
        self.assertEqual(skip_semsim_param.default, True)

    def test_skip_semsim_true_returns_results_directly(self):
        """With skip_semsim=True, results are returned without semsim verification."""
        edge = hedge('(is/P mary/C happy/C)')
        pattern = hedge('(is/P * *)')

        # Normal matching should work
        matcher = Matcher(edge, pattern, skip_semsim=True)
        self.assertEqual(len(matcher.results), 1)

    def test_return_semsim_instances_flag(self):
        """return_semsim_instances flag works correctly."""
        from graphbrain.patterns.entrypoints import match_pattern

        edge = hedge('(is/P mary/C happy/C)')
        pattern = hedge('(is/P * *)')

        # Without flag, returns just results
        results = match_pattern(edge, pattern)
        self.assertIsInstance(results, list)

        # With flag, returns tuple
        results, instances = match_pattern(edge, pattern, return_semsim_instances=True)
        self.assertIsInstance(results, list)
        self.assertIsInstance(instances, list)


if __name__ == '__main__':
    unittest.main()
