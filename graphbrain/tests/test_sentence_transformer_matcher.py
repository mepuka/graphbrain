"""Tests for SentenceTransformerMatcher."""
import unittest
from unittest.mock import patch, MagicMock

from graphbrain.semsim.matcher.matcher import SemSimConfig


class TestSentenceTransformerAvailability(unittest.TestCase):
    """Tests for sentence-transformer availability."""

    def test_import_works(self):
        """Verify sentence-transformer matcher can be imported."""
        from graphbrain.semsim.matcher.sentence_transformer_matcher import (
            SentenceTransformerMatcher,
            SENTENCE_TRANSFORMER_AVAILABLE
        )
        self.assertTrue(SENTENCE_TRANSFORMER_AVAILABLE)
        self.assertIsNotNone(SentenceTransformerMatcher)


class TestSentenceTransformerMatcher(unittest.TestCase):
    """Tests for SentenceTransformerMatcher functionality."""

    def setUp(self):
        """Set up test fixtures."""
        from graphbrain.semsim.matcher.sentence_transformer_matcher import (
            SentenceTransformerMatcher,
            SENTENCE_TRANSFORMER_AVAILABLE
        )
        if not SENTENCE_TRANSFORMER_AVAILABLE:
            self.skipTest("sentence-transformers not available")
        self.SentenceTransformerMatcher = SentenceTransformerMatcher

    def test_lazy_loading(self):
        """Model is not loaded during initialization."""
        config = SemSimConfig(model_name='e5-base', similarity_threshold=0.5)

        # Patch the actual SentenceTransformer import
        with patch('graphbrain.semsim.matcher.sentence_transformer_matcher.SentenceTransformer') as mock_st:
            from graphbrain.semsim.matcher.sentence_transformer_matcher import SentenceTransformerMatcher

            # Create matcher - should not load model yet
            matcher = SentenceTransformerMatcher(config)
            mock_st.assert_not_called()

            # Model should be None
            self.assertIsNone(matcher._model)

    def test_model_loaded_on_access(self):
        """Model is loaded when accessed via property."""
        config = SemSimConfig(model_name='e5-base', similarity_threshold=0.5)

        with patch('graphbrain.semsim.matcher.sentence_transformer_matcher.SentenceTransformer') as mock_st:
            matcher = self.SentenceTransformerMatcher(config)

            # Model not loaded yet
            mock_st.assert_not_called()

            # Access model property - this triggers loading
            _ = matcher.model

            # Verify model was loaded with correct ID
            mock_st.assert_called_once_with('intfloat/e5-base-v2')

    def test_filter_oov_returns_all_words(self):
        """filter_oov returns all words (no OOV with transformers)."""
        config = SemSimConfig(model_name='e5-base', similarity_threshold=0.5)

        with patch('graphbrain.semsim.matcher.sentence_transformer_matcher.SentenceTransformer'):
            matcher = self.SentenceTransformerMatcher(config)
            words = ['kubernetes', 'blockchain', 'nonexistentword123']
            filtered = matcher.filter_oov(words)
            self.assertEqual(filtered, words)

    def test_embedding_prefix_for_e5_models(self):
        """E5 models use 'query: ' prefix."""
        config = SemSimConfig(model_name='e5-base', similarity_threshold=0.5)

        with patch('graphbrain.semsim.matcher.sentence_transformer_matcher.SentenceTransformer'):
            matcher = self.SentenceTransformerMatcher(config)
            self.assertEqual(matcher.embedding_prefix, 'query: ')

    def test_no_prefix_for_minilm(self):
        """MiniLM models don't use prefix."""
        config = SemSimConfig(model_name='minilm', similarity_threshold=0.4)

        with patch('graphbrain.semsim.matcher.sentence_transformer_matcher.SentenceTransformer'):
            matcher = self.SentenceTransformerMatcher(config)
            self.assertEqual(matcher.embedding_prefix, '')

    def test_default_threshold_from_model_config(self):
        """Default threshold is taken from model config."""
        config = SemSimConfig(model_name='e5-base')  # No threshold specified

        with patch('graphbrain.semsim.matcher.sentence_transformer_matcher.SentenceTransformer'):
            matcher = self.SentenceTransformerMatcher(config)
            self.assertEqual(matcher._similarity_threshold, 0.5)

    def test_cache_info(self):
        """Cache info method works correctly."""
        config = SemSimConfig(model_name='e5-base', similarity_threshold=0.5)

        with patch('graphbrain.semsim.matcher.sentence_transformer_matcher.SentenceTransformer'):
            matcher = self.SentenceTransformerMatcher(config)
            info = matcher.cache_info()

            self.assertIn('hits', info)
            self.assertIn('misses', info)
            self.assertIn('hit_rate', info)

    def test_clear_cache(self):
        """clear_cache method clears the embedding cache."""
        config = SemSimConfig(model_name='e5-base', similarity_threshold=0.5)

        with patch('graphbrain.semsim.matcher.sentence_transformer_matcher.SentenceTransformer'):
            matcher = self.SentenceTransformerMatcher(config)
            # Should not raise
            matcher.clear_cache()


class TestModelConfigs(unittest.TestCase):
    """Tests for model configuration registry."""

    def test_get_model_config_by_short_name(self):
        """get_model_config works with short names."""
        from graphbrain.semsim.matcher.sentence_transformer_matcher import get_model_config

        config = get_model_config('e5-base')
        self.assertEqual(config['id'], 'intfloat/e5-base-v2')
        self.assertEqual(config['dims'], 768)

    def test_get_model_config_by_full_id(self):
        """get_model_config works with full model IDs."""
        from graphbrain.semsim.matcher.sentence_transformer_matcher import get_model_config

        config = get_model_config('intfloat/e5-base-v2')
        self.assertEqual(config['id'], 'intfloat/e5-base-v2')

    def test_get_model_config_unknown_model(self):
        """get_model_config returns default for unknown models."""
        from graphbrain.semsim.matcher.sentence_transformer_matcher import get_model_config

        config = get_model_config('unknown-model-xyz')
        self.assertEqual(config['id'], 'unknown-model-xyz')
        self.assertEqual(config['default_threshold'], 0.5)


class TestInterfaceIntegration(unittest.TestCase):
    """Tests for interface.py integration."""

    def test_backend_enum_exists(self):
        """MatcherBackend enum is defined."""
        from graphbrain.semsim.interface import MatcherBackend

        self.assertEqual(MatcherBackend.AUTO.value, "AUTO")
        self.assertEqual(MatcherBackend.SENTENCE_TRANSFORMER.value, "SENTENCE_TRANSFORMER")
        self.assertEqual(MatcherBackend.GENSIM.value, "GENSIM")

    def test_sentence_transformer_available_flag(self):
        """SENTENCE_TRANSFORMER_AVAILABLE flag is correctly set."""
        from graphbrain.semsim.interface import SENTENCE_TRANSFORMER_AVAILABLE
        self.assertTrue(SENTENCE_TRANSFORMER_AVAILABLE)

    def test_set_fix_backend(self):
        """set_fix_backend function works."""
        from graphbrain.semsim.interface import set_fix_backend, MatcherBackend, _fix_backend

        original = _fix_backend
        try:
            set_fix_backend(MatcherBackend.GENSIM)
            # Import again to check the change
            from graphbrain.semsim import interface
            self.assertEqual(interface._fix_backend, MatcherBackend.GENSIM)
        finally:
            # Reset to original
            set_fix_backend(original)

    def test_default_fix_config_uses_sentence_transformers(self):
        """Default FIX config uses sentence-transformers when available."""
        from graphbrain.semsim.interface import (
            SENTENCE_TRANSFORMER_AVAILABLE,
            get_default_fix_config
        )

        config = get_default_fix_config()
        if SENTENCE_TRANSFORMER_AVAILABLE:
            self.assertEqual(config.model_name, 'intfloat/e5-base-v2')
            self.assertEqual(config.similarity_threshold, 0.5)
        else:
            self.assertEqual(config.model_name, 'word2vec-google-news-300')
            self.assertEqual(config.similarity_threshold, 0.2)


class TestSimilarityComputation(unittest.TestCase):
    """Tests for actual similarity computation (requires model download)."""

    @classmethod
    def setUpClass(cls):
        """Check if we can run integration tests."""
        try:
            from sentence_transformers import SentenceTransformer
            cls.skip_integration = False
        except ImportError:
            cls.skip_integration = True

    def setUp(self):
        if self.skip_integration:
            self.skipTest("sentence-transformers not available")

    def test_real_similarity_computation(self):
        """Test actual similarity computation with real model."""
        from graphbrain.semsim.matcher.sentence_transformer_matcher import SentenceTransformerMatcher

        # Use smallest model for speed
        config = SemSimConfig(model_name='minilm', similarity_threshold=0.3)
        matcher = SentenceTransformerMatcher(config)

        # Test similar words
        result = matcher._similarities(
            cand_word='king',
            ref_words=['queen', 'prince', 'banana']
        )

        self.assertIsNotNone(result)
        self.assertIn('queen', result)
        self.assertIn('prince', result)
        self.assertIn('banana', result)

        # Royal words should be more similar than banana
        self.assertGreater(result['queen'], result['banana'])
        self.assertGreater(result['prince'], result['banana'])

    def test_similar_method(self):
        """Test the similar() method works correctly."""
        from graphbrain.semsim.matcher.sentence_transformer_matcher import SentenceTransformerMatcher

        config = SemSimConfig(model_name='minilm', similarity_threshold=0.3)
        matcher = SentenceTransformerMatcher(config)

        # King should be similar to royalty
        result = matcher.similar(
            cand_word='king',
            ref_words=['queen', 'monarch']
        )
        self.assertTrue(result)

        # King should not be similar to unrelated words at high threshold
        result = matcher.similar(
            threshold=0.9,  # Very high threshold
            cand_word='king',
            ref_words=['banana', 'bicycle']
        )
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
