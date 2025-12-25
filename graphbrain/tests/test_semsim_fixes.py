"""Tests for Phase 4.1 semsim fixes."""
import unittest
from unittest.mock import patch, MagicMock

from graphbrain.semsim.matcher.context_matcher import (
    _average_pool,
    _get_trf_tok_idxes,
    ContextEmbeddingMatcher,
)
from graphbrain.semsim.matcher.matcher import SemSimConfig


class TestNormalizationFix(unittest.TestCase):
    """Tests for the E5 normalization bug fix."""

    def test_average_pool_normalizes_by_default(self):
        """Verify _average_pool normalizes embeddings by default for E5 compatibility."""
        import torch

        # Create test tensor with known values
        hidden_states = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        attention_mask = torch.tensor([[1, 1]])

        # Call with default (should normalize)
        result = _average_pool(hidden_states, attention_mask)

        # Verify it's normalized (L2 norm should be 1.0)
        norm = torch.norm(result, p=2).item()
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_average_pool_can_skip_normalization(self):
        """Verify _average_pool can still skip normalization if explicitly requested."""
        import torch

        hidden_states = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        attention_mask = torch.tensor([[1, 1]])

        # Call with normalize=False
        result = _average_pool(hidden_states, attention_mask, normalize=False)

        # Verify it's NOT normalized (L2 norm should NOT be 1.0)
        norm = torch.norm(result, p=2).item()
        self.assertNotAlmostEqual(norm, 1.0, places=1)


class TestTokenAlignmentDeduplication(unittest.TestCase):
    """Tests for the O(1) deduplication fix in token alignment."""

    def test_get_trf_tok_idxes_deduplicates(self):
        """Verify transformer token indexes are deduplicated efficiently."""
        # Create mock alignment data
        class MockRagged:
            def __init__(self, lengths, dataXd):
                self.lengths = lengths
                self.dataXd = dataXd

        # 3 lexical tokens, each mapping to some transformer tokens
        # Token 0 -> [0, 1]
        # Token 1 -> [1, 2]  (note: 1 is duplicated)
        # Token 2 -> [2, 3]  (note: 2 is duplicated)
        alignment = MockRagged(
            lengths=[2, 2, 2],
            dataXd=[0, 1, 1, 2, 2, 3]
        )

        # Get transformer indexes for lexical tokens 0, 1, 2
        result = _get_trf_tok_idxes(3, (0, 1, 2), alignment)

        # Should be deduplicated: [0, 1, 2, 3]
        self.assertEqual(result, (0, 1, 2, 3))

    def test_get_trf_tok_idxes_preserves_order(self):
        """Verify transformer token indexes preserve insertion order."""
        class MockRagged:
            def __init__(self, lengths, dataXd):
                self.lengths = lengths
                self.dataXd = dataXd

        # Token 0 -> [5]
        # Token 1 -> [3]
        # Token 2 -> [1]
        alignment = MockRagged(
            lengths=[1, 1, 1],
            dataXd=[5, 3, 1]
        )

        result = _get_trf_tok_idxes(3, (0, 1, 2), alignment)

        # Should preserve order: (5, 3, 1), NOT sorted
        self.assertEqual(result, (5, 3, 1))


class TestLazyLoading(unittest.TestCase):
    """Tests for lazy loading of the spacy pipeline."""

    def test_init_does_not_load_pipeline(self):
        """Verify ContextEmbeddingMatcher.__init__ does not load the pipeline."""
        config = SemSimConfig(
            model_name='test-model',
            similarity_threshold=0.5,
        )

        # Patch _create_spacy_pipeline to track calls
        with patch.object(ContextEmbeddingMatcher, '_create_spacy_pipeline') as mock_create:
            matcher = ContextEmbeddingMatcher(config)

            # Pipeline should NOT be created during __init__
            mock_create.assert_not_called()

            # Internal state should be None
            self.assertIsNone(matcher._spacy_pipe)

    def test_spacy_pipe_property_loads_on_access(self):
        """Verify spacy_pipe property triggers lazy loading."""
        config = SemSimConfig(
            model_name='test-model',
            similarity_threshold=0.5,
        )

        mock_pipeline = MagicMock()

        with patch.object(ContextEmbeddingMatcher, '_create_spacy_pipeline', return_value=mock_pipeline) as mock_create:
            matcher = ContextEmbeddingMatcher(config)

            # Access the property
            pipe = matcher.spacy_pipe

            # Pipeline should be created
            mock_create.assert_called_once_with('test-model')
            self.assertIs(pipe, mock_pipeline)

    def test_spacy_pipe_property_caches_result(self):
        """Verify spacy_pipe property only loads once."""
        config = SemSimConfig(
            model_name='test-model',
            similarity_threshold=0.5,
        )

        mock_pipeline = MagicMock()

        with patch.object(ContextEmbeddingMatcher, '_create_spacy_pipeline', return_value=mock_pipeline) as mock_create:
            matcher = ContextEmbeddingMatcher(config)

            # Access the property multiple times
            pipe1 = matcher.spacy_pipe
            pipe2 = matcher.spacy_pipe
            pipe3 = matcher.spacy_pipe

            # Pipeline should only be created once
            mock_create.assert_called_once()
            self.assertIs(pipe1, pipe2)
            self.assertIs(pipe2, pipe3)

    def test_embedding_prefix_tokens_lazy_loaded(self):
        """Verify embedding_prefix_tokens are lazily loaded."""
        config = SemSimConfig(
            model_name='test-model',
            similarity_threshold=0.5,
            embedding_prefix='query:'
        )

        with patch.object(ContextEmbeddingMatcher, '_create_spacy_pipeline'):
            matcher = ContextEmbeddingMatcher(config)

            # Should be None initially
            self.assertIsNone(matcher._embedding_prefix_tokens)

            # Access the property
            tokens = matcher.embedding_prefix_tokens

            # Should be populated now
            self.assertIsNotNone(tokens)
            self.assertEqual(tokens, ['query', ':'])


class TestCacheSizes(unittest.TestCase):
    """Tests for increased cache sizes."""

    def test_spacy_doc_cache_size_increased(self):
        """Verify spacy doc cache size is increased to 10000."""
        self.assertEqual(ContextEmbeddingMatcher._SPACY_DOC_CACHE_SIZE, 10000)

    def test_embedding_cache_size_increased(self):
        """Verify embedding cache size is increased to 10000."""
        self.assertEqual(ContextEmbeddingMatcher._EMBEDDING_CACHE_SIZE, 10000)


if __name__ == '__main__':
    unittest.main()
