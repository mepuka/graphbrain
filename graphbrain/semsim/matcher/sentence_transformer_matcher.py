"""Sentence-transformer based semantic similarity matcher.

This module provides a modern alternative to the Word2Vec-based FixedEmbeddingMatcher.
It uses sentence-transformers with models like intfloat/e5-base-v2 which:
- Are 7x smaller than Word2Vec (500MB vs 3.6GB)
- Have better OOV handling (subword tokenization)
- Produce contextual embeddings even for single words
- Handle modern terminology (e.g., "kubernetes", "blockchain")

Usage:
    from graphbrain.semsim.matcher.sentence_transformer_matcher import (
        SentenceTransformerMatcher,
        SENTENCE_TRANSFORMER_AVAILABLE
    )

    if SENTENCE_TRANSFORMER_AVAILABLE:
        config = SemSimConfig(model_name='intfloat/e5-base-v2', similarity_threshold=0.5)
        matcher = SentenceTransformerMatcher(config)
        result = matcher.similar(cand_word='king', ref_words=['queen', 'prince'])
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Union

from graphbrain.semsim.matcher.matcher import SemSimMatcher, SemSimConfig

logger: logging.Logger = logging.getLogger(__name__)

# Check if sentence-transformers is available
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    SentenceTransformer = None
    np = None


# Supported models with their configurations
EMBEDDING_MODELS = {
    'e5-base': {
        'id': 'intfloat/e5-base-v2',
        'dims': 768,
        'size_mb': 500,
        'default_threshold': 0.5,
        'prefix': 'query: ',  # E5 models use prefixes
    },
    'e5-large': {
        'id': 'intfloat/e5-large-v2',
        'dims': 1024,
        'size_mb': 1300,
        'default_threshold': 0.55,
        'prefix': 'query: ',
    },
    'e5-small': {
        'id': 'intfloat/e5-small-v2',
        'dims': 384,
        'size_mb': 130,
        'default_threshold': 0.45,
        'prefix': 'query: ',
    },
    'minilm': {
        'id': 'all-MiniLM-L6-v2',
        'dims': 384,
        'size_mb': 80,
        'default_threshold': 0.4,
        'prefix': '',  # MiniLM doesn't use prefixes
    },
    'mpnet': {
        'id': 'all-mpnet-base-v2',
        'dims': 768,
        'size_mb': 420,
        'default_threshold': 0.5,
        'prefix': '',
    },
}


def get_model_config(model_name: str) -> dict:
    """Get model configuration by name or model ID."""
    # Check if it's a short name
    if model_name in EMBEDDING_MODELS:
        return EMBEDDING_MODELS[model_name]

    # Check if it's a full model ID
    for config in EMBEDDING_MODELS.values():
        if config['id'] == model_name:
            return config

    # Unknown model - return default config
    return {
        'id': model_name,
        'dims': None,
        'size_mb': None,
        'default_threshold': 0.5,
        'prefix': '',
    }


class SentenceTransformerMatcher(SemSimMatcher):
    """Semantic similarity matcher using sentence-transformers.

    This matcher uses modern transformer-based embeddings for word/phrase
    similarity. It's designed as a drop-in replacement for FixedEmbeddingMatcher
    with several advantages:

    - No OOV issues (subword tokenization handles any word)
    - Much smaller models (500MB vs 3.6GB for Word2Vec)
    - Better semantic understanding
    - Handles modern terminology

    The model is lazily loaded on first use to avoid startup delays.
    """

    _EMBEDDING_CACHE_SIZE: int = 10000

    def __init__(self, config: SemSimConfig):
        """Initialize the matcher with configuration.

        Args:
            config: SemSimConfig with model_name and optional similarity_threshold.
                   model_name can be a short name (e.g., 'e5-base') or full
                   HuggingFace model ID (e.g., 'intfloat/e5-base-v2').
        """
        if not SENTENCE_TRANSFORMER_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerMatcher. "
                "Install it with: pip install sentence-transformers"
            )

        super().__init__(config=config)
        self._model_name: str = config.model_name
        self._model_config: dict = get_model_config(config.model_name)
        self._model: SentenceTransformer | None = None  # Lazy loaded

        # Use model's default threshold if not specified
        if self._similarity_threshold is None:
            self._similarity_threshold = self._model_config['default_threshold']

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the sentence-transformer model on first access."""
        if self._model is None:
            model_id = self._model_config['id']
            logger.info(f"Loading sentence-transformer model: {model_id}")
            self._model = SentenceTransformer(model_id)
            logger.info(f"Loaded sentence-transformer model: {model_id}")
        return self._model

    @property
    def embedding_prefix(self) -> str:
        """Get the embedding prefix for the model (e.g., 'query: ' for E5)."""
        return self._model_config.get('prefix', '')

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text string.

        Uses caching to avoid redundant computation.
        """
        # Add prefix for models that require it (like E5)
        prefixed_text = f"{self.embedding_prefix}{text}"
        return self._get_embedding_cached(prefixed_text)

    @lru_cache(maxsize=_EMBEDDING_CACHE_SIZE)
    def _get_embedding_cached(self, text: str) -> np.ndarray:
        """Cached embedding computation."""
        return self.model.encode(text, normalize_embeddings=True)

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two normalized embeddings."""
        # Since embeddings are normalized, dot product = cosine similarity
        return float(np.dot(emb1, emb2))

    def _similarities(
            self,
            cand_word: str = None,
            ref_words: list[str] = None,
            **kwargs
    ) -> Union[dict[str, float], None]:
        """Compute similarities between candidate and reference words.

        Args:
            cand_word: The candidate word to compare.
            ref_words: List of reference words to compare against.
            **kwargs: Additional arguments (ignored for compatibility).

        Returns:
            Dictionary mapping reference words to similarity scores,
            or None if inputs are invalid.
        """
        if cand_word is None or ref_words is None:
            logger.warning(f"Missing required arguments: {cand_word=}, {ref_words=}")
            return None

        if not ref_words:
            logger.warning("Empty reference words list")
            return None

        logger.debug(f"Computing similarities: cand={cand_word}, refs={ref_words}")

        # Get candidate embedding
        cand_emb = self._get_embedding(cand_word)

        # Compute similarities with all reference words
        similarities = {}
        for ref_word in ref_words:
            ref_emb = self._get_embedding(ref_word)
            sim = self._cosine_similarity(cand_emb, ref_emb)
            similarities[ref_word] = sim
            logger.debug(f"  {cand_word} <-> {ref_word}: {sim:.4f}")

        return similarities

    def filter_oov(self, words: list[str]) -> list[str]:
        """Filter out-of-vocabulary words.

        For sentence-transformers, all words are in-vocabulary due to
        subword tokenization, so this returns the input unchanged.
        """
        # Sentence transformers handle all words via subword tokenization
        return words

    def clear_cache(self):
        """Clear the embedding cache."""
        self._get_embedding_cached.cache_clear()

    def cache_info(self) -> dict:
        """Get cache statistics."""
        info = self._get_embedding_cached.cache_info()
        return {
            'hits': info.hits,
            'misses': info.misses,
            'maxsize': info.maxsize,
            'currsize': info.currsize,
            'hit_rate': info.hits / (info.hits + info.misses) if (info.hits + info.misses) > 0 else 0.0
        }
