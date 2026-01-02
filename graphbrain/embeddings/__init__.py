"""Embedding utilities for graphbrain.

This module provides configurable embedding support with:
- Multiple distance metrics (cosine, L2, inner product)
- Configurable index types (HNSW, IVFFlat)
- Batch processing utilities
- LRU caching for in-memory embeddings
"""

from graphbrain.embeddings.config import EmbeddingConfig
from graphbrain.embeddings.batch import BatchEmbeddingProcessor
from graphbrain.embeddings.cache import LRUEmbeddingCache

__all__ = [
    "EmbeddingConfig",
    "BatchEmbeddingProcessor",
    "LRUEmbeddingCache",
]
