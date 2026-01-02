"""LRU cache for embeddings.

Thread-safe LRU cache implementation for storing embeddings in memory.
Used by SQLite backend where embeddings are computed in-process.
"""

import threading
from collections import OrderedDict
from typing import Optional


class LRUEmbeddingCache:
    """Thread-safe LRU cache for embedding vectors.

    This cache stores embedding vectors in memory with a maximum size limit.
    When the cache is full, the least recently used items are evicted.

    The cache is thread-safe and can be used from multiple threads.

    Example:
        >>> cache = LRUEmbeddingCache(max_size=10000)
        >>> cache.put("edge_key_1", [0.1, 0.2, 0.3, ...])
        >>> embedding = cache.get("edge_key_1")
        >>> if embedding is None:
        ...     # Cache miss, compute embedding
        ...     embedding = compute_embedding(text)
        ...     cache.put("edge_key_1", embedding)
    """

    def __init__(self, max_size: int = 10000):
        """Initialize the LRU cache.

        Args:
            max_size: Maximum number of embeddings to cache.
                      When exceeded, least recently used items are evicted.
        """
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[list[float]]:
        """Get an embedding from the cache.

        If found, moves the item to the end (most recently used).

        Args:
            key: The cache key (typically edge_key)

        Returns:
            The embedding vector if found, None otherwise
        """
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, embedding: list[float]) -> None:
        """Store an embedding in the cache.

        If the cache is full, evicts the least recently used item.

        Args:
            key: The cache key (typically edge_key)
            embedding: The embedding vector to store
        """
        with self._lock:
            # If key already exists, update and move to end
            if key in self._cache:
                self._cache[key] = embedding
                self._cache.move_to_end(key)
                return

            # Add new item
            self._cache[key] = embedding

            # Evict oldest if over capacity
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def remove(self, key: str) -> bool:
        """Remove an embedding from the cache.

        Args:
            key: The cache key to remove

        Returns:
            True if the key was found and removed, False otherwise
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all items from the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def contains(self, key: str) -> bool:
        """Check if a key exists in the cache without affecting LRU order.

        Args:
            key: The cache key to check

        Returns:
            True if the key exists in the cache
        """
        with self._lock:
            return key in self._cache

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator."""
        return self.contains(key)

    def __len__(self) -> int:
        """Return the number of items in the cache."""
        with self._lock:
            return len(self._cache)

    @property
    def max_size(self) -> int:
        """Get the maximum cache size."""
        return self._max_size

    @max_size.setter
    def max_size(self, value: int) -> None:
        """Set the maximum cache size, evicting items if necessary."""
        with self._lock:
            self._max_size = value
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics:
            - size: Current number of items
            - max_size: Maximum capacity
            - hits: Number of cache hits
            - misses: Number of cache misses
            - hit_rate: Hit rate as a percentage (0-100)
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }

    def keys(self) -> list[str]:
        """Get all keys in the cache (in LRU order, oldest first).

        Returns:
            List of cache keys
        """
        with self._lock:
            return list(self._cache.keys())

    def items(self):
        """Iterate over (key, embedding) pairs in LRU order.

        Note: Creates a copy to avoid holding the lock during iteration.

        Yields:
            Tuples of (key, embedding)
        """
        with self._lock:
            items = list(self._cache.items())
        yield from items

    def load_bulk(self, items: dict[str, list[float]]) -> int:
        """Load multiple embeddings into the cache at once.

        More efficient than calling put() repeatedly.

        Args:
            items: Dictionary of key -> embedding

        Returns:
            Number of items added
        """
        with self._lock:
            count = 0
            for key, embedding in items.items():
                if key not in self._cache:
                    self._cache[key] = embedding
                    count += 1
                else:
                    # Update existing
                    self._cache[key] = embedding
                    self._cache.move_to_end(key)

            # Evict oldest if over capacity
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

            return count

    def export(self) -> dict[str, list[float]]:
        """Export all cached embeddings as a dictionary.

        Returns:
            Dictionary of key -> embedding
        """
        with self._lock:
            return dict(self._cache)
