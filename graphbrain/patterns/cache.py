"""Pattern matching caching system.

Provides LRU caches for:
- Pattern compilation results
- Atomic pattern match results
- Lemma lookups
"""

import logging
import weakref
from functools import lru_cache
from typing import Optional, Tuple

from graphbrain.hyperedge import Hyperedge


logger = logging.getLogger(__name__)


# Configuration for cache sizes
PATTERN_CACHE_SIZE = 1000
ATOMIC_MATCH_CACHE_SIZE = 10000
LEMMA_CACHE_SIZE = 5000


@lru_cache(maxsize=PATTERN_CACHE_SIZE)
def parse_argroles(argroles_str: str) -> Tuple[str, str, bool]:
    """Parse argroles string into components.

    Args:
        argroles_str: The argroles string from a pattern.

    Returns:
        Tuple of (required_roles, optional_roles, match_by_order).
    """
    if not argroles_str:
        return ('', '', True)

    argroles_posopt = argroles_str.split('-')[0]

    if len(argroles_posopt) > 0 and argroles_posopt[0] == '{':
        match_by_order = False
        argroles_posopt = argroles_posopt[1:-1]
    else:
        match_by_order = True

    argroles = argroles_posopt.split(',')[0]
    argroles_opt = argroles_posopt.replace(',', '')

    return (argroles, argroles_opt, match_by_order)


@lru_cache(maxsize=ATOMIC_MATCH_CACHE_SIZE)
def cached_atomic_match(edge_str: str, pattern_str: str) -> bool:
    """Cache atomic pattern matching results.

    Uses string keys for hashability.

    Args:
        edge_str: String representation of edge.
        pattern_str: String representation of pattern.

    Returns:
        Whether the edge matches the pattern.
    """
    from graphbrain.hyperedge import hedge
    from graphbrain.patterns.atoms import _matches_atomic_pattern

    edge = hedge(edge_str)
    pattern = hedge(pattern_str)
    return _matches_atomic_pattern(edge, pattern)


class LemmaCache:
    """Weak-reference cache for lemma lookups.

    Uses weak references to avoid holding hypergraph in memory.
    """

    def __init__(self, maxsize: int = LEMMA_CACHE_SIZE):
        self._cache: dict = {}
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0

    def get(self, hg, edge) -> Optional[Hyperedge]:
        """Get cached lemma for edge.

        Args:
            hg: Hypergraph to look up lemma in.
            edge: Edge to get lemma for.

        Returns:
            Cached lemma or None if not cached.
        """
        key = (id(hg), edge.to_str())
        result = self._cache.get(key)
        if result is not None:
            self._hits += 1
            return result
        self._misses += 1
        return None

    def put(self, hg, edge, lemma_edge: Hyperedge) -> None:
        """Cache lemma for edge.

        Args:
            hg: Hypergraph the lemma was looked up in.
            edge: Edge the lemma is for.
            lemma_edge: The lemma edge to cache.
        """
        # Evict oldest entries if cache is full
        if len(self._cache) >= self._maxsize:
            # Remove 10% of oldest entries
            to_remove = self._maxsize // 10
            keys = list(self._cache.keys())[:to_remove]
            for key in keys:
                del self._cache[key]

        key = (id(hg), edge.to_str())
        self._cache[key] = lemma_edge

    def stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'size': len(self._cache),
            'maxsize': self._maxsize,
        }

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0


# Global lemma cache instance
_lemma_cache = LemmaCache()


def get_lemma_cache() -> LemmaCache:
    """Get the global lemma cache."""
    return _lemma_cache


def clear_all_caches() -> None:
    """Clear all pattern matching caches."""
    parse_argroles.cache_clear()
    cached_atomic_match.cache_clear()
    _lemma_cache.clear()
    logger.debug("All pattern caches cleared")


def get_cache_stats() -> dict:
    """Get statistics for all caches."""
    argroles_info = parse_argroles.cache_info()
    atomic_info = cached_atomic_match.cache_info()

    return {
        'argroles': {
            'hits': argroles_info.hits,
            'misses': argroles_info.misses,
            'size': argroles_info.currsize,
            'maxsize': argroles_info.maxsize,
        },
        'atomic_match': {
            'hits': atomic_info.hits,
            'misses': atomic_info.misses,
            'size': atomic_info.currsize,
            'maxsize': atomic_info.maxsize,
        },
        'lemma': _lemma_cache.stats(),
    }
