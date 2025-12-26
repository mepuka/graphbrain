"""Abstract base class for search backends.

Defines the interface for BM25 and semantic search operations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class SearchResult:
    """Result from hybrid search.

    Attributes:
        edge_key: The edge key (string representation)
        combined_score: Combined score from fusion
        bm25_score: BM25/full-text score component
        semantic_score: Semantic similarity score component
        text_content: Original text content (if available)
    """
    edge_key: str
    combined_score: float
    bm25_score: Optional[float] = None
    semantic_score: Optional[float] = None
    text_content: Optional[str] = None


class SearchBackend(ABC):
    """Abstract base class for search backends.

    Implementations must provide:
    - BM25/full-text search
    - Semantic similarity search (optional)
    - Hybrid search combining both
    """

    DEFAULT_WEIGHTS = {"bm25": 0.3, "semantic": 0.7}

    @abstractmethod
    def close(self):
        """Close any database connections."""
        pass

    @abstractmethod
    def bm25_search(
        self,
        query: str,
        limit: int = 100,
        min_rank: float = 0.0,
    ) -> Iterator[SearchResult]:
        """Search using full-text search (BM25-like ranking).

        Args:
            query: Search query
            limit: Maximum number of results
            min_rank: Minimum rank score to include

        Yields:
            SearchResult with bm25_score populated
        """
        pass

    def semantic_search(
        self,
        embedding: list[float],
        limit: int = 100,
        min_similarity: float = 0.0,
    ) -> Iterator[SearchResult]:
        """Search using semantic similarity.

        Default implementation returns empty iterator.
        PostgreSQL uses pgvector, SQLite can use in-process computation.

        Args:
            embedding: Query embedding vector
            limit: Maximum number of results
            min_similarity: Minimum cosine similarity to include (0-1)

        Yields:
            SearchResult with semantic_score populated
        """
        return iter([])

    def encode(self, text: str) -> list[float]:
        """Encode text into an embedding vector.

        Default implementation raises NotImplementedError.
        Subclasses with semantic search support should override.

        Args:
            text: Text to encode

        Returns:
            Embedding vector as list of floats
        """
        raise NotImplementedError("Semantic encoding not available in this backend")

    @abstractmethod
    def search(
        self,
        query: str,
        weights: Optional[dict] = None,
        limit: int = 100,
        use_rrf: bool = False,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        """Hybrid search combining BM25 and semantic similarity.

        Args:
            query: Search query text
            weights: Fusion weights {"bm25": float, "semantic": float}
            limit: Maximum number of results
            use_rrf: Use Reciprocal Rank Fusion instead of weighted sum
            rrf_k: RRF constant (default 60)

        Returns:
            List of SearchResults sorted by combined_score descending
        """
        pass

    def search_by_class(
        self,
        class_id: str,
        query: str,
        weights: Optional[dict] = None,
        limit: int = 100,
    ) -> list[SearchResult]:
        """Search within edges classified into a specific semantic class.

        Default implementation does full search then filters.
        Subclasses may optimize with database-level filtering.

        Args:
            class_id: Semantic class ID to filter by
            query: Search query text
            weights: Fusion weights
            limit: Maximum results

        Returns:
            Filtered and scored SearchResults
        """
        # Default: no class filtering (subclasses should override)
        return self.search(query, weights=weights, limit=limit)

    def build_bm25_query(self, predicates: list[str]) -> str:
        """Build a BM25 query from a list of predicates.

        Args:
            predicates: List of predicate lemmas (e.g., ["say", "claim", "announce"])

        Returns:
            Query string suitable for bm25_search
        """
        return " ".join(predicates)

    def get_stats(self) -> dict:
        """Get search statistics.

        Returns:
            Dictionary with statistics about searchable content
        """
        return {}

    @property
    def supports_semantic_search(self) -> bool:
        """Whether this backend supports semantic search."""
        return False

    @property
    def supports_bm25(self) -> bool:
        """Whether this backend supports BM25/full-text search."""
        return True
