"""Hybrid BM25 + Semantic search for edge retrieval.

Combines PostgreSQL full-text search (BM25-like ranking via ts_rank)
with semantic similarity (via pgvector) for improved retrieval.
"""

import logging
from dataclasses import dataclass
from typing import Iterator, Optional

try:
    import psycopg2
    _PSYCOPG2_AVAILABLE = True
except ImportError:
    _PSYCOPG2_AVAILABLE = False

# Optional: sentence-transformers for embedding generation
try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


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


class HybridSearcher:
    """Combines BM25 and semantic similarity for edge retrieval.

    Uses PostgreSQL's full-text search (tsvector/ts_rank) for lexical
    matching and pgvector for semantic similarity. Results are fused
    using configurable weights or Reciprocal Rank Fusion (RRF).

    Usage:
        searcher = HybridSearcher(connection_string)

        # Search with default weights
        results = searcher.search("seattle mayor announcement")

        # Search with custom weights
        results = searcher.search(
            "seattle mayor",
            weights={"bm25": 0.5, "semantic": 0.5}
        )

        # BM25-only search
        results = searcher.bm25_search("said OR claimed OR announced")

        # Semantic-only search
        results = searcher.semantic_search(query_embedding)
    """

    DEFAULT_MODEL = "intfloat/e5-base-v2"
    DEFAULT_WEIGHTS = {"bm25": 0.3, "semantic": 0.7}

    def __init__(
        self,
        connection_string: str,
        embedding_model: Optional[str] = None,
        default_weights: Optional[dict] = None,
    ):
        """Initialize hybrid searcher.

        Args:
            connection_string: PostgreSQL connection URI
            embedding_model: Sentence transformer model name (optional)
            default_weights: Default fusion weights {"bm25": float, "semantic": float}
        """
        if not _PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 not available. Install with: pip install psycopg2-binary")

        # Normalize connection string
        if connection_string.startswith('postgres://'):
            connection_string = 'postgresql://' + connection_string[len('postgres://'):]

        self._conn_string = connection_string
        self._conn = psycopg2.connect(connection_string)
        self._conn.autocommit = True

        self._default_weights = default_weights or self.DEFAULT_WEIGHTS
        self._encoder = None
        self._model_name = embedding_model or self.DEFAULT_MODEL

    def _get_encoder(self):
        """Lazy load the sentence transformer model."""
        if self._encoder is None:
            if not _SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "sentence-transformers not available. "
                    "Install with: pip install sentence-transformers"
                )
            logger.info(f"Loading embedding model: {self._model_name}")
            self._encoder = SentenceTransformer(self._model_name)
        return self._encoder

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def encode(self, text: str) -> list[float]:
        """Encode text into an embedding vector."""
        encoder = self._get_encoder()
        embedding = encoder.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def bm25_search(
        self,
        query: str,
        limit: int = 100,
        min_rank: float = 0.0,
    ) -> Iterator[SearchResult]:
        """Search using PostgreSQL full-text search (BM25-like ranking).

        Args:
            query: Search query (can use tsquery syntax: word1 & word2 | word3)
            limit: Maximum number of results
            min_rank: Minimum ts_rank score to include

        Yields:
            SearchResult with bm25_score populated
        """
        with self._conn.cursor() as cur:
            # Use plainto_tsquery for simple queries, or websearch_to_tsquery for advanced
            cur.execute(
                """
                SELECT edge_key, text_content,
                       ts_rank(to_tsvector('english', COALESCE(text_content, '')),
                               plainto_tsquery('english', %s)) as rank
                FROM edges
                WHERE to_tsvector('english', COALESCE(text_content, ''))
                      @@ plainto_tsquery('english', %s)
                  AND ts_rank(to_tsvector('english', COALESCE(text_content, '')),
                              plainto_tsquery('english', %s)) >= %s
                ORDER BY rank DESC
                LIMIT %s
                """,
                (query, query, query, min_rank, limit)
            )
            for row in cur:
                yield SearchResult(
                    edge_key=row[0],
                    combined_score=row[2],
                    bm25_score=row[2],
                    semantic_score=None,
                    text_content=row[1],
                )

    def semantic_search(
        self,
        embedding: list[float],
        limit: int = 100,
        min_similarity: float = 0.0,
    ) -> Iterator[SearchResult]:
        """Search using pgvector semantic similarity.

        Args:
            embedding: Query embedding vector
            limit: Maximum number of results
            min_similarity: Minimum cosine similarity to include (0-1)

        Yields:
            SearchResult with semantic_score populated
        """
        with self._conn.cursor() as cur:
            try:
                # Cosine similarity: 1 - cosine distance
                cur.execute(
                    """
                    SELECT edge_key, text_content,
                           1 - (embedding <=> %s::vector) as similarity
                    FROM edges
                    WHERE embedding IS NOT NULL
                      AND 1 - (embedding <=> %s::vector) >= %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (embedding, embedding, min_similarity, embedding, limit)
                )
                for row in cur:
                    yield SearchResult(
                        edge_key=row[0],
                        combined_score=row[2],
                        bm25_score=None,
                        semantic_score=row[2],
                        text_content=row[1],
                    )
            except psycopg2.Error as e:
                logger.warning(f"Semantic search failed (pgvector may not be available): {e}")

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
        weights = weights or self._default_weights
        bm25_weight = weights.get("bm25", 0.3)
        semantic_weight = weights.get("semantic", 0.7)

        # Get BM25 results
        bm25_results = {
            r.edge_key: r for r in self.bm25_search(query, limit=limit * 2)
        }

        # Get semantic results
        try:
            query_embedding = self.encode(query)
            semantic_results = {
                r.edge_key: r for r in self.semantic_search(query_embedding, limit=limit * 2)
            }
        except ImportError:
            logger.warning("Semantic search not available, using BM25 only")
            semantic_results = {}

        # Combine results
        all_keys = set(bm25_results.keys()) | set(semantic_results.keys())
        combined = []

        if use_rrf:
            # Reciprocal Rank Fusion
            bm25_ranks = {k: i + 1 for i, k in enumerate(bm25_results.keys())}
            semantic_ranks = {k: i + 1 for i, k in enumerate(semantic_results.keys())}

            for edge_key in all_keys:
                bm25_rank = bm25_ranks.get(edge_key, len(bm25_results) + 1)
                semantic_rank = semantic_ranks.get(edge_key, len(semantic_results) + 1)

                rrf_score = (
                    bm25_weight / (rrf_k + bm25_rank) +
                    semantic_weight / (rrf_k + semantic_rank)
                )

                bm25_result = bm25_results.get(edge_key)
                semantic_result = semantic_results.get(edge_key)

                combined.append(SearchResult(
                    edge_key=edge_key,
                    combined_score=rrf_score,
                    bm25_score=bm25_result.bm25_score if bm25_result else None,
                    semantic_score=semantic_result.semantic_score if semantic_result else None,
                    text_content=(bm25_result or semantic_result).text_content,
                ))
        else:
            # Weighted sum (normalize scores to 0-1 range)
            max_bm25 = max((r.bm25_score for r in bm25_results.values()), default=1.0) or 1.0
            max_semantic = max((r.semantic_score for r in semantic_results.values()), default=1.0) or 1.0

            for edge_key in all_keys:
                bm25_result = bm25_results.get(edge_key)
                semantic_result = semantic_results.get(edge_key)

                bm25_score = (bm25_result.bm25_score / max_bm25) if bm25_result else 0.0
                semantic_score = (semantic_result.semantic_score / max_semantic) if semantic_result else 0.0

                combined_score = bm25_weight * bm25_score + semantic_weight * semantic_score

                combined.append(SearchResult(
                    edge_key=edge_key,
                    combined_score=combined_score,
                    bm25_score=bm25_result.bm25_score if bm25_result else None,
                    semantic_score=semantic_result.semantic_score if semantic_result else None,
                    text_content=(bm25_result or semantic_result).text_content,
                ))

        # Sort by combined score and limit
        combined.sort(key=lambda r: r.combined_score, reverse=True)
        return combined[:limit]

    def search_by_class(
        self,
        class_id: str,
        query: str,
        weights: Optional[dict] = None,
        limit: int = 100,
    ) -> list[SearchResult]:
        """Search within edges classified into a specific semantic class.

        Args:
            class_id: Semantic class ID to filter by
            query: Search query text
            weights: Fusion weights
            limit: Maximum results

        Returns:
            Filtered and scored SearchResults
        """
        weights = weights or self._default_weights

        with self._conn.cursor() as cur:
            # First get edges in the class
            cur.execute(
                """
                SELECT e.edge_key, e.text_content, ec.confidence
                FROM edges e
                JOIN edge_classifications ec ON e.edge_key = ec.edge_key
                WHERE ec.class_id = %s
                """,
                (class_id,)
            )
            class_edges = {row[0]: (row[1], row[2]) for row in cur}

        if not class_edges:
            return []

        # Get hybrid search results
        all_results = self.search(query, weights=weights, limit=limit * 5)

        # Filter to class edges and boost by classification confidence
        filtered = []
        for result in all_results:
            if result.edge_key in class_edges:
                text_content, class_confidence = class_edges[result.edge_key]
                # Boost by classification confidence
                result.combined_score *= class_confidence
                result.text_content = text_content
                filtered.append(result)

        filtered.sort(key=lambda r: r.combined_score, reverse=True)
        return filtered[:limit]

    def build_bm25_query(self, predicates: list[str]) -> str:
        """Build a BM25 query from a list of predicates.

        Args:
            predicates: List of predicate lemmas (e.g., ["say", "claim", "announce"])

        Returns:
            tsquery-compatible query string
        """
        # Create OR query: (said | claimed | announced)
        return " | ".join(predicates)

    def get_stats(self) -> dict:
        """Get search statistics."""
        stats = {}
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM edges WHERE text_content IS NOT NULL"
            )
            stats["edges_with_text"] = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM edges WHERE embedding IS NOT NULL"
            )
            stats["edges_with_embedding"] = cur.fetchone()[0]

        return stats
