"""PostgreSQL search backend using tsvector and pgvector.

Provides:
- BM25-like full-text search via ts_rank
- Semantic similarity via pgvector cosine distance
- Hybrid fusion with configurable weights or RRF
"""

import logging
from typing import Iterator, Optional

from graphbrain.classification.search.base import SearchBackend, SearchResult

try:
    import psycopg2
    _PSYCOPG2_AVAILABLE = True
except ImportError:
    _PSYCOPG2_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class PostgresSearchBackend(SearchBackend):
    """PostgreSQL search backend using tsvector and pgvector."""

    DEFAULT_MODEL = "intfloat/e5-base-v2"

    def __init__(
        self,
        connection_string: str,
        embedding_model: Optional[str] = None,
        default_weights: Optional[dict] = None,
    ):
        """Initialize with PostgreSQL connection.

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
        self._has_pgvector = self._check_pgvector()

    def _check_pgvector(self) -> bool:
        """Check if pgvector extension is available."""
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                return cur.fetchone() is not None
        except Exception:
            return False

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

    @property
    def supports_semantic_search(self) -> bool:
        """Whether this backend supports semantic search."""
        return self._has_pgvector and _SENTENCE_TRANSFORMERS_AVAILABLE

    def bm25_search(
        self,
        query: str,
        limit: int = 100,
        min_rank: float = 0.0,
    ) -> Iterator[SearchResult]:
        """Search using PostgreSQL full-text search (BM25-like ranking)."""
        with self._conn.cursor() as cur:
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
        """Search using pgvector semantic similarity."""
        if not self._has_pgvector:
            logger.warning("pgvector not available, skipping semantic search")
            return

        with self._conn.cursor() as cur:
            try:
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
                logger.warning(f"Semantic search failed: {e}")

    def search(
        self,
        query: str,
        weights: Optional[dict] = None,
        limit: int = 100,
        use_rrf: bool = False,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        """Hybrid search combining BM25 and semantic similarity."""
        weights = weights or self._default_weights
        bm25_weight = weights.get("bm25", 0.3)
        semantic_weight = weights.get("semantic", 0.7)

        # Get BM25 results
        bm25_results = {
            r.edge_key: r for r in self.bm25_search(query, limit=limit * 2)
        }

        # Get semantic results if available
        semantic_results = {}
        if self.supports_semantic_search:
            try:
                query_embedding = self.encode(query)
                semantic_results = {
                    r.edge_key: r for r in self.semantic_search(query_embedding, limit=limit * 2)
                }
            except ImportError:
                logger.warning("Semantic search not available, using BM25 only")

        # If no semantic results, just return BM25
        if not semantic_results:
            results = list(bm25_results.values())
            results.sort(key=lambda r: r.combined_score, reverse=True)
            return results[:limit]

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

        combined.sort(key=lambda r: r.combined_score, reverse=True)
        return combined[:limit]

    def search_by_class(
        self,
        class_id: str,
        query: str,
        weights: Optional[dict] = None,
        limit: int = 100,
    ) -> list[SearchResult]:
        """Search within edges classified into a specific semantic class."""
        weights = weights or self._default_weights

        with self._conn.cursor() as cur:
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
                result.combined_score *= class_confidence
                result.text_content = text_content
                filtered.append(result)

        filtered.sort(key=lambda r: r.combined_score, reverse=True)
        return filtered[:limit]

    def build_bm25_query(self, predicates: list[str]) -> str:
        """Build a BM25 query from a list of predicates."""
        return " | ".join(predicates)

    def get_stats(self) -> dict:
        """Get search statistics."""
        stats = {}
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM edges WHERE text_content IS NOT NULL"
            )
            stats["edges_with_text"] = cur.fetchone()[0]

            if self._has_pgvector:
                cur.execute(
                    "SELECT COUNT(*) FROM edges WHERE embedding IS NOT NULL"
                )
                stats["edges_with_embedding"] = cur.fetchone()[0]
            else:
                stats["edges_with_embedding"] = 0

            stats["pgvector_available"] = self._has_pgvector

        return stats
