"""SQLite search backend with FTS5 full-text search.

Provides:
- BM25 ranking via SQLite FTS5 with snippet/highlight support
- Phrase search, prefix search, and boolean operators
- Optional in-process semantic similarity with LRU caching
- Hybrid fusion with configurable weights (weighted sum or RRF)
"""

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from typing import Iterator, Optional, TYPE_CHECKING, Literal

from graphbrain.classification.search.base import SearchBackend, SearchResult
from graphbrain.embeddings.cache import LRUEmbeddingCache

if TYPE_CHECKING:
    from graphbrain.embeddings.config import EmbeddingConfig

logger = logging.getLogger(__name__)


@dataclass
class FTSConfig:
    """Configuration for FTS5 search behavior."""
    # Snippet configuration
    snippet_size: int = 64  # Approximate tokens per snippet
    highlight_start: str = '<mark>'
    highlight_end: str = '</mark>'
    snippet_ellipsis: str = '...'

    # BM25 column weights (key, text_content)
    bm25_weights: tuple = (0.0, 1.0)

    # Query preprocessing
    auto_prefix: bool = False  # Auto-add * to words for prefix matching
    default_operator: Literal['AND', 'OR'] = 'AND'

# Try to import sentence-transformers for semantic search
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
    np = None


# SQL for creating search tables
_CREATE_SEARCH_SCHEMA = """
-- Searchable text content for edges
CREATE TABLE IF NOT EXISTS edge_text (
    edge_key TEXT PRIMARY KEY,
    text_content TEXT NOT NULL,
    embedding TEXT
);

-- FTS5 virtual table for full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS edge_text_fts USING fts5(
    edge_key,
    text_content,
    content='edge_text',
    content_rowid='rowid'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS edge_text_ai AFTER INSERT ON edge_text BEGIN
    INSERT INTO edge_text_fts(rowid, edge_key, text_content)
    VALUES (new.rowid, new.edge_key, new.text_content);
END;

CREATE TRIGGER IF NOT EXISTS edge_text_ad AFTER DELETE ON edge_text BEGIN
    INSERT INTO edge_text_fts(edge_text_fts, rowid, edge_key, text_content)
    VALUES ('delete', old.rowid, old.edge_key, old.text_content);
END;

CREATE TRIGGER IF NOT EXISTS edge_text_au AFTER UPDATE ON edge_text BEGIN
    INSERT INTO edge_text_fts(edge_text_fts, rowid, edge_key, text_content)
    VALUES ('delete', old.rowid, old.edge_key, old.text_content);
    INSERT INTO edge_text_fts(rowid, edge_key, text_content)
    VALUES (new.rowid, new.edge_key, new.text_content);
END;
"""


@dataclass
class SearchResultWithSnippet(SearchResult):
    """Extended search result with snippet support."""
    snippet: Optional[str] = None
    highlighted_text: Optional[str] = None


class SqliteSearchBackend(SearchBackend):
    """SQLite search backend using FTS5 with LRU caching for embeddings."""

    DEFAULT_MODEL = "intfloat/e5-base-v2"
    DEFAULT_CACHE_SIZE = 10000

    def __init__(
        self,
        db_path: str,
        embedding_model: Optional[str] = None,
        default_weights: Optional[dict] = None,
        embedding_config: Optional["EmbeddingConfig"] = None,
        cache_max_size: Optional[int] = None,
        fts_config: Optional[FTSConfig] = None,
    ):
        """Initialize with SQLite database path.

        Args:
            db_path: Path to SQLite database file
            embedding_model: Sentence transformer model name (optional)
            default_weights: Default fusion weights {"bm25": float, "semantic": float}
            embedding_config: Optional embedding configuration for cache settings
            cache_max_size: Maximum number of embeddings to cache (default 10000)
            fts_config: Optional FTS5 configuration for snippets and highlighting
        """
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Performance pragmas - WAL mode for concurrent access
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA cache_size = -32000")  # 32MB cache

        self._default_weights = default_weights or self.DEFAULT_WEIGHTS
        self._encoder = None
        self._fts_config = fts_config or FTSConfig()

        # Use embedding config if provided, otherwise use defaults
        if embedding_config:
            self._model_name = embedding_config.model_name
            max_size = embedding_config.cache_max_size
        else:
            self._model_name = embedding_model or self.DEFAULT_MODEL
            max_size = cache_max_size or self.DEFAULT_CACHE_SIZE

        # Use LRU cache instead of plain dict
        self._embeddings_cache = LRUEmbeddingCache(max_size=max_size)

        self._init_schema()

    def _init_schema(self):
        """Initialize search tables."""
        try:
            self._conn.executescript(_CREATE_SEARCH_SCHEMA)
            self._conn.commit()
            logger.debug("SQLite search schema initialized")
        except sqlite3.OperationalError as e:
            # FTS5 might not be available
            logger.warning(f"Could not create FTS5 tables: {e}")

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
        return _SENTENCE_TRANSFORMERS_AVAILABLE

    @property
    def supports_bm25(self) -> bool:
        """Whether this backend supports BM25/full-text search."""
        # Check if FTS5 table exists
        try:
            cur = self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='edge_text_fts'"
            )
            return cur.fetchone() is not None
        except Exception:
            return False

    def add_text(self, edge_key: str, text_content: str, embedding: list[float] = None):
        """Add or update searchable text for an edge.

        Args:
            edge_key: The edge key
            text_content: Text content to index
            embedding: Optional pre-computed embedding
        """
        embedding_json = json.dumps(embedding) if embedding else None

        self._conn.execute(
            """
            INSERT OR REPLACE INTO edge_text (edge_key, text_content, embedding)
            VALUES (?, ?, ?)
            """,
            (edge_key, text_content, embedding_json)
        )
        self._conn.commit()

        # Update LRU cache
        if embedding:
            self._embeddings_cache.put(edge_key, embedding)

    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for FTS5.

        Handles:
        - Phrase search (quoted strings)
        - Prefix search (word*)
        - Boolean operators (AND, OR, NOT)
        - Auto-prefix mode
        """
        # If query contains FTS5 operators, use as-is
        if any(op in query.upper() for op in ['AND', 'OR', 'NOT', 'NEAR']):
            return query

        # Check for quoted phrases
        if '"' in query:
            return query

        # Check for explicit prefix wildcards
        if '*' in query:
            return query

        # Apply auto-prefix if configured
        if self._fts_config.auto_prefix:
            words = query.split()
            return ' '.join(f'{word}*' for word in words if word)

        # Default: join words with configured operator
        words = query.split()
        if len(words) > 1 and self._fts_config.default_operator == 'AND':
            return ' AND '.join(words)

        return query

    def bm25_search(
        self,
        query: str,
        limit: int = 100,
        min_rank: float = 0.0,
        return_snippets: bool = False,
        return_highlights: bool = False,
    ) -> Iterator[SearchResult]:
        """Search using SQLite FTS5 BM25 ranking.

        Supports FTS5 query syntax:
        - Simple words: "hello world" (uses default_operator)
        - Phrase search: '"exact phrase"'
        - Prefix search: "hel*" matches words starting with "hel"
        - Boolean: "hello AND world", "hello OR world", "hello NOT world"
        - NEAR: "hello NEAR world" (within 10 tokens)

        Args:
            query: Search query (plain text or FTS5 syntax)
            limit: Maximum results
            min_rank: Minimum BM25 score threshold
            return_snippets: Include highlighted snippets in results
            return_highlights: Include full highlighted text in results
        """
        if not self.supports_bm25:
            # Fallback to simple LIKE search
            yield from self._simple_search(query, limit)
            return

        # Preprocess query for FTS5
        fts_query = self._preprocess_query(query)

        try:
            if return_snippets:
                cur = self._conn.execute(
                    """
                    SELECT edge_key, text_content, bm25(edge_text_fts) as rank,
                           snippet(edge_text_fts, 1, ?, ?, ?, ?) as snippet
                    FROM edge_text_fts
                    WHERE edge_text_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (
                        self._fts_config.highlight_start,
                        self._fts_config.highlight_end,
                        self._fts_config.snippet_ellipsis,
                        self._fts_config.snippet_size,
                        fts_query,
                        limit
                    )
                )
            elif return_highlights:
                cur = self._conn.execute(
                    """
                    SELECT edge_key, text_content, bm25(edge_text_fts) as rank,
                           highlight(edge_text_fts, 1, ?, ?) as highlighted
                    FROM edge_text_fts
                    WHERE edge_text_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (
                        self._fts_config.highlight_start,
                        self._fts_config.highlight_end,
                        fts_query,
                        limit
                    )
                )
            else:
                cur = self._conn.execute(
                    """
                    SELECT edge_key, text_content, bm25(edge_text_fts) as rank
                    FROM edge_text_fts
                    WHERE edge_text_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, limit)
                )

            for row in cur:
                # FTS5 bm25() returns negative values (lower is better)
                # Convert to positive score (higher is better)
                score = -row["rank"] if row["rank"] else 0.0
                if score >= min_rank:
                    if return_snippets or return_highlights:
                        extra_field = row[3] if len(row) > 3 else None
                        yield SearchResultWithSnippet(
                            edge_key=row["edge_key"],
                            combined_score=score,
                            bm25_score=score,
                            semantic_score=None,
                            text_content=row["text_content"],
                            snippet=extra_field if return_snippets else None,
                            highlighted_text=extra_field if return_highlights else None,
                        )
                    else:
                        yield SearchResult(
                            edge_key=row["edge_key"],
                            combined_score=score,
                            bm25_score=score,
                            semantic_score=None,
                            text_content=row["text_content"],
                        )
        except sqlite3.OperationalError as e:
            logger.warning(f"FTS5 search failed: {e}")
            yield from self._simple_search(query, limit)

    def phrase_search(
        self,
        phrase: str,
        limit: int = 100,
        return_snippets: bool = True,
    ) -> Iterator[SearchResult]:
        """Search for an exact phrase.

        Args:
            phrase: Exact phrase to search for
            limit: Maximum results
            return_snippets: Include highlighted snippets
        """
        # Wrap in quotes for phrase search
        fts_query = f'"{phrase}"'
        yield from self.bm25_search(
            fts_query,
            limit=limit,
            return_snippets=return_snippets,
        )

    def prefix_search(
        self,
        prefix: str,
        limit: int = 100,
    ) -> Iterator[SearchResult]:
        """Search for words starting with a prefix.

        Args:
            prefix: Word prefix to match
            limit: Maximum results
        """
        fts_query = f'{prefix}*'
        yield from self.bm25_search(fts_query, limit=limit)

    def _simple_search(self, query: str, limit: int) -> Iterator[SearchResult]:
        """Fallback simple text search using LIKE."""
        # Split query into words for OR matching
        words = query.lower().split()
        if not words:
            return

        # Build LIKE conditions
        conditions = " OR ".join(["LOWER(text_content) LIKE ?" for _ in words])
        params = [f"%{word}%" for word in words]
        params.append(limit)

        cur = self._conn.execute(
            f"""
            SELECT edge_key, text_content
            FROM edge_text
            WHERE {conditions}
            LIMIT ?
            """,
            params
        )

        for i, row in enumerate(cur):
            yield SearchResult(
                edge_key=row["edge_key"],
                combined_score=1.0 / (i + 1),  # Rank by position
                bm25_score=1.0 / (i + 1),
                semantic_score=None,
                text_content=row["text_content"],
            )

    def semantic_search(
        self,
        embedding: list[float],
        limit: int = 100,
        min_similarity: float = 0.0,
    ) -> Iterator[SearchResult]:
        """Search using in-process cosine similarity with LRU caching."""
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            return

        # Load embeddings from database if cache is empty
        if len(self._embeddings_cache) == 0:
            self._load_embeddings()

        if len(self._embeddings_cache) == 0:
            return

        # Compute similarities
        query_vec = np.array(embedding)
        query_norm = np.linalg.norm(query_vec)

        similarities = []
        for edge_key, emb in self._embeddings_cache.items():
            emb_vec = np.array(emb)
            emb_norm = np.linalg.norm(emb_vec)
            if emb_norm > 0 and query_norm > 0:
                similarity = np.dot(query_vec, emb_vec) / (query_norm * emb_norm)
                if similarity >= min_similarity:
                    similarities.append((edge_key, float(similarity)))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get text content and yield results
        for edge_key, similarity in similarities[:limit]:
            cur = self._conn.execute(
                "SELECT text_content FROM edge_text WHERE edge_key = ?",
                (edge_key,)
            )
            row = cur.fetchone()
            text_content = row["text_content"] if row else None

            yield SearchResult(
                edge_key=edge_key,
                combined_score=similarity,
                bm25_score=None,
                semantic_score=similarity,
                text_content=text_content,
            )

    def _load_embeddings(self):
        """Load embeddings from database into LRU cache.

        Loads embeddings in bulk for efficiency. The cache will automatically
        evict least recently used items if max_size is exceeded.
        """
        cur = self._conn.execute(
            "SELECT edge_key, embedding FROM edge_text WHERE embedding IS NOT NULL"
        )

        # Collect embeddings for bulk loading
        embeddings_to_load = {}
        for row in cur:
            try:
                embedding = json.loads(row["embedding"])
                embeddings_to_load[row["edge_key"]] = embedding
            except json.JSONDecodeError:
                pass

        # Bulk load into cache
        if embeddings_to_load:
            count = self._embeddings_cache.load_bulk(embeddings_to_load)
            logger.info(f"Loaded {count} embeddings into LRU cache (max_size={self._embeddings_cache.max_size})")

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
                pass

        # If no results at all, return empty
        if not bm25_results and not semantic_results:
            return []

        # If only BM25 results, return those
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
            # Weighted sum
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

    def get_stats(self) -> dict:
        """Get search statistics including LRU cache metrics."""
        stats = {}

        try:
            cur = self._conn.execute("SELECT COUNT(*) FROM edge_text")
            stats["edges_with_text"] = cur.fetchone()[0]
        except sqlite3.OperationalError:
            stats["edges_with_text"] = 0

        try:
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM edge_text WHERE embedding IS NOT NULL"
            )
            stats["edges_with_embedding"] = cur.fetchone()[0]
        except sqlite3.OperationalError:
            stats["edges_with_embedding"] = 0

        stats["fts5_available"] = self.supports_bm25

        # Include LRU cache statistics
        cache_stats = self._embeddings_cache.get_stats()
        stats["embeddings_cached"] = cache_stats["size"]
        stats["cache_max_size"] = cache_stats["max_size"]
        stats["cache_hit_rate"] = cache_stats["hit_rate"]
        stats["cache_hits"] = cache_stats["hits"]
        stats["cache_misses"] = cache_stats["misses"]

        return stats
