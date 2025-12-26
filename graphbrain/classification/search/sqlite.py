"""SQLite search backend with FTS5 full-text search.

Provides:
- BM25 ranking via SQLite FTS5
- Optional in-process semantic similarity
- Hybrid fusion with configurable weights
"""

import json
import logging
import sqlite3
from typing import Iterator, Optional

from graphbrain.classification.search.base import SearchBackend, SearchResult

logger = logging.getLogger(__name__)

# Try to import sentence-transformers for semantic search
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False


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


class SqliteSearchBackend(SearchBackend):
    """SQLite search backend using FTS5."""

    DEFAULT_MODEL = "intfloat/e5-base-v2"

    def __init__(
        self,
        db_path: str,
        embedding_model: Optional[str] = None,
        default_weights: Optional[dict] = None,
    ):
        """Initialize with SQLite database path.

        Args:
            db_path: Path to SQLite database file
            embedding_model: Sentence transformer model name (optional)
            default_weights: Default fusion weights {"bm25": float, "semantic": float}
        """
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        self._default_weights = default_weights or self.DEFAULT_WEIGHTS
        self._encoder = None
        self._model_name = embedding_model or self.DEFAULT_MODEL
        self._embeddings_cache: dict[str, list[float]] = {}

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

        # Update cache
        if embedding:
            self._embeddings_cache[edge_key] = embedding

    def bm25_search(
        self,
        query: str,
        limit: int = 100,
        min_rank: float = 0.0,
    ) -> Iterator[SearchResult]:
        """Search using SQLite FTS5 BM25 ranking."""
        if not self.supports_bm25:
            # Fallback to simple LIKE search
            yield from self._simple_search(query, limit)
            return

        try:
            cur = self._conn.execute(
                """
                SELECT edge_key, text_content, bm25(edge_text_fts) as rank
                FROM edge_text_fts
                WHERE edge_text_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit)
            )
            for row in cur:
                # FTS5 bm25() returns negative values (lower is better)
                # Convert to positive score (higher is better)
                score = -row["rank"] if row["rank"] else 0.0
                if score >= min_rank:
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
        """Search using in-process cosine similarity."""
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            return

        # Load embeddings from database if not cached
        if not self._embeddings_cache:
            self._load_embeddings()

        if not self._embeddings_cache:
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
        """Load all embeddings from database into cache."""
        cur = self._conn.execute(
            "SELECT edge_key, embedding FROM edge_text WHERE embedding IS NOT NULL"
        )
        for row in cur:
            try:
                embedding = json.loads(row["embedding"])
                self._embeddings_cache[row["edge_key"]] = embedding
            except json.JSONDecodeError:
                pass

        logger.info(f"Loaded {len(self._embeddings_cache)} embeddings into cache")

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
        """Get search statistics."""
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
        stats["embeddings_cached"] = len(self._embeddings_cache)

        return stats
