"""PostgreSQL backend for classification storage.

Uses psycopg2 for PostgreSQL connection and supports:
- JSONB for hybrid_weights
- pgvector for embeddings (optional, with HNSW or IVFFlat indexes)
- pg_trgm for fuzzy predicate search (optional)
"""

import json
import logging
from datetime import datetime
from typing import Iterator, Optional, TYPE_CHECKING

from graphbrain.classification.backends.base import ClassificationBackend
from graphbrain.classification.models import (
    SemanticClass,
    PredicateBankEntry,
    ClassPattern,
    EdgeClassification,
    ClassificationFeedback,
)

if TYPE_CHECKING:
    from graphbrain.embeddings.config import EmbeddingConfig

try:
    import psycopg2
    import psycopg2.extras
    _PSYCOPG2_AVAILABLE = True
except ImportError:
    _PSYCOPG2_AVAILABLE = False

logger = logging.getLogger(__name__)


# SQL for creating classification tables
_CREATE_CLASSIFICATION_SCHEMA = """
-- Semantic classes
CREATE TABLE IF NOT EXISTS semantic_classes (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    domain TEXT DEFAULT 'default',
    provenance TEXT DEFAULT 'seed',
    confidence REAL DEFAULT 1.0,
    bm25_query TEXT,
    hybrid_weights JSONB DEFAULT '{"bm25": 0.3, "semantic": 0.7}',
    version INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sc_name ON semantic_classes (name);
CREATE INDEX IF NOT EXISTS idx_sc_domain ON semantic_classes (domain);
CREATE INDEX IF NOT EXISTS idx_sc_domain_name ON semantic_classes (domain, name);

-- Predicate banks with full-text search
CREATE TABLE IF NOT EXISTS predicate_banks (
    id SERIAL PRIMARY KEY,
    class_id TEXT NOT NULL REFERENCES semantic_classes(id) ON DELETE CASCADE,
    lemma TEXT NOT NULL,
    similarity_score REAL,
    frequency INTEGER DEFAULT 0,
    is_seed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(class_id, lemma)
);

CREATE INDEX IF NOT EXISTS idx_pb_class ON predicate_banks (class_id);
CREATE INDEX IF NOT EXISTS idx_pb_lemma ON predicate_banks (lemma);
CREATE INDEX IF NOT EXISTS idx_pb_class_freq ON predicate_banks (class_id, is_seed DESC, frequency DESC);

-- Pattern definitions
CREATE TABLE IF NOT EXISTS class_patterns (
    id SERIAL PRIMARY KEY,
    class_id TEXT NOT NULL REFERENCES semantic_classes(id) ON DELETE CASCADE,
    pattern TEXT NOT NULL,
    pattern_type TEXT DEFAULT 'structural',
    priority INTEGER DEFAULT 0,
    match_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(class_id, pattern)
);

CREATE INDEX IF NOT EXISTS idx_cp_class ON class_patterns (class_id);

-- Edge classifications
CREATE TABLE IF NOT EXISTS edge_classifications (
    id SERIAL PRIMARY KEY,
    edge_key TEXT NOT NULL,
    class_id TEXT NOT NULL REFERENCES semantic_classes(id) ON DELETE CASCADE,
    confidence REAL NOT NULL,
    method TEXT NOT NULL,
    bm25_score REAL,
    semantic_score REAL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(edge_key, class_id)
);

CREATE INDEX IF NOT EXISTS idx_ec_edge ON edge_classifications (edge_key);
CREATE INDEX IF NOT EXISTS idx_ec_class ON edge_classifications (class_id);
CREATE INDEX IF NOT EXISTS idx_ec_confidence ON edge_classifications (confidence DESC);
CREATE INDEX IF NOT EXISTS idx_ec_class_confidence ON edge_classifications (class_id, confidence DESC);

-- Classification feedback for active learning
CREATE TABLE IF NOT EXISTS classification_feedback (
    id SERIAL PRIMARY KEY,
    review_id TEXT UNIQUE NOT NULL,
    predicate TEXT NOT NULL,
    original_class TEXT NOT NULL,
    correct_class TEXT NOT NULL,
    confidence_adjustment REAL,
    reviewer_id TEXT,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cf_status ON classification_feedback (status);
CREATE INDEX IF NOT EXISTS idx_cf_status_created ON classification_feedback (status, created_at);
"""

# Template for adding embedding columns - dimensions filled in at runtime
_ADD_EMBEDDING_COLUMNS_TEMPLATE = """
-- Add embedding column to semantic_classes
ALTER TABLE semantic_classes ADD COLUMN IF NOT EXISTS embedding vector({dimensions});
-- Add embedding column to predicate_banks
ALTER TABLE predicate_banks ADD COLUMN IF NOT EXISTS embedding vector({dimensions});
"""


def _get_pgvector_version(conn) -> Optional[tuple]:
    """Get the installed pgvector version as a tuple.

    Returns:
        Version tuple (major, minor, patch) or None if not available.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT extversion FROM pg_extension WHERE extname = 'vector'
            """)
            row = cur.fetchone()
            if row:
                version_str = row[0]
                parts = version_str.split('.')
                return tuple(int(p) for p in parts[:3])
    except Exception:
        pass
    return None


def _get_classification_embedding_index_sql(
    table: str,
    index_type: str,
    distance_metric: str,
    hnsw_m: int = 16,
    hnsw_ef_construction: int = 64,
    ivfflat_lists: int = 100,
) -> str:
    """Generate SQL for creating a vector index on classification tables.

    Args:
        table: Table name ('semantic_classes' or 'predicate_banks')
        index_type: 'hnsw', 'ivfflat', or 'none'
        distance_metric: 'cosine', 'l2', or 'inner_product'
        hnsw_m: HNSW max connections per layer
        hnsw_ef_construction: HNSW construction exploration factor
        ivfflat_lists: Number of IVFFlat clusters

    Returns:
        SQL CREATE INDEX statement, or empty string if index_type is 'none'.
    """
    if index_type == "none":
        return ""

    ops_map = {
        "cosine": "vector_cosine_ops",
        "l2": "vector_l2_ops",
        "inner_product": "vector_ip_ops",
    }
    ops_class = ops_map.get(distance_metric, "vector_cosine_ops")
    prefix = "sc" if table == "semantic_classes" else "pb"

    if index_type == "hnsw":
        return f"""
DO $$
BEGIN
    CREATE INDEX IF NOT EXISTS idx_{prefix}_embedding_hnsw
        ON {table} USING hnsw (embedding {ops_class})
        WITH (m = {hnsw_m}, ef_construction = {hnsw_ef_construction});
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Could not create HNSW index on {table}: %', SQLERRM;
END $$;
"""
    elif index_type == "ivfflat":
        return f"""
DO $$
BEGIN
    CREATE INDEX IF NOT EXISTS idx_{prefix}_embedding_ivfflat
        ON {table} USING ivfflat (embedding {ops_class})
        WITH (lists = {ivfflat_lists});
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Could not create IVFFlat index on {table}: %', SQLERRM;
END $$;
"""
    else:
        return ""

# Add trigram index for fuzzy search
_ADD_TRGM_INDEX = """
DO $$
BEGIN
    CREATE INDEX IF NOT EXISTS idx_pb_lemma_trgm
        ON predicate_banks USING gin (lemma gin_trgm_ops);
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Could not add trigram index (pg_trgm may not be available)';
END $$;
"""


class PostgresBackend(ClassificationBackend):
    """PostgreSQL backend for classification storage.

    Supports configurable embedding dimensions and index types (HNSW or IVFFlat).
    HNSW indexes require pgvector >= 0.5.0; falls back to IVFFlat automatically.
    """

    def __init__(
        self,
        connection_string: str,
        embedding_config: Optional["EmbeddingConfig"] = None,
    ):
        """Initialize with PostgreSQL connection.

        Args:
            connection_string: PostgreSQL connection URI
            embedding_config: Optional embedding configuration for pgvector setup.
                             If not provided, uses default 768-dim with HNSW index.
        """
        if not _PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 not available. Install with: pip install psycopg2-binary")

        # Normalize connection string
        if connection_string.startswith('postgres://'):
            connection_string = 'postgresql://' + connection_string[len('postgres://'):]

        self._conn_string = connection_string
        self._conn = psycopg2.connect(connection_string)
        self._conn.autocommit = True

        # Store embedding config (create default if not provided)
        if embedding_config is None:
            from graphbrain.embeddings.config import EmbeddingConfig
            embedding_config = EmbeddingConfig()
        self.embedding_config = embedding_config

        # Track pgvector version
        self._pgvector_version: Optional[tuple] = None

        self._init_schema()

    def _init_schema(self):
        """Initialize database schema."""
        with self._conn.cursor() as cur:
            cur.execute(_CREATE_CLASSIFICATION_SCHEMA)

            # Check pgvector version - it's required for PostgreSQL backend
            self._pgvector_version = _get_pgvector_version(self._conn)
            if self._pgvector_version is None:
                raise RuntimeError(
                    "pgvector extension is required for PostgreSQL backend. "
                    "Install with: CREATE EXTENSION vector; "
                    "See https://github.com/pgvector/pgvector for installation instructions."
                )

            # Determine index type (fall back to ivfflat if HNSW not supported)
            index_type = self.embedding_config.index_type
            if index_type == "hnsw" and self._pgvector_version < (0, 5, 0):
                # HNSW requires pgvector >= 0.5.0
                logger.warning(
                    f"pgvector {'.'.join(map(str, self._pgvector_version))} "
                    "doesn't support HNSW; falling back to IVFFlat"
                )
                index_type = "ivfflat"

            # Add embedding columns
            add_columns_sql = _ADD_EMBEDDING_COLUMNS_TEMPLATE.format(
                dimensions=self.embedding_config.dimensions
            )
            cur.execute(add_columns_sql)

            # Create indexes for both tables
            for table in ["semantic_classes", "predicate_banks"]:
                index_sql = _get_classification_embedding_index_sql(
                    table=table,
                    index_type=index_type,
                    distance_metric=self.embedding_config.distance_metric,
                    hnsw_m=self.embedding_config.hnsw_m,
                    hnsw_ef_construction=self.embedding_config.hnsw_ef_construction,
                    ivfflat_lists=self.embedding_config.ivfflat_lists,
                )
                if index_sql:
                    cur.execute(index_sql)

            cur.execute(_ADD_TRGM_INDEX)

        logger.debug(
            f"Classification schema initialized (pgvector: "
            f"{'.'.join(map(str, self._pgvector_version))}, "
            f"index: {index_type})"
        )

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # ===================================
    # Semantic Class operations
    # ===================================

    def save_class(self, sem_class: SemanticClass) -> SemanticClass:
        """Save or update a semantic class."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO semantic_classes
                    (id, name, description, domain, provenance, confidence,
                     bm25_query, hybrid_weights, embedding, version, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    domain = EXCLUDED.domain,
                    provenance = EXCLUDED.provenance,
                    confidence = EXCLUDED.confidence,
                    bm25_query = EXCLUDED.bm25_query,
                    hybrid_weights = EXCLUDED.hybrid_weights,
                    embedding = EXCLUDED.embedding,
                    version = semantic_classes.version + 1,
                    updated_at = NOW()
                RETURNING version, updated_at
                """,
                (
                    sem_class.id,
                    sem_class.name,
                    sem_class.description,
                    sem_class.domain,
                    sem_class.provenance,
                    sem_class.confidence,
                    sem_class.bm25_query,
                    json.dumps(sem_class.hybrid_weights),
                    sem_class.embedding,
                    sem_class.version,
                    sem_class.created_at or datetime.now(),
                    sem_class.updated_at or datetime.now(),
                )
            )
            row = cur.fetchone()
            sem_class.version = row[0]
            sem_class.updated_at = row[1]
        return sem_class

    def get_class(self, class_id: str) -> Optional[SemanticClass]:
        """Get a semantic class by ID."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, description, domain, provenance, confidence,
                       bm25_query, hybrid_weights, embedding, version, created_at, updated_at
                FROM semantic_classes
                WHERE id = %s
                """,
                (class_id,)
            )
            row = cur.fetchone()
            if row:
                return self._row_to_class(row)
        return None

    def get_class_by_name(self, name: str, domain: str = "default") -> Optional[SemanticClass]:
        """Get a semantic class by name and domain."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, description, domain, provenance, confidence,
                       bm25_query, hybrid_weights, embedding, version, created_at, updated_at
                FROM semantic_classes
                WHERE name = %s AND domain = %s
                """,
                (name, domain)
            )
            row = cur.fetchone()
            if row:
                return self._row_to_class(row)
        return None

    def list_classes(self, domain: Optional[str] = None) -> Iterator[SemanticClass]:
        """List all semantic classes, optionally filtered by domain."""
        with self._conn.cursor() as cur:
            if domain:
                cur.execute(
                    """
                    SELECT id, name, description, domain, provenance, confidence,
                           bm25_query, hybrid_weights, embedding, version, created_at, updated_at
                    FROM semantic_classes
                    WHERE domain = %s
                    ORDER BY name
                    """,
                    (domain,)
                )
            else:
                cur.execute(
                    """
                    SELECT id, name, description, domain, provenance, confidence,
                           bm25_query, hybrid_weights, embedding, version, created_at, updated_at
                    FROM semantic_classes
                    ORDER BY domain, name
                    """
                )
            for row in cur:
                yield self._row_to_class(row)

    def delete_class(self, class_id: str) -> bool:
        """Delete a semantic class and all associated data."""
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM semantic_classes WHERE id = %s", (class_id,))
            return cur.rowcount > 0

    def _row_to_class(self, row) -> SemanticClass:
        """Convert a database row to SemanticClass."""
        return SemanticClass(
            id=row[0],
            name=row[1],
            description=row[2],
            domain=row[3],
            provenance=row[4],
            confidence=row[5],
            bm25_query=row[6],
            hybrid_weights=row[7] if isinstance(row[7], dict) else json.loads(row[7] or "{}"),
            embedding=row[8],
            version=row[9],
            created_at=row[10],
            updated_at=row[11],
        )

    # ===================================
    # Predicate Bank operations
    # ===================================

    def save_predicate(self, predicate: PredicateBankEntry) -> PredicateBankEntry:
        """Save or update a predicate bank entry."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO predicate_banks
                    (class_id, lemma, similarity_score, frequency, is_seed, embedding, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (class_id, lemma) DO UPDATE SET
                    similarity_score = EXCLUDED.similarity_score,
                    frequency = EXCLUDED.frequency,
                    is_seed = EXCLUDED.is_seed,
                    embedding = EXCLUDED.embedding
                RETURNING id, created_at
                """,
                (
                    predicate.class_id,
                    predicate.lemma,
                    predicate.similarity_score,
                    predicate.frequency,
                    predicate.is_seed,
                    predicate.embedding,
                    predicate.created_at or datetime.now(),
                )
            )
            row = cur.fetchone()
            predicate.id = row[0]
            predicate.created_at = row[1]
        return predicate

    def get_predicates_by_class(self, class_id: str) -> Iterator[PredicateBankEntry]:
        """Get all predicates for a semantic class."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, class_id, lemma, similarity_score, frequency, is_seed, embedding, created_at
                FROM predicate_banks
                WHERE class_id = %s
                ORDER BY is_seed DESC, frequency DESC
                """,
                (class_id,)
            )
            for row in cur:
                yield self._row_to_predicate(row)

    def find_predicate(self, lemma: str) -> Iterator[tuple[PredicateBankEntry, SemanticClass]]:
        """Find all classes that contain a predicate lemma."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT pb.id, pb.class_id, pb.lemma, pb.similarity_score, pb.frequency,
                       pb.is_seed, pb.embedding, pb.created_at,
                       sc.id, sc.name, sc.description, sc.domain, sc.provenance, sc.confidence,
                       sc.bm25_query, sc.hybrid_weights, sc.embedding, sc.version,
                       sc.created_at, sc.updated_at
                FROM predicate_banks pb
                JOIN semantic_classes sc ON pb.class_id = sc.id
                WHERE pb.lemma = %s
                ORDER BY pb.similarity_score DESC NULLS LAST
                """,
                (lemma,)
            )
            for row in cur:
                predicate = self._row_to_predicate(row[:8])
                sem_class = SemanticClass(
                    id=row[8],
                    name=row[9],
                    description=row[10],
                    domain=row[11],
                    provenance=row[12],
                    confidence=row[13],
                    bm25_query=row[14],
                    hybrid_weights=row[15] if isinstance(row[15], dict) else json.loads(row[15] or "{}"),
                    embedding=row[16],
                    version=row[17],
                    created_at=row[18],
                    updated_at=row[19],
                )
                yield predicate, sem_class

    def find_similar_predicates(self, lemma: str, limit: int = 10) -> Iterator[PredicateBankEntry]:
        """Find predicates similar to the given lemma using trigram similarity."""
        with self._conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    SELECT id, class_id, lemma, similarity_score, frequency, is_seed,
                           embedding, created_at, similarity(lemma, %s) as sim
                    FROM predicate_banks
                    WHERE lemma %% %s
                    ORDER BY sim DESC
                    LIMIT %s
                    """,
                    (lemma, lemma, limit)
                )
                for row in cur:
                    yield self._row_to_predicate(row[:8])
            except psycopg2.Error as e:
                logger.warning(f"Trigram search failed (pg_trgm may not be available): {e}")

    def increment_predicate_frequency(self, class_id: str, lemma: str) -> bool:
        """Increment the frequency counter for a predicate."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE predicate_banks
                SET frequency = frequency + 1
                WHERE class_id = %s AND lemma = %s
                """,
                (class_id, lemma)
            )
            return cur.rowcount > 0

    def _row_to_predicate(self, row) -> PredicateBankEntry:
        """Convert a database row to PredicateBankEntry."""
        return PredicateBankEntry(
            id=row[0],
            class_id=row[1],
            lemma=row[2],
            similarity_score=row[3],
            frequency=row[4],
            is_seed=row[5],
            embedding=row[6],
            created_at=row[7],
        )

    # ===================================
    # Pattern operations
    # ===================================

    def save_pattern(self, pattern: ClassPattern) -> ClassPattern:
        """Save or update a class pattern."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO class_patterns
                    (class_id, pattern, pattern_type, priority, match_count, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (class_id, pattern) DO UPDATE SET
                    pattern_type = EXCLUDED.pattern_type,
                    priority = EXCLUDED.priority,
                    match_count = EXCLUDED.match_count
                RETURNING id, created_at
                """,
                (
                    pattern.class_id,
                    pattern.pattern,
                    pattern.pattern_type,
                    pattern.priority,
                    pattern.match_count,
                    pattern.created_at or datetime.now(),
                )
            )
            row = cur.fetchone()
            pattern.id = row[0]
            pattern.created_at = row[1]
        return pattern

    def get_patterns_by_class(self, class_id: str) -> Iterator[ClassPattern]:
        """Get all patterns for a semantic class."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, class_id, pattern, pattern_type, priority, match_count, created_at
                FROM class_patterns
                WHERE class_id = %s
                ORDER BY priority DESC, match_count DESC
                """,
                (class_id,)
            )
            for row in cur:
                yield ClassPattern(
                    id=row[0],
                    class_id=row[1],
                    pattern=row[2],
                    pattern_type=row[3],
                    priority=row[4],
                    match_count=row[5],
                    created_at=row[6],
                )

    def increment_pattern_match_count(self, class_id: str, pattern: str) -> bool:
        """Increment the match count for a pattern."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE class_patterns
                SET match_count = match_count + 1
                WHERE class_id = %s AND pattern = %s
                """,
                (class_id, pattern)
            )
            return cur.rowcount > 0

    def get_all_patterns(self) -> Iterator[tuple[str, list[ClassPattern]]]:
        """Get all patterns grouped by class_id."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT class_id, id, pattern, pattern_type, priority, match_count, created_at
                FROM class_patterns
                ORDER BY class_id, priority DESC
                """
            )
            current_class = None
            patterns = []
            for row in cur:
                if current_class != row[0]:
                    if current_class is not None:
                        yield current_class, patterns
                    current_class = row[0]
                    patterns = []
                patterns.append(ClassPattern(
                    id=row[1],
                    class_id=row[0],
                    pattern=row[2],
                    pattern_type=row[3],
                    priority=row[4],
                    match_count=row[5],
                    created_at=row[6],
                ))
            if current_class is not None:
                yield current_class, patterns

    # ===================================
    # Classification operations
    # ===================================

    def save_classification(self, classification: EdgeClassification) -> EdgeClassification:
        """Save or update an edge classification."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO edge_classifications
                    (edge_key, class_id, confidence, method, bm25_score, semantic_score, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (edge_key, class_id) DO UPDATE SET
                    confidence = EXCLUDED.confidence,
                    method = EXCLUDED.method,
                    bm25_score = EXCLUDED.bm25_score,
                    semantic_score = EXCLUDED.semantic_score
                RETURNING id, created_at
                """,
                (
                    classification.edge_key,
                    classification.class_id,
                    classification.confidence,
                    classification.method,
                    classification.bm25_score,
                    classification.semantic_score,
                    classification.created_at or datetime.now(),
                )
            )
            row = cur.fetchone()
            classification.id = row[0]
            classification.created_at = row[1]
        return classification

    def get_classifications(self, edge_key: str) -> Iterator[EdgeClassification]:
        """Get all classifications for an edge."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, edge_key, class_id, confidence, method, bm25_score, semantic_score, created_at
                FROM edge_classifications
                WHERE edge_key = %s
                ORDER BY confidence DESC
                """,
                (edge_key,)
            )
            for row in cur:
                yield EdgeClassification(
                    id=row[0],
                    edge_key=row[1],
                    class_id=row[2],
                    confidence=row[3],
                    method=row[4],
                    bm25_score=row[5],
                    semantic_score=row[6],
                    created_at=row[7],
                )

    def get_edges_by_class(
        self,
        class_id: str,
        min_confidence: float = 0.0,
        limit: int = 100
    ) -> Iterator[EdgeClassification]:
        """Get edges classified into a semantic class."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, edge_key, class_id, confidence, method, bm25_score, semantic_score, created_at
                FROM edge_classifications
                WHERE class_id = %s AND confidence >= %s
                ORDER BY confidence DESC
                LIMIT %s
                """,
                (class_id, min_confidence, limit)
            )
            for row in cur:
                yield EdgeClassification(
                    id=row[0],
                    edge_key=row[1],
                    class_id=row[2],
                    confidence=row[3],
                    method=row[4],
                    bm25_score=row[5],
                    semantic_score=row[6],
                    created_at=row[7],
                )

    # ===================================
    # Feedback operations
    # ===================================

    def save_feedback(self, feedback: ClassificationFeedback) -> ClassificationFeedback:
        """Save classification feedback."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO classification_feedback
                    (review_id, predicate, original_class, correct_class,
                     confidence_adjustment, reviewer_id, status, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (review_id) DO UPDATE SET
                    correct_class = EXCLUDED.correct_class,
                    confidence_adjustment = EXCLUDED.confidence_adjustment,
                    status = EXCLUDED.status
                RETURNING id, created_at
                """,
                (
                    feedback.review_id,
                    feedback.predicate,
                    feedback.original_class,
                    feedback.correct_class,
                    feedback.confidence_adjustment,
                    feedback.reviewer_id,
                    feedback.status,
                    feedback.created_at or datetime.now(),
                )
            )
            row = cur.fetchone()
            feedback.id = row[0]
            feedback.created_at = row[1]
        return feedback

    def get_pending_feedback(self, limit: int = 100) -> Iterator[ClassificationFeedback]:
        """Get pending feedback entries for review."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, review_id, predicate, original_class, correct_class,
                       confidence_adjustment, reviewer_id, status, created_at
                FROM classification_feedback
                WHERE status = 'pending'
                ORDER BY created_at
                LIMIT %s
                """,
                (limit,)
            )
            for row in cur:
                yield ClassificationFeedback(
                    id=row[0],
                    review_id=row[1],
                    predicate=row[2],
                    original_class=row[3],
                    correct_class=row[4],
                    confidence_adjustment=row[5],
                    reviewer_id=row[6],
                    status=row[7],
                    created_at=row[8],
                )

    def apply_feedback(self, review_id: str) -> bool:
        """Mark feedback as applied."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE classification_feedback
                SET status = 'applied'
                WHERE review_id = %s AND status = 'pending'
                """,
                (review_id,)
            )
            return cur.rowcount > 0

    # ===================================
    # Utility methods
    # ===================================

    def get_stats(self) -> dict:
        """Get statistics about the classification database."""
        stats = {}
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM semantic_classes")
            stats["semantic_classes"] = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM predicate_banks")
            stats["predicates"] = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM class_patterns")
            stats["patterns"] = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM edge_classifications")
            stats["classifications"] = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM classification_feedback WHERE status = 'pending'")
            stats["pending_feedback"] = cur.fetchone()[0]

        return stats

    def clear_all(self):
        """Clear all classification data (use with caution!)."""
        with self._conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE classification_feedback")
            cur.execute("TRUNCATE TABLE edge_classifications")
            cur.execute("TRUNCATE TABLE class_patterns")
            cur.execute("TRUNCATE TABLE predicate_banks")
            cur.execute("TRUNCATE TABLE semantic_classes CASCADE")
        logger.warning("All classification data cleared")
