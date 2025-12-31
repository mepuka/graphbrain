"""SQLite backend for classification storage.

Provides the same interface as PostgreSQL backend but using SQLite.
Some features use fallback implementations:
- Fuzzy search: Uses LIKE queries + Python's difflib.SequenceMatcher
  instead of PostgreSQL's pg_trgm trigram similarity
- Embeddings: Stored as JSON text (no native vector operations)
- hybrid_weights: Stored as JSON text
"""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Iterator, Optional

from graphbrain.classification.backends.base import ClassificationBackend
from graphbrain.classification.models import (
    SemanticClass,
    PredicateBankEntry,
    ClassPattern,
    EdgeClassification,
    ClassificationFeedback,
)

logger = logging.getLogger(__name__)


# SQL for creating classification tables (SQLite syntax)
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
    hybrid_weights TEXT DEFAULT '{"bm25": 0.3, "semantic": 0.7}',
    embedding TEXT,
    version INTEGER DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sc_name ON semantic_classes (name);
CREATE INDEX IF NOT EXISTS idx_sc_domain ON semantic_classes (domain);

-- Predicate banks
CREATE TABLE IF NOT EXISTS predicate_banks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id TEXT NOT NULL REFERENCES semantic_classes(id) ON DELETE CASCADE,
    lemma TEXT NOT NULL,
    similarity_score REAL,
    frequency INTEGER DEFAULT 0,
    is_seed INTEGER DEFAULT 0,
    embedding TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(class_id, lemma)
);

CREATE INDEX IF NOT EXISTS idx_pb_class ON predicate_banks (class_id);
CREATE INDEX IF NOT EXISTS idx_pb_lemma ON predicate_banks (lemma);
-- Composite index for predicate frequency queries
CREATE INDEX IF NOT EXISTS idx_pb_class_freq ON predicate_banks (class_id, frequency DESC);

-- Pattern definitions
CREATE TABLE IF NOT EXISTS class_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id TEXT NOT NULL REFERENCES semantic_classes(id) ON DELETE CASCADE,
    pattern TEXT NOT NULL,
    pattern_type TEXT DEFAULT 'structural',
    priority INTEGER DEFAULT 0,
    match_count INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(class_id, pattern)
);

CREATE INDEX IF NOT EXISTS idx_cp_class ON class_patterns (class_id);

-- Edge classifications
CREATE TABLE IF NOT EXISTS edge_classifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    edge_key TEXT NOT NULL,
    class_id TEXT NOT NULL REFERENCES semantic_classes(id) ON DELETE CASCADE,
    confidence REAL NOT NULL,
    method TEXT NOT NULL,
    bm25_score REAL,
    semantic_score REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(edge_key, class_id)
);

CREATE INDEX IF NOT EXISTS idx_ec_edge ON edge_classifications (edge_key);
CREATE INDEX IF NOT EXISTS idx_ec_class ON edge_classifications (class_id);
CREATE INDEX IF NOT EXISTS idx_ec_confidence ON edge_classifications (confidence DESC);
-- Composite index for get_edges_by_class with confidence filter
CREATE INDEX IF NOT EXISTS idx_ec_class_conf ON edge_classifications (class_id, confidence DESC);

-- Classification feedback for active learning
CREATE TABLE IF NOT EXISTS classification_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id TEXT UNIQUE NOT NULL,
    predicate TEXT NOT NULL,
    original_class TEXT NOT NULL,
    correct_class TEXT NOT NULL,
    confidence_adjustment REAL,
    reviewer_id TEXT,
    status TEXT DEFAULT 'pending',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cf_status ON classification_feedback (status);
"""


class SqliteBackend(ClassificationBackend):
    """SQLite backend for classification storage."""

    def __init__(self, db_path: str):
        """Initialize with SQLite database path.

        Args:
            db_path: Path to SQLite database file
        """
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Performance pragmas - WAL mode for concurrent access
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA cache_size = -32000")  # 32MB cache
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema."""
        self._conn.executescript(_CREATE_CLASSIFICATION_SCHEMA)
        self._conn.commit()
        logger.debug("SQLite classification schema initialized")

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _now(self) -> str:
        """Get current timestamp in SQLite format."""
        return datetime.now().isoformat()

    # ===================================
    # Semantic Class operations
    # ===================================

    def save_class(self, sem_class: SemanticClass) -> SemanticClass:
        """Save or update a semantic class."""
        now = self._now()
        embedding_json = json.dumps(sem_class.embedding) if sem_class.embedding else None

        # Try update first
        cur = self._conn.execute(
            """
            UPDATE semantic_classes SET
                name = ?, description = ?, domain = ?, provenance = ?,
                confidence = ?, bm25_query = ?, hybrid_weights = ?,
                embedding = ?, version = version + 1, updated_at = ?
            WHERE id = ?
            """,
            (
                sem_class.name, sem_class.description, sem_class.domain,
                sem_class.provenance, sem_class.confidence, sem_class.bm25_query,
                json.dumps(sem_class.hybrid_weights), embedding_json, now, sem_class.id
            )
        )

        if cur.rowcount == 0:
            # Insert new
            self._conn.execute(
                """
                INSERT INTO semantic_classes
                    (id, name, description, domain, provenance, confidence,
                     bm25_query, hybrid_weights, embedding, version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sem_class.id, sem_class.name, sem_class.description,
                    sem_class.domain, sem_class.provenance, sem_class.confidence,
                    sem_class.bm25_query, json.dumps(sem_class.hybrid_weights),
                    embedding_json, sem_class.version or 1,
                    sem_class.created_at or now, now
                )
            )

        self._conn.commit()

        # Fetch updated version
        cur = self._conn.execute(
            "SELECT version, updated_at FROM semantic_classes WHERE id = ?",
            (sem_class.id,)
        )
        row = cur.fetchone()
        if row:
            sem_class.version = row["version"]
            sem_class.updated_at = row["updated_at"]

        return sem_class

    def get_class(self, class_id: str) -> Optional[SemanticClass]:
        """Get a semantic class by ID."""
        cur = self._conn.execute(
            """
            SELECT id, name, description, domain, provenance, confidence,
                   bm25_query, hybrid_weights, embedding, version, created_at, updated_at
            FROM semantic_classes
            WHERE id = ?
            """,
            (class_id,)
        )
        row = cur.fetchone()
        if row:
            return self._row_to_class(row)
        return None

    def get_class_by_name(self, name: str, domain: str = "default") -> Optional[SemanticClass]:
        """Get a semantic class by name and domain."""
        cur = self._conn.execute(
            """
            SELECT id, name, description, domain, provenance, confidence,
                   bm25_query, hybrid_weights, embedding, version, created_at, updated_at
            FROM semantic_classes
            WHERE name = ? AND domain = ?
            """,
            (name, domain)
        )
        row = cur.fetchone()
        if row:
            return self._row_to_class(row)
        return None

    def list_classes(self, domain: Optional[str] = None) -> Iterator[SemanticClass]:
        """List all semantic classes, optionally filtered by domain."""
        if domain:
            cur = self._conn.execute(
                """
                SELECT id, name, description, domain, provenance, confidence,
                       bm25_query, hybrid_weights, embedding, version, created_at, updated_at
                FROM semantic_classes
                WHERE domain = ?
                ORDER BY name
                """,
                (domain,)
            )
        else:
            cur = self._conn.execute(
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
        cur = self._conn.execute("DELETE FROM semantic_classes WHERE id = ?", (class_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def _row_to_class(self, row) -> SemanticClass:
        """Convert a database row to SemanticClass."""
        embedding = None
        if row["embedding"]:
            try:
                embedding = json.loads(row["embedding"])
            except json.JSONDecodeError:
                pass

        hybrid_weights = {"bm25": 0.3, "semantic": 0.7}
        if row["hybrid_weights"]:
            try:
                hybrid_weights = json.loads(row["hybrid_weights"])
            except json.JSONDecodeError:
                pass

        return SemanticClass(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            domain=row["domain"],
            provenance=row["provenance"],
            confidence=row["confidence"],
            bm25_query=row["bm25_query"],
            hybrid_weights=hybrid_weights,
            embedding=embedding,
            version=row["version"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ===================================
    # Predicate Bank operations
    # ===================================

    def save_predicate(self, predicate: PredicateBankEntry) -> PredicateBankEntry:
        """Save or update a predicate bank entry."""
        now = self._now()
        embedding_json = json.dumps(predicate.embedding) if predicate.embedding else None

        # Try update first
        cur = self._conn.execute(
            """
            UPDATE predicate_banks SET
                similarity_score = ?, frequency = ?, is_seed = ?, embedding = ?
            WHERE class_id = ? AND lemma = ?
            """,
            (
                predicate.similarity_score, predicate.frequency,
                1 if predicate.is_seed else 0, embedding_json,
                predicate.class_id, predicate.lemma
            )
        )

        if cur.rowcount == 0:
            # Insert new
            self._conn.execute(
                """
                INSERT INTO predicate_banks
                    (class_id, lemma, similarity_score, frequency, is_seed, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    predicate.class_id, predicate.lemma, predicate.similarity_score,
                    predicate.frequency, 1 if predicate.is_seed else 0,
                    embedding_json, predicate.created_at or now
                )
            )

        self._conn.commit()

        # Fetch ID
        cur = self._conn.execute(
            "SELECT id, created_at FROM predicate_banks WHERE class_id = ? AND lemma = ?",
            (predicate.class_id, predicate.lemma)
        )
        row = cur.fetchone()
        if row:
            predicate.id = row["id"]
            predicate.created_at = row["created_at"]

        return predicate

    def get_predicates_by_class(self, class_id: str) -> Iterator[PredicateBankEntry]:
        """Get all predicates for a semantic class."""
        cur = self._conn.execute(
            """
            SELECT id, class_id, lemma, similarity_score, frequency, is_seed, embedding, created_at
            FROM predicate_banks
            WHERE class_id = ?
            ORDER BY is_seed DESC, frequency DESC
            """,
            (class_id,)
        )
        for row in cur:
            yield self._row_to_predicate(row)

    def find_predicate(self, lemma: str) -> Iterator[tuple[PredicateBankEntry, SemanticClass]]:
        """Find all classes that contain a predicate lemma."""
        cur = self._conn.execute(
            """
            SELECT pb.id, pb.class_id, pb.lemma, pb.similarity_score, pb.frequency,
                   pb.is_seed, pb.embedding, pb.created_at,
                   sc.id as sc_id, sc.name, sc.description, sc.domain, sc.provenance, sc.confidence,
                   sc.bm25_query, sc.hybrid_weights, sc.embedding as sc_embedding, sc.version,
                   sc.created_at as sc_created_at, sc.updated_at
            FROM predicate_banks pb
            JOIN semantic_classes sc ON pb.class_id = sc.id
            WHERE pb.lemma = ?
            ORDER BY pb.similarity_score DESC
            """,
            (lemma,)
        )
        for row in cur:
            predicate = self._row_to_predicate(row)

            embedding = None
            if row["sc_embedding"]:
                try:
                    embedding = json.loads(row["sc_embedding"])
                except json.JSONDecodeError:
                    pass

            hybrid_weights = {"bm25": 0.3, "semantic": 0.7}
            if row["hybrid_weights"]:
                try:
                    hybrid_weights = json.loads(row["hybrid_weights"])
                except json.JSONDecodeError:
                    pass

            sem_class = SemanticClass(
                id=row["sc_id"],
                name=row["name"],
                description=row["description"],
                domain=row["domain"],
                provenance=row["provenance"],
                confidence=row["confidence"],
                bm25_query=row["bm25_query"],
                hybrid_weights=hybrid_weights,
                embedding=embedding,
                version=row["version"],
                created_at=row["sc_created_at"],
                updated_at=row["updated_at"],
            )
            yield predicate, sem_class

    def find_similar_predicates(self, lemma: str, limit: int = 10) -> Iterator[PredicateBankEntry]:
        """Find predicates similar to the given lemma.

        SQLite doesn't have pg_trgm, so we use a combination of:
        1. LIKE queries for prefix/substring matching
        2. Python's difflib.SequenceMatcher for fuzzy similarity scoring

        This provides reasonable results for finding similar predicates,
        though not as sophisticated as PostgreSQL's trigram similarity.
        """
        from difflib import SequenceMatcher

        logger.debug(f"find_similar_predicates (SQLite fallback) for: {lemma}")

        # First, get all predicates that might be similar
        # Use LIKE for basic substring matching to reduce candidates
        candidates = []

        # Strategy 1: Exact prefix match (e.g., "say" matches "saying")
        # Strategy 2: Contains the query (e.g., "say" matches "unsay")
        # Strategy 3: Query contains the predicate (e.g., "saying" matches "say")
        cur = self._conn.execute(
            """
            SELECT id, class_id, lemma, similarity_score, frequency, is_seed, embedding, created_at
            FROM predicate_banks
            WHERE lemma LIKE ? || '%'
               OR lemma LIKE '%' || ?
               OR ? LIKE lemma || '%'
            """,
            (lemma, lemma, lemma)
        )

        for row in cur:
            entry = self._row_to_predicate(row)
            candidates.append(entry)

        # If we didn't find enough candidates, get all predicates and filter
        if len(candidates) < limit * 2:
            cur = self._conn.execute(
                """
                SELECT id, class_id, lemma, similarity_score, frequency, is_seed, embedding, created_at
                FROM predicate_banks
                ORDER BY frequency DESC
                LIMIT 500
                """
            )
            existing_lemmas = {c.lemma for c in candidates}
            for row in cur:
                entry = self._row_to_predicate(row)
                if entry.lemma not in existing_lemmas:
                    candidates.append(entry)
                    existing_lemmas.add(entry.lemma)

        # Score all candidates using SequenceMatcher
        scored = []
        for entry in candidates:
            if entry.lemma == lemma:
                continue  # Skip exact match
            # SequenceMatcher ratio returns 0.0-1.0 similarity
            sim = SequenceMatcher(None, lemma.lower(), entry.lemma.lower()).ratio()
            if sim >= 0.3:  # Only include reasonably similar predicates
                scored.append((sim, entry))

        # Sort by similarity score descending
        scored.sort(key=lambda x: -x[0])

        # Yield top results
        for sim, entry in scored[:limit]:
            yield entry

    def increment_predicate_frequency(self, class_id: str, lemma: str) -> bool:
        """Increment the frequency counter for a predicate."""
        cur = self._conn.execute(
            """
            UPDATE predicate_banks
            SET frequency = frequency + 1
            WHERE class_id = ? AND lemma = ?
            """,
            (class_id, lemma)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def _row_to_predicate(self, row) -> PredicateBankEntry:
        """Convert a database row to PredicateBankEntry."""
        embedding = None
        if row["embedding"]:
            try:
                embedding = json.loads(row["embedding"])
            except json.JSONDecodeError:
                pass

        return PredicateBankEntry(
            id=row["id"],
            class_id=row["class_id"],
            lemma=row["lemma"],
            similarity_score=row["similarity_score"],
            frequency=row["frequency"],
            is_seed=bool(row["is_seed"]),
            embedding=embedding,
            created_at=row["created_at"],
        )

    # ===================================
    # Pattern operations
    # ===================================

    def save_pattern(self, pattern: ClassPattern) -> ClassPattern:
        """Save or update a class pattern."""
        now = self._now()

        # Try update first
        cur = self._conn.execute(
            """
            UPDATE class_patterns SET
                pattern_type = ?, priority = ?, match_count = ?
            WHERE class_id = ? AND pattern = ?
            """,
            (
                pattern.pattern_type, pattern.priority, pattern.match_count,
                pattern.class_id, pattern.pattern
            )
        )

        if cur.rowcount == 0:
            # Insert new
            self._conn.execute(
                """
                INSERT INTO class_patterns
                    (class_id, pattern, pattern_type, priority, match_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    pattern.class_id, pattern.pattern, pattern.pattern_type,
                    pattern.priority, pattern.match_count, pattern.created_at or now
                )
            )

        self._conn.commit()

        # Fetch ID
        cur = self._conn.execute(
            "SELECT id, created_at FROM class_patterns WHERE class_id = ? AND pattern = ?",
            (pattern.class_id, pattern.pattern)
        )
        row = cur.fetchone()
        if row:
            pattern.id = row["id"]
            pattern.created_at = row["created_at"]

        return pattern

    def get_patterns_by_class(self, class_id: str) -> Iterator[ClassPattern]:
        """Get all patterns for a semantic class."""
        cur = self._conn.execute(
            """
            SELECT id, class_id, pattern, pattern_type, priority, match_count, created_at
            FROM class_patterns
            WHERE class_id = ?
            ORDER BY priority DESC, match_count DESC
            """,
            (class_id,)
        )
        for row in cur:
            yield ClassPattern(
                id=row["id"],
                class_id=row["class_id"],
                pattern=row["pattern"],
                pattern_type=row["pattern_type"],
                priority=row["priority"],
                match_count=row["match_count"],
                created_at=row["created_at"],
            )

    def increment_pattern_match_count(self, class_id: str, pattern: str) -> bool:
        """Increment the match count for a pattern."""
        cur = self._conn.execute(
            """
            UPDATE class_patterns
            SET match_count = match_count + 1
            WHERE class_id = ? AND pattern = ?
            """,
            (class_id, pattern)
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ===================================
    # Classification operations
    # ===================================

    def save_classification(self, classification: EdgeClassification) -> EdgeClassification:
        """Save or update an edge classification."""
        now = self._now()

        # Try update first
        cur = self._conn.execute(
            """
            UPDATE edge_classifications SET
                confidence = ?, method = ?, bm25_score = ?, semantic_score = ?
            WHERE edge_key = ? AND class_id = ?
            """,
            (
                classification.confidence, classification.method,
                classification.bm25_score, classification.semantic_score,
                classification.edge_key, classification.class_id
            )
        )

        if cur.rowcount == 0:
            # Insert new
            self._conn.execute(
                """
                INSERT INTO edge_classifications
                    (edge_key, class_id, confidence, method, bm25_score, semantic_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    classification.edge_key, classification.class_id,
                    classification.confidence, classification.method,
                    classification.bm25_score, classification.semantic_score,
                    classification.created_at or now
                )
            )

        self._conn.commit()

        # Fetch ID
        cur = self._conn.execute(
            "SELECT id, created_at FROM edge_classifications WHERE edge_key = ? AND class_id = ?",
            (classification.edge_key, classification.class_id)
        )
        row = cur.fetchone()
        if row:
            classification.id = row["id"]
            classification.created_at = row["created_at"]

        return classification

    def get_classifications(self, edge_key: str) -> Iterator[EdgeClassification]:
        """Get all classifications for an edge."""
        cur = self._conn.execute(
            """
            SELECT id, edge_key, class_id, confidence, method, bm25_score, semantic_score, created_at
            FROM edge_classifications
            WHERE edge_key = ?
            ORDER BY confidence DESC
            """,
            (edge_key,)
        )
        for row in cur:
            yield EdgeClassification(
                id=row["id"],
                edge_key=row["edge_key"],
                class_id=row["class_id"],
                confidence=row["confidence"],
                method=row["method"],
                bm25_score=row["bm25_score"],
                semantic_score=row["semantic_score"],
                created_at=row["created_at"],
            )

    def get_edges_by_class(
        self,
        class_id: str,
        min_confidence: float = 0.0,
        limit: int = 100
    ) -> Iterator[EdgeClassification]:
        """Get edges classified into a semantic class."""
        cur = self._conn.execute(
            """
            SELECT id, edge_key, class_id, confidence, method, bm25_score, semantic_score, created_at
            FROM edge_classifications
            WHERE class_id = ? AND confidence >= ?
            ORDER BY confidence DESC
            LIMIT ?
            """,
            (class_id, min_confidence, limit)
        )
        for row in cur:
            yield EdgeClassification(
                id=row["id"],
                edge_key=row["edge_key"],
                class_id=row["class_id"],
                confidence=row["confidence"],
                method=row["method"],
                bm25_score=row["bm25_score"],
                semantic_score=row["semantic_score"],
                created_at=row["created_at"],
            )

    # ===================================
    # Feedback operations
    # ===================================

    def save_feedback(self, feedback: ClassificationFeedback) -> ClassificationFeedback:
        """Save classification feedback."""
        now = self._now()

        # Try update first
        cur = self._conn.execute(
            """
            UPDATE classification_feedback SET
                correct_class = ?, confidence_adjustment = ?, status = ?
            WHERE review_id = ?
            """,
            (
                feedback.correct_class, feedback.confidence_adjustment,
                feedback.status, feedback.review_id
            )
        )

        if cur.rowcount == 0:
            # Insert new
            self._conn.execute(
                """
                INSERT INTO classification_feedback
                    (review_id, predicate, original_class, correct_class,
                     confidence_adjustment, reviewer_id, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback.review_id, feedback.predicate, feedback.original_class,
                    feedback.correct_class, feedback.confidence_adjustment,
                    feedback.reviewer_id, feedback.status, feedback.created_at or now
                )
            )

        self._conn.commit()

        # Fetch ID
        cur = self._conn.execute(
            "SELECT id, created_at FROM classification_feedback WHERE review_id = ?",
            (feedback.review_id,)
        )
        row = cur.fetchone()
        if row:
            feedback.id = row["id"]
            feedback.created_at = row["created_at"]

        return feedback

    def get_pending_feedback(self, limit: int = 100) -> Iterator[ClassificationFeedback]:
        """Get pending feedback entries for review."""
        cur = self._conn.execute(
            """
            SELECT id, review_id, predicate, original_class, correct_class,
                   confidence_adjustment, reviewer_id, status, created_at
            FROM classification_feedback
            WHERE status = 'pending'
            ORDER BY created_at
            LIMIT ?
            """,
            (limit,)
        )
        for row in cur:
            yield ClassificationFeedback(
                id=row["id"],
                review_id=row["review_id"],
                predicate=row["predicate"],
                original_class=row["original_class"],
                correct_class=row["correct_class"],
                confidence_adjustment=row["confidence_adjustment"],
                reviewer_id=row["reviewer_id"],
                status=row["status"],
                created_at=row["created_at"],
            )

    def apply_feedback(self, review_id: str) -> bool:
        """Mark feedback as applied."""
        cur = self._conn.execute(
            """
            UPDATE classification_feedback
            SET status = 'applied'
            WHERE review_id = ? AND status = 'pending'
            """,
            (review_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ===================================
    # Utility methods
    # ===================================

    def get_stats(self) -> dict:
        """Get statistics about the classification database."""
        stats = {}

        cur = self._conn.execute("SELECT COUNT(*) FROM semantic_classes")
        stats["semantic_classes"] = cur.fetchone()[0]

        cur = self._conn.execute("SELECT COUNT(*) FROM predicate_banks")
        stats["predicates"] = cur.fetchone()[0]

        cur = self._conn.execute("SELECT COUNT(*) FROM class_patterns")
        stats["patterns"] = cur.fetchone()[0]

        cur = self._conn.execute("SELECT COUNT(*) FROM edge_classifications")
        stats["classifications"] = cur.fetchone()[0]

        cur = self._conn.execute("SELECT COUNT(*) FROM classification_feedback WHERE status = 'pending'")
        stats["pending_feedback"] = cur.fetchone()[0]

        return stats

    def clear_all(self):
        """Clear all classification data (use with caution!)."""
        self._conn.execute("DELETE FROM classification_feedback")
        self._conn.execute("DELETE FROM edge_classifications")
        self._conn.execute("DELETE FROM class_patterns")
        self._conn.execute("DELETE FROM predicate_banks")
        self._conn.execute("DELETE FROM semantic_classes")
        self._conn.commit()
        logger.warning("All classification data cleared")
