"""Domain models for the classification system.

These dataclasses represent semantic classes and their associated
predicates, patterns, and classifications.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid


def _generate_id() -> str:
    """Generate a short unique ID."""
    return uuid.uuid4().hex[:8]


@dataclass
class SemanticClass:
    """A semantic class representing a category of edges/predicates.

    A semantic class (e.g., "claims about seattle mayor") maps to multiple
    retrieval methods:
    - predicate_bank: list of predicates associated with this class
    - patterns: structural patterns for matching
    - embedding: class centroid vector for semantic similarity
    - bm25_query: pre-built lexical search query

    Attributes:
        id: Unique identifier for the class
        name: Human-readable name (e.g., "claim", "conflict", "action")
        description: Detailed description of what this class represents
        domain: Domain context (e.g., "urbanist", "politics", "default")
        provenance: How this class was created ("seed", "discovered", "user")
        confidence: Confidence score for discovered classes (0.0-1.0)
        bm25_query: Pre-built BM25 query for lexical search
        hybrid_weights: Weights for combining BM25 and semantic search
        embedding: Class centroid embedding vector
        version: Version number for tracking updates
        created_at: Timestamp of creation
        updated_at: Timestamp of last update
    """
    name: str
    id: str = field(default_factory=_generate_id)
    description: Optional[str] = None
    domain: str = "default"
    provenance: str = "seed"
    confidence: float = 1.0
    bm25_query: Optional[str] = None
    hybrid_weights: dict = field(default_factory=lambda: {"bm25": 0.3, "semantic": 0.7})
    embedding: Optional[list[float]] = None
    version: int = 1
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def create(cls, name: str, domain: str = "default", **kwargs) -> "SemanticClass":
        """Factory method for creating a new SemanticClass."""
        now = datetime.now()
        return cls(
            name=name,
            domain=domain,
            created_at=now,
            updated_at=now,
            **kwargs
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "provenance": self.provenance,
            "confidence": self.confidence,
            "bm25_query": self.bm25_query,
            "hybrid_weights": self.hybrid_weights,
            "embedding": self.embedding,
            "version": self.version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SemanticClass":
        """Create from dictionary."""
        created_at = data.get("created_at")
        updated_at = data.get("updated_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)

        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description"),
            domain=data.get("domain", "default"),
            provenance=data.get("provenance", "seed"),
            confidence=data.get("confidence", 1.0),
            bm25_query=data.get("bm25_query"),
            hybrid_weights=data.get("hybrid_weights", {"bm25": 0.3, "semantic": 0.7}),
            embedding=data.get("embedding"),
            version=data.get("version", 1),
            created_at=created_at,
            updated_at=updated_at,
        )


@dataclass
class PredicateBankEntry:
    """An entry in a predicate bank associated with a semantic class.

    Predicate banks contain lemmas (e.g., "say", "claim", "announce") that
    are associated with a semantic class. Each entry can have:
    - similarity_score: How similar this predicate is to the class centroid
    - frequency: How often this predicate appears in the corpus
    - is_seed: Whether this was a seed predicate or discovered

    Attributes:
        class_id: ID of the semantic class this predicate belongs to
        lemma: The predicate lemma (e.g., "say", "claim")
        id: Unique identifier for this entry
        similarity_score: Semantic similarity to class centroid
        frequency: Occurrence count in corpus
        is_seed: True if this was a seed predicate
        embedding: Embedding vector for this predicate
        created_at: Timestamp of creation
    """
    class_id: str
    lemma: str
    id: Optional[int] = None
    similarity_score: Optional[float] = None
    frequency: int = 0
    is_seed: bool = False
    embedding: Optional[list[float]] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "class_id": self.class_id,
            "lemma": self.lemma,
            "similarity_score": self.similarity_score,
            "frequency": self.frequency,
            "is_seed": self.is_seed,
            "embedding": self.embedding,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PredicateBankEntry":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return cls(
            id=data.get("id"),
            class_id=data["class_id"],
            lemma=data["lemma"],
            similarity_score=data.get("similarity_score"),
            frequency=data.get("frequency", 0),
            is_seed=data.get("is_seed", False),
            embedding=data.get("embedding"),
            created_at=created_at,
        )


@dataclass
class ClassPattern:
    """A structural pattern associated with a semantic class.

    Patterns use graphbrain's pattern syntax with wildcards, variables,
    and argrole constraints to match edges structurally.

    Examples:
        - "(*/Pd.{sr} */Cp *)" - predicates with source and recipient roles
        - "(says/Pd * *)" - edges with "says" predicate
        - "(*/Pd.so * *)" - predicates with subject and object

    Attributes:
        class_id: ID of the semantic class this pattern belongs to
        pattern: The pattern string in graphbrain syntax
        id: Unique identifier for this entry
        pattern_type: Type of pattern ("structural", "semantic", "hybrid")
        priority: Priority for pattern matching (higher = checked first)
        match_count: Number of times this pattern has matched
        created_at: Timestamp of creation
    """
    class_id: str
    pattern: str
    id: Optional[int] = None
    pattern_type: str = "structural"
    priority: int = 0
    match_count: int = 0
    created_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "class_id": self.class_id,
            "pattern": self.pattern,
            "pattern_type": self.pattern_type,
            "priority": self.priority,
            "match_count": self.match_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClassPattern":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return cls(
            id=data.get("id"),
            class_id=data["class_id"],
            pattern=data["pattern"],
            pattern_type=data.get("pattern_type", "structural"),
            priority=data.get("priority", 0),
            match_count=data.get("match_count", 0),
            created_at=created_at,
        )


@dataclass
class EdgeClassification:
    """A classification of an edge into a semantic class.

    Records when an edge is classified into a semantic class, including
    the method used and confidence scores.

    Attributes:
        edge_key: String representation of the classified edge
        class_id: ID of the semantic class
        confidence: Overall confidence score (0.0-1.0)
        method: Classification method ("bm25", "semantic", "hybrid", "pattern")
        id: Unique identifier for this classification
        bm25_score: BM25 component score
        semantic_score: Semantic similarity component score
        created_at: Timestamp of classification
    """
    edge_key: str
    class_id: str
    confidence: float
    method: str
    id: Optional[int] = None
    bm25_score: Optional[float] = None
    semantic_score: Optional[float] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "edge_key": self.edge_key,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "method": self.method,
            "bm25_score": self.bm25_score,
            "semantic_score": self.semantic_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EdgeClassification":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return cls(
            id=data.get("id"),
            edge_key=data["edge_key"],
            class_id=data["class_id"],
            confidence=data["confidence"],
            method=data["method"],
            bm25_score=data.get("bm25_score"),
            semantic_score=data.get("semantic_score"),
            created_at=created_at,
        )


@dataclass
class ClassificationFeedback:
    """Human feedback on a classification for active learning.

    Used to track corrections to classifications for iterative
    improvement of the classification system.

    Attributes:
        review_id: Unique identifier for this review
        predicate: The predicate that was classified
        original_class: The original classification
        correct_class: The correct classification (from human)
        id: Unique identifier for this feedback entry
        confidence_adjustment: Suggested adjustment to confidence
        reviewer_id: Identifier of the reviewer
        status: Status of this feedback ("pending", "applied", "rejected")
        created_at: Timestamp of feedback submission
    """
    review_id: str
    predicate: str
    original_class: str
    correct_class: str
    id: Optional[int] = None
    confidence_adjustment: Optional[float] = None
    reviewer_id: Optional[str] = None
    status: str = "pending"
    created_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "review_id": self.review_id,
            "predicate": self.predicate,
            "original_class": self.original_class,
            "correct_class": self.correct_class,
            "confidence_adjustment": self.confidence_adjustment,
            "reviewer_id": self.reviewer_id,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClassificationFeedback":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return cls(
            id=data.get("id"),
            review_id=data["review_id"],
            predicate=data["predicate"],
            original_class=data["original_class"],
            correct_class=data["correct_class"],
            confidence_adjustment=data.get("confidence_adjustment"),
            reviewer_id=data.get("reviewer_id"),
            status=data.get("status", "pending"),
            created_at=created_at,
        )
