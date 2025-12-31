"""Abstract base class for classification storage backends.

Defines the interface that PostgreSQL and SQLite backends must implement.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional

from graphbrain.classification.models import (
    SemanticClass,
    PredicateBankEntry,
    ClassPattern,
    EdgeClassification,
    ClassificationFeedback,
)


class ClassificationBackend(ABC):
    """Abstract base class for classification storage backends.

    Implementations must provide CRUD operations for:
    - Semantic classes
    - Predicate banks
    - Class patterns
    - Edge classifications
    - Classification feedback
    """

    @abstractmethod
    def close(self):
        """Close the database connection."""
        pass

    # ===================================
    # Semantic Class operations
    # ===================================

    @abstractmethod
    def save_class(self, sem_class: SemanticClass) -> SemanticClass:
        """Save or update a semantic class."""
        pass

    @abstractmethod
    def get_class(self, class_id: str) -> Optional[SemanticClass]:
        """Get a semantic class by ID."""
        pass

    @abstractmethod
    def get_class_by_name(self, name: str, domain: str = "default") -> Optional[SemanticClass]:
        """Get a semantic class by name and domain."""
        pass

    @abstractmethod
    def list_classes(self, domain: Optional[str] = None) -> Iterator[SemanticClass]:
        """List all semantic classes, optionally filtered by domain."""
        pass

    @abstractmethod
    def delete_class(self, class_id: str) -> bool:
        """Delete a semantic class and all associated data."""
        pass

    # ===================================
    # Predicate Bank operations
    # ===================================

    @abstractmethod
    def save_predicate(self, predicate: PredicateBankEntry) -> PredicateBankEntry:
        """Save or update a predicate bank entry."""
        pass

    @abstractmethod
    def get_predicates_by_class(self, class_id: str) -> Iterator[PredicateBankEntry]:
        """Get all predicates for a semantic class."""
        pass

    @abstractmethod
    def find_predicate(self, lemma: str) -> Iterator[tuple[PredicateBankEntry, SemanticClass]]:
        """Find all classes that contain a predicate lemma."""
        pass

    def find_similar_predicates(self, lemma: str, limit: int = 10) -> Iterator[PredicateBankEntry]:
        """Find predicates similar to the given lemma.

        Backend-specific implementations:
        - PostgreSQL: Uses pg_trgm trigram similarity (fast, index-backed)
        - SQLite: Uses LIKE queries + Python's difflib.SequenceMatcher

        Default implementation returns empty iterator for backends that
        don't support similarity search.
        """
        return iter([])

    @abstractmethod
    def increment_predicate_frequency(self, class_id: str, lemma: str) -> bool:
        """Increment the frequency counter for a predicate."""
        pass

    # ===================================
    # Pattern operations
    # ===================================

    @abstractmethod
    def save_pattern(self, pattern: ClassPattern) -> ClassPattern:
        """Save or update a class pattern."""
        pass

    @abstractmethod
    def get_patterns_by_class(self, class_id: str) -> Iterator[ClassPattern]:
        """Get all patterns for a semantic class."""
        pass

    @abstractmethod
    def increment_pattern_match_count(self, class_id: str, pattern: str) -> bool:
        """Increment the match count for a pattern."""
        pass

    def get_all_patterns(self) -> Iterator[tuple[str, list[ClassPattern]]]:
        """Get all patterns grouped by class_id.

        Default implementation iterates over all classes.
        """
        for sem_class in self.list_classes():
            patterns = list(self.get_patterns_by_class(sem_class.id))
            if patterns:
                yield sem_class.id, patterns

    # ===================================
    # Classification operations
    # ===================================

    @abstractmethod
    def save_classification(self, classification: EdgeClassification) -> EdgeClassification:
        """Save or update an edge classification."""
        pass

    @abstractmethod
    def get_classifications(self, edge_key: str) -> Iterator[EdgeClassification]:
        """Get all classifications for an edge."""
        pass

    @abstractmethod
    def get_edges_by_class(
        self,
        class_id: str,
        min_confidence: float = 0.0,
        limit: int = 100
    ) -> Iterator[EdgeClassification]:
        """Get edges classified into a semantic class."""
        pass

    # ===================================
    # Feedback operations
    # ===================================

    @abstractmethod
    def save_feedback(self, feedback: ClassificationFeedback) -> ClassificationFeedback:
        """Save classification feedback."""
        pass

    @abstractmethod
    def get_pending_feedback(self, limit: int = 100) -> Iterator[ClassificationFeedback]:
        """Get pending feedback entries for review."""
        pass

    @abstractmethod
    def apply_feedback(self, review_id: str) -> bool:
        """Mark feedback as applied."""
        pass

    # ===================================
    # Utility methods
    # ===================================

    @abstractmethod
    def get_stats(self) -> dict:
        """Get statistics about the classification database."""
        pass

    def clear_all(self):
        """Clear all classification data (use with caution!).

        Default implementation raises NotImplementedError.
        """
        raise NotImplementedError("clear_all not implemented for this backend")
