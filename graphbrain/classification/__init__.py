"""Classification infrastructure for graphbrain.

This module provides semantic class-based classification with hybrid
BM25 + semantic retrieval for domain-adaptive pattern matching.

Supports both PostgreSQL and SQLite backends via the backend factories.

Key components:
- SemanticClass: Represents a classification category with predicates, patterns, embeddings
- PredicateBankEntry: Individual predicates associated with a semantic class
- get_classification_backend: Factory for auto-detecting and creating classification backends
- get_search_backend: Factory for auto-detecting and creating search backends

Usage:
    from graphbrain.classification import (
        get_classification_backend,
        get_search_backend,
        SemanticClass,
        PredicateBankEntry,
    )

    # Create backends (auto-detects PostgreSQL or SQLite)
    repo = get_classification_backend('postgresql://localhost/graphbrain')
    # or: repo = get_classification_backend('/path/to/knowledge.db')

    # Create semantic class
    sem_class = SemanticClass.create(name="claim", domain="urbanist")
    repo.save_class(sem_class)

    # Add predicates
    repo.save_predicate(PredicateBankEntry(class_id=sem_class.id, lemma="say"))

    # Search
    searcher = get_search_backend('postgresql://localhost/graphbrain')
    results = searcher.search("mayor announced new policy")

Legacy imports (PostgreSQL-only):
    from graphbrain.classification import ClassificationRepository, HybridSearcher
"""

from graphbrain.classification.models import (
    SemanticClass,
    PredicateBankEntry,
    ClassPattern,
    EdgeClassification,
    ClassificationFeedback,
)

# Backend factories (preferred, support both PostgreSQL and SQLite)
from graphbrain.classification.backends import (
    ClassificationBackend,
    get_classification_backend,
)
from graphbrain.classification.search import (
    SearchBackend,
    SearchResult,
    get_search_backend,
)

# Legacy PostgreSQL-only imports (for backwards compatibility)
from graphbrain.classification.repository import ClassificationRepository
from graphbrain.classification.hybrid_search import HybridSearcher

from graphbrain.classification.migration import (
    migrate_seed_predicates,
    migrate_hypergraph_to_postgres,
    export_to_seed_format,
    SEED_PREDICATES,
)

__all__ = [
    # Models
    'SemanticClass',
    'PredicateBankEntry',
    'ClassPattern',
    'EdgeClassification',
    'ClassificationFeedback',
    # Backend factories (preferred)
    'ClassificationBackend',
    'get_classification_backend',
    'SearchBackend',
    'SearchResult',
    'get_search_backend',
    # Legacy (PostgreSQL-only, for backwards compatibility)
    'ClassificationRepository',
    'HybridSearcher',
    # Migration
    'migrate_seed_predicates',
    'migrate_hypergraph_to_postgres',
    'export_to_seed_format',
    'SEED_PREDICATES',
]
