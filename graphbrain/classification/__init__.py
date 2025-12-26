"""Classification infrastructure for graphbrain.

This module provides semantic class-based classification with hybrid
BM25 + semantic retrieval for domain-adaptive pattern matching.

Key components:
- SemanticClass: Represents a classification category with predicates, patterns, embeddings
- PredicateBankEntry: Individual predicates associated with a semantic class
- ClassificationRepository: PostgreSQL-backed storage for classification data
- HybridSearcher: Combines BM25 and semantic similarity for retrieval

Usage:
    from graphbrain.classification import (
        ClassificationRepository,
        SemanticClass,
        PredicateBankEntry,
        HybridSearcher,
    )

    # Create repository
    repo = ClassificationRepository('postgresql://localhost/graphbrain')

    # Create semantic class
    sem_class = SemanticClass.create(name="claim", domain="urbanist")
    repo.save_class(sem_class)

    # Add predicates
    repo.save_predicate(PredicateBankEntry(class_id=sem_class.id, lemma="say"))

    # Search
    searcher = HybridSearcher('postgresql://localhost/graphbrain')
    results = searcher.search("mayor announced new policy")
"""

from graphbrain.classification.models import (
    SemanticClass,
    PredicateBankEntry,
    ClassPattern,
    EdgeClassification,
    ClassificationFeedback,
)
from graphbrain.classification.repository import ClassificationRepository
from graphbrain.classification.hybrid_search import HybridSearcher, SearchResult
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
    # Repository
    'ClassificationRepository',
    # Search
    'HybridSearcher',
    'SearchResult',
    # Migration
    'migrate_seed_predicates',
    'migrate_hypergraph_to_postgres',
    'export_to_seed_format',
    'SEED_PREDICATES',
]
