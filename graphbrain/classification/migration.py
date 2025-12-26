"""Migration utilities for the classification system.

Provides tools to migrate from the hardcoded SEED_PREDICATES
to the PostgreSQL-based classification system.
"""

import logging
from datetime import datetime
from typing import Optional

from graphbrain.classification.models import SemanticClass, PredicateBankEntry, ClassPattern
from graphbrain.classification.repository import ClassificationRepository

logger = logging.getLogger(__name__)


# Default seed predicates from adaptive_predicates.py
SEED_PREDICATES = {
    'claim': ['say', 'claim', 'state', 'assert', 'announce', 'declare', 'report',
              'tell', 'mention', 'note', 'add', 'explain', 'argue', 'suggest',
              'indicate', 'reveal', 'confirm', 'acknowledge', 'admit', 'deny',
              'insist', 'maintain', 'contend', 'allege', 'believe', 'think',
              'feel', 'consider', 'find', 'conclude', 'determine', 'decide'],
    'conflict': ['attack', 'blame', 'accuse', 'condemn', 'warn', 'threaten',
                 'criticize', 'oppose', 'reject', 'dispute', 'challenge',
                 'confront', 'fight', 'resist', 'protest', 'denounce',
                 'complain', 'object', 'disagree', 'contradict'],
    'action': ['do', 'make', 'create', 'build', 'develop', 'launch', 'start',
               'begin', 'establish', 'introduce', 'implement', 'execute',
               'perform', 'conduct', 'carry', 'run', 'operate', 'manage',
               'handle', 'lead', 'organize', 'arrange', 'prepare', 'plan',
               'design', 'produce', 'generate', 'form', 'construct'],
    'change': ['change', 'transform', 'convert', 'modify', 'alter', 'adjust',
               'adapt', 'revise', 'update', 'improve', 'enhance', 'expand',
               'extend', 'increase', 'grow', 'reduce', 'decrease', 'cut',
               'shrink', 'limit', 'restrict', 'reform', 'restructure'],
    'support': ['support', 'help', 'assist', 'aid', 'back', 'endorse', 'approve',
                'favor', 'advocate', 'promote', 'encourage', 'enable', 'allow',
                'permit', 'authorize', 'accept', 'agree', 'welcome', 'praise',
                'commend', 'appreciate', 'thank', 'acknowledge'],
    'movement': ['go', 'come', 'move', 'travel', 'walk', 'run', 'drive', 'fly',
                 'arrive', 'leave', 'depart', 'return', 'enter', 'exit',
                 'approach', 'reach', 'pass', 'cross', 'follow', 'lead'],
    'possession': ['have', 'own', 'possess', 'hold', 'keep', 'maintain', 'retain',
                   'get', 'obtain', 'acquire', 'receive', 'gain', 'earn', 'win',
                   'lose', 'give', 'provide', 'offer', 'grant', 'donate',
                   'contribute', 'share', 'distribute', 'allocate', 'assign'],
    'cognition': ['know', 'understand', 'realize', 'recognize', 'learn', 'discover',
                  'find', 'see', 'notice', 'observe', 'perceive', 'sense',
                  'remember', 'recall', 'forget', 'imagine', 'expect', 'anticipate',
                  'predict', 'assume', 'suppose', 'guess', 'wonder', 'doubt'],
}

# Default patterns for semantic classes
DEFAULT_PATTERNS = {
    'claim': [
        '(*/Pd.{sr} * *)',     # Predicate with source and recipient roles
        '(*/Pd.so * *)',       # Predicate with subject and object
        '(says/Pd * *)',       # Explicit "says" pattern
    ],
    'conflict': [
        '(*/Pd.{so} * *)',     # Predicate with subject and object (attack X)
        '(against/T * *)',     # Against trigger
    ],
    'action': [
        '(*/Pd.{so} * *)',     # Action on object
        '(*/Pd.{sx} * *)',     # Action with specifier
    ],
    'possession': [
        '(*/Pd.{so} * *)',     # Has/owns object
        "(have/Pd * *)",
    ],
}


def migrate_seed_predicates(
    repo: ClassificationRepository,
    domain: str = "default",
    clear_existing: bool = False,
) -> dict:
    """Migrate SEED_PREDICATES to the PostgreSQL classification system.

    Args:
        repo: Classification repository instance
        domain: Domain name for the semantic classes
        clear_existing: If True, clear existing data first

    Returns:
        Statistics about the migration
    """
    if clear_existing:
        repo.clear_all()
        logger.info("Cleared existing classification data")

    stats = {
        "classes_created": 0,
        "predicates_created": 0,
        "patterns_created": 0,
    }

    for class_name, predicates in SEED_PREDICATES.items():
        # Check if class already exists
        existing = repo.get_class_by_name(class_name, domain)
        if existing:
            logger.debug(f"Class '{class_name}' already exists, skipping")
            continue

        # Create semantic class
        sem_class = SemanticClass.create(
            name=class_name,
            domain=domain,
            description=f"Predicates related to {class_name}",
            provenance="seed",
            confidence=1.0,
        )
        repo.save_class(sem_class)
        stats["classes_created"] += 1
        logger.info(f"Created semantic class: {class_name}")

        # Add predicates
        for lemma in predicates:
            entry = PredicateBankEntry(
                class_id=sem_class.id,
                lemma=lemma,
                is_seed=True,
                frequency=0,
                created_at=datetime.now(),
            )
            repo.save_predicate(entry)
            stats["predicates_created"] += 1

        # Add patterns if available
        if class_name in DEFAULT_PATTERNS:
            for pattern_str in DEFAULT_PATTERNS[class_name]:
                pattern = ClassPattern(
                    class_id=sem_class.id,
                    pattern=pattern_str,
                    pattern_type="structural",
                    priority=1,
                    created_at=datetime.now(),
                )
                repo.save_pattern(pattern)
                stats["patterns_created"] += 1

    logger.info(f"Migration complete: {stats}")
    return stats


def migrate_hypergraph_to_postgres(
    source_path: str,
    pg_connection: str,
    batch_size: int = 1000,
) -> dict:
    """Migrate a SQLite/LevelDB hypergraph to PostgreSQL.

    Args:
        source_path: Path to source hypergraph (SQLite .db or LevelDB folder)
        pg_connection: PostgreSQL connection string
        batch_size: Number of edges per batch

    Returns:
        Statistics about the migration
    """
    from graphbrain import hgraph

    logger.info(f"Migrating from {source_path} to PostgreSQL")

    # Open source
    src = hgraph(source_path)

    # Open destination
    dst = hgraph(pg_connection)
    dst.destroy()  # Clear destination

    stats = {
        "edges_migrated": 0,
        "atoms_migrated": 0,
        "errors": 0,
    }

    batch = []
    for edge, attrs in src.all_attributes():
        batch.append((edge, attrs))

        if len(batch) >= batch_size:
            dst.begin_transaction()
            for e, a in batch:
                try:
                    dst.add_with_attributes(e, a)
                    if e.atom:
                        stats["atoms_migrated"] += 1
                    else:
                        stats["edges_migrated"] += 1
                except Exception as ex:
                    logger.warning(f"Error migrating edge {e}: {ex}")
                    stats["errors"] += 1
            dst.end_transaction()
            batch = []
            logger.info(f"Progress: {stats['edges_migrated']} edges, {stats['atoms_migrated']} atoms")

    # Final batch
    if batch:
        dst.begin_transaction()
        for e, a in batch:
            try:
                dst.add_with_attributes(e, a)
                if e.atom:
                    stats["atoms_migrated"] += 1
                else:
                    stats["edges_migrated"] += 1
            except Exception as ex:
                logger.warning(f"Error migrating edge {e}: {ex}")
                stats["errors"] += 1
        dst.end_transaction()

    src.close()
    dst.close()

    logger.info(f"Migration complete: {stats}")
    return stats


def export_to_seed_format(
    repo: ClassificationRepository,
    domain: Optional[str] = None,
) -> dict[str, list[str]]:
    """Export semantic classes back to SEED_PREDICATES format.

    Useful for verification and backwards compatibility.

    Args:
        repo: Classification repository
        domain: Optional domain filter

    Returns:
        Dictionary mapping class names to predicate lists
    """
    result = {}

    for sem_class in repo.list_classes(domain):
        predicates = [
            p.lemma for p in repo.get_predicates_by_class(sem_class.id)
        ]
        result[sem_class.name] = predicates

    return result
