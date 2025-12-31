"""Predicate discovery tools for MCP server.

Provides tools for discovering and analyzing predicates.
"""

import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

from graphbrain.mcp.errors import (
    not_found_error,
    service_unavailable_error,
    error_response,
    ErrorCode,
)

logger = logging.getLogger(__name__)


def register_predicate_tools(server: FastMCP):
    """Register predicate tools with the MCP server."""

    @server.tool(
        name="discover_predicates",
        description="""
Discover predicates from the hypergraph that aren't yet in predicate banks.

Scans edges in the hypergraph to find predicate atoms (type Pd) and
returns those that aren't already classified. Useful for expanding
the semantic class coverage.

Args:
    min_frequency: Minimum occurrence count to include (default 5)
    limit: Maximum predicates to return (default 100)

Returns:
  - predicates: list of {lemma, frequency, suggested_class}
  - total_scanned: total predicates found in graph
  - min_frequency: the frequency threshold used
""",
    )
    async def discover_predicates(
        min_frequency: int = 5,
        limit: int = 100,
    ) -> dict:
        """Discover unclassified predicates."""
        logger.debug(f"discover_predicates: min_frequency={min_frequency}, limit={limit}")

        import graphbrain.hyperedge as he

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        hg = lifespan_data["hg"]
        repo = lifespan_data["repo"]

        # Count predicate occurrences
        predicate_counts = {}

        for edge in hg.all():
            if not edge.atom and len(edge) > 0:
                connector = edge[0]
                if connector.atom:
                    # Check if it's a predicate type
                    atom_type = connector.type()
                    if atom_type and atom_type.startswith('P'):  # Pd, Pm, etc.
                        lemma = connector.root()
                        predicate_counts[lemma] = predicate_counts.get(lemma, 0) + 1

        # Filter by frequency and check if already classified
        discovered = []
        for lemma, count in sorted(predicate_counts.items(), key=lambda x: -x[1]):
            if count < min_frequency:
                continue
            if len(discovered) >= limit:
                break

            # Check if already in a predicate bank
            existing = list(repo.find_predicate(lemma))
            if not existing:
                discovered.append({
                    "lemma": lemma,
                    "frequency": count,
                    "suggested_class": None,
                })
            # If exists but has low frequency in banks, might still be interesting
            elif existing[0][0].frequency < count // 2:
                discovered.append({
                    "lemma": lemma,
                    "frequency": count,
                    "suggested_class": existing[0][1].name,
                    "existing_frequency": existing[0][0].frequency,
                })

        logger.info(f"discover_predicates: found {len(discovered)} unclassified predicates from {len(predicate_counts)} total")
        return {
            "status": "success",
            "predicates": discovered[:limit],
            "total_scanned": len(predicate_counts),
            "min_frequency": min_frequency,
        }

    @server.tool(
        name="find_similar_predicates",
        description="""
Find predicates similar to a given predicate.

Uses lexical similarity to find related predicates that might belong
to the same semantic class. Useful for predicate discovery and expansion.

Backend-specific behavior:
- PostgreSQL: Uses pg_trgm trigram similarity (fast, index-backed)
- SQLite: Uses LIKE queries + Python's SequenceMatcher (reasonable fallback)

Args:
    predicate: Reference predicate lemma
    limit: Maximum results (default 10)

Returns:
  - reference: the input predicate
  - similar: list of {lemma, class_name, class_id, is_seed, frequency}
  - total: number of similar predicates found
  - backend: which backend implementation was used
""",
    )
    async def find_similar_predicates(
        predicate: str,
        limit: int = 10,
    ) -> dict:
        """Find similar predicates."""
        logger.debug(f"find_similar_predicates: predicate='{predicate}', limit={limit}")

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        repo = lifespan_data["repo"]

        # Detect backend type for response
        backend_type = type(repo).__name__
        if "Sqlite" in backend_type:
            backend = "sqlite"
        elif "Postgres" in backend_type:
            backend = "postgresql"
        else:
            backend = "unknown"

        similar = []
        try:
            for entry in repo.find_similar_predicates(predicate, limit=limit):
                # Get the class for this predicate
                sem_class = repo.get_class(entry.class_id)
                similar.append({
                    "lemma": entry.lemma,
                    "class_name": sem_class.name if sem_class else None,
                    "class_id": entry.class_id,
                    "is_seed": entry.is_seed,
                    "frequency": entry.frequency,
                })
        except Exception as e:
            logger.warning(f"find_similar_predicates: similarity search failed - {e}")
            return service_unavailable_error("similarity_search", str(e))

        logger.info(f"find_similar_predicates: found {len(similar)} similar to '{predicate}' (backend={backend})")
        return {
            "status": "success",
            "reference": predicate,
            "similar": similar,
            "total": len(similar),
            "backend": backend,
        }

    @server.tool(
        name="get_predicate_classes",
        description="""
Get all semantic classes that a predicate belongs to.

Returns detailed information about each class the predicate is
associated with, including confidence scores.

Args:
    predicate: Predicate lemma to look up

Returns:
  - predicate: the input predicate
  - classes: list of {class_id, class_name, domain, is_seed, frequency, similarity_score}
  - total: number of classes
  - is_classified: true if predicate is in at least one class
""",
    )
    async def get_predicate_classes(
        predicate: str,
    ) -> dict:
        """Get classes for a predicate."""
        logger.debug(f"get_predicate_classes: predicate='{predicate}'")

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        repo = lifespan_data["repo"]

        classes = []
        for entry, sem_class in repo.find_predicate(predicate):
            classes.append({
                "class_id": sem_class.id,
                "class_name": sem_class.name,
                "domain": sem_class.domain,
                "is_seed": entry.is_seed,
                "frequency": entry.frequency,
                "similarity_score": entry.similarity_score,
            })

        logger.info(f"get_predicate_classes: '{predicate}' belongs to {len(classes)} classes")
        return {
            "status": "success",
            "predicate": predicate,
            "classes": classes,
            "total": len(classes),
            "is_classified": len(classes) > 0,
        }

    @server.tool(
        name="list_predicates_by_class",
        description="""
List all predicates in a semantic class.

Returns predicates sorted by whether they're seeds and by frequency.

Args:
    class_id: The class ID (or use name+domain)
    name: Class name (requires domain)
    domain: Domain (default "default")

Returns:
  - class_id, class_name: the class
  - predicates: list of {lemma, is_seed, frequency, similarity_score}
  - total: number of predicates
  - seed_count: number of seed predicates
""",
    )
    async def list_predicates_by_class(
        class_id: Optional[str] = None,
        name: Optional[str] = None,
        domain: str = "default",
    ) -> dict:
        """List predicates in a class."""
        logger.debug(f"list_predicates_by_class: class_id={class_id}, name={name}, domain={domain}")

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        repo = lifespan_data["repo"]

        if class_id:
            sem_class = repo.get_class(class_id)
        elif name:
            sem_class = repo.get_class_by_name(name, domain)
        else:
            logger.warning("list_predicates_by_class: missing class_id or name")
            return error_response(ErrorCode.MISSING_PARAMETER, "Must provide either class_id or name")

        if not sem_class:
            logger.debug("list_predicates_by_class: class not found")
            return not_found_error("semantic_class", class_id or name)

        predicates = []
        seed_count = 0
        for entry in repo.get_predicates_by_class(sem_class.id):
            predicates.append({
                "lemma": entry.lemma,
                "is_seed": entry.is_seed,
                "frequency": entry.frequency,
                "similarity_score": entry.similarity_score,
            })
            if entry.is_seed:
                seed_count += 1

        logger.info(f"list_predicates_by_class: '{sem_class.name}' has {len(predicates)} predicates ({seed_count} seeds)")
        return {
            "status": "success",
            "class_id": sem_class.id,
            "class_name": sem_class.name,
            "predicates": predicates,
            "total": len(predicates),
            "seed_count": seed_count,
        }

    @server.tool(
        name="update_predicate_frequency",
        description="""
Increment the frequency counter for a predicate.

Called when a predicate is observed in new data to track usage patterns.

Args:
    class_id: The class ID
    lemma: Predicate lemma

Returns:
  - class_id: the class ID
  - lemma: the predicate
  - updated: true if successful
""",
    )
    async def update_predicate_frequency(
        class_id: str,
        lemma: str,
    ) -> dict:
        """Update predicate frequency."""
        logger.debug(f"update_predicate_frequency: class_id={class_id}, lemma='{lemma}'")

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        repo = lifespan_data["repo"]

        updated = repo.increment_predicate_frequency(class_id, lemma)

        if updated:
            logger.info(f"update_predicate_frequency: incremented '{lemma}' in class {class_id}")
        else:
            logger.debug(f"update_predicate_frequency: predicate '{lemma}' not found in class {class_id}")

        return {
            "status": "success",
            "class_id": class_id,
            "lemma": lemma,
            "updated": updated,
        }
