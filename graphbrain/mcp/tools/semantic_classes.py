"""Semantic class tools for MCP server.

Provides CRUD operations for semantic classes.
"""

import logging
from typing import Optional
from datetime import datetime

from mcp.server.fastmcp import FastMCP

from graphbrain.mcp.errors import (
    already_exists_error,
    not_found_error,
    invalid_pattern_error,
    error_response,
    ErrorCode,
)

logger = logging.getLogger(__name__)


def register_semantic_class_tools(server: FastMCP):
    """Register semantic class tools with the MCP server."""

    @server.tool(
        name="create_semantic_class",
        description="""
Create a new semantic class for classification.

A semantic class represents a category of predicates/edges (e.g., "claim",
"conflict", "action"). Classes can have seed predicates and structural patterns.

Args:
    name: Class name (e.g., "claim", "conflict")
    domain: Domain for organization (default "default")
    description: Human-readable description
    seed_predicates: Initial predicates for the class (e.g., ["say", "claim", "announce"])
    patterns: Structural patterns in SH notation

Returns:
  - id: the new class ID
  - name, domain, description: class metadata
  - predicates_added: count of seed predicates added
  - patterns_added: count of patterns added
""",
    )
    async def create_semantic_class(
        name: str,
        domain: str = "default",
        description: Optional[str] = None,
        seed_predicates: Optional[list[str]] = None,
        patterns: Optional[list[str]] = None,
    ) -> dict:
        """Create a new semantic class."""
        logger.debug(f"create_semantic_class: name='{name}', domain='{domain}'")

        from graphbrain.classification.models import SemanticClass, PredicateBankEntry, ClassPattern

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        repo = lifespan_data["repo"]

        # Check if class already exists
        existing = repo.get_class_by_name(name, domain)
        if existing:
            logger.warning(f"create_semantic_class: class '{name}' already exists")
            return already_exists_error("semantic_class", f"{name} (domain: {domain})")

        # Create the class
        sem_class = SemanticClass.create(
            name=name,
            domain=domain,
            description=description or f"Predicates related to {name}",
            provenance="user",
        )
        repo.save_class(sem_class)

        predicates_added = 0
        patterns_added = 0

        # Add seed predicates
        if seed_predicates:
            for lemma in seed_predicates:
                entry = PredicateBankEntry(
                    class_id=sem_class.id,
                    lemma=lemma,
                    is_seed=True,
                    frequency=0,
                    created_at=datetime.now(),
                )
                repo.save_predicate(entry)
                predicates_added += 1

        # Add patterns
        if patterns:
            for pattern_str in patterns:
                pattern = ClassPattern(
                    class_id=sem_class.id,
                    pattern=pattern_str,
                    pattern_type="structural",
                    priority=1,
                    created_at=datetime.now(),
                )
                repo.save_pattern(pattern)
                patterns_added += 1

        logger.info(f"create_semantic_class: created '{name}' with {predicates_added} predicates, {patterns_added} patterns")
        return {
            "status": "success",
            "id": sem_class.id,
            "name": sem_class.name,
            "domain": sem_class.domain,
            "description": sem_class.description,
            "predicates_added": predicates_added,
            "patterns_added": patterns_added,
        }

    @server.tool(
        name="get_semantic_class",
        description="""
Get details about a semantic class.

Returns class information including predicates and patterns.

Args:
    class_id: The class ID (or use name+domain)
    name: Class name (requires domain)
    domain: Domain (required if using name)

Returns:
  - id, name, domain, description: class metadata
  - predicates: list of predicate lemmas
  - patterns: list of pattern strings
  - provenance, confidence, version: class metadata
""",
    )
    async def get_semantic_class(
        class_id: Optional[str] = None,
        name: Optional[str] = None,
        domain: str = "default",
    ) -> dict:
        """Get semantic class details."""
        logger.debug(f"get_semantic_class: class_id={class_id}, name={name}, domain={domain}")

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        repo = lifespan_data["repo"]

        if class_id:
            sem_class = repo.get_class(class_id)
        elif name:
            sem_class = repo.get_class_by_name(name, domain)
        else:
            logger.warning("get_semantic_class: missing class_id or name")
            return error_response(ErrorCode.MISSING_PARAMETER, "Must provide either class_id or name")

        if not sem_class:
            logger.debug(f"get_semantic_class: class not found")
            return not_found_error("semantic_class", class_id or name)

        # Get predicates
        predicates = [p.lemma for p in repo.get_predicates_by_class(sem_class.id)]

        # Get patterns
        patterns = [p.pattern for p in repo.get_patterns_by_class(sem_class.id)]

        logger.info(f"get_semantic_class: found '{sem_class.name}' with {len(predicates)} predicates, {len(patterns)} patterns")
        return {
            "status": "success",
            "id": sem_class.id,
            "name": sem_class.name,
            "domain": sem_class.domain,
            "description": sem_class.description,
            "provenance": sem_class.provenance,
            "confidence": sem_class.confidence,
            "predicates": predicates,
            "patterns": patterns,
            "version": sem_class.version,
            "created_at": sem_class.created_at.isoformat() if sem_class.created_at else None,
            "updated_at": sem_class.updated_at.isoformat() if sem_class.updated_at else None,
        }

    @server.tool(
        name="list_semantic_classes",
        description="""
List all semantic classes.

Returns a list of all classes, optionally filtered by domain.

Args:
    domain: Optional domain filter

Returns:
  - classes: list of class summaries with predicate/pattern counts
  - total: number of classes
  - domain_filter: the domain filter used (if any)
""",
    )
    async def list_semantic_classes(
        domain: Optional[str] = None,
    ) -> dict:
        """List semantic classes."""
        logger.debug(f"list_semantic_classes: domain={domain}")

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        repo = lifespan_data["repo"]

        classes = []
        for sem_class in repo.list_classes(domain):
            # Count predicates and patterns
            pred_count = len(list(repo.get_predicates_by_class(sem_class.id)))
            pattern_count = len(list(repo.get_patterns_by_class(sem_class.id)))

            classes.append({
                "id": sem_class.id,
                "name": sem_class.name,
                "domain": sem_class.domain,
                "description": sem_class.description,
                "predicate_count": pred_count,
                "pattern_count": pattern_count,
                "provenance": sem_class.provenance,
                "confidence": sem_class.confidence,
            })

        logger.info(f"list_semantic_classes: found {len(classes)} classes")
        return {
            "status": "success",
            "classes": classes,
            "total": len(classes),
            "domain_filter": domain,
        }

    @server.tool(
        name="add_predicate_to_class",
        description="""
Add a predicate to a semantic class.

Args:
    class_id: The class ID
    lemma: Predicate lemma to add
    is_seed: Whether this is a seed predicate (default False)

Returns:
  - class_id, class_name: the class
  - lemma: the predicate added
  - is_seed: whether it's a seed predicate
  - added: true if successful
""",
    )
    async def add_predicate_to_class(
        class_id: str,
        lemma: str,
        is_seed: bool = False,
    ) -> dict:
        """Add a predicate to a class."""
        logger.debug(f"add_predicate_to_class: class_id={class_id}, lemma='{lemma}'")

        from graphbrain.classification.models import PredicateBankEntry

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        repo = lifespan_data["repo"]

        # Verify class exists
        sem_class = repo.get_class(class_id)
        if not sem_class:
            logger.warning(f"add_predicate_to_class: class '{class_id}' not found")
            return not_found_error("semantic_class", class_id)

        entry = PredicateBankEntry(
            class_id=class_id,
            lemma=lemma,
            is_seed=is_seed,
            frequency=0,
            created_at=datetime.now(),
        )
        repo.save_predicate(entry)

        logger.info(f"add_predicate_to_class: added '{lemma}' to class '{sem_class.name}'")
        return {
            "status": "success",
            "class_id": class_id,
            "class_name": sem_class.name,
            "lemma": lemma,
            "is_seed": is_seed,
            "added": True,
        }

    @server.tool(
        name="add_pattern_to_class",
        description="""
Add a structural pattern to a semantic class.

Args:
    class_id: The class ID
    pattern: Pattern in SH notation
    priority: Pattern priority (higher = checked first)

Returns:
  - class_id, class_name: the class
  - pattern: the pattern added
  - priority: the pattern priority
  - added: true if successful
""",
    )
    async def add_pattern_to_class(
        class_id: str,
        pattern: str,
        priority: int = 0,
    ) -> dict:
        """Add a pattern to a class."""
        logger.debug(f"add_pattern_to_class: class_id={class_id}, pattern='{pattern}'")

        from graphbrain.classification.models import ClassPattern
        import graphbrain.hyperedge as he

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        repo = lifespan_data["repo"]

        # Verify class exists
        sem_class = repo.get_class(class_id)
        if not sem_class:
            logger.warning(f"add_pattern_to_class: class '{class_id}' not found")
            return not_found_error("semantic_class", class_id)

        # Validate pattern syntax
        try:
            he.hedge(pattern)
        except Exception as e:
            logger.warning(f"add_pattern_to_class: invalid pattern syntax - {e}")
            return invalid_pattern_error(pattern, e)

        pattern_obj = ClassPattern(
            class_id=class_id,
            pattern=pattern,
            pattern_type="structural",
            priority=priority,
            created_at=datetime.now(),
        )
        repo.save_pattern(pattern_obj)

        logger.info(f"add_pattern_to_class: added pattern to class '{sem_class.name}'")
        return {
            "status": "success",
            "class_id": class_id,
            "class_name": sem_class.name,
            "pattern": pattern,
            "priority": priority,
            "added": True,
        }

    @server.tool(
        name="delete_semantic_class",
        description="""
Delete a semantic class and all associated data.

This permanently removes the class, its predicates, patterns, and classifications.
Use with caution!

Args:
    class_id: The class ID to delete

Returns:
  - deleted: true if successful
  - class_id: the deleted class ID
  - class_name: the deleted class name
""",
    )
    async def delete_semantic_class(
        class_id: str,
    ) -> dict:
        """Delete a semantic class."""
        logger.debug(f"delete_semantic_class: class_id={class_id}")

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        repo = lifespan_data["repo"]

        # Get class info before deletion
        sem_class = repo.get_class(class_id)
        if not sem_class:
            logger.warning(f"delete_semantic_class: class '{class_id}' not found")
            return not_found_error("semantic_class", class_id)

        deleted = repo.delete_class(class_id)

        logger.info(f"delete_semantic_class: deleted class '{sem_class.name}'")
        return {
            "status": "success",
            "deleted": deleted,
            "class_id": class_id,
            "class_name": sem_class.name,
        }

    @server.tool(
        name="classification_stats",
        description="""
Get statistics about the classification system.

Returns:
  - total_classes: number of semantic classes
  - total_predicates: number of predicates in banks
  - total_patterns: number of patterns
  - pending_feedback: number of pending feedback items
""",
    )
    async def classification_stats() -> dict:
        """Get classification statistics."""
        logger.debug("classification_stats: computing statistics")

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        repo = lifespan_data["repo"]

        stats = repo.get_stats()
        logger.info(f"classification_stats: {stats}")
        return {"status": "success", **stats}
