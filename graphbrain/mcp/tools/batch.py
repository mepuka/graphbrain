"""Batch processing tools for MCP server.

Provides tools for bulk operations on edges and classifications.
"""

import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

from graphbrain.mcp.errors import (
    invalid_edge_error,
    service_unavailable_error,
)
from graphbrain.mcp.utils import validate_limit, calculate_confidence

logger = logging.getLogger(__name__)


def register_batch_tools(server: FastMCP):
    """Register batch processing tools with the MCP server."""

    @server.tool(
        name="batch_add_edges",
        description="""
Batch add multiple edges to the hypergraph.

More efficient than calling add_edge multiple times.

Args:
    edges: List of edge specifications, each with:
        - edge: Edge in SH notation (required)
        - text: Optional source text
        - primary: Mark as primary edge (default True)

Returns:
  - added: count of newly added edges
  - updated: count of existing edges that were updated
  - failed: list of edges that failed with error messages
  - total: total edges processed
""",
    )
    async def batch_add_edges(
        edges: list[dict],
    ) -> dict:
        """Batch add edges to hypergraph."""
        logger.debug(f"batch_add_edges: {len(edges)} edges")

        import graphbrain.hyperedge as he

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        hg = lifespan_data["hg"]
        searcher = lifespan_data.get("searcher")

        added = 0
        updated = 0
        failed = []

        # Wrap entire batch in a transaction for atomicity and performance
        try:
            hg.begin_transaction()
        except (AttributeError, NotImplementedError):
            pass  # Backend doesn't support transactions (unlikely)

        try:
            for i, edge_spec in enumerate(edges):
                edge_str = edge_spec.get("edge")
                if not edge_str:
                    failed.append({"index": i, "error": "Missing 'edge' field"})
                    continue

                text = edge_spec.get("text")
                primary = edge_spec.get("primary", True)

                try:
                    parsed_edge = he.hedge(edge_str)
                    existed = hg.exists(parsed_edge)

                    if primary:
                        hg.add(parsed_edge, primary=True)
                    else:
                        hg.add(parsed_edge)

                    if existed:
                        updated += 1
                    else:
                        added += 1

                    # Index text if provided
                    if text:
                        hg.set_attribute(parsed_edge, "text", text)
                        if searcher:
                            try:
                                searcher.add_text(parsed_edge.to_str(), text)
                            except Exception:
                                pass  # Non-critical

                except Exception as e:
                    failed.append({"index": i, "edge": edge_str, "error": str(e)})

            # Commit the transaction
            try:
                hg.end_transaction()
            except (AttributeError, NotImplementedError):
                pass
        except Exception as e:
            # Rollback on error
            try:
                hg.rollback()
            except (AttributeError, NotImplementedError):
                pass
            logger.error(f"batch_add_edges: transaction failed - {e}")
            raise

        logger.info(f"batch_add_edges: added={added}, updated={updated}, failed={len(failed)}")
        return {
            "status": "success",
            "added": added,
            "updated": updated,
            "failed": failed,
            "total": len(edges),
        }

    @server.tool(
        name="batch_classify_predicates",
        description="""
Batch classify multiple predicates.

More efficient than calling classify_predicate multiple times.

Args:
    predicates: List of predicate lemmas to classify
    threshold: Minimum confidence threshold (default 0.5)

Returns:
  - results: list of {predicate, classifications, needs_review}
  - classified: count of predicates with classifications
  - needs_review: count of predicates needing review
  - total: total predicates processed
""",
    )
    async def batch_classify_predicates(
        predicates: list[str],
        threshold: float = 0.5,
    ) -> dict:
        """Batch classify predicates."""
        logger.debug(f"batch_classify_predicates: {len(predicates)} predicates")

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        repo = lifespan_data["repo"]

        results = []
        classified_count = 0
        review_count = 0

        for predicate in predicates:
            classifications = []
            needs_review = False

            for entry, sem_class in repo.find_predicate(predicate):
                confidence = calculate_confidence(entry)
                classifications.append({
                    "class_name": sem_class.name,
                    "class_id": sem_class.id,
                    "confidence": confidence,
                    "is_seed": entry.is_seed,
                })
                if confidence < threshold:
                    needs_review = True

            if not classifications:
                needs_review = True
                review_count += 1
            else:
                classified_count += 1
                if needs_review:
                    review_count += 1

            results.append({
                "predicate": predicate,
                "classifications": classifications,
                "needs_review": needs_review,
            })

        logger.info(f"batch_classify_predicates: classified={classified_count}, needs_review={review_count}")
        return {
            "status": "success",
            "results": results,
            "classified": classified_count,
            "needs_review": review_count,
            "total": len(predicates),
        }

    @server.tool(
        name="batch_add_predicates",
        description="""
Batch add multiple predicates to a semantic class.

More efficient than calling add_predicate_to_class multiple times.

Args:
    class_id: The class ID to add predicates to
    lemmas: List of predicate lemmas to add
    is_seed: Whether these are seed predicates (default False)

Returns:
  - added: count of predicates added
  - class_id: the class ID
  - class_name: the class name
""",
    )
    async def batch_add_predicates(
        class_id: str,
        lemmas: list[str],
        is_seed: bool = False,
    ) -> dict:
        """Batch add predicates to a class."""
        logger.debug(f"batch_add_predicates: {len(lemmas)} lemmas to class {class_id}")

        from datetime import datetime
        from graphbrain.classification.models import PredicateBankEntry
        from graphbrain.mcp.errors import not_found_error

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        repo = lifespan_data["repo"]

        # Verify class exists
        sem_class = repo.get_class(class_id)
        if not sem_class:
            return not_found_error("semantic_class", class_id)

        added = 0
        for lemma in lemmas:
            entry = PredicateBankEntry(
                class_id=class_id,
                lemma=lemma,
                is_seed=is_seed,
                frequency=0,
                created_at=datetime.now(),
            )
            repo.save_predicate(entry)
            added += 1

        logger.info(f"batch_add_predicates: added {added} predicates to class '{sem_class.name}'")
        return {
            "status": "success",
            "added": added,
            "class_id": class_id,
            "class_name": sem_class.name,
        }

    logger.info("Registered 3 batch processing tools")
