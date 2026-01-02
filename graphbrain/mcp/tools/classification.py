"""Classification tools for MCP server.

Provides tools for classifying predicates and edges using hybrid search.
"""

import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

from graphbrain.mcp.errors import (
    invalid_edge_error,
    service_unavailable_error,
)
from graphbrain.mcp.utils import validate_threshold, validate_limit, calculate_confidence

logger = logging.getLogger(__name__)


def register_classification_tools(server: FastMCP):
    """Register classification tools with the MCP server."""

    @server.tool(
        name="classify_predicate",
        description="""
Classify a predicate lemma into semantic classes.

Looks up the predicate in the predicate banks to determine which
semantic classes it belongs to (e.g., "say" -> "claim", "attack" -> "conflict").

Args:
    predicate: Predicate lemma to classify (e.g., "say", "announce", "attack")
    threshold: Minimum confidence threshold (default 0.5)

Returns:
  - predicate: the input predicate
  - classifications: list of {class_name, class_id, confidence, method, is_seed}
  - needs_review: true if low confidence or no match found
  - threshold: the threshold used
""",
    )
    async def classify_predicate(
        predicate: str,
        threshold: float = 0.5,
    ) -> dict:
        """Classify a predicate lemma."""
        # Validate inputs
        if error := validate_threshold(threshold):
            return error

        logger.debug(f"classify_predicate: predicate='{predicate}', threshold={threshold}")

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        repo = lifespan_data["repo"]

        classifications = []
        needs_review = False

        # Find predicate in predicate banks
        for entry, sem_class in repo.find_predicate(predicate):
            # Calculate confidence using centralized helper
            confidence = calculate_confidence(entry)
            classifications.append({
                "class_name": sem_class.name,
                "class_id": sem_class.id,
                "confidence": confidence,
                "method": "predicate_bank",
                "is_seed": entry.is_seed,
            })
            if confidence < threshold:
                needs_review = True

        # If not found in banks, flag for review
        if not classifications:
            needs_review = True
            logger.debug(f"classify_predicate: no classifications found for '{predicate}'")
        else:
            logger.info(f"classify_predicate: {len(classifications)} classifications for '{predicate}'")

        return {
            "status": "success",
            "predicate": predicate,
            "classifications": classifications,
            "needs_review": needs_review,
            "threshold": threshold,
        }

    def _extract_predicates(edge):
        """Extract all predicate lemmas from an edge, including nested ones.

        Traverses the edge structure recursively to find all predicates.
        A predicate is identified by its type starting with 'P' (predicate)
        or being the connector of a relation.

        Returns list of (lemma, subedge_str, depth) tuples.
        """
        predicates = []

        def traverse(e, depth=0):
            if e.atom:
                # Check if this atom is a predicate type
                try:
                    t = e.type()
                    if t and t[0] == 'P':
                        predicates.append((e.root(), str(e), depth))
                except Exception as ex:
                    logger.debug(f"_extract_predicates: failed to get type for atom {e}: {ex}")
                return

            # Non-atomic edge - check connector
            if len(e) > 0:
                connector = e[0]
                # Get the inner connector atom
                try:
                    conn_atom = e.connector_atom()
                    if conn_atom:
                        t = conn_atom.type()
                        if t and t[0] == 'P':
                            predicates.append((conn_atom.root(), str(e), depth))
                except Exception as ex:
                    logger.debug(f"_extract_predicates: failed to get connector for edge {e}: {ex}")

                # Recursively traverse all elements
                for item in e:
                    traverse(item, depth + 1)

        traverse(edge)
        return predicates

    @server.tool(
        name="classify_edge",
        description="""
Classify an edge into semantic classes.

Analyzes the edge structure and predicate to determine semantic classes.
Uses both predicate classification and structural pattern matching.
Performs DEEP TRAVERSAL to find predicates nested within conjunctions
and other complex structures.

Args:
    edge: Edge in SH notation
    threshold: Minimum confidence threshold (default 0.5)
    deep: If True, traverse nested structures to find all predicates (default True)

Returns:
  - edge: the input edge
  - classifications: list of {class_name, class_id, confidence, method}
  - methods: list of methods used (predicate, pattern, deep_predicate)
  - needs_review: true if low confidence or no match found
  - predicates_found: list of predicates discovered in the edge
""",
    )
    async def classify_edge(
        edge: str,
        threshold: float = 0.5,
        deep: bool = True,
    ) -> dict:
        """Classify an edge."""
        # Validate inputs
        if error := validate_threshold(threshold):
            return error

        logger.debug(f"classify_edge: edge='{edge}', threshold={threshold}, deep={deep}")

        import graphbrain.hyperedge as he

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        repo = lifespan_data["repo"]

        try:
            parsed_edge = he.hedge(edge)
        except Exception as e:
            logger.warning(f"classify_edge: invalid edge syntax - {e}")
            return invalid_edge_error(edge, e)

        classifications = []
        methods_used = []
        predicates_found = []
        seen_classes = set()  # Avoid duplicate classifications

        def classify_predicate_lemma(lemma, method_name, source=None):
            """Helper to classify a predicate lemma and add to classifications."""
            for entry, sem_class in repo.find_predicate(lemma):
                # Skip if we've already classified this class
                class_key = (sem_class.id, method_name)
                if class_key in seen_classes:
                    continue
                seen_classes.add(class_key)

                # Calculate confidence using centralized helper
                confidence = calculate_confidence(entry)

                classification = {
                    "class_name": sem_class.name,
                    "class_id": sem_class.id,
                    "confidence": confidence,
                    "method": method_name,
                }
                if source:
                    classification["source_predicate"] = source
                classifications.append(classification)

                if method_name not in methods_used:
                    methods_used.append(method_name)

        # Get the main predicate if it's a relation (shallow check)
        if not parsed_edge.atom and len(parsed_edge) > 0:
            connector = parsed_edge[0]
            if connector.atom:
                lemma = connector.root()
                predicates_found.append({"lemma": lemma, "depth": 0, "source": str(connector)})
                classify_predicate_lemma(lemma, "predicate")

        # Deep traversal to find nested predicates
        if deep:
            nested_predicates = _extract_predicates(parsed_edge)
            for lemma, source, depth in nested_predicates:
                # Skip depth 0 - already handled above
                if depth > 0:
                    predicates_found.append({"lemma": lemma, "depth": depth, "source": source})
                    classify_predicate_lemma(lemma, "deep_predicate", source)

        # Check structural patterns
        from graphbrain.patterns import match_pattern

        for class_id, patterns in repo.get_all_patterns():
            for pattern_obj in patterns:
                try:
                    pattern_edge = he.hedge(pattern_obj.pattern)
                    # match_pattern returns List[Dict], non-empty means match
                    bindings_list = match_pattern(parsed_edge, pattern_edge)
                    if bindings_list:
                        # Pattern matched - get the class
                        sem_class = repo.get_class(class_id)
                        if sem_class:
                            class_key = (sem_class.id, "pattern")
                            if class_key not in seen_classes:
                                seen_classes.add(class_key)
                                classifications.append({
                                    "class_name": sem_class.name,
                                    "class_id": sem_class.id,
                                    "confidence": 0.8,  # Pattern match confidence
                                    "method": "pattern",
                                    "pattern": pattern_obj.pattern,
                                })
                                if "pattern" not in methods_used:
                                    methods_used.append("pattern")
                except Exception as e:
                    logger.debug(f"Pattern match error: {e}")

        needs_review = len(classifications) == 0 or any(c["confidence"] < threshold for c in classifications)
        logger.info(f"classify_edge: {len(classifications)} classifications, methods={methods_used}, needs_review={needs_review}")

        return {
            "status": "success",
            "edge": edge,
            "classifications": classifications,
            "methods": methods_used,
            "needs_review": needs_review,
            "predicates_found": predicates_found,
        }

    @server.tool(
        name="hybrid_search",
        description="""
Search edges using hybrid BM25 + semantic similarity.

Combines lexical search (BM25) with semantic embedding similarity
for more accurate retrieval. Results are fused using configurable weights.

Args:
    query: Search query text
    class_id: Optional - filter results to a semantic class
    bm25_weight: Weight for BM25 component (default 0.3)
    semantic_weight: Weight for semantic component (default 0.7)
    limit: Maximum results (default 50)

Returns:
  - results: list of {edge_key, combined_score, bm25_score, semantic_score, text_content}
  - query: the search query
  - weights: the weights used
  - class_id: the class filter (if any)
  - total: number of results
""",
    )
    async def hybrid_search(
        query: str,
        class_id: Optional[str] = None,
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7,
        limit: int = 50,
    ) -> dict:
        """Hybrid BM25 + semantic search."""
        # Validate inputs
        if error := validate_limit(limit, max_limit=1000):
            return error
        if error := validate_threshold(bm25_weight, "bm25_weight"):
            return error
        if error := validate_threshold(semantic_weight, "semantic_weight"):
            return error

        logger.debug(f"hybrid_search: query='{query}', class_id={class_id}, limit={limit}")

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        searcher = lifespan_data.get("searcher")

        if searcher is None:
            logger.warning("hybrid_search: searcher not available")
            return service_unavailable_error("hybrid_searcher", "not initialized")

        weights = {"bm25": bm25_weight, "semantic": semantic_weight}

        try:
            if class_id:
                results = searcher.search_by_class(class_id, query, weights=weights, limit=limit)
            else:
                results = searcher.search(query, weights=weights, limit=limit)

            logger.info(f"hybrid_search: found {len(results)} results for '{query}'")
            return {
                "status": "success",
                "results": [
                    {
                        "edge_key": r.edge_key,
                        "combined_score": r.combined_score,
                        "bm25_score": r.bm25_score,
                        "semantic_score": r.semantic_score,
                        "text_content": r.text_content,
                    }
                    for r in results
                ],
                "query": query,
                "weights": weights,
                "class_id": class_id,
                "total": len(results),
            }
        except Exception as e:
            logger.error(f"hybrid_search: error - {e}")
            return service_unavailable_error("hybrid_search", str(e))

    @server.tool(
        name="bm25_search",
        description="""
Search edges using BM25 (lexical) search only.

Uses PostgreSQL full-text search with ts_rank for BM25-like ranking.
Faster than hybrid search but less semantic.

Args:
    query: Search query (can use tsquery syntax: word1 & word2 | word3)
    limit: Maximum results (default 100)

Returns:
  - results: list of {edge_key, score, text_content}
  - query: the search query
  - total: number of results
""",
    )
    async def bm25_search(
        query: str,
        limit: int = 100,
    ) -> dict:
        """BM25-only search."""
        # Validate inputs
        if error := validate_limit(limit, max_limit=1000):
            return error

        logger.debug(f"bm25_search: query='{query}', limit={limit}")

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        searcher = lifespan_data.get("searcher")

        if searcher is None:
            logger.warning("bm25_search: searcher not available")
            return service_unavailable_error("bm25_searcher", "not initialized")

        try:
            results = list(searcher.bm25_search(query, limit=limit))
            logger.info(f"bm25_search: found {len(results)} results for '{query}'")
            return {
                "status": "success",
                "results": [
                    {
                        "edge_key": r.edge_key,
                        "score": r.bm25_score,
                        "text_content": r.text_content,
                    }
                    for r in results
                ],
                "query": query,
                "total": len(results),
            }
        except Exception as e:
            logger.error(f"bm25_search: error - {e}")
            return service_unavailable_error("bm25_search", str(e))
