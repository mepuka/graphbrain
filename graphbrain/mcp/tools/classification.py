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
        logger.debug(f"classify_predicate: predicate='{predicate}', threshold={threshold}")

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        repo = lifespan_data["repo"]

        classifications = []
        needs_review = False

        # Find predicate in predicate banks
        for entry, sem_class in repo.find_predicate(predicate):
            # Handle confidence: use similarity_score if set, otherwise default based on is_seed
            if entry.similarity_score is not None:
                confidence = entry.similarity_score
            elif entry.is_seed:
                confidence = 1.0
            else:
                confidence = 0.7
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

    @server.tool(
        name="classify_edge",
        description="""
Classify an edge into semantic classes.

Analyzes the edge structure and predicate to determine semantic classes.
Uses both predicate classification and structural pattern matching.

Args:
    edge: Edge in SH notation
    threshold: Minimum confidence threshold (default 0.5)

Returns:
  - edge: the input edge
  - classifications: list of {class_name, class_id, confidence, method}
  - methods: list of methods used (predicate, pattern)
  - needs_review: true if low confidence or no match found
""",
    )
    async def classify_edge(
        edge: str,
        threshold: float = 0.5,
    ) -> dict:
        """Classify an edge."""
        logger.debug(f"classify_edge: edge='{edge}', threshold={threshold}")

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

        # Get the main predicate if it's a relation
        if not parsed_edge.atom and len(parsed_edge) > 0:
            connector = parsed_edge[0]
            if connector.atom:
                # Get lemma from the connector
                lemma = connector.root()

                # Classify the predicate
                for entry, sem_class in repo.find_predicate(lemma):
                    # Handle confidence: use similarity_score if set, otherwise default based on is_seed
                    if entry.similarity_score is not None:
                        confidence = entry.similarity_score
                    elif entry.is_seed:
                        confidence = 1.0
                    else:
                        confidence = 0.7
                    classifications.append({
                        "class_name": sem_class.name,
                        "class_id": sem_class.id,
                        "confidence": confidence,
                        "method": "predicate",
                    })
                    if "predicate" not in methods_used:
                        methods_used.append("predicate")

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
