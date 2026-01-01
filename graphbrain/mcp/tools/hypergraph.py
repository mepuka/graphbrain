"""Hypergraph tools for MCP server.

Provides tools for searching, adding, and pattern matching edges.
"""

import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

import graphbrain.hyperedge as he
from graphbrain.mcp.errors import (
    invalid_edge_error,
    invalid_pattern_error,
    database_error,
)

logger = logging.getLogger(__name__)


def register_hypergraph_tools(server: FastMCP):
    """Register hypergraph tools with the MCP server."""

    @server.tool(
        name="search_edges",
        description="""
Search for edges in the hypergraph by text content.

Uses full-text search (BM25-like ranking) to find edges
containing the query terms. Results are sorted by relevance score.

Returns:
  - edges: list of {edge, text, score}
  - total: number of results
  - query: the search query used
""",
    )
    async def search_edges(
        query: str,
        limit: int = 50,
    ) -> dict:
        """Search edges by text content."""
        logger.debug(f"search_edges: query='{query}', limit={limit}")

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        searcher = lifespan_data.get("searcher")

        if not searcher:
            return {
                "status": "error",
                "code": "service_unavailable",
                "message": "Search backend not available. Text content may not be indexed.",
                "edges": [],
                "total": 0,
                "query": query,
            }

        results = []
        count = 0

        try:
            # Use the search backend's BM25 search
            for result in searcher.bm25_search(query, limit=limit):
                results.append({
                    "edge": result.edge_key,
                    "text": result.text_content,
                    "score": float(result.bm25_score) if result.bm25_score else 0.0,
                })
                count += 1

            logger.info(f"search_edges: found {count} results for '{query}'")
        except Exception as e:
            logger.error(f"search_edges: search error - {e}")
            return database_error("search_edges", e)

        return {
            "status": "success",
            "edges": results,
            "total": count,
            "query": query,
        }

    @server.tool(
        name="pattern_match",
        description="""
Find edges matching a structural pattern.

Uses graphbrain's pattern matching syntax with wildcards and constraints.

Pattern examples:
- "(*/Pd * *)" - Any predicate with any arguments
- "(says/Pd.sr * *)" - "says" predicate with source/recipient roles
- "(*/Pd.{so} */Cp *)" - Predicate with subject/object, concept as first arg
- "(*/* * (*/Pd * *))" - Nested predicate patterns

Special pattern symbols:
- * : Match any single atom or edge
- . : Match any single atom only
- (*) : Match any edge only
- */T : Match any atom of type T (Pd=predicate, Cp=proper concept, etc.)
- {xy} : Require argument roles x and y
- ... : Open-ended pattern (matches additional elements)

Variable capture (use UPPERCASE names with type):
- SPEAKER/Cp : Capture a proper noun to SPEAKER
- MESSAGE/* : Capture anything to MESSAGE
- PRED/Pd : Capture a declarative predicate to PRED

Returns:
  - edges: list of {edge, bindings} where bindings maps VAR names to matched values
  - pattern: the pattern used
  - total: number of matches found
""",
    )
    async def pattern_match(
        pattern: str,
        limit: int = 100,
    ) -> dict:
        """Match edges against a structural pattern."""
        logger.debug(f"pattern_match: pattern='{pattern}', limit={limit}")

        from graphbrain.patterns import match_pattern

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        hg = lifespan_data["hg"]

        try:
            pattern_edge = he.hedge(pattern)
        except Exception as e:
            logger.warning(f"pattern_match: invalid pattern '{pattern}' - {e}")
            return invalid_pattern_error(pattern, e)

        results = []
        count = 0

        for edge in hg.all():
            if count >= limit:
                break

            # match_pattern returns List[Dict] of bindings, or empty list if no match
            bindings_list = match_pattern(edge, pattern_edge)
            if bindings_list:
                # Take first binding set (there may be multiple valid bindings)
                bindings = bindings_list[0]
                results.append({
                    "edge": edge.to_str(),
                    "bindings": {k: v.to_str() for k, v in bindings.items()},
                })
                count += 1

        logger.info(f"pattern_match: found {count} matches for pattern")
        return {
            "status": "success",
            "edges": results,
            "pattern": pattern,
            "total": count,
        }

    @server.tool(
        name="add_edge",
        description="""
Add an edge to the hypergraph.

Parses the edge from SH (Semantic Hypergraph) notation and adds it
to the database. Optionally associates source text.

SH notation examples:
- "(says/Pd.sr john/Cp hello/C)" - John says hello
- "(is/Pd.sc sky/Cc blue/Ca)" - The sky is blue
- "(of/Br.ma capital/Cc france/Cp)" - Capital of France

Args:
    edge: Edge in SH notation
    text: Optional source text
    primary: Mark as primary edge (default True)
""",
    )
    async def add_edge(
        edge: str,
        text: Optional[str] = None,
        primary: bool = True,
    ) -> dict:
        """Add an edge to the hypergraph."""
        logger.debug(f"add_edge: edge='{edge}', primary={primary}")

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        hg = lifespan_data["hg"]

        try:
            parsed_edge = he.hedge(edge)
        except Exception as e:
            logger.warning(f"add_edge: invalid edge syntax - {e}")
            return invalid_edge_error(edge, e)

        existed = hg.exists(parsed_edge)

        if primary:
            hg.add(parsed_edge, primary=True)
        else:
            hg.add(parsed_edge)

        # Add text content if provided
        if text:
            hg.set_attribute(parsed_edge, "text", text)
            # Also index for full-text search
            searcher = lifespan_data.get("searcher")
            if searcher:
                try:
                    searcher.add_text(parsed_edge.to_str(), text)
                except Exception as e:
                    logger.warning(f"add_edge: failed to index text - {e}")

        logger.info(f"add_edge: {'updated' if existed else 'added'} edge")
        return {
            "status": "success",
            "edge": parsed_edge.to_str(),
            "added": True,
            "existed": existed,
        }

    @server.tool(
        name="get_edge",
        description="""
Get details about a specific edge.

Returns the edge's attributes, degree (number of containing edges),
and existence status.

Args:
    edge: Edge in SH notation
""",
    )
    async def get_edge(edge: str) -> dict:
        """Get edge details."""
        logger.debug(f"get_edge: edge='{edge}'")

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        hg = lifespan_data["hg"]

        try:
            parsed_edge = he.hedge(edge)
        except Exception as e:
            logger.warning(f"get_edge: invalid edge syntax - {e}")
            return invalid_edge_error(edge, e)

        if not hg.exists(parsed_edge):
            logger.debug(f"get_edge: edge not found")
            return {
                "status": "success",
                "exists": False,
                "edge": edge,
            }

        attrs = hg.get_attributes(parsed_edge)
        degree = hg.degree(parsed_edge)

        return {
            "status": "success",
            "exists": True,
            "edge": parsed_edge.to_str(),
            "degree": degree,
            "attributes": attrs,
        }

    @server.tool(
        name="edges_with_root",
        description="""
Find edges that have a specific atom as their root (first element).

Useful for finding all predicates of a certain type, all concepts
in a relation, etc.

Args:
    root: The root atom (e.g., "says/Pd", "is/Pd", "of/Br")
    limit: Maximum results (default 100)

Returns:
  - edges: list of {edge}
  - root: the root atom used
  - total: number of results
""",
    )
    async def edges_with_root(
        root: str,
        limit: int = 100,
    ) -> dict:
        """Find edges with a specific root."""
        logger.debug(f"edges_with_root: root='{root}', limit={limit}")

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        hg = lifespan_data["hg"]

        try:
            root_atom = he.hedge(root)
        except Exception as e:
            logger.warning(f"edges_with_root: invalid root '{root}' - {e}")
            return invalid_edge_error(root, e)

        results = []
        count = 0

        for edge in hg.edges_with_edges([root_atom], root=root_atom):
            if count >= limit:
                break
            results.append({
                "edge": edge.to_str(),
            })
            count += 1

        logger.info(f"edges_with_root: found {count} edges with root '{root}'")
        return {
            "status": "success",
            "edges": results,
            "root": root,
            "total": count,
        }

    @server.tool(
        name="hypergraph_stats",
        description="""
Get statistics about the hypergraph.

Returns:
  - total_entries: total count of all entries
  - atoms: count of atomic entries
  - edges: count of compound edges
  - primary_edges: count of primary (top-level) edges
""",
    )
    async def hypergraph_stats() -> dict:
        """Get hypergraph statistics."""
        logger.debug("hypergraph_stats: computing statistics")

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        hg = lifespan_data["hg"]

        total = 0
        atoms = 0
        edges_count = 0
        primary = 0

        for edge in hg.all():
            total += 1
            if edge.atom:
                atoms += 1
            else:
                edges_count += 1
            if hg.is_primary(edge):
                primary += 1

        logger.info(f"hypergraph_stats: {total} entries ({atoms} atoms, {edges_count} edges, {primary} primary)")
        return {
            "status": "success",
            "total_entries": total,
            "atoms": atoms,
            "edges": edges_count,
            "primary_edges": primary,
        }
