"""Algorithm tools for MCP server.

Provides tools for graph algorithms including centrality, community detection,
path finding, and graph statistics.
"""

import logging
from typing import Optional, Literal

from mcp.server.fastmcp import FastMCP

import graphbrain.hyperedge as he
from graphbrain.mcp.errors import invalid_edge_error


logger = logging.getLogger(__name__)


def register_algorithm_tools(server: FastMCP):
    """Register algorithm tools with the MCP server."""

    @server.tool(
        name="compute_centrality",
        description="""
Compute centrality scores for atoms in the hypergraph.

Centrality measures identify the most important or influential nodes
in the knowledge graph.

Algorithms:
- pagerank: Google's PageRank (default). Good for finding authoritative entities.
- betweenness: Betweenness centrality. Finds brokers/bridges between communities.
- closeness: Closeness centrality. Finds central, well-connected nodes.
- degree: Degree centrality. Simple count of connections.
- eigenvector: Eigenvector centrality. Influence from well-connected neighbors.

Args:
    algorithm: Centrality algorithm to use
    top_k: Return only top K results (default 50)
    store: If true, store scores as attributes on atoms

Returns:
  - results: list of {atom, score} sorted by score descending
  - algorithm: the algorithm used
  - total: number of atoms scored
""",
    )
    async def compute_centrality(
        algorithm: Literal["pagerank", "betweenness", "closeness", "degree", "eigenvector"] = "pagerank",
        top_k: int = 50,
        store: bool = False,
    ) -> dict:
        """Compute centrality scores."""
        logger.debug(f"compute_centrality: algorithm={algorithm}, top_k={top_k}")

        from graphbrain.algorithms import (
            pagerank,
            betweenness_centrality,
            closeness_centrality,
            degree_centrality,
            eigenvector_centrality,
        )

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        hg = lifespan_data["hg"]

        try:
            if algorithm == "pagerank":
                scores = pagerank(hg, store=store)
            elif algorithm == "betweenness":
                scores = betweenness_centrality(hg, store=store)
            elif algorithm == "closeness":
                scores = closeness_centrality(hg, store=store)
            elif algorithm == "degree":
                scores = degree_centrality(hg, store=store)
            elif algorithm == "eigenvector":
                scores = eigenvector_centrality(hg, store=store)
            else:
                return {
                    "status": "error",
                    "code": "invalid_algorithm",
                    "message": f"Unknown algorithm: {algorithm}",
                }

            # Sort by score descending and take top_k
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            results = [
                {"atom": str(atom), "score": float(score)}
                for atom, score in sorted_scores[:top_k]
            ]

            logger.info(f"compute_centrality: computed {len(scores)} scores, returning top {len(results)}")
            return {
                "status": "success",
                "results": results,
                "algorithm": algorithm,
                "total": len(scores),
            }
        except Exception as e:
            logger.error(f"compute_centrality: error - {e}")
            return {
                "status": "error",
                "code": "algorithm_error",
                "message": str(e),
            }

    @server.tool(
        name="find_communities",
        description="""
Detect communities (clusters) of related atoms in the hypergraph.

Community detection groups atoms that are densely connected to each other
but sparsely connected to other groups.

Algorithms:
- louvain: Louvain method (default). Fast, good quality, adjustable resolution.
- label_propagation: Label propagation. Very fast, semi-random results.
- greedy_modularity: Greedy modularity optimization. Deterministic.
- connected_components: Find disconnected components. Simple but fundamental.

Args:
    algorithm: Community detection algorithm
    min_size: Minimum community size to return (default 1)
    resolution: Resolution parameter for louvain (higher = more communities)

Returns:
  - communities: list of {id, size, members} where members are atom strings
  - algorithm: the algorithm used
  - total: number of communities found
""",
    )
    async def find_communities(
        algorithm: Literal["louvain", "label_propagation", "greedy_modularity", "connected_components"] = "louvain",
        min_size: int = 1,
        resolution: float = 1.0,
    ) -> dict:
        """Detect communities."""
        logger.debug(f"find_communities: algorithm={algorithm}, min_size={min_size}")

        from graphbrain.algorithms import (
            louvain_communities,
            label_propagation,
            greedy_modularity_communities,
            connected_components,
        )

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        hg = lifespan_data["hg"]

        try:
            if algorithm == "louvain":
                communities = louvain_communities(hg, resolution=resolution)
            elif algorithm == "label_propagation":
                communities = label_propagation(hg)
            elif algorithm == "greedy_modularity":
                communities = greedy_modularity_communities(hg)
            elif algorithm == "connected_components":
                communities = connected_components(hg)
            else:
                return {
                    "status": "error",
                    "code": "invalid_algorithm",
                    "message": f"Unknown algorithm: {algorithm}",
                }

            # Filter by min_size and format
            results = []
            for i, community in enumerate(communities):
                if len(community) >= min_size:
                    results.append({
                        "id": i,
                        "size": len(community),
                        "members": [str(atom) for atom in list(community)[:100]],  # Limit members shown
                    })

            # Sort by size descending
            results.sort(key=lambda x: x["size"], reverse=True)

            logger.info(f"find_communities: found {len(results)} communities (min_size={min_size})")
            return {
                "status": "success",
                "communities": results,
                "algorithm": algorithm,
                "total": len(results),
            }
        except Exception as e:
            logger.error(f"find_communities: error - {e}")
            return {
                "status": "error",
                "code": "algorithm_error",
                "message": str(e),
            }

    @server.tool(
        name="find_path",
        description="""
Find the shortest path between two atoms in the hypergraph.

Uses the co-occurrence projection where atoms are connected if they
appear together in the same hyperedge.

Args:
    source: Source atom (e.g., "alice/C", "says/Pd")
    target: Target atom
    max_depth: Maximum path length (default 10)

Returns:
  - path: list of atoms forming the shortest path, or null if no path
  - length: path length (number of edges)
  - exists: whether a path exists
""",
    )
    async def find_path(
        source: str,
        target: str,
        max_depth: int = 10,
    ) -> dict:
        """Find shortest path between atoms."""
        logger.debug(f"find_path: source='{source}', target='{target}'")

        from graphbrain.algorithms import shortest_path, has_path

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        hg = lifespan_data["hg"]

        try:
            source_atom = he.hedge(source)
            target_atom = he.hedge(target)
        except Exception as e:
            return invalid_edge_error(f"{source} or {target}", e)

        try:
            # Check if path exists first
            if not has_path(hg, source_atom, target_atom):
                return {
                    "status": "success",
                    "path": None,
                    "length": None,
                    "exists": False,
                }

            path = shortest_path(hg, source_atom, target_atom)
            if path:
                path_strs = [str(atom) for atom in path]
                logger.info(f"find_path: found path of length {len(path) - 1}")
                return {
                    "status": "success",
                    "path": path_strs,
                    "length": len(path) - 1,
                    "exists": True,
                }
            else:
                return {
                    "status": "success",
                    "path": None,
                    "length": None,
                    "exists": False,
                }
        except Exception as e:
            logger.error(f"find_path: error - {e}")
            return {
                "status": "error",
                "code": "algorithm_error",
                "message": str(e),
            }

    @server.tool(
        name="get_graph_stats",
        description="""
Get comprehensive statistics about the hypergraph structure.

Returns metrics about the graph topology including connectivity,
clustering, and structure.

Returns:
  - nodes: number of atoms
  - edges: number of graph edges (co-occurrences)
  - components: number of connected components
  - density: graph density (0-1)
  - clustering: average clustering coefficient
  - transitivity: graph transitivity
  - diameter: graph diameter (longest shortest path)
""",
    )
    async def get_graph_stats() -> dict:
        """Get graph statistics."""
        logger.debug("get_graph_stats: computing statistics")

        from graphbrain.algorithms import (
            summary,
            connected_components,
        )

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        hg = lifespan_data["hg"]

        try:
            stats = summary(hg)
            components = connected_components(hg)

            result = {
                "status": "success",
                "nodes": stats.get("nodes", 0),
                "edges": stats.get("edges", 0),
                "components": len(components),
                "density": stats.get("density", 0.0),
                "clustering": stats.get("average_clustering", 0.0),
                "transitivity": stats.get("transitivity", 0.0),
                "diameter": stats.get("diameter"),
            }

            logger.info(f"get_graph_stats: {result['nodes']} nodes, {result['edges']} edges, {result['components']} components")
            return result
        except Exception as e:
            logger.error(f"get_graph_stats: error - {e}")
            return {
                "status": "error",
                "code": "algorithm_error",
                "message": str(e),
            }

    @server.tool(
        name="extract_subgraph",
        description="""
Extract a subgraph around a center atom.

Useful for exploring the local neighborhood of an entity.

Args:
    center: The center atom
    radius: How many hops from center to include (default 1)
    include_edges: Whether to return the connecting hyperedges

Returns:
  - atoms: list of atoms in the subgraph
  - size: number of atoms
  - edges: list of hyperedges (if include_edges=true)
""",
    )
    async def extract_subgraph(
        center: str,
        radius: int = 1,
        include_edges: bool = False,
    ) -> dict:
        """Extract ego subgraph around center atom."""
        logger.debug(f"extract_subgraph: center='{center}', radius={radius}")

        from graphbrain.algorithms import ego_atoms, ego_graph

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        hg = lifespan_data["hg"]

        try:
            center_atom = he.hedge(center)
        except Exception as e:
            return invalid_edge_error(center, e)

        try:
            atoms = ego_atoms(hg, center_atom, depth=radius)
            atom_strs = [str(atom) for atom in atoms]

            result = {
                "status": "success",
                "atoms": atom_strs,
                "size": len(atoms),
            }

            if include_edges:
                edges = list(ego_graph(hg, center_atom, depth=radius))
                result["edges"] = [e.to_str() for e in edges[:500]]  # Limit to 500
                result["edge_count"] = len(edges)

            logger.info(f"extract_subgraph: found {len(atoms)} atoms")
            return result
        except Exception as e:
            logger.error(f"extract_subgraph: error - {e}")
            return {
                "status": "error",
                "code": "algorithm_error",
                "message": str(e),
            }

    @server.tool(
        name="find_neighbors",
        description="""
Find atoms that co-occur with a given atom in hyperedges.

Neighbors are atoms that appear in the same hyperedge as the given atom.

Args:
    atom: The atom to find neighbors for
    limit: Maximum neighbors to return (default 100)

Returns:
  - neighbors: list of neighbor atoms
  - total: number of neighbors found
""",
    )
    async def find_neighbors(
        atom: str,
        limit: int = 100,
    ) -> dict:
        """Find neighboring atoms."""
        logger.debug(f"find_neighbors: atom='{atom}', limit={limit}")

        from graphbrain.algorithms import neighbors

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        hg = lifespan_data["hg"]

        try:
            atom_edge = he.hedge(atom)
        except Exception as e:
            return invalid_edge_error(atom, e)

        try:
            neighbor_set = neighbors(hg, atom_edge)
            neighbor_list = [str(n) for n in list(neighbor_set)[:limit]]

            logger.info(f"find_neighbors: found {len(neighbor_set)} neighbors")
            return {
                "status": "success",
                "neighbors": neighbor_list,
                "total": len(neighbor_set),
            }
        except Exception as e:
            logger.error(f"find_neighbors: error - {e}")
            return {
                "status": "error",
                "code": "algorithm_error",
                "message": str(e),
            }

    @server.tool(
        name="find_reachable",
        description="""
Find all atoms reachable from a source atom within a given depth.

Useful for finding the transitive closure of relationships.

Args:
    source: The source atom
    max_depth: Maximum depth to search (default: no limit)

Returns:
  - atoms: list of reachable atoms
  - total: number of reachable atoms
""",
    )
    async def find_reachable(
        source: str,
        max_depth: Optional[int] = None,
    ) -> dict:
        """Find all reachable atoms."""
        logger.debug(f"find_reachable: source='{source}', max_depth={max_depth}")

        from graphbrain.algorithms import reachable

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        hg = lifespan_data["hg"]

        try:
            source_atom = he.hedge(source)
        except Exception as e:
            return invalid_edge_error(source, e)

        try:
            reachable_set = reachable(hg, source_atom, max_depth=max_depth)
            atom_list = [str(a) for a in list(reachable_set)[:1000]]  # Limit to 1000

            logger.info(f"find_reachable: found {len(reachable_set)} reachable atoms")
            return {
                "status": "success",
                "atoms": atom_list,
                "total": len(reachable_set),
            }
        except Exception as e:
            logger.error(f"find_reachable: error - {e}")
            return {
                "status": "error",
                "code": "algorithm_error",
                "message": str(e),
            }
