"""Path finding algorithms for hypergraphs.

This module provides algorithms for finding paths and computing
reachability between atoms in a hypergraph.
"""

from typing import Callable, Dict, List, Optional, Set, Union

import networkx as nx

from graphbrain.hypergraph import Hypergraph
from graphbrain.hyperedge import Hyperedge, hedge
from graphbrain.algorithms.projections import (
    AtomCooccurrenceProjection,
    HypergraphProjection,
)


def _get_projection(
    hg: Hypergraph,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> nx.Graph:
    """Get a NetworkX graph from a hypergraph projection."""
    if projection is not None:
        return projection.to_networkx()
    return AtomCooccurrenceProjection(hg, filter_fn).to_networkx()


def _to_hedge(edge: Union[str, Hyperedge]) -> Hyperedge:
    """Convert to Hyperedge if necessary."""
    if isinstance(edge, str):
        return hedge(edge)
    return edge


def shortest_path(
    hg: Hypergraph,
    source: Union[str, Hyperedge],
    target: Union[str, Hyperedge],
    weight: Optional[str] = None,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> Optional[List]:
    """Find the shortest path between two atoms.

    Args:
        hg: The hypergraph.
        source: Starting atom.
        target: Ending atom.
        weight: Edge attribute to use as weight. None for unweighted.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        List of atoms forming the path from source to target,
        or None if no path exists.
    """
    g = _get_projection(hg, projection, filter_fn)
    source = _to_hedge(source)
    target = _to_hedge(target)

    if source not in g or target not in g:
        return None

    try:
        return nx.shortest_path(g, source, target, weight=weight)
    except nx.NetworkXNoPath:
        return None


def shortest_path_length(
    hg: Hypergraph,
    source: Union[str, Hyperedge],
    target: Union[str, Hyperedge],
    weight: Optional[str] = None,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> Optional[int]:
    """Find the length of the shortest path between two atoms.

    Args:
        hg: The hypergraph.
        source: Starting atom.
        target: Ending atom.
        weight: Edge attribute to use as weight. None for unweighted.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        Length of the shortest path, or None if no path exists.
    """
    g = _get_projection(hg, projection, filter_fn)
    source = _to_hedge(source)
    target = _to_hedge(target)

    if source not in g or target not in g:
        return None

    try:
        return nx.shortest_path_length(g, source, target, weight=weight)
    except nx.NetworkXNoPath:
        return None


def all_shortest_paths(
    hg: Hypergraph,
    source: Union[str, Hyperedge],
    target: Union[str, Hyperedge],
    weight: Optional[str] = None,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> List[List]:
    """Find all shortest paths between two atoms.

    Args:
        hg: The hypergraph.
        source: Starting atom.
        target: Ending atom.
        weight: Edge attribute to use as weight. None for unweighted.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        List of paths, where each path is a list of atoms.
        Empty list if no path exists.
    """
    g = _get_projection(hg, projection, filter_fn)
    source = _to_hedge(source)
    target = _to_hedge(target)

    if source not in g or target not in g:
        return []

    try:
        return list(nx.all_shortest_paths(g, source, target, weight=weight))
    except nx.NetworkXNoPath:
        return []


def reachable(
    hg: Hypergraph,
    source: Union[str, Hyperedge],
    max_depth: Optional[int] = None,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> Set:
    """Find all atoms reachable from source within a given depth.

    Args:
        hg: The hypergraph.
        source: Starting atom.
        max_depth: Maximum path length to consider. None for unlimited.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        Set of atoms reachable from source (including source itself).
    """
    g = _get_projection(hg, projection, filter_fn)
    source = _to_hedge(source)

    if source not in g:
        return {source}

    if max_depth is None:
        # Get all nodes in the same connected component
        return set(nx.node_connected_component(g, source))
    else:
        # Use BFS with depth limit
        reachable_nodes = {source}
        frontier = {source}

        for _ in range(max_depth):
            next_frontier = set()
            for node in frontier:
                for neighbor in g.neighbors(node):
                    if neighbor not in reachable_nodes:
                        reachable_nodes.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
            if not frontier:
                break

        return reachable_nodes


def has_path(
    hg: Hypergraph,
    source: Union[str, Hyperedge],
    target: Union[str, Hyperedge],
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> bool:
    """Check if a path exists between two atoms.

    More efficient than finding the actual path if you only need
    to check existence.

    Args:
        hg: The hypergraph.
        source: Starting atom.
        target: Ending atom.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        True if a path exists, False otherwise.
    """
    g = _get_projection(hg, projection, filter_fn)
    source = _to_hedge(source)
    target = _to_hedge(target)

    if source not in g or target not in g:
        return False

    return nx.has_path(g, source, target)


def all_pairs_shortest_path_length(
    hg: Hypergraph,
    cutoff: Optional[int] = None,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> Dict:
    """Compute shortest path lengths between all pairs of atoms.

    Warning: This can be expensive for large graphs.

    Args:
        hg: The hypergraph.
        cutoff: Maximum path length to compute. None for unlimited.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        Dictionary of dictionaries where result[u][v] is the
        shortest path length from u to v.
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return {}

    return dict(nx.all_pairs_shortest_path_length(g, cutoff=cutoff))


def diameter(
    hg: Hypergraph,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> Optional[int]:
    """Compute the diameter of the hypergraph.

    The diameter is the maximum shortest path length between any
    pair of atoms. Only computed on the largest connected component
    if the graph is disconnected.

    Args:
        hg: The hypergraph.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        The diameter, or None if the graph is empty.
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return None

    # If graph is disconnected, compute diameter of largest component
    if not nx.is_connected(g):
        largest_cc = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_cc)

    return nx.diameter(g)


def radius(
    hg: Hypergraph,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> Optional[int]:
    """Compute the radius of the hypergraph.

    The radius is the minimum eccentricity of any node (the minimum
    of the maximum distances from each node to all others).

    Args:
        hg: The hypergraph.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        The radius, or None if the graph is empty.
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return None

    # If graph is disconnected, compute radius of largest component
    if not nx.is_connected(g):
        largest_cc = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_cc)

    return nx.radius(g)


def center(
    hg: Hypergraph,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> List:
    """Find the center atoms of the hypergraph.

    The center is the set of atoms with minimum eccentricity (equal to
    the radius).

    Args:
        hg: The hypergraph.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        List of atoms in the center.
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return []

    # If graph is disconnected, compute center of largest component
    if not nx.is_connected(g):
        largest_cc = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_cc)

    return list(nx.center(g))


def periphery(
    hg: Hypergraph,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> List:
    """Find the periphery atoms of the hypergraph.

    The periphery is the set of atoms with maximum eccentricity (equal
    to the diameter).

    Args:
        hg: The hypergraph.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        List of atoms in the periphery.
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return []

    # If graph is disconnected, compute periphery of largest component
    if not nx.is_connected(g):
        largest_cc = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_cc)

    return list(nx.periphery(g))


def eccentricity(
    hg: Hypergraph,
    atom: Optional[Union[str, Hyperedge]] = None,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> Union[int, Dict]:
    """Compute eccentricity of atoms.

    The eccentricity of an atom is its maximum distance to any other atom.

    Args:
        hg: The hypergraph.
        atom: Specific atom to compute eccentricity for. If None,
              computes for all atoms.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        If atom is specified: eccentricity as an integer.
        Otherwise: dictionary mapping atoms to their eccentricities.
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return {} if atom is None else 0

    # If graph is disconnected, use largest component
    if not nx.is_connected(g):
        largest_cc = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_cc)

    if atom is not None:
        atom = _to_hedge(atom)
        if atom not in g:
            return 0
        return nx.eccentricity(g, v=atom)
    else:
        return nx.eccentricity(g)
