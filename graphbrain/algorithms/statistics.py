"""Graph statistics and metrics for hypergraphs.

This module provides functions to compute various statistics and
structural metrics of a hypergraph.
"""

from typing import Callable, Dict, Optional

import networkx as nx

from graphbrain.hypergraph import Hypergraph
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


def summary(
    hg: Hypergraph,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> Dict:
    """Compute a summary of graph statistics.

    Returns a dictionary with key metrics about the hypergraph structure.

    Args:
        hg: The hypergraph.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        Dictionary containing:
        - node_count: Number of atoms
        - edge_count: Number of edges in projection
        - hyperedge_count: Number of non-atomic edges in original hypergraph
        - density: Graph density (ratio of actual to possible edges)
        - component_count: Number of connected components
        - largest_component_size: Size of the largest connected component
        - avg_degree: Average degree of nodes
        - avg_clustering: Average clustering coefficient
        - is_connected: Whether the graph is fully connected
    """
    g = _get_projection(hg, projection, filter_fn)

    node_count = g.number_of_nodes()
    edge_count = g.number_of_edges()

    # Count hyperedges in original hypergraph
    hyperedge_count = sum(1 for e in hg.all() if e.not_atom)

    # Basic stats
    stats = {
        'node_count': node_count,
        'edge_count': edge_count,
        'hyperedge_count': hyperedge_count,
    }

    if node_count == 0:
        stats.update({
            'density': 0.0,
            'component_count': 0,
            'largest_component_size': 0,
            'avg_degree': 0.0,
            'avg_clustering': 0.0,
            'is_connected': True,
        })
        return stats

    # Compute density
    stats['density'] = nx.density(g)

    # Connected components
    components = list(nx.connected_components(g))
    stats['component_count'] = len(components)
    stats['largest_component_size'] = max(len(c) for c in components) if components else 0
    stats['is_connected'] = len(components) == 1

    # Degree statistics
    degrees = [d for _, d in g.degree()]
    stats['avg_degree'] = sum(degrees) / len(degrees) if degrees else 0.0

    # Clustering coefficient
    stats['avg_clustering'] = nx.average_clustering(g)

    return stats


def density(
    hg: Hypergraph,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> float:
    """Compute the density of the hypergraph projection.

    Density is the ratio of existing edges to the maximum possible
    number of edges.

    Args:
        hg: The hypergraph.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        Density value between 0 and 1.
    """
    g = _get_projection(hg, projection, filter_fn)
    return nx.density(g)


def average_clustering(
    hg: Hypergraph,
    weight: Optional[str] = None,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> float:
    """Compute the average clustering coefficient.

    The clustering coefficient measures the degree to which atoms
    cluster together (form triangles).

    Args:
        hg: The hypergraph.
        weight: Edge attribute for weighted clustering. None for unweighted.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        Average clustering coefficient between 0 and 1.
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return 0.0

    return nx.average_clustering(g, weight=weight)


def clustering(
    hg: Hypergraph,
    weight: Optional[str] = None,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> Dict:
    """Compute the clustering coefficient for each atom.

    Args:
        hg: The hypergraph.
        weight: Edge attribute for weighted clustering. None for unweighted.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        Dictionary mapping atoms to their clustering coefficients.
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return {}

    return nx.clustering(g, weight=weight)


def transitivity(
    hg: Hypergraph,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> float:
    """Compute the transitivity (global clustering coefficient).

    Transitivity is the fraction of all possible triangles that exist.
    It's based on the number of triangles in the graph.

    Args:
        hg: The hypergraph.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        Transitivity value between 0 and 1.
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return 0.0

    return nx.transitivity(g)


def degree_histogram(
    hg: Hypergraph,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> list:
    """Compute the degree histogram of the graph.

    Args:
        hg: The hypergraph.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        List where index i gives the number of nodes with degree i.
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return []

    return nx.degree_histogram(g)


def degree_sequence(
    hg: Hypergraph,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> list:
    """Get the degree sequence of the graph.

    Args:
        hg: The hypergraph.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        List of degrees sorted in descending order.
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return []

    degrees = [d for _, d in g.degree()]
    return sorted(degrees, reverse=True)


def average_shortest_path_length(
    hg: Hypergraph,
    weight: Optional[str] = None,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> Optional[float]:
    """Compute the average shortest path length.

    Only computed on the largest connected component if the graph
    is disconnected.

    Args:
        hg: The hypergraph.
        weight: Edge attribute for weighted paths. None for unweighted.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        Average shortest path length, or None if the graph is empty.
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() < 2:
        return None

    # Use largest connected component if disconnected
    if not nx.is_connected(g):
        largest_cc = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_cc)

    if g.number_of_nodes() < 2:
        return None

    return nx.average_shortest_path_length(g, weight=weight)


def number_of_triangles(
    hg: Hypergraph,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> int:
    """Count the total number of triangles in the graph.

    Args:
        hg: The hypergraph.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        Number of triangles.
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return 0

    # nx.triangles returns count for each node (each triangle counted 3 times)
    return sum(nx.triangles(g).values()) // 3


def rich_club_coefficient(
    hg: Hypergraph,
    normalized: bool = True,
    seed: Optional[int] = None,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> Dict:
    """Compute the rich-club coefficient.

    The rich-club coefficient measures the extent to which high-degree
    nodes connect to each other.

    Args:
        hg: The hypergraph.
        normalized: Whether to normalize by random graph expectation.
        seed: Random seed for normalization.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        Dictionary mapping degree to rich-club coefficient.
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return {}

    if normalized:
        return nx.rich_club_coefficient(g, normalized=True, seed=seed)
    else:
        return nx.rich_club_coefficient(g, normalized=False)


def small_world_sigma(
    hg: Hypergraph,
    niter: int = 100,
    nrand: int = 10,
    seed: Optional[int] = None,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> Optional[float]:
    """Compute the small-world coefficient sigma.

    Sigma > 1 indicates small-world properties. Computed as:
    sigma = (C / C_rand) / (L / L_rand)

    where C is clustering coefficient and L is average path length.

    Warning: This can be slow for large graphs.

    Args:
        hg: The hypergraph.
        niter: Approximate number of rewiring per edge for random graphs.
        nrand: Number of random graphs to average over.
        seed: Random seed for reproducibility.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        Small-world coefficient sigma, or None if it cannot be computed.
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() < 4:
        return None

    # Use largest connected component if disconnected
    if not nx.is_connected(g):
        largest_cc = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_cc).copy()

    if g.number_of_nodes() < 4:
        return None

    try:
        return nx.sigma(g, niter=niter, nrand=nrand, seed=seed)
    except (nx.NetworkXError, ZeroDivisionError):
        return None


def small_world_omega(
    hg: Hypergraph,
    niter: int = 5,
    nrand: int = 10,
    seed: Optional[int] = None,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> Optional[float]:
    """Compute the small-world coefficient omega.

    Omega ranges from -1 to 1. Values close to 0 indicate small-world
    properties. Negative values indicate lattice-like, positive indicates
    random-like.

    Warning: This can be slow for large graphs.

    Args:
        hg: The hypergraph.
        niter: Approximate number of rewiring per edge.
        nrand: Number of random graphs to average over.
        seed: Random seed for reproducibility.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        Small-world coefficient omega, or None if it cannot be computed.
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() < 4:
        return None

    # Use largest connected component if disconnected
    if not nx.is_connected(g):
        largest_cc = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_cc).copy()

    if g.number_of_nodes() < 4:
        return None

    try:
        return nx.omega(g, niter=niter, nrand=nrand, seed=seed)
    except (nx.NetworkXError, ZeroDivisionError):
        return None


def assortativity(
    hg: Hypergraph,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> Optional[float]:
    """Compute degree assortativity coefficient.

    Measures the tendency of nodes to connect to nodes of similar degree.
    Positive values indicate assortative mixing (high-degree nodes connect
    to high-degree nodes), negative values indicate disassortative mixing.

    Args:
        hg: The hypergraph.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        Assortativity coefficient between -1 and 1, or None if undefined.
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_edges() == 0:
        return None

    try:
        return nx.degree_assortativity_coefficient(g)
    except (ValueError, ZeroDivisionError):
        return None
