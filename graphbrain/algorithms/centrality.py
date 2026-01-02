"""Centrality algorithms for hypergraphs.

This module provides various centrality measures for hypergraph atoms,
computed via projection to NetworkX graphs.
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
    """Get a NetworkX graph from a hypergraph projection.

    Args:
        hg: The hypergraph.
        projection: Optional custom projection. If None, uses AtomCooccurrenceProjection.
        filter_fn: Optional filter function for the default projection.

    Returns:
        A NetworkX Graph.
    """
    if projection is not None:
        return projection.to_networkx()
    return AtomCooccurrenceProjection(hg, filter_fn).to_networkx()


def pagerank(
    hg: Hypergraph,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1.0e-6,
    weight: str = 'weight',
    store: bool = False,
    attribute: str = 'pagerank',
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> Dict:
    """Compute PageRank centrality for hypergraph atoms.

    PageRank measures the importance of atoms based on the structure of
    their connections, simulating a random walker on the graph.

    Args:
        hg: The hypergraph.
        alpha: Damping parameter (probability of following an edge). Default 0.85.
        max_iter: Maximum number of iterations. Default 100.
        tol: Convergence tolerance. Default 1.0e-6.
        weight: Edge attribute to use as weight. Default 'weight'.
        store: If True, store results as edge attributes. Default False.
        attribute: Attribute name for storing results. Default 'pagerank'.
        projection: Optional custom projection. Uses AtomCooccurrenceProjection if None.
        filter_fn: Optional filter function for edges.

    Returns:
        Dictionary mapping atoms to their PageRank scores.
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return {}

    scores = nx.pagerank(g, alpha=alpha, max_iter=max_iter, tol=tol, weight=weight)

    if store:
        for atom, score in scores.items():
            hg.set_attribute(atom, attribute, score)

    return scores


def betweenness_centrality(
    hg: Hypergraph,
    normalized: bool = True,
    weight: Optional[str] = None,
    endpoints: bool = False,
    store: bool = False,
    attribute: str = 'betweenness',
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> Dict:
    """Compute betweenness centrality for hypergraph atoms.

    Betweenness centrality measures how often an atom lies on the shortest
    path between other pairs of atoms. High betweenness indicates a
    "bridge" role in the network.

    Args:
        hg: The hypergraph.
        normalized: If True, normalize by 2/((n-1)(n-2)) for graphs. Default True.
        weight: Edge attribute to use as weight. None means unweighted.
        endpoints: If True, include endpoints in shortest path count. Default False.
        store: If True, store results as edge attributes. Default False.
        attribute: Attribute name for storing results. Default 'betweenness'.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        Dictionary mapping atoms to their betweenness centrality scores.
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return {}

    scores = nx.betweenness_centrality(
        g, normalized=normalized, weight=weight, endpoints=endpoints
    )

    if store:
        for atom, score in scores.items():
            hg.set_attribute(atom, attribute, score)

    return scores


def closeness_centrality(
    hg: Hypergraph,
    distance: Optional[str] = None,
    wf_improved: bool = True,
    store: bool = False,
    attribute: str = 'closeness',
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> Dict:
    """Compute closeness centrality for hypergraph atoms.

    Closeness centrality measures how close an atom is to all other atoms
    in the network. Higher values indicate more central positions.

    Args:
        hg: The hypergraph.
        distance: Edge attribute to use as distance. None uses hop count.
        wf_improved: Use Wasserman-Faust improved formula. Default True.
        store: If True, store results as edge attributes. Default False.
        attribute: Attribute name for storing results. Default 'closeness'.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        Dictionary mapping atoms to their closeness centrality scores.
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return {}

    scores = nx.closeness_centrality(g, distance=distance, wf_improved=wf_improved)

    if store:
        for atom, score in scores.items():
            hg.set_attribute(atom, attribute, score)

    return scores


def degree_centrality(
    hg: Hypergraph,
    store: bool = False,
    attribute: str = 'degree_centrality',
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> Dict:
    """Compute degree centrality for hypergraph atoms.

    Degree centrality is the fraction of nodes each atom is connected to.
    It's the simplest centrality measure.

    Args:
        hg: The hypergraph.
        store: If True, store results as edge attributes. Default False.
        attribute: Attribute name for storing results. Default 'degree_centrality'.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        Dictionary mapping atoms to their degree centrality scores.
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return {}

    scores = nx.degree_centrality(g)

    if store:
        for atom, score in scores.items():
            hg.set_attribute(atom, attribute, score)

    return scores


def eigenvector_centrality(
    hg: Hypergraph,
    max_iter: int = 100,
    tol: float = 1.0e-6,
    weight: Optional[str] = 'weight',
    store: bool = False,
    attribute: str = 'eigenvector',
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> Dict:
    """Compute eigenvector centrality for hypergraph atoms.

    Eigenvector centrality measures influence by considering both the
    number of connections and the importance of connected nodes.
    Similar to PageRank but without the damping factor.

    Args:
        hg: The hypergraph.
        max_iter: Maximum number of iterations. Default 100.
        tol: Convergence tolerance. Default 1.0e-6.
        weight: Edge attribute to use as weight. Default 'weight'.
        store: If True, store results as edge attributes. Default False.
        attribute: Attribute name for storing results. Default 'eigenvector'.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        Dictionary mapping atoms to their eigenvector centrality scores.
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return {}

    try:
        scores = nx.eigenvector_centrality(
            g, max_iter=max_iter, tol=tol, weight=weight
        )
    except nx.PowerIterationFailedConvergence:
        # Fall back to numpy-based computation if power iteration fails
        scores = nx.eigenvector_centrality_numpy(g, weight=weight)

    if store:
        for atom, score in scores.items():
            hg.set_attribute(atom, attribute, score)

    return scores


def katz_centrality(
    hg: Hypergraph,
    alpha: float = 0.1,
    beta: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1.0e-6,
    weight: Optional[str] = 'weight',
    store: bool = False,
    attribute: str = 'katz',
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> Dict:
    """Compute Katz centrality for hypergraph atoms.

    Katz centrality computes influence by considering the total number of
    walks between nodes, with longer walks weighted less than shorter ones.

    Args:
        hg: The hypergraph.
        alpha: Attenuation factor. Must be less than 1/max(eigenvalue). Default 0.1.
        beta: Weight attributed to immediate neighbors. Default 1.0.
        max_iter: Maximum number of iterations. Default 1000.
        tol: Convergence tolerance. Default 1.0e-6.
        weight: Edge attribute to use as weight. Default 'weight'.
        store: If True, store results as edge attributes. Default False.
        attribute: Attribute name for storing results. Default 'katz'.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        Dictionary mapping atoms to their Katz centrality scores.
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return {}

    try:
        scores = nx.katz_centrality(
            g, alpha=alpha, beta=beta, max_iter=max_iter, tol=tol, weight=weight
        )
    except nx.PowerIterationFailedConvergence:
        # Fall back to numpy-based computation
        scores = nx.katz_centrality_numpy(g, alpha=alpha, beta=beta, weight=weight)

    if store:
        for atom, score in scores.items():
            hg.set_attribute(atom, attribute, score)

    return scores


def top_centrality(
    hg: Hypergraph,
    algorithm: str = 'pagerank',
    top_k: int = 10,
    **kwargs
) -> list:
    """Get top-k atoms by centrality.

    Convenience function to compute centrality and return the top results.

    Args:
        hg: The hypergraph.
        algorithm: Centrality algorithm to use. One of:
                   'pagerank', 'betweenness', 'closeness', 'degree',
                   'eigenvector', 'katz'. Default 'pagerank'.
        top_k: Number of top results to return. Default 10.
        **kwargs: Additional arguments passed to the centrality function.

    Returns:
        List of (atom, score) tuples sorted by score descending.
    """
    algorithms = {
        'pagerank': pagerank,
        'betweenness': betweenness_centrality,
        'closeness': closeness_centrality,
        'degree': degree_centrality,
        'eigenvector': eigenvector_centrality,
        'katz': katz_centrality,
    }

    if algorithm not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm}. "
                        f"Choose from: {list(algorithms.keys())}")

    scores = algorithms[algorithm](hg, **kwargs)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_scores[:top_k]
