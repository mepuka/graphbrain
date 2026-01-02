"""Community detection algorithms for hypergraphs.

This module provides algorithms for identifying communities or clusters
of closely related atoms in a hypergraph.
"""

from typing import Callable, List, Optional, Set

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


def connected_components(
    hg: Hypergraph,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> List[Set]:
    """Find connected components in the hypergraph.

    A connected component is a maximal set of atoms where every pair
    is connected by some path through the hypergraph.

    Args:
        hg: The hypergraph.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        List of sets, each containing the atoms in a connected component.
        Sorted by size (largest first).
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return []

    components = [set(c) for c in nx.connected_components(g)]
    # Sort by size, largest first
    components.sort(key=len, reverse=True)

    return components


def louvain_communities(
    hg: Hypergraph,
    resolution: float = 1.0,
    threshold: float = 0.0000001,
    seed: Optional[int] = None,
    weight: str = 'weight',
    store: bool = False,
    attribute: str = 'community',
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> List[Set]:
    """Detect communities using the Louvain algorithm.

    The Louvain method is a hierarchical community detection algorithm
    that optimizes modularity. It's efficient and produces high-quality
    partitions.

    Args:
        hg: The hypergraph.
        resolution: Resolution parameter for modularity. Higher values
                    result in more communities. Default 1.0.
        threshold: Modularity gain threshold for merging. Default 0.0000001.
        seed: Random seed for reproducibility. Default None.
        weight: Edge attribute to use as weight. Default 'weight'.
        store: If True, store community labels as edge attributes. Default False.
        attribute: Attribute name for storing results. Default 'community'.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        List of sets, each containing atoms in the same community.
        Sorted by size (largest first).
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return []

    communities = list(nx.community.louvain_communities(
        g, resolution=resolution, threshold=threshold, seed=seed, weight=weight
    ))

    # Sort by size, largest first
    communities.sort(key=len, reverse=True)

    if store:
        for i, community in enumerate(communities):
            for atom in community:
                hg.set_attribute(atom, attribute, i)

    return communities


def label_propagation(
    hg: Hypergraph,
    store: bool = False,
    attribute: str = 'community',
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> List[Set]:
    """Detect communities using label propagation.

    Label propagation is a fast, near-linear time algorithm that
    assigns labels based on the majority labels of neighbors.
    Results may vary between runs due to random tie-breaking.

    Args:
        hg: The hypergraph.
        store: If True, store community labels as edge attributes. Default False.
        attribute: Attribute name for storing results. Default 'community'.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        List of sets, each containing atoms in the same community.
        Sorted by size (largest first).
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return []

    communities = list(nx.community.label_propagation_communities(g))

    # Sort by size, largest first
    communities.sort(key=len, reverse=True)

    if store:
        for i, community in enumerate(communities):
            for atom in community:
                hg.set_attribute(atom, attribute, i)

    return communities


def greedy_modularity_communities(
    hg: Hypergraph,
    weight: Optional[str] = 'weight',
    resolution: float = 1.0,
    cutoff: int = 1,
    best_n: Optional[int] = None,
    store: bool = False,
    attribute: str = 'community',
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> List[Set]:
    """Detect communities using greedy modularity optimization.

    This algorithm uses a greedy approach to maximize modularity
    by iteratively merging communities.

    Args:
        hg: The hypergraph.
        weight: Edge attribute for weights. Default 'weight'.
        resolution: Resolution parameter. Default 1.0.
        cutoff: Lower bound on number of communities. Default 1.
        best_n: Upper bound on number of communities. None for unlimited.
        store: If True, store community labels as edge attributes. Default False.
        attribute: Attribute name for storing results. Default 'community'.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        List of sets, each containing atoms in the same community.
        Sorted by size (largest first).
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return []

    communities = list(nx.community.greedy_modularity_communities(
        g, weight=weight, resolution=resolution, cutoff=cutoff, best_n=best_n
    ))

    # Sort by size, largest first
    communities.sort(key=len, reverse=True)

    if store:
        for i, community in enumerate(communities):
            for atom in community:
                hg.set_attribute(atom, attribute, i)

    return communities


def girvan_newman(
    hg: Hypergraph,
    num_communities: Optional[int] = None,
    store: bool = False,
    attribute: str = 'community',
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> List[Set]:
    """Detect communities using the Girvan-Newman algorithm.

    This algorithm progressively removes edges with highest betweenness
    to reveal community structure. It's more expensive than other methods
    but can reveal hierarchical structure.

    Args:
        hg: The hypergraph.
        num_communities: Target number of communities. If None, returns the
                        partition with highest modularity.
        store: If True, store community labels as edge attributes. Default False.
        attribute: Attribute name for storing results. Default 'community'.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        List of sets, each containing atoms in the same community.
        Sorted by size (largest first).
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return []

    # Get the generator
    comp_gen = nx.community.girvan_newman(g)

    if num_communities is not None:
        # Iterate until we have the desired number of communities
        communities = None
        for partition in comp_gen:
            if len(partition) >= num_communities:
                communities = [set(c) for c in partition]
                break
        if communities is None:
            # If we didn't reach the target, use the last partition
            communities = [set(c) for c in partition]
    else:
        # Find partition with highest modularity
        best_modularity = -1
        best_communities = None

        for partition in comp_gen:
            # Convert to format expected by modularity function
            partition_list = [frozenset(c) for c in partition]
            mod = nx.community.modularity(g, partition_list)

            if mod > best_modularity:
                best_modularity = mod
                best_communities = [set(c) for c in partition]

            # Early stopping if modularity starts decreasing significantly
            if best_communities is not None and mod < best_modularity - 0.1:
                break

        communities = best_communities or []

    # Sort by size, largest first
    communities.sort(key=len, reverse=True)

    if store:
        for i, community in enumerate(communities):
            for atom in community:
                hg.set_attribute(atom, attribute, i)

    return communities


def k_clique_communities(
    hg: Hypergraph,
    k: int = 3,
    store: bool = False,
    attribute: str = 'community',
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> List[Set]:
    """Find k-clique communities in the hypergraph.

    A k-clique community is a maximal set of k-cliques that can be
    reached from each other through adjacent k-cliques (k-cliques
    that share k-1 nodes).

    Args:
        hg: The hypergraph.
        k: Size of cliques to use. Default 3.
        store: If True, store community labels as edge attributes. Default False.
        attribute: Attribute name for storing results. Default 'community'.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        List of frozensets, each containing atoms in the same community.
        Sorted by size (largest first).
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return []

    try:
        communities = list(nx.community.k_clique_communities(g, k))
    except nx.NetworkXError:
        # No k-cliques found
        return []

    # Convert to sets and sort by size
    communities = [set(c) for c in communities]
    communities.sort(key=len, reverse=True)

    if store:
        for i, community in enumerate(communities):
            for atom in community:
                # Note: atoms may be in multiple overlapping communities
                existing = hg.get_str_attribute(atom, attribute)
                if existing:
                    hg.set_attribute(atom, attribute, f"{existing},{i}")
                else:
                    hg.set_attribute(atom, attribute, str(i))

    return communities


def modularity(
    hg: Hypergraph,
    communities: List[Set],
    weight: str = 'weight',
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> float:
    """Compute the modularity of a community partition.

    Modularity measures the quality of a partition, with higher values
    indicating stronger community structure.

    Args:
        hg: The hypergraph.
        communities: List of sets representing the partition.
        weight: Edge attribute for weights. Default 'weight'.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        Modularity score (typically between -0.5 and 1.0).
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return 0.0

    # Convert to frozensets as required by networkx
    partition = [frozenset(c) for c in communities]

    return nx.community.modularity(g, partition, weight=weight)
