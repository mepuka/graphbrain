"""Subgraph extraction algorithms for hypergraphs.

This module provides functions for extracting subgraphs from hypergraphs,
including ego networks, induced subgraphs, and k-core decomposition.
"""

from typing import Callable, Dict, Generator, Optional, Set, Union

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


def ego_graph(
    hg: Hypergraph,
    center: Union[str, Hyperedge],
    radius: int = 1,
    include_center: bool = True
) -> Set:
    """Extract an ego network around a center atom.

    The ego network includes all atoms and hyperedges within a certain
    number of hops from the center atom. This works directly on the
    hypergraph structure without projection.

    Args:
        hg: The hypergraph.
        center: The center atom for the ego network.
        radius: Maximum number of hops from center. Default 1.
        include_center: Whether to include the center in the result. Default True.

    Returns:
        Set of hyperedges in the ego network.
    """
    center = _to_hedge(center)

    visited_edges = set()
    visited_atoms = set()
    frontier = {center}

    for _ in range(radius):
        next_frontier = set()

        for atom in frontier:
            visited_atoms.add(atom)

            # Get all hyperedges containing this atom
            for edge in hg.star(atom):
                if edge not in visited_edges:
                    visited_edges.add(edge)

                    # Add all atoms from this edge to next frontier
                    for contained_atom in edge.atoms():
                        if contained_atom not in visited_atoms:
                            next_frontier.add(contained_atom)

        frontier = next_frontier
        if not frontier:
            break

    if include_center and center not in visited_atoms:
        visited_atoms.add(center)

    return visited_edges


def ego_atoms(
    hg: Hypergraph,
    center: Union[str, Hyperedge],
    radius: int = 1,
    include_center: bool = True
) -> Set:
    """Get all atoms within an ego network.

    Similar to ego_graph but returns atoms instead of hyperedges.

    Args:
        hg: The hypergraph.
        center: The center atom for the ego network.
        radius: Maximum number of hops from center. Default 1.
        include_center: Whether to include the center. Default True.

    Returns:
        Set of atoms in the ego network.
    """
    center = _to_hedge(center)

    visited_atoms = {center} if include_center else set()
    current_frontier = {center}

    for _ in range(radius):
        next_frontier = set()

        for atom in current_frontier:
            for edge in hg.star(atom):
                for contained_atom in edge.atoms():
                    if contained_atom not in visited_atoms:
                        next_frontier.add(contained_atom)
                        visited_atoms.add(contained_atom)

        current_frontier = next_frontier

        if not current_frontier:
            break

    return visited_atoms


def induced_subgraph(
    hg: Hypergraph,
    atoms: Set[Union[str, Hyperedge]]
) -> Generator:
    """Generate hyperedges from the induced subgraph.

    The induced subgraph contains all hyperedges where ALL atoms in the
    hyperedge are in the specified set.

    Args:
        hg: The hypergraph.
        atoms: Set of atoms to include.

    Yields:
        Hyperedges where all atoms are in the specified set.
    """
    # Convert strings to hedges if necessary
    atom_set = {_to_hedge(a) for a in atoms}

    for edge in hg.all():
        if edge.not_atom:
            edge_atoms = edge.atoms()
            if edge_atoms.issubset(atom_set):
                yield edge


def partial_induced_subgraph(
    hg: Hypergraph,
    atoms: Set[Union[str, Hyperedge]],
    min_overlap: float = 0.5
) -> Generator:
    """Generate hyperedges with partial overlap with atom set.

    More flexible than induced_subgraph - includes hyperedges where
    at least a certain fraction of atoms are in the specified set.

    Args:
        hg: The hypergraph.
        atoms: Set of atoms to check against.
        min_overlap: Minimum fraction of edge atoms that must be in set.
                     Default 0.5 (at least half).

    Yields:
        Hyperedges meeting the overlap criterion.
    """
    atom_set = {_to_hedge(a) for a in atoms}

    for edge in hg.all():
        if edge.not_atom:
            edge_atoms = edge.atoms()
            if len(edge_atoms) > 0:
                overlap = len(edge_atoms.intersection(atom_set)) / len(edge_atoms)
                if overlap >= min_overlap:
                    yield edge


def k_core(
    hg: Hypergraph,
    k: int,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> Set:
    """Find the k-core of the hypergraph.

    The k-core is the maximal subgraph where every node has degree at
    least k.

    Args:
        hg: The hypergraph.
        k: Minimum degree for nodes in the k-core.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        Set of atoms in the k-core.
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return set()

    core_numbers = nx.core_number(g)
    return {node for node, core in core_numbers.items() if core >= k}


def k_shell(
    hg: Hypergraph,
    k: int,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> Set:
    """Find the k-shell of the hypergraph.

    The k-shell is the subgraph of nodes in the k-core but not in the
    (k+1)-core.

    Args:
        hg: The hypergraph.
        k: Shell number.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        Set of atoms in the k-shell.
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return set()

    core_numbers = nx.core_number(g)
    return {node for node, core in core_numbers.items() if core == k}


def core_number(
    hg: Hypergraph,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> Dict:
    """Compute core number for each atom.

    The core number of an atom is the largest k for which it is in
    the k-core.

    Args:
        hg: The hypergraph.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        Dictionary mapping atoms to their core numbers.
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return {}

    return nx.core_number(g)


def k_truss(
    hg: Hypergraph,
    k: int,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> Set:
    """Find the k-truss of the hypergraph.

    The k-truss is the maximal subgraph where every edge is part of at
    least (k-2) triangles.

    Args:
        hg: The hypergraph.
        k: Truss number.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        Set of atoms in the k-truss.
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return set()

    try:
        truss_g = nx.k_truss(g, k)
        return set(truss_g.nodes())
    except nx.NetworkXError:
        return set()


def neighbors(
    hg: Hypergraph,
    atom: Union[str, Hyperedge],
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> Set:
    """Get immediate neighbors of an atom in the projected graph.

    Args:
        hg: The hypergraph.
        atom: The atom to find neighbors for.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        Set of neighbor atoms.
    """
    g = _get_projection(hg, projection, filter_fn)
    atom = _to_hedge(atom)

    if atom not in g:
        return set()

    return set(g.neighbors(atom))


def common_neighbors(
    hg: Hypergraph,
    atom1: Union[str, Hyperedge],
    atom2: Union[str, Hyperedge],
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> Set:
    """Find common neighbors of two atoms.

    Args:
        hg: The hypergraph.
        atom1: First atom.
        atom2: Second atom.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        Set of atoms that are neighbors of both atom1 and atom2.
    """
    g = _get_projection(hg, projection, filter_fn)
    atom1 = _to_hedge(atom1)
    atom2 = _to_hedge(atom2)

    if atom1 not in g or atom2 not in g:
        return set()

    return set(nx.common_neighbors(g, atom1, atom2))


def cliques_containing(
    hg: Hypergraph,
    atom: Union[str, Hyperedge],
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> list:
    """Find all maximal cliques containing a specific atom.

    A clique is a complete subgraph (every pair of nodes is connected).

    Args:
        hg: The hypergraph.
        atom: The atom to find cliques for.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        List of cliques (each clique is a frozenset of atoms).
    """
    g = _get_projection(hg, projection, filter_fn)
    atom = _to_hedge(atom)

    if atom not in g:
        return []

    cliques = []
    for clique in nx.find_cliques(g):
        if atom in clique:
            cliques.append(frozenset(clique))

    return cliques


def largest_clique(
    hg: Hypergraph,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> Set:
    """Find the largest clique in the graph.

    Note: Finding the maximum clique is NP-hard, so this may be slow
    for large graphs.

    Args:
        hg: The hypergraph.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        Set of atoms in the largest clique.
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return set()

    # Find all maximal cliques
    cliques = list(nx.find_cliques(g))
    if not cliques:
        return set()

    # Return the largest one
    return set(max(cliques, key=len))


def bridges(
    hg: Hypergraph,
    projection: Optional[HypergraphProjection] = None,
    filter_fn: Optional[Callable] = None
) -> list:
    """Find bridge atoms (articulation points).

    A bridge atom is one whose removal would disconnect the graph.
    These are critical nodes for graph connectivity.

    Args:
        hg: The hypergraph.
        projection: Optional custom projection.
        filter_fn: Optional filter function for edges.

    Returns:
        List of bridge atoms.
    """
    g = _get_projection(hg, projection, filter_fn)

    if g.number_of_nodes() == 0:
        return []

    return list(nx.articulation_points(g))
