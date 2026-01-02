"""Hypergraph to NetworkX projection utilities.

This module provides various methods to project a hypergraph onto a standard
graph representation using NetworkX, enabling the use of classical graph
algorithms on hypergraph data.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional

import networkx as nx

from graphbrain.hypergraph import Hypergraph


class HypergraphProjection(ABC):
    """Abstract base class for hypergraph projections.

    A projection converts a hypergraph into a NetworkX graph by defining
    specific rules for how atoms and hyperedges map to nodes and edges.

    Args:
        hg: The hypergraph to project.
        filter_fn: Optional function to filter which hyperedges to include.
                   Takes a hyperedge and returns True if it should be included.
    """

    def __init__(self, hg: Hypergraph, filter_fn: Optional[Callable] = None):
        self.hg = hg
        self.filter_fn = filter_fn

    def _should_include(self, edge) -> bool:
        """Check if an edge should be included in the projection."""
        if self.filter_fn is None:
            return True
        return self.filter_fn(edge)

    @abstractmethod
    def to_networkx(self) -> nx.Graph:
        """Convert the hypergraph to a NetworkX graph.

        Returns:
            A NetworkX Graph instance representing the projection.
        """
        raise NotImplementedError()


class AtomCooccurrenceProjection(HypergraphProjection):
    """Project hypergraph based on atom co-occurrence.

    Two atoms are connected if they appear together in the same hyperedge.
    Edge weights represent the number of hyperedges where both atoms co-occur.

    This is the most common projection and is useful for understanding
    relationships between entities in the knowledge base.
    """

    def to_networkx(self) -> nx.Graph:
        """Create a graph where atoms are nodes and edges represent co-occurrence.

        Returns:
            A weighted NetworkX Graph where:
            - Nodes are atoms (as Hyperedge objects)
            - Edges connect atoms that appear in the same hyperedge
            - Edge weights count co-occurrence frequency
        """
        g = nx.Graph()

        for edge in self.hg.all():
            # Skip atoms - we only want non-atomic edges (hyperedges)
            if edge.atom:
                continue

            # Check if this edge should be included
            if not self._should_include(edge):
                continue

            # Get all atoms in this hyperedge
            atoms = list(edge.atoms())

            # Create edges between all pairs of atoms
            for i, a1 in enumerate(atoms):
                # Ensure the atom is added as a node
                if not g.has_node(a1):
                    g.add_node(a1)

                for a2 in atoms[i+1:]:
                    if not g.has_node(a2):
                        g.add_node(a2)

                    if g.has_edge(a1, a2):
                        g[a1][a2]['weight'] += 1
                    else:
                        g.add_edge(a1, a2, weight=1)

        return g


class BipartiteProjection(HypergraphProjection):
    """Project hypergraph as a bipartite graph of atoms and hyperedges.

    Creates a bipartite graph where one set contains atoms and the other
    contains non-atomic hyperedges. An edge connects an atom to a hyperedge
    if the atom appears in that hyperedge.

    This projection preserves more structure than co-occurrence but is
    less intuitive for some analyses.
    """

    def to_networkx(self) -> nx.Graph:
        """Create a bipartite graph of atoms and hyperedges.

        Returns:
            A NetworkX Graph where:
            - Nodes are either atoms (bipartite=0) or hyperedges (bipartite=1)
            - Edges connect atoms to the hyperedges they appear in
        """
        g = nx.Graph()

        for edge in self.hg.all():
            # Skip atoms for now - we add them when processing hyperedges
            if edge.atom:
                continue

            # Check if this edge should be included
            if not self._should_include(edge):
                continue

            # Add the hyperedge as a node
            edge_id = edge.to_str()
            g.add_node(edge_id, bipartite=1, edge=edge)

            # Connect all atoms in this hyperedge to it
            for atom in edge.atoms():
                if not g.has_node(atom):
                    g.add_node(atom, bipartite=0)
                g.add_edge(atom, edge_id)

        return g


class PredicateProjection(HypergraphProjection):
    """Project hypergraph based on predicate-argument structure.

    Creates a directed graph where edges represent predicate relationships.
    For a hyperedge like (says/Pd mary/C hello/C), this creates directed
    edges from 'mary' to 'hello' labeled with 'says'.

    This is useful for analyzing causal or directional relationships.
    """

    def to_networkx(self) -> nx.DiGraph:
        """Create a directed graph based on predicate structure.

        Returns:
            A NetworkX DiGraph where:
            - Nodes are concept atoms (type C)
            - Directed edges represent predicate relationships
            - Edge attributes include the predicate and full hyperedge
        """
        g = nx.DiGraph()

        for edge in self.hg.all():
            if edge.atom:
                continue

            if not self._should_include(edge):
                continue

            # Only process relation edges (those with predicate connectors)
            if len(edge) < 2:
                continue

            connector = edge[0]
            if connector.atom:
                conn_type = connector.mtype() if hasattr(connector, 'mtype') else None
            else:
                conn_type = connector.connector_mtype() if hasattr(connector, 'connector_mtype') else None

            # Check if this is a relation (predicate-based)
            if conn_type != 'P':
                continue

            # Get the arguments (concepts after the predicate)
            args = [arg for arg in edge[1:] if not arg.atom or arg.mtype() == 'C']

            if len(args) >= 2:
                # Create edges from first argument to all subsequent arguments
                source = args[0]
                for target in args[1:]:
                    # Use string representation for node identification
                    src_str = source.to_str()
                    tgt_str = target.to_str()

                    if not g.has_node(src_str):
                        g.add_node(src_str, edge=source)
                    if not g.has_node(tgt_str):
                        g.add_node(tgt_str, edge=target)

                    # Add or update edge
                    if g.has_edge(src_str, tgt_str):
                        g[src_str][tgt_str]['weight'] += 1
                        g[src_str][tgt_str]['predicates'].append(connector)
                    else:
                        g.add_edge(src_str, tgt_str, weight=1,
                                  predicates=[connector], hyperedge=edge)

        return g


class ConnectorProjection(HypergraphProjection):
    """Project hypergraph by connector type.

    Creates a graph where atoms are connected based on specific connector
    types. Useful for analyzing specific relationship types (e.g., only
    builder relationships, only predicate relationships).

    Args:
        hg: The hypergraph to project.
        connector_types: Set of connector main types to include (e.g., {'P', 'B'}).
        filter_fn: Optional additional filter function.
    """

    def __init__(self, hg: Hypergraph, connector_types: set = None,
                 filter_fn: Optional[Callable] = None):
        super().__init__(hg, filter_fn)
        self.connector_types = connector_types or {'P', 'B', 'M', 'T', 'J'}

    def to_networkx(self) -> nx.Graph:
        """Create a graph filtered by connector type.

        Returns:
            A weighted NetworkX Graph similar to AtomCooccurrenceProjection
            but only including hyperedges with specified connector types.
        """
        g = nx.Graph()

        for edge in self.hg.all():
            if edge.atom:
                continue

            if not self._should_include(edge):
                continue

            # Check connector type
            if len(edge) < 1:
                continue

            connector = edge[0]
            if connector.atom:
                conn_type = connector.mtype() if hasattr(connector, 'mtype') else None
            else:
                conn_type = connector.connector_mtype() if hasattr(connector, 'connector_mtype') else None

            if conn_type not in self.connector_types:
                continue

            # Same logic as AtomCooccurrenceProjection
            atoms = list(edge.atoms())
            for i, a1 in enumerate(atoms):
                if not g.has_node(a1):
                    g.add_node(a1)

                for a2 in atoms[i+1:]:
                    if not g.has_node(a2):
                        g.add_node(a2)

                    if g.has_edge(a1, a2):
                        g[a1][a2]['weight'] += 1
                    else:
                        g.add_edge(a1, a2, weight=1, connector_type=conn_type)

        return g
