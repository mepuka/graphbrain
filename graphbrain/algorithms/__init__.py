"""Graph algorithms for hypergraphs.

This module provides graph algorithms that operate on hypergraphs by
projecting them to NetworkX graphs. It includes:

- **Projections**: Convert hypergraphs to NetworkX graphs
- **Centrality**: PageRank, betweenness, closeness, degree, eigenvector
- **Community Detection**: Louvain, label propagation, connected components
- **Path Finding**: Shortest paths, reachability, diameter
- **Statistics**: Density, clustering, transitivity, small-world metrics
- **Subgraphs**: Ego networks, induced subgraphs, k-core decomposition

Example usage:

    from graphbrain import hgraph
    from graphbrain.algorithms import pagerank, louvain_communities, shortest_path

    # Open a hypergraph
    hg = hgraph('my_knowledge.db')

    # Compute PageRank centrality
    scores = pagerank(hg)
    top_atoms = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]

    # Detect communities
    communities = louvain_communities(hg)
    print(f"Found {len(communities)} communities")

    # Find shortest path between atoms
    path = shortest_path(hg, 'alice/C', 'bob/C')
    if path:
        print(f"Path: {' -> '.join(str(a) for a in path)}")
"""

# Projections
from graphbrain.algorithms.projections import (
    HypergraphProjection,
    AtomCooccurrenceProjection,
    BipartiteProjection,
    PredicateProjection,
    ConnectorProjection,
)

# Centrality algorithms
from graphbrain.algorithms.centrality import (
    pagerank,
    betweenness_centrality,
    closeness_centrality,
    degree_centrality,
    eigenvector_centrality,
    katz_centrality,
    top_centrality,
)

# Community detection
from graphbrain.algorithms.community import (
    connected_components,
    louvain_communities,
    label_propagation,
    greedy_modularity_communities,
    girvan_newman,
    k_clique_communities,
    modularity,
)

# Path finding
from graphbrain.algorithms.paths import (
    shortest_path,
    shortest_path_length,
    all_shortest_paths,
    reachable,
    has_path,
    all_pairs_shortest_path_length,
    diameter,
    radius,
    center,
    periphery,
    eccentricity,
)

# Statistics
from graphbrain.algorithms.statistics import (
    summary,
    density,
    average_clustering,
    clustering,
    transitivity,
    degree_histogram,
    degree_sequence,
    average_shortest_path_length,
    number_of_triangles,
    rich_club_coefficient,
    small_world_sigma,
    small_world_omega,
    assortativity,
)

# Subgraph extraction
from graphbrain.algorithms.subgraphs import (
    ego_graph,
    ego_atoms,
    induced_subgraph,
    partial_induced_subgraph,
    k_core,
    k_shell,
    core_number,
    k_truss,
    neighbors,
    common_neighbors,
    cliques_containing,
    largest_clique,
    bridges,
)

__all__ = [
    # Projections
    'HypergraphProjection',
    'AtomCooccurrenceProjection',
    'BipartiteProjection',
    'PredicateProjection',
    'ConnectorProjection',
    # Centrality
    'pagerank',
    'betweenness_centrality',
    'closeness_centrality',
    'degree_centrality',
    'eigenvector_centrality',
    'katz_centrality',
    'top_centrality',
    # Community
    'connected_components',
    'louvain_communities',
    'label_propagation',
    'greedy_modularity_communities',
    'girvan_newman',
    'k_clique_communities',
    'modularity',
    # Paths
    'shortest_path',
    'shortest_path_length',
    'all_shortest_paths',
    'reachable',
    'has_path',
    'all_pairs_shortest_path_length',
    'diameter',
    'radius',
    'center',
    'periphery',
    'eccentricity',
    # Statistics
    'summary',
    'density',
    'average_clustering',
    'clustering',
    'transitivity',
    'degree_histogram',
    'degree_sequence',
    'average_shortest_path_length',
    'number_of_triangles',
    'rich_club_coefficient',
    'small_world_sigma',
    'small_world_omega',
    'assortativity',
    # Subgraphs
    'ego_graph',
    'ego_atoms',
    'induced_subgraph',
    'partial_induced_subgraph',
    'k_core',
    'k_shell',
    'core_number',
    'k_truss',
    'neighbors',
    'common_neighbors',
    'cliques_containing',
    'largest_clique',
    'bridges',
]
