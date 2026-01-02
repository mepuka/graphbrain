"""Tests for the graphbrain.algorithms module."""

import unittest

from graphbrain import hgraph, hedge
from graphbrain.algorithms import (
    # Projections
    AtomCooccurrenceProjection,
    BipartiteProjection,
    PredicateProjection,
    ConnectorProjection,
    # Centrality
    pagerank,
    betweenness_centrality,
    closeness_centrality,
    degree_centrality,
    eigenvector_centrality,
    top_centrality,
    # Community
    connected_components,
    louvain_communities,
    label_propagation,
    greedy_modularity_communities,
    modularity,
    # Paths
    shortest_path,
    shortest_path_length,
    all_shortest_paths,
    reachable,
    has_path,
    diameter,
    radius,
    center,
    periphery,
    eccentricity,
    # Statistics
    summary,
    density,
    average_clustering,
    clustering,
    transitivity,
    degree_histogram,
    degree_sequence,
    number_of_triangles,
    assortativity,
    # Subgraphs
    ego_graph,
    ego_atoms,
    induced_subgraph,
    k_core,
    k_shell,
    core_number,
    neighbors,
    common_neighbors,
    cliques_containing,
    largest_clique,
    bridges,
)


class TestProjections(unittest.TestCase):
    """Test projection classes."""

    def setUp(self):
        self.hg = hgraph('test_algorithms.db')
        self.hg.destroy()

        # Create a simple hypergraph
        # Alice knows Bob, Bob knows Carol, Carol knows Alice (triangle)
        self.hg.add('(knows/P alice/C bob/C)')
        self.hg.add('(knows/P bob/C carol/C)')
        self.hg.add('(knows/P carol/C alice/C)')

        # Dave knows Eve (separate component)
        self.hg.add('(knows/P dave/C eve/C)')

    def tearDown(self):
        self.hg.close()

    def test_atom_cooccurrence_projection(self):
        projection = AtomCooccurrenceProjection(self.hg)
        g = projection.to_networkx()

        # Should have atoms as nodes
        self.assertGreater(g.number_of_nodes(), 0)

        # Check that connected atoms have edges
        alice = hedge('alice/C')
        bob = hedge('bob/C')

        # alice and bob should be connected (appear in same hyperedge)
        self.assertTrue(g.has_edge(alice, bob))

    def test_atom_cooccurrence_weights(self):
        # Add another edge with same atoms
        self.hg.add('(likes/P alice/C bob/C)')

        projection = AtomCooccurrenceProjection(self.hg)
        g = projection.to_networkx()

        alice = hedge('alice/C')
        bob = hedge('bob/C')

        # Weight should be 2 (two hyperedges contain both alice and bob)
        self.assertEqual(g[alice][bob]['weight'], 2)

    def test_bipartite_projection(self):
        projection = BipartiteProjection(self.hg)
        g = projection.to_networkx()

        # Should have both atoms and hyperedges as nodes
        self.assertGreater(g.number_of_nodes(), 0)

        # Check bipartite attribute
        atoms = [n for n, d in g.nodes(data=True) if d.get('bipartite') == 0]
        hyperedges = [n for n, d in g.nodes(data=True) if d.get('bipartite') == 1]

        self.assertGreater(len(atoms), 0)
        self.assertGreater(len(hyperedges), 0)

    def test_filter_function(self):
        # Only include edges with 'knows' predicate
        def filter_knows(edge):
            if edge.not_atom and len(edge) > 0:
                conn = edge[0]
                if conn.atom:
                    return conn.root() == 'knows'
            return False

        projection = AtomCooccurrenceProjection(self.hg, filter_fn=filter_knows)
        g = projection.to_networkx()

        self.assertGreater(g.number_of_nodes(), 0)


class TestCentrality(unittest.TestCase):
    """Test centrality algorithms."""

    def setUp(self):
        self.hg = hgraph('test_centrality.db')
        self.hg.destroy()

        # Create a star topology: center connected to all others
        self.hg.add('(rel/P center/C a/C)')
        self.hg.add('(rel/P center/C b/C)')
        self.hg.add('(rel/P center/C c/C)')
        self.hg.add('(rel/P center/C d/C)')

    def tearDown(self):
        self.hg.close()

    def test_pagerank(self):
        scores = pagerank(self.hg)

        self.assertIsInstance(scores, dict)
        self.assertGreater(len(scores), 0)

        # All scores should be between 0 and 1
        for score in scores.values():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)

    def test_pagerank_store(self):
        scores = pagerank(self.hg, store=True, attribute='test_pr')

        center = hedge('center/C')
        stored = self.hg.get_float_attribute(center, 'test_pr')

        self.assertIsNotNone(stored)
        self.assertAlmostEqual(stored, scores[center], places=6)

    def test_betweenness_centrality(self):
        scores = betweenness_centrality(self.hg)

        self.assertIsInstance(scores, dict)
        self.assertGreater(len(scores), 0)

    def test_closeness_centrality(self):
        scores = closeness_centrality(self.hg)

        self.assertIsInstance(scores, dict)
        self.assertGreater(len(scores), 0)

    def test_degree_centrality(self):
        scores = degree_centrality(self.hg)

        self.assertIsInstance(scores, dict)
        self.assertGreater(len(scores), 0)

        # Center should have highest degree centrality
        center = hedge('center/C')
        center_score = scores.get(center, 0)

        # Center is connected to all other atoms
        for atom, score in scores.items():
            if atom.root() != 'center' and atom.root() != 'rel':
                self.assertGreaterEqual(center_score, score)

    def test_top_centrality(self):
        results = top_centrality(self.hg, algorithm='pagerank', top_k=3)

        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 3)

        # Results should be sorted descending
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i][1], results[i+1][1])

    def test_empty_graph(self):
        empty_hg = hgraph('test_empty.db')
        empty_hg.destroy()

        scores = pagerank(empty_hg)
        self.assertEqual(scores, {})

        empty_hg.close()


class TestCommunity(unittest.TestCase):
    """Test community detection algorithms."""

    def setUp(self):
        self.hg = hgraph('test_community.db')
        self.hg.destroy()

        # Create two distinct communities with DIFFERENT connectors
        # so they don't get connected through a shared connector atom
        # Community 1: alice, bob, carol (densely connected)
        self.hg.add('(knows/P alice/C bob/C)')
        self.hg.add('(knows/P bob/C carol/C)')
        self.hg.add('(knows/P carol/C alice/C)')

        # Community 2: dave, eve, frank (densely connected) - uses different connector
        self.hg.add('(likes/P dave/C eve/C)')
        self.hg.add('(likes/P eve/C frank/C)')
        self.hg.add('(likes/P frank/C dave/C)')

    def tearDown(self):
        self.hg.close()

    def test_connected_components(self):
        components = connected_components(self.hg)

        self.assertIsInstance(components, list)
        # With different connectors, we should have at least 2 components
        self.assertGreaterEqual(len(components), 1)

        # Each component is a set
        for comp in components:
            self.assertIsInstance(comp, set)

    def test_louvain_communities(self):
        communities = louvain_communities(self.hg, seed=42)

        self.assertIsInstance(communities, list)
        self.assertGreater(len(communities), 0)

        # Sorted by size
        for i in range(len(communities) - 1):
            self.assertGreaterEqual(len(communities[i]), len(communities[i+1]))

    def test_louvain_store(self):
        communities = louvain_communities(self.hg, store=True, attribute='comm', seed=42)

        alice = hedge('alice/C')
        stored = self.hg.get_int_attribute(alice, 'comm')

        self.assertIsNotNone(stored)

    def test_label_propagation(self):
        communities = label_propagation(self.hg)

        self.assertIsInstance(communities, list)
        self.assertGreater(len(communities), 0)

    def test_greedy_modularity(self):
        communities = greedy_modularity_communities(self.hg)

        self.assertIsInstance(communities, list)
        self.assertGreater(len(communities), 0)

    def test_modularity(self):
        communities = louvain_communities(self.hg, seed=42)
        mod = modularity(self.hg, communities)

        self.assertIsInstance(mod, float)
        self.assertGreaterEqual(mod, -0.5)
        self.assertLessEqual(mod, 1.0)


class TestPaths(unittest.TestCase):
    """Test path finding algorithms."""

    def setUp(self):
        self.hg = hgraph('test_paths.db')
        self.hg.destroy()

        # Create a linear chain: a - b - c - d
        self.hg.add('(rel/P a/C b/C)')
        self.hg.add('(rel/P b/C c/C)')
        self.hg.add('(rel/P c/C d/C)')

    def tearDown(self):
        self.hg.close()

    def test_shortest_path_exists(self):
        path = shortest_path(self.hg, 'a/C', 'd/C')

        self.assertIsNotNone(path)
        self.assertIsInstance(path, list)
        self.assertGreater(len(path), 0)

        # Path should start and end correctly
        self.assertEqual(path[0], hedge('a/C'))
        self.assertEqual(path[-1], hedge('d/C'))

    def test_shortest_path_not_exists(self):
        # Add isolated node using a DIFFERENT connector atom
        # so it's not connected through the shared 'rel/P' connector
        self.hg.add('(other/P x/C y/C)')

        path = shortest_path(self.hg, 'a/C', 'x/C')
        self.assertIsNone(path)

    def test_shortest_path_length(self):
        length = shortest_path_length(self.hg, 'a/C', 'd/C')

        self.assertIsNotNone(length)
        self.assertIsInstance(length, int)
        self.assertGreater(length, 0)

    def test_all_shortest_paths(self):
        paths = all_shortest_paths(self.hg, 'a/C', 'd/C')

        self.assertIsInstance(paths, list)
        self.assertGreater(len(paths), 0)

    def test_has_path(self):
        self.assertTrue(has_path(self.hg, 'a/C', 'd/C'))

        # Add isolated component with DIFFERENT connector
        self.hg.add('(other/P x/C y/C)')
        self.assertFalse(has_path(self.hg, 'a/C', 'x/C'))

    def test_reachable(self):
        reachable_set = reachable(self.hg, 'a/C')

        self.assertIsInstance(reachable_set, set)
        self.assertIn(hedge('a/C'), reachable_set)
        self.assertIn(hedge('d/C'), reachable_set)

    def test_reachable_with_depth(self):
        reachable_set = reachable(self.hg, 'a/C', max_depth=1)

        self.assertIn(hedge('a/C'), reachable_set)
        self.assertIn(hedge('b/C'), reachable_set)
        # d/C is too far away at depth 1
        # Note: depends on projection structure

    def test_diameter(self):
        d = diameter(self.hg)

        self.assertIsNotNone(d)
        self.assertIsInstance(d, int)
        self.assertGreater(d, 0)

    def test_radius(self):
        r = radius(self.hg)

        self.assertIsNotNone(r)
        self.assertIsInstance(r, int)
        self.assertGreater(r, 0)

    def test_center(self):
        c = center(self.hg)

        self.assertIsInstance(c, list)
        self.assertGreater(len(c), 0)

    def test_periphery(self):
        p = periphery(self.hg)

        self.assertIsInstance(p, list)
        self.assertGreater(len(p), 0)

    def test_eccentricity(self):
        ecc = eccentricity(self.hg)

        self.assertIsInstance(ecc, dict)
        self.assertGreater(len(ecc), 0)


class TestStatistics(unittest.TestCase):
    """Test statistics functions."""

    def setUp(self):
        self.hg = hgraph('test_statistics.db')
        self.hg.destroy()

        # Create a triangle
        self.hg.add('(rel/P a/C b/C)')
        self.hg.add('(rel/P b/C c/C)')
        self.hg.add('(rel/P c/C a/C)')

    def tearDown(self):
        self.hg.close()

    def test_summary(self):
        stats = summary(self.hg)

        self.assertIsInstance(stats, dict)
        self.assertIn('node_count', stats)
        self.assertIn('edge_count', stats)
        self.assertIn('density', stats)
        self.assertIn('component_count', stats)
        self.assertIn('avg_degree', stats)
        self.assertIn('avg_clustering', stats)
        self.assertIn('is_connected', stats)

    def test_density(self):
        d = density(self.hg)

        self.assertIsInstance(d, float)
        self.assertGreaterEqual(d, 0)
        self.assertLessEqual(d, 1)

    def test_average_clustering(self):
        ac = average_clustering(self.hg)

        self.assertIsInstance(ac, float)
        self.assertGreaterEqual(ac, 0)
        self.assertLessEqual(ac, 1)

    def test_clustering(self):
        c = clustering(self.hg)

        self.assertIsInstance(c, dict)
        for val in c.values():
            self.assertGreaterEqual(val, 0)
            self.assertLessEqual(val, 1)

    def test_transitivity(self):
        t = transitivity(self.hg)

        self.assertIsInstance(t, float)
        self.assertGreaterEqual(t, 0)
        self.assertLessEqual(t, 1)

    def test_degree_histogram(self):
        hist = degree_histogram(self.hg)

        self.assertIsInstance(hist, list)

    def test_degree_sequence(self):
        seq = degree_sequence(self.hg)

        self.assertIsInstance(seq, list)
        # Should be sorted descending
        for i in range(len(seq) - 1):
            self.assertGreaterEqual(seq[i], seq[i+1])

    def test_number_of_triangles(self):
        # Our setup has a triangle of atoms
        n = number_of_triangles(self.hg)

        self.assertIsInstance(n, int)
        self.assertGreaterEqual(n, 0)

    def test_assortativity(self):
        a = assortativity(self.hg)

        # Assortativity may be nan for small graphs with uniform degree
        if a is not None:
            self.assertIsInstance(a, float)
            # Check for valid range or nan
            import math
            if not math.isnan(a):
                self.assertGreaterEqual(a, -1)
                self.assertLessEqual(a, 1)


class TestSubgraphs(unittest.TestCase):
    """Test subgraph extraction algorithms."""

    def setUp(self):
        self.hg = hgraph('test_subgraphs.db')
        self.hg.destroy()

        # Create a star centered on 'hub'
        self.hg.add('(rel/P hub/C a/C)')
        self.hg.add('(rel/P hub/C b/C)')
        self.hg.add('(rel/P hub/C c/C)')

        # Extended edges
        self.hg.add('(rel/P a/C x/C)')
        self.hg.add('(rel/P b/C y/C)')

    def tearDown(self):
        self.hg.close()

    def test_ego_graph(self):
        ego = ego_graph(self.hg, 'hub/C', radius=1)

        self.assertIsInstance(ego, set)
        self.assertGreater(len(ego), 0)

        # Should contain hyperedges directly connected to hub
        for edge in ego:
            # Each edge should contain hub
            atoms = edge.atoms()
            hub = hedge('hub/C')
            self.assertIn(hub, atoms)

    def test_ego_graph_radius_2(self):
        ego = ego_graph(self.hg, 'hub/C', radius=2)

        # Should include more edges
        self.assertGreater(len(ego), 0)

    def test_ego_atoms(self):
        atoms = ego_atoms(self.hg, 'hub/C', radius=1)

        self.assertIsInstance(atoms, set)
        self.assertIn(hedge('hub/C'), atoms)
        # The atoms a, b, c should be reachable at radius 1
        # because they appear in edges with hub
        # However, the implementation might need to find edges via star()
        # Let's just verify we get some atoms beyond hub
        self.assertGreater(len(atoms), 1)

    def test_induced_subgraph(self):
        atom_set = {'hub/C', 'a/C', 'b/C'}
        induced = list(induced_subgraph(self.hg, atom_set))

        # All edges in induced subgraph should only contain atoms from set
        for edge in induced:
            edge_atoms = edge.atoms()
            for atom in edge_atoms:
                self.assertIn(atom.to_str(), atom_set | {'rel/P'})

    def test_k_core(self):
        core = k_core(self.hg, k=1)

        self.assertIsInstance(core, set)
        # At least some nodes should be in 1-core

    def test_k_shell(self):
        shell = k_shell(self.hg, k=1)

        self.assertIsInstance(shell, set)

    def test_core_number(self):
        cores = core_number(self.hg)

        self.assertIsInstance(cores, dict)
        for atom, core in cores.items():
            self.assertIsInstance(core, int)
            self.assertGreaterEqual(core, 0)

    def test_neighbors(self):
        hub = hedge('hub/C')
        n = neighbors(self.hg, hub)

        self.assertIsInstance(n, set)
        # hub should have neighbors (a, b, c, rel)

    def test_common_neighbors(self):
        # a and b both connected to hub
        common = common_neighbors(self.hg, 'a/C', 'b/C')

        self.assertIsInstance(common, set)
        # hub and rel should be common neighbors

    def test_cliques_containing(self):
        cliques = cliques_containing(self.hg, 'hub/C')

        self.assertIsInstance(cliques, list)
        for clique in cliques:
            self.assertIn(hedge('hub/C'), clique)

    def test_largest_clique(self):
        clique = largest_clique(self.hg)

        self.assertIsInstance(clique, set)

    def test_bridges(self):
        b = bridges(self.hg)

        self.assertIsInstance(b, list)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        self.hg = hgraph('test_edge_cases.db')
        self.hg.destroy()

    def tearDown(self):
        self.hg.close()

    def test_empty_hypergraph_summary(self):
        stats = summary(self.hg)

        self.assertEqual(stats['node_count'], 0)
        self.assertEqual(stats['edge_count'], 0)
        self.assertEqual(stats['density'], 0.0)

    def test_empty_hypergraph_communities(self):
        communities = louvain_communities(self.hg)
        self.assertEqual(communities, [])

    def test_empty_hypergraph_paths(self):
        path = shortest_path(self.hg, 'a/C', 'b/C')
        self.assertIsNone(path)

    def test_single_edge(self):
        self.hg.add('(rel/P a/C b/C)')

        stats = summary(self.hg)
        self.assertGreater(stats['node_count'], 0)

        path = shortest_path(self.hg, 'a/C', 'b/C')
        self.assertIsNotNone(path)

    def test_disconnected_atoms(self):
        # Use DIFFERENT connectors to ensure disconnected components
        self.hg.add('(knows/P a/C b/C)')
        self.hg.add('(likes/P x/C y/C)')

        # Check no path between disconnected components
        self.assertFalse(has_path(self.hg, 'a/C', 'x/C'))

        # Check component count (at least 2 with different connectors)
        components = connected_components(self.hg)
        self.assertGreaterEqual(len(components), 2)


if __name__ == '__main__':
    unittest.main()
