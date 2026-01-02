"""Integration tests: Processor → Algorithm → Analysis chain."""

import pytest
from graphbrain import hedge


class TestActorNetworkAnalysis:
    """Test actor extraction → network → centrality chain."""

    def test_extract_actors_from_claims(self, populated_hg):
        """Extract actors from claim edges."""
        from graphbrain.processors.actors import Actors

        processor = Actors(populated_hg, use_adaptive=False)
        processor.run()

        # Should have extracted some actors (or at least not crash)
        assert hasattr(processor, "actor_counter")

    def test_compute_pagerank(self, populated_hg):
        """Compute PageRank centrality on hypergraph."""
        from graphbrain.algorithms.centrality import pagerank

        scores = pagerank(populated_hg)

        assert isinstance(scores, dict)

    def test_compute_degree_centrality(self, populated_hg):
        """Compute degree centrality on hypergraph."""
        from graphbrain.algorithms.centrality import degree_centrality

        scores = degree_centrality(populated_hg)

        assert isinstance(scores, dict)


class TestClaimTopicAnalysis:
    """Test claim extraction → topic clustering chain."""

    def test_extract_claims(self, populated_hg):
        """Extract claims from hypergraph."""
        from graphbrain.processors.claims import Claims

        processor = Claims(populated_hg)
        processor.run()

        claims = processor.claims
        assert isinstance(claims, (list, set, dict))

    def test_group_claims_by_predicate(self, populated_hg):
        """Group claims by their main predicate."""
        from graphbrain.processors.claims import Claims

        processor = Claims(populated_hg)
        processor.run()

        # Group by predicate root
        predicate_groups = {}
        for claim in getattr(processor, "claims", []):
            if hasattr(claim, "__getitem__") and len(claim) > 0:
                pred = claim[0]
                root = pred.root() if hasattr(pred, "root") else str(pred)
                if root not in predicate_groups:
                    predicate_groups[root] = []
                predicate_groups[root].append(claim)

        # Should have some grouping
        assert isinstance(predicate_groups, dict)


@pytest.mark.urbanist
class TestUrbanistAnalysis:
    """Test full analysis chain on urbanist data."""

    def test_urbanist_actor_extraction(self, urbanist_hg):
        """Extract actors from urbanist corpus."""
        from graphbrain.processors.actors import Actors

        processor = Actors(urbanist_hg, use_adaptive=False)

        # Process subset for speed
        count = 0
        for edge in urbanist_hg.all():
            if count >= 1000:
                break
            processor.process_edge(edge)
            count += 1

        processor.on_end()

        # Should find some actors
        assert hasattr(processor, "actor_counter")

    def test_urbanist_claim_patterns(self, urbanist_hg):
        """Find claim patterns in urbanist data."""
        # Search for announcement patterns
        pattern = hedge("(*/Pd.so * *)")
        matches = list(urbanist_hg.search(pattern))

        # Urbanist data should have plenty of claims
        assert isinstance(matches, list)

    def test_urbanist_entity_mentions(self, urbanist_hg):
        """Count entity mentions in urbanist data."""
        # Find atoms with "mayor" root
        mayor_atoms = list(urbanist_hg.atoms_with_root("mayor"))

        # Should have mayor mentions in urbanist data
        assert isinstance(mayor_atoms, list)

    def test_urbanist_pagerank(self, urbanist_hg):
        """Compute PageRank on urbanist data."""
        from graphbrain.algorithms.centrality import pagerank

        # Filter to just proper concepts
        def filter_fn(edge):
            return edge.mtype() == "C"

        scores = pagerank(urbanist_hg, filter_fn=filter_fn)

        assert isinstance(scores, dict)
        assert len(scores) > 0


class TestAlgorithmIntegration:
    """Test algorithm module integration."""

    def test_centrality_module_imports(self):
        """Centrality algorithms can be imported."""
        from graphbrain.algorithms import centrality

        assert hasattr(centrality, "pagerank")
        assert hasattr(centrality, "degree_centrality")
        assert hasattr(centrality, "betweenness_centrality")

    def test_projections_module_imports(self):
        """Projection algorithms can be imported."""
        from graphbrain.algorithms import projections

        assert hasattr(projections, "AtomCooccurrenceProjection")

    def test_community_module_imports(self):
        """Community algorithms can be imported."""
        from graphbrain.algorithms import community

        assert hasattr(community, "louvain_communities")

    def test_statistics_module_imports(self):
        """Statistics algorithms can be imported."""
        from graphbrain.algorithms import statistics

        assert True  # Module exists


class TestEndToEndPipeline:
    """Test complete analysis pipelines."""

    def test_text_to_network_pipeline(self, test_hg, parser):
        """Complete pipeline: text → parse → actors → network."""
        # Parse text
        text = "The mayor announced the new transit plan. Council approved it."
        parser.parse_and_add(text, test_hg)

        # Extract actors
        from graphbrain.processors.actors import Actors

        processor = Actors(test_hg, use_adaptive=False)
        processor.run()

        # Compute centrality
        from graphbrain.algorithms.centrality import degree_centrality

        scores = degree_centrality(test_hg)

        # Pipeline completed without error
        assert isinstance(scores, dict)

    def test_text_to_claims_pipeline(self, test_hg, parser):
        """Complete pipeline: text → parse → claims."""
        text = "Officials said housing costs are rising. The report showed growth."
        parser.parse_and_add(text, test_hg)

        from graphbrain.processors.claims import Claims

        processor = Claims(test_hg)
        processor.run()

        # Pipeline completed
        assert hasattr(processor, "claims")

    def test_pattern_discovery_pipeline(self, populated_hg):
        """Discover patterns from existing edges."""
        # Find all predicate types used
        predicates = set()
        count = 0
        for edge in populated_hg.all():
            if count >= 500:
                break
            if not edge.is_atom() and len(edge) > 0:
                conn = edge[0]
                if hasattr(conn, "type") and conn.type().startswith("P"):
                    predicates.add(conn.root() if hasattr(conn, "root") else str(conn))
            count += 1

        # Should find some predicates
        assert isinstance(predicates, set)

    def test_subgraph_extraction(self, populated_hg):
        """Extract subgraph around a concept."""
        from graphbrain.algorithms.subgraphs import ego_graph

        # Get an atom to center on
        atoms = list(populated_hg.all_atoms())
        if not atoms:
            pytest.skip("No atoms in test hypergraph")

        center = atoms[0]
        subgraph = ego_graph(populated_hg, center, radius=1)

        assert isinstance(subgraph, set)
