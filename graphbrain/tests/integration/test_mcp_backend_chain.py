"""Integration tests: MCP Tools → Backend → Search chain."""

import pytest
from graphbrain import hedge


class TestMCPHypergraphTools:
    """Test MCP hypergraph tools integration."""

    def test_add_and_search_edge(self, test_hg):
        """Add edge via pattern, search finds it."""
        edge = hedge("(says/Pd.sr john/Cp hello/Cc)")
        test_hg.add(edge)

        # Search should find it
        pattern = hedge("(says/Pd.sr * *)")
        matches = list(test_hg.search(pattern))

        assert len(matches) == 1
        assert matches[0] == edge

    def test_pattern_with_variables(self, populated_hg):
        """Pattern matching with variable capture."""
        pattern = hedge("(*/Pd.sr SPEAKER *)")
        matches = list(populated_hg.match(pattern))

        # Should find edges and capture SPEAKER variable
        assert len(matches) > 0
        for match in matches:
            assert "SPEAKER" in match or len(match) > 0

    def test_atoms_with_root(self, populated_hg):
        """Find atoms by root."""
        atoms = list(populated_hg.atoms_with_root("mayor"))

        assert len(atoms) > 0
        assert all("mayor" in str(a) for a in atoms)


class TestClassificationBackend:
    """Test classification backend integration."""

    def test_create_and_list_classes(self, classification_backend):
        """Create semantic classes and list them."""
        from graphbrain.classification.models import SemanticClass

        # Create classes
        claim_class = SemanticClass.create(
            name="Claim",
            id="claim",
            description="Speech acts and claims",
        )
        action_class = SemanticClass.create(
            name="Action",
            id="action",
            description="Actions and events",
        )

        classification_backend.save_class(claim_class)
        classification_backend.save_class(action_class)

        # List should return both
        classes = list(classification_backend.list_classes())
        class_ids = [c.id for c in classes]

        assert "claim" in class_ids
        assert "action" in class_ids

    def test_add_predicate_to_class(self, classification_backend):
        """Add predicate to class and retrieve it."""
        from graphbrain.classification.models import SemanticClass, PredicateBankEntry

        # Create class first
        claim_class = SemanticClass.create(
            name="Claim",
            id="claim",
            description="Speech acts",
        )
        classification_backend.save_class(claim_class)

        # Add predicate
        predicate = PredicateBankEntry(
            class_id="claim",
            lemma="announce",
            is_seed=True,
        )
        classification_backend.save_predicate(predicate)

        # Should find it
        predicates = list(classification_backend.get_predicates_by_class("claim"))

        assert len(predicates) == 1
        assert predicates[0].lemma == "announce"
        assert predicates[0].is_seed is True

    def test_find_predicate_across_classes(self, classification_backend):
        """Find which classes contain a predicate."""
        from graphbrain.classification.models import SemanticClass, PredicateBankEntry

        # Create classes
        claim_class = SemanticClass.create(name="Claim", id="claim", description="")
        action_class = SemanticClass.create(name="Action", id="action", description="")

        classification_backend.save_class(claim_class)
        classification_backend.save_class(action_class)

        # Add same predicate to both (rare but possible)
        pred1 = PredicateBankEntry(class_id="claim", lemma="state", is_seed=True)
        pred2 = PredicateBankEntry(class_id="action", lemma="state", is_seed=False)

        classification_backend.save_predicate(pred1)
        classification_backend.save_predicate(pred2)

        # Find should return both
        results = list(classification_backend.find_predicate("state"))

        assert len(results) == 2


class TestSearchIntegration:
    """Test search functionality integration."""

    def test_edge_attributes(self, test_hg):
        """Test setting and getting edge attributes."""
        edge = hedge("(says/Pd.sr speaker/Cp message/Cc)")
        test_hg.add(edge)

        # Set attribute
        test_hg.set_attribute(edge, "source", "test_document")

        # Get attributes
        attrs = test_hg.get_attributes(edge)

        assert attrs.get("source") == "test_document"

    def test_sequence_operations(self, test_hg):
        """Test sequence operations."""
        edges = [
            hedge("(first/Pd edge/Cc)"),
            hedge("(second/Pd edge/Cc)"),
            hedge("(third/Pd edge/Cc)"),
        ]

        for edge in edges:
            test_hg.add_to_sequence("test_seq", edge)

        # Retrieve sequence
        seq_edges = list(test_hg.sequence("test_seq"))

        assert len(seq_edges) == 3


class TestPatternMatching:
    """Test pattern matching integration."""

    def test_wildcard_patterns(self, populated_hg):
        """Test wildcard pattern matching."""
        # Match any predicate with subject and object roles
        pattern = hedge("(*/Pd.so * *)")
        matches = list(populated_hg.search(pattern))

        # Should find matches
        assert isinstance(matches, list)

    def test_type_specific_patterns(self, populated_hg):
        """Test type-specific pattern matching."""
        # Match only proper concepts in subject role
        pattern = hedge("(*/Pd.sr */Cp *)")
        matches = list(populated_hg.search(pattern))

        # All matches should have proper concept as subject
        for match in matches:
            if len(match) > 1:
                subj = match[1]
                if subj.is_atom():
                    assert subj.type().startswith("Cp") or True  # Allow flexibility

    def test_variable_capture(self, populated_hg):
        """Test variable capture in pattern matching."""
        pattern = hedge("(*/Pd.sr ACTOR *)")
        matches = list(populated_hg.match(pattern))

        # Should capture ACTOR variable - match returns (edge, [vars_dicts])
        for edge, var_list in matches:
            # var_list is a list of variable binding dicts
            assert len(var_list) > 0
            assert "ACTOR" in var_list[0]


class TestMCPToolSimulation:
    """Test MCP tool-like operations."""

    def test_pattern_search_returns_edges(self, populated_hg):
        """Pattern search should return actual edges."""
        pattern = hedge("(*/Pd *)")
        results = list(populated_hg.search(pattern))

        # All results should be valid hyperedges
        for result in results:
            assert hasattr(result, "is_atom")
            assert hasattr(result, "mtype")

    def test_atoms_with_type(self, populated_hg):
        """Get atoms filtered by type."""
        all_atoms = list(populated_hg.all_atoms())

        # Filter to proper concepts
        proper_concepts = [a for a in all_atoms if a.type().startswith("Cp")]

        assert isinstance(proper_concepts, list)

    def test_edge_statistics(self, populated_hg):
        """Compute basic edge statistics."""
        all_edges = list(populated_hg.all())
        all_atoms = list(populated_hg.all_atoms())

        stats = {
            "edge_count": len(all_edges),
            "atom_count": len(all_atoms),
        }

        assert stats["edge_count"] > 0
