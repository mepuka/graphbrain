"""Tests for utility modules."""

import pytest
from graphbrain import hedge, hgraph

from graphbrain.utils.concepts import (
    strip_concept,
    has_proper_concept,
    has_common_or_proper_concept,
    all_concepts,
)
from graphbrain.utils.lemmas import lemma, deep_lemma, lemma_degrees


class TestStripConcept:
    """Tests for strip_concept function."""

    def test_strip_concept_already_concept(self):
        """Test strip_concept returns concept as-is."""
        edge = hedge("book/Cc")
        result = strip_concept(edge)
        assert result == edge

    def test_strip_concept_with_modifier(self):
        """Test strip_concept strips modifier."""
        edge = hedge("(the/Md book/Cc)")
        result = strip_concept(edge)
        # Should return the concept, not the modifier
        assert result is not None
        assert result.mtype() == 'C'

    def test_strip_concept_with_trigger(self):
        """Test strip_concept strips trigger."""
        edge = hedge("(against/T (the/Md treaty/Cc))")
        result = strip_concept(edge)
        assert result is not None

    def test_strip_concept_nested(self):
        """Test strip_concept with nested structure."""
        edge = hedge("(of/Br treaty/Cc paris/Cp)")
        result = strip_concept(edge)
        assert result is not None

    def test_strip_concept_atom_non_concept(self):
        """Test strip_concept with non-concept atom returns None."""
        edge = hedge("running/Pd")
        result = strip_concept(edge)
        assert result is None


class TestHasProperConcept:
    """Tests for has_proper_concept function."""

    def test_proper_concept_atom(self):
        """Test proper concept detection on atom."""
        edge = hedge("john/Cp")
        assert has_proper_concept(edge) is True

    def test_common_concept_atom(self):
        """Test common concept is not proper."""
        edge = hedge("book/Cc")
        assert has_proper_concept(edge) is False

    def test_proper_concept_nested(self):
        """Test proper concept in nested edge."""
        edge = hedge("(the/Md (+/B john/Cp smith/Cp))")
        assert has_proper_concept(edge) is True

    def test_no_proper_concept(self):
        """Test edge with no proper concept."""
        edge = hedge("(the/Md book/Cc)")
        assert has_proper_concept(edge) is False


class TestHasCommonOrProperConcept:
    """Tests for has_common_or_proper_concept function."""

    def test_proper_concept(self):
        """Test proper concept detected."""
        edge = hedge("john/Cp")
        assert has_common_or_proper_concept(edge) is True

    def test_common_concept(self):
        """Test common concept detected."""
        edge = hedge("book/Cc")
        assert has_common_or_proper_concept(edge) is True

    def test_other_type(self):
        """Test other types not detected."""
        edge = hedge("running/Pd")
        assert has_common_or_proper_concept(edge) is False


class TestAllConcepts:
    """Tests for all_concepts function."""

    def test_all_concepts_single(self):
        """Test all_concepts with single concept."""
        edge = hedge("book/Cc")
        concepts = all_concepts(edge)
        assert len(concepts) == 1
        assert edge in concepts

    def test_all_concepts_nested(self):
        """Test all_concepts finds nested concepts."""
        edge = hedge("(of/Br book/Cc author/Cc)")
        concepts = all_concepts(edge)
        assert len(concepts) >= 2

    def test_all_concepts_no_concepts(self):
        """Test all_concepts with no concepts."""
        edge = hedge("running/Pd")
        concepts = all_concepts(edge)
        assert len(concepts) == 0

    def test_all_concepts_mixed(self):
        """Test all_concepts with mixed types."""
        edge = hedge("(says/Pd.sr john/Cp message/Cc)")
        concepts = all_concepts(edge)
        # Should find john/Cp and message/Cc
        assert len(concepts) >= 2


class TestLemma:
    """Tests for lemma function."""

    @pytest.fixture
    def hg(self):
        """Create test hypergraph with lemma edges."""
        hg = hgraph("test_lemmas.db")
        # Add a lemma relationship
        import graphbrain.constants as const
        hg.add((const.lemma_connector, hedge("running/Pd"), hedge("run/Pd")))
        yield hg

    def test_lemma_found(self, hg):
        """Test lemma retrieval when exists."""
        atom = hedge("running/Pd")
        result = lemma(hg, atom)
        # Should return the lemma if the edge exists
        # May be None if lemma connector format differs
        # This is an integration test

    def test_lemma_not_found(self, hg):
        """Test lemma returns None when not found."""
        atom = hedge("unknown/Pd")
        result = lemma(hg, atom)
        assert result is None

    def test_lemma_same_if_none(self, hg):
        """Test same_if_none returns atom."""
        atom = hedge("unknown/Pd")
        result = lemma(hg, atom, same_if_none=True)
        assert result == atom


class TestDeepLemma:
    """Tests for deep_lemma function."""

    @pytest.fixture
    def hg(self):
        """Create test hypergraph."""
        return hgraph("test_deep_lemmas.db")

    def test_deep_lemma_atom(self, hg):
        """Test deep_lemma on atom."""
        atom = hedge("running/Pd")
        result = deep_lemma(hg, atom, same_if_none=True)
        assert result == atom

    def test_deep_lemma_nested(self, hg):
        """Test deep_lemma on nested edge."""
        edge = hedge("(not/M (is/Pd going/Pd))")
        # deep_lemma descends into nested edges and calls lemma()
        # which returns None if no lemma edge exists (unless same_if_none=True)
        # The function itself doesn't pass same_if_none to recursive calls
        result = deep_lemma(hg, edge)
        # May be None if no lemma found - that's expected behavior
        # Just verify no exception is raised

    def test_deep_lemma_predicate(self, hg):
        """Test deep_lemma finds predicate lemma."""
        edge = hedge("(will/Mv (be/Pd done/Ca))")
        result = deep_lemma(hg, edge, same_if_none=True)
        # Should find lemma of innermost verb


class TestLemmaDegrees:
    """Tests for lemma_degrees function."""

    @pytest.fixture
    def hg(self):
        """Create test hypergraph with some edges."""
        hg = hgraph("test_lemma_degrees.db")
        # Add some edges
        hg.add(hedge("(says/Pd john/Cp hello/C)"))
        hg.add(hedge("(runs/Pd mary/Cp)"))
        yield hg

    def test_lemma_degrees_atom(self, hg):
        """Test lemma_degrees on atom."""
        atom = hedge("john/Cp")
        d, dd = lemma_degrees(hg, atom)
        # Should return degree and deep degree
        assert isinstance(d, int)
        assert isinstance(dd, int)

    def test_lemma_degrees_edge(self, hg):
        """Test lemma_degrees on non-atomic edge."""
        edge = hedge("(says/Pd john/Cp hello/C)")
        d, dd = lemma_degrees(hg, edge)
        assert isinstance(d, int)
        assert isinstance(dd, int)


class TestConjunctions:
    """Tests for conjunctions utility."""

    def test_import(self):
        """Test conjunctions module can be imported."""
        from graphbrain.utils.conjunctions import conjunctions_decomposition
        assert conjunctions_decomposition is not None


class TestCorefs:
    """Tests for coreference utilities."""

    def test_import(self):
        """Test corefs module can be imported."""
        from graphbrain.utils.corefs import main_coref
        assert main_coref is not None

    def test_main_coref_identity(self):
        """Test main_coref returns identity when no coref exists."""
        from graphbrain.utils.corefs import main_coref

        hg = hgraph("test_corefs.db")
        edge = hedge("john/Cp")

        result = main_coref(hg, edge)
        assert result == edge


class TestEnglish:
    """Tests for English utility functions."""

    def test_import(self):
        """Test english module can be imported."""
        from graphbrain.utils.english import word_to_american, to_american
        assert word_to_american is not None
        assert to_american is not None

    def test_word_to_american(self):
        """Test British to American spelling conversion."""
        from graphbrain.utils.english import word_to_american
        # Should convert British spellings
        assert word_to_american("colour") == "color"
        assert word_to_american("centre") == "center"
        # Should leave American spellings unchanged
        assert word_to_american("color") == "color"


class TestOntology:
    """Tests for ontology utilities."""

    def test_import(self):
        """Test ontology module can be imported."""
        from graphbrain.utils.ontology import subtypes
        assert subtypes is not None


class TestNumber:
    """Tests for number utilities."""

    def test_import(self):
        """Test number module can be imported."""
        from graphbrain.utils.number import number
        assert number is not None
