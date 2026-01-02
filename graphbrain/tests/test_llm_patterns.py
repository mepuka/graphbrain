"""Tests for LLM-enhanced pattern matching and discovery."""

import pytest
from unittest.mock import MagicMock, patch

from graphbrain import hedge, hgraph

from graphbrain.patterns.llm_patterns import (
    PatternCandidate,
    PatternDiscovery,
    NaturalLanguagePatterns,
    PatternExplainer,
    AdaptivePatternMatcher,
    PATTERN_SYNTAX_GUIDE,
)


class TestPatternCandidate:
    """Tests for PatternCandidate dataclass."""

    def test_create_candidate(self):
        """Test creating a pattern candidate."""
        candidate = PatternCandidate(
            pattern="(*/P * *)",
            frequency=10,
            examples=[hedge("(says/P john/C hello/C)")],
            confidence=0.8,
            description="Test pattern"
        )
        assert candidate.pattern == "(*/P * *)"
        assert candidate.frequency == 10
        assert candidate.confidence == 0.8
        assert len(candidate.examples) == 1

    def test_default_description(self):
        """Test default empty description."""
        candidate = PatternCandidate(
            pattern="(*/P *)",
            frequency=5,
            examples=[],
            confidence=0.5
        )
        assert candidate.description == ""


class TestPatternDiscovery:
    """Tests for PatternDiscovery class."""

    @pytest.fixture
    def hg(self):
        """Create test hypergraph with sample edges."""
        hg = hgraph("test_pattern_discovery.db")
        # Add sample edges
        hg.add(hedge("(says/Pd john/Cp hello/C)"))
        hg.add(hedge("(says/Pd mary/Cp goodbye/C)"))
        hg.add(hedge("(says/Pd bob/Cp message/C)"))
        hg.add(hedge("(is/Pd.sc sky/Cc blue/Ca)"))
        hg.add(hedge("(is/Pd.sc grass/Cc green/Ca)"))
        yield hg

    @pytest.fixture
    def discovery(self, hg):
        """Create PatternDiscovery instance."""
        return PatternDiscovery(hg)

    def test_init(self, hg):
        """Test initialization."""
        discovery = PatternDiscovery(hg)
        assert discovery.hg == hg

    def test_edge_to_pattern_atom(self, discovery):
        """Test pattern generation for atoms."""
        edge = hedge("john/Cp")
        pattern = discovery._edge_to_pattern(edge, 2)
        assert "C" in pattern or pattern == "."

    def test_edge_to_pattern_simple(self, discovery):
        """Test pattern generation for simple edge."""
        edge = hedge("(says/Pd john/Cp hello/C)")
        pattern = discovery._edge_to_pattern(edge, 2)
        assert pattern.startswith("(")
        assert "P" in pattern

    def test_edge_to_pattern_depth_zero(self, discovery):
        """Test pattern at depth 0 returns generic."""
        edge = hedge("(says/Pd john/Cp hello/C)")
        pattern = discovery._edge_to_pattern(edge, 0)
        assert pattern == "(*)"

    def test_describe_pattern_predicate(self, discovery):
        """Test pattern description for predicates."""
        desc = discovery._describe_pattern("(*/Pd * *)")
        assert "action" in desc.lower() or "event" in desc.lower()

    def test_describe_pattern_concept(self, discovery):
        """Test pattern description for concepts."""
        desc = discovery._describe_pattern("(*/P */Cp *)")
        assert "proper" in desc.lower()

    def test_describe_pattern_modifier(self, discovery):
        """Test pattern description for modifiers."""
        desc = discovery._describe_pattern("(*/M *)")
        assert "modifier" in desc.lower()

    def test_discover_from_examples_empty(self, discovery):
        """Test discovering from empty examples."""
        result = discovery.discover_from_examples([])
        assert result == []

    def test_discover_from_examples(self, discovery):
        """Test discovering from example edges."""
        examples = [
            hedge("(says/Pd john/Cp hello/C)"),
            hedge("(says/Pd mary/Cp goodbye/C)"),
        ]
        result = discovery.discover_from_examples(examples)
        assert len(result) >= 1
        assert result[0].examples == examples[:5]

    def test_rule_based_suggestions_claims(self, discovery):
        """Test rule-based suggestions for claims."""
        suggestions = discovery._rule_based_suggestions("find all claims")
        assert len(suggestions) > 0
        assert any("say" in s.lower() or "P" in s for s in suggestions)

    def test_rule_based_suggestions_conflict(self, discovery):
        """Test rule-based suggestions for conflict."""
        suggestions = discovery._rule_based_suggestions("attack against")
        assert len(suggestions) > 0

    def test_rule_based_suggestions_location(self, discovery):
        """Test rule-based suggestions for location."""
        suggestions = discovery._rule_based_suggestions("where is the location")
        assert len(suggestions) > 0

    def test_rule_based_suggestions_default(self, discovery):
        """Test default suggestions."""
        suggestions = discovery._rule_based_suggestions("random query")
        assert len(suggestions) > 0
        assert len(suggestions) <= 3


class TestNaturalLanguagePatterns:
    """Tests for NaturalLanguagePatterns class."""

    @pytest.fixture
    def hg(self):
        """Create test hypergraph."""
        return hgraph("test_nl_patterns.db")

    @pytest.fixture
    def nl_patterns(self, hg):
        """Create NaturalLanguagePatterns instance."""
        return NaturalLanguagePatterns(hg)

    def test_init(self, hg):
        """Test initialization."""
        nl = NaturalLanguagePatterns(hg)
        assert nl.hg == hg

    def test_rule_translate_say(self, nl_patterns):
        """Test translation of 'say' queries."""
        pattern = nl_patterns._rule_translate("who says what")
        assert pattern is not None
        assert "say" in pattern.lower() or "P" in pattern

    def test_rule_translate_predicate(self, nl_patterns):
        """Test translation of predicate queries."""
        pattern = nl_patterns._rule_translate("all predicates")
        assert pattern is not None
        assert "P" in pattern

    def test_rule_translate_concept(self, nl_patterns):
        """Test translation of concept queries."""
        pattern = nl_patterns._rule_translate("find concepts")
        assert pattern is not None
        assert "C" in pattern

    def test_rule_translate_relationship(self, nl_patterns):
        """Test translation of relationship queries."""
        pattern = nl_patterns._rule_translate("find relationships")
        assert pattern is not None

    def test_rule_translate_default(self, nl_patterns):
        """Test default translation."""
        pattern = nl_patterns._rule_translate("unknown query")
        assert pattern is not None

    def test_query_to_pattern_no_llm(self, nl_patterns):
        """Test query translation without LLM."""
        pattern = nl_patterns.query_to_pattern("who says something")
        assert pattern is not None

    def test_pattern_to_description(self, nl_patterns):
        """Test pattern to description conversion."""
        desc = nl_patterns.pattern_to_description("(*/P * *)")
        assert desc is not None
        assert len(desc) > 0


class TestPatternExplainer:
    """Tests for PatternExplainer class."""

    def test_explain_predicate_pattern(self):
        """Test explaining predicate pattern."""
        explanation = PatternExplainer.explain("(*/P * *)")
        assert "predicate" in explanation.lower() or "action" in explanation.lower()

    def test_explain_wildcard(self):
        """Test explaining wildcard."""
        explanation = PatternExplainer.explain("*")
        assert "any" in explanation.lower()

    def test_explain_atom_wildcard(self):
        """Test explaining atom wildcard."""
        explanation = PatternExplainer.explain(".")
        assert "atom" in explanation.lower()

    def test_explain_type_specifier(self):
        """Test explaining type specifier."""
        explanation = PatternExplainer.explain("*/P")
        assert "predicate" in explanation.lower()

    def test_explain_lemma_pattern(self):
        """Test explaining lemma pattern."""
        explanation = PatternExplainer.explain("(lemma say/P SPEAKER *)")
        assert "say" in explanation.lower() or "form" in explanation.lower()

    def test_explain_variable_capture(self):
        """Test explaining variable capture."""
        explanation = PatternExplainer.explain("(*/P SPEAKER *)")
        assert "SPEAKER" in explanation

    def test_explain_match(self):
        """Test explaining a match."""
        edge = hedge("(says/Pd john/Cp hello/C)")
        pattern = "(*/P SPEAKER *)"
        variables = {"SPEAKER": hedge("john/Cp")}

        explanation = PatternExplainer.explain_match(edge, pattern, variables)

        assert "says" in explanation
        assert "SPEAKER" in explanation
        assert "john" in explanation


class TestAdaptivePatternMatcher:
    """Tests for AdaptivePatternMatcher class."""

    @pytest.fixture
    def hg(self):
        """Create test hypergraph."""
        hg = hgraph("test_adaptive_matcher.db")
        hg.add(hedge("(says/Pd john/Cp hello/C)"))
        hg.add(hedge("(says/Pd mary/Cp world/C)"))
        yield hg

    @pytest.fixture
    def matcher(self, hg):
        """Create AdaptivePatternMatcher instance."""
        return AdaptivePatternMatcher(hg)

    def test_init(self, hg):
        """Test initialization."""
        matcher = AdaptivePatternMatcher(hg)
        assert matcher.hg == hg
        assert matcher.discovery is not None
        assert matcher.nl_patterns is not None

    def test_init_with_llm(self, hg):
        """Test initialization with LLM client."""
        mock_llm = MagicMock()
        matcher = AdaptivePatternMatcher(hg, llm_client=mock_llm)
        assert matcher.llm_client == mock_llm

    def test_suggest_related_patterns_generalize(self, matcher):
        """Test suggesting more general patterns."""
        suggestions = matcher.suggest_related_patterns("(*/P * *)")
        assert len(suggestions) > 0

    def test_suggest_related_patterns_specialize(self, matcher):
        """Test suggesting more specific patterns."""
        suggestions = matcher.suggest_related_patterns("(*/P * *)")
        # Should suggest */C or */P versions
        assert any("C" in s or "P" in s for s in suggestions)

    def test_suggest_related_patterns_capture(self, matcher):
        """Test suggesting patterns with variable capture."""
        suggestions = matcher.suggest_related_patterns("(*/P * *)")
        assert any("CAPTURE" in s for s in suggestions)


class TestPatternSyntaxGuide:
    """Tests for pattern syntax documentation."""

    def test_guide_exists(self):
        """Test that syntax guide is defined."""
        assert PATTERN_SYNTAX_GUIDE is not None
        assert len(PATTERN_SYNTAX_GUIDE) > 100

    def test_guide_contains_wildcards(self):
        """Test guide documents wildcards."""
        assert "*" in PATTERN_SYNTAX_GUIDE
        assert "." in PATTERN_SYNTAX_GUIDE
        assert "..." in PATTERN_SYNTAX_GUIDE

    def test_guide_contains_types(self):
        """Test guide documents types."""
        assert "*/P" in PATTERN_SYNTAX_GUIDE
        assert "*/C" in PATTERN_SYNTAX_GUIDE
        assert "*/M" in PATTERN_SYNTAX_GUIDE

    def test_guide_contains_functions(self):
        """Test guide documents functions."""
        assert "lemma" in PATTERN_SYNTAX_GUIDE
        assert "any" in PATTERN_SYNTAX_GUIDE
        assert "not" in PATTERN_SYNTAX_GUIDE

    def test_guide_contains_examples(self):
        """Test guide contains examples."""
        assert "EXAMPLES" in PATTERN_SYNTAX_GUIDE
        assert "(*/P * *)" in PATTERN_SYNTAX_GUIDE


class TestDomainPatterns:
    """Tests for domain-specific patterns."""

    def test_domain_patterns_import(self):
        """Test domain patterns can be imported."""
        from graphbrain.patterns.llm_patterns import DOMAIN_PATTERNS
        assert DOMAIN_PATTERNS is not None
        assert len(DOMAIN_PATTERNS) > 0

    def test_claims_domain(self):
        """Test claims domain is defined."""
        from graphbrain.patterns.llm_patterns import DOMAIN_PATTERNS
        assert "claims" in DOMAIN_PATTERNS
        assert "patterns" in DOMAIN_PATTERNS["claims"]
        assert "keywords" in DOMAIN_PATTERNS["claims"]
        assert "say" in DOMAIN_PATTERNS["claims"]["keywords"]

    def test_conflict_domain(self):
        """Test conflict domain is defined."""
        from graphbrain.patterns.llm_patterns import DOMAIN_PATTERNS
        assert "conflict" in DOMAIN_PATTERNS
        assert "attack" in DOMAIN_PATTERNS["conflict"]["keywords"]

    def test_support_domain(self):
        """Test support domain is defined."""
        from graphbrain.patterns.llm_patterns import DOMAIN_PATTERNS
        assert "support" in DOMAIN_PATTERNS
        assert "endorse" in DOMAIN_PATTERNS["support"]["keywords"]

    def test_policy_domain(self):
        """Test policy domain is defined."""
        from graphbrain.patterns.llm_patterns import DOMAIN_PATTERNS
        assert "policy" in DOMAIN_PATTERNS
        assert "policy" in DOMAIN_PATTERNS["policy"]["keywords"]


class TestSemanticSearch:
    """Tests for semantic search functionality."""

    @pytest.fixture
    def hg(self):
        """Create test hypergraph."""
        return hgraph("test_semantic_search.db")

    @pytest.fixture
    def matcher(self, hg):
        """Create AdaptivePatternMatcher instance."""
        return AdaptivePatternMatcher(hg)

    def test_list_domains(self, matcher):
        """Test listing available domains."""
        domains = matcher.list_domains()
        assert len(domains) > 0
        assert any(d["name"] == "claims" for d in domains)

    def test_get_domain_patterns(self, matcher):
        """Test getting patterns for a domain."""
        patterns = matcher.get_domain_patterns("claims")
        assert len(patterns) > 0
        assert any("say" in p for p in patterns)

    def test_get_domain_patterns_unknown(self, matcher):
        """Test getting patterns for unknown domain."""
        patterns = matcher.get_domain_patterns("nonexistent")
        assert patterns == []

    def test_explain_query_claims(self, matcher):
        """Test query explanation for claims."""
        # Use a query that clearly triggers claims domain without other keywords
        explanation = matcher.explain_query("what did they say or claim")
        assert "claims" in explanation.lower() or "say" in explanation.lower()

    def test_explain_query_conflict(self, matcher):
        """Test query explanation for conflict."""
        explanation = matcher.explain_query("attacks on the policy")
        assert "conflict" in explanation.lower() or "attack" in explanation.lower()

    def test_search_semantic_empty(self, matcher):
        """Test semantic search on empty hypergraph."""
        results = matcher.search_semantic("claims about policy")
        assert isinstance(results, list)
