"""Tests for agent skills."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from graphbrain.agents.skills.base import BaseSkill, SkillResult
from graphbrain.agents.skills.extraction import ExtractionSkill, ExtractionResult
from graphbrain.agents.skills.query import QuerySkill, QueryResult
from graphbrain.agents.skills.classification import (
    ClassificationSkill,
    ClassificationResult,
    ClassificationAction,
)
from graphbrain.agents.skills.analysis import (
    AnalysisSkill,
    AnalysisResult,
    AnalysisMode,
    ActorStats,
)
from graphbrain.agents.skills.feedback import (
    FeedbackSkill,
    ReviewItem,
    ReviewStatus,
    FeedbackDecision,
    FeedbackSession,
)


class TestSkillResult:
    """Tests for SkillResult dataclass."""

    def test_success_result(self):
        """Test creating a success result."""
        result = SkillResult(
            success=True,
            data={"edge": "(test/P)"},
            confidence=0.95,
        )

        assert result.success is True
        assert result.data["edge"] == "(test/P)"
        assert result.confidence == 0.95
        assert result.error is None

    def test_error_result(self):
        """Test creating an error result."""
        result = SkillResult(
            success=False,
            error="Parse failed",
        )

        assert result.success is False
        assert result.error == "Parse failed"

    def test_to_dict(self):
        """Test serialization."""
        result = SkillResult(
            success=True,
            data="test",
            confidence=0.8,
            metadata={"source": "test"}
        )

        d = result.to_dict()
        assert d["success"] is True
        assert d["data"] == "test"
        assert d["confidence"] == 0.8
        assert d["metadata"]["source"] == "test"


class TestExtractionSkill:
    """Tests for ExtractionSkill."""

    @pytest.fixture
    def skill(self):
        return ExtractionSkill()

    def test_skill_name(self, skill):
        """Test skill name."""
        assert skill.SKILL_NAME == "extraction"

    def test_get_tools(self, skill):
        """Test getting tool list."""
        tools = skill.get_tools()

        assert "mcp__graphbrain__add_edge" in tools
        assert "mcp__graphbrain__pattern_match" in tools
        assert "mcp__graphbrain__flag_for_review" in tools

    def test_validate_edge_valid(self, skill):
        """Test validating a valid edge."""
        is_valid, error = skill.validate_edge("(is/Pd sky/C blue/C)")

        assert is_valid is True
        assert error is None

    def test_validate_edge_invalid(self, skill):
        """Test validating an invalid edge."""
        is_valid, error = skill.validate_edge("(broken syntax")

        assert is_valid is False
        assert error is not None

    def test_format_extraction(self, skill):
        """Test formatting an extraction result."""
        result = skill.format_extraction(
            edge_str="(is/Pd sky/C blue/C)",
            original_text="The sky is blue",
            confidence=0.85,
            source_url="https://example.com",
            reasoning="Clear declarative statement",
        )

        assert result.edge == "(is/Pd sky/C blue/C)"
        assert result.original_text == "The sky is blue"
        assert result.confidence == 0.85
        assert result.needs_review is False  # 0.85 > 0.7 threshold
        assert result.metadata["source"] == "https://example.com"

    def test_format_extraction_needs_review(self, skill):
        """Test that low confidence triggers review."""
        result = skill.format_extraction(
            edge_str="(unclear/P thing/C)",
            original_text="Unclear text",
            confidence=0.5,
        )

        assert result.needs_review is True

    def test_build_add_edge_params(self, skill):
        """Test building add_edge parameters."""
        extraction = ExtractionResult(
            edge="(test/P)",
            original_text="Test",
            confidence=0.9,
        )

        params = skill.build_add_edge_params(extraction)

        assert params["edge"] == "(test/P)"
        assert params["text"] == "Test"
        assert params["primary"] is True

    def test_estimate_confidence(self, skill):
        """Test confidence estimation."""
        # Full confidence for complete extraction
        conf = skill.estimate_confidence(
            "(is/Pd sky/C blue/C)",
            has_subject=True,
            has_predicate=True,
        )
        assert conf >= 0.8

        # Lower confidence for missing subject
        conf_no_subj = skill.estimate_confidence(
            "(is/Pd blue/C)",
            has_subject=False,
            has_predicate=True,
        )
        assert conf_no_subj < conf

    def test_get_extraction_patterns(self, skill):
        """Test getting extraction patterns."""
        patterns = skill.get_extraction_patterns()

        assert "attributed_claim" in patterns
        assert "declarative" in patterns
        assert "action_object" in patterns


class TestQuerySkill:
    """Tests for QuerySkill."""

    @pytest.fixture
    def skill(self):
        return QuerySkill()

    def test_skill_name(self, skill):
        """Test skill name."""
        assert skill.SKILL_NAME == "query"

    def test_get_tools(self, skill):
        """Test getting tool list."""
        tools = skill.get_tools()

        assert "mcp__graphbrain__hybrid_search" in tools
        assert "mcp__graphbrain__pattern_match" in tools
        assert "mcp__graphbrain__edges_with_root" in tools

    def test_translate_who_said_question(self, skill):
        """Test translating 'who said' question."""
        result = skill.translate_question("Who said anything about housing?")

        assert result["method"] == "pattern_match"
        assert "SPEAKER" in result["params"]["pattern"]

    def test_translate_what_did_question(self, skill):
        """Test translating 'what did' question."""
        result = skill.translate_question("What did Mayor Harrell do?")

        assert result["method"] == "edges_with_root"
        assert "mayor" in result["params"]["root"].lower()

    def test_translate_claims_question(self, skill):
        """Test translating claims question."""
        result = skill.translate_question("Find claims about transit")

        assert result["method"] == "hybrid_search"
        assert result["params"]["class_id"] == "claim"

    def test_translate_relationship_question(self, skill):
        """Test translating relationship question."""
        result = skill.translate_question("How is Seattle related to Housing?")

        assert "entities" in result or result["method"] == "pattern_match"

    def test_extract_entity(self, skill):
        """Test entity extraction from question."""
        entity = skill._extract_entity("What did Mayor Harrell do?")
        assert entity is not None
        assert "mayor" in entity.lower() or "harrell" in entity.lower()

    def test_extract_entities(self, skill):
        """Test extracting multiple entities."""
        entities = skill._extract_entities("How are Seattle and Portland related?")
        assert len(entities) >= 1

    def test_build_hybrid_search_params(self, skill):
        """Test building hybrid search params."""
        params = skill.build_hybrid_search_params(
            "housing policy",
            class_id="claim",
            limit=25
        )

        assert params["query"] == "housing policy"
        assert params["class_id"] == "claim"
        assert params["limit"] == 25

    def test_get_query_patterns(self, skill):
        """Test getting query patterns."""
        patterns = skill.get_query_patterns()

        assert "claims_by_speaker" in patterns
        assert "actions_by_subject" in patterns


class TestClassificationSkill:
    """Tests for ClassificationSkill."""

    @pytest.fixture
    def skill(self):
        return ClassificationSkill()

    def test_skill_name(self, skill):
        """Test skill name."""
        assert skill.SKILL_NAME == "classification"

    def test_get_tools(self, skill):
        """Test getting tool list."""
        tools = skill.get_tools()

        assert "mcp__graphbrain__classify_predicate" in tools
        assert "mcp__graphbrain__discover_predicates" in tools

    def test_determine_action_auto_apply(self, skill):
        """Test high confidence triggers auto-apply."""
        action = skill.determine_action(0.95)
        assert action == ClassificationAction.AUTO_APPLY

    def test_determine_action_apply_with_log(self, skill):
        """Test moderate-high confidence triggers apply with log."""
        action = skill.determine_action(0.85)
        assert action == ClassificationAction.APPLY_WITH_LOG

    def test_determine_action_flag_optional(self, skill):
        """Test moderate confidence triggers optional flag."""
        action = skill.determine_action(0.75)
        assert action == ClassificationAction.FLAG_OPTIONAL

    def test_determine_action_require_review(self, skill):
        """Test low confidence requires review."""
        action = skill.determine_action(0.6)
        assert action == ClassificationAction.REQUIRE_REVIEW

    def test_determine_action_reject(self, skill):
        """Test very low confidence rejects."""
        action = skill.determine_action(0.3)
        assert action == ClassificationAction.REJECT

    def test_format_classification(self, skill):
        """Test formatting classification result."""
        result = skill.format_classification(
            predicate="announce",
            suggested_class="claim",
            confidence=0.92,
            method="predicate_bank",
            alternatives=[("action", 0.7), ("support", 0.5)],
        )

        assert result.predicate == "announce"
        assert result.suggested_class == "claim"
        assert result.action == ClassificationAction.AUTO_APPLY
        assert len(result.alternatives) == 2

    def test_merge_similar_results(self, skill):
        """Test merging similar predicate results."""
        similar = [
            {"predicate": "declare", "class_id": "claim", "similarity": 0.9},
            {"predicate": "state", "class_id": "claim", "similarity": 0.85},
            {"predicate": "do", "class_id": "action", "similarity": 0.7},
        ]

        suggested, confidence, alts = skill.merge_similar_results(similar)

        assert suggested == "claim"
        assert confidence > 0.8
        assert any(a[0] == "action" for a in alts)

    def test_get_classification_summary(self, skill):
        """Test getting classification summary."""
        results = [
            ClassificationResult("pred1", "claim", 0.95, "bank", ClassificationAction.AUTO_APPLY),
            ClassificationResult("pred2", "claim", 0.85, "semantic", ClassificationAction.APPLY_WITH_LOG),
            ClassificationResult("pred3", "action", 0.65, "semantic", ClassificationAction.REQUIRE_REVIEW),
        ]

        summary = skill.get_classification_summary(results)

        assert summary["total"] == 3
        assert summary["auto_applied"] == 1
        assert summary["by_class"]["claim"] == 2


class TestAnalysisSkill:
    """Tests for AnalysisSkill."""

    @pytest.fixture
    def skill(self):
        return AnalysisSkill()

    def test_skill_name(self, skill):
        """Test skill name."""
        assert skill.SKILL_NAME == "analysis"

    def test_get_tools(self, skill):
        """Test getting tool list."""
        tools = skill.get_tools()

        assert "mcp__graphbrain__pattern_match" in tools
        assert "mcp__graphbrain__hypergraph_stats" in tools

    def test_get_analysis_patterns(self, skill):
        """Test getting patterns for analysis modes."""
        actor_patterns = skill.get_analysis_patterns(AnalysisMode.ACTOR)
        assert len(actor_patterns) > 0

        claim_patterns = skill.get_analysis_patterns(AnalysisMode.CLAIM)
        assert len(claim_patterns) > 0

    def test_aggregate_actors(self, skill):
        """Test aggregating actor statistics."""
        edges = [
            {"bindings": {"ACTOR": "mayor/Cp"}, "edge": "(said/Pd mayor/Cp hello/C)"},
            {"bindings": {"ACTOR": "mayor/Cp"}, "edge": "(did/Pd mayor/Cp thing/C)"},
            {"bindings": {"SPEAKER": "council/Cc"}, "edge": "(stated/Pd council/Cc claim/C)"},
        ]

        actors = skill.aggregate_actors(edges)

        assert len(actors) == 2
        # Mayor should be first (more mentions)
        assert actors[0].mention_count == 2

    def test_detect_conflicts_empty(self, skill):
        """Test conflict detection with no conflicts."""
        claims = [
            {"predicate": "said", "topic": "housing"},
            {"predicate": "announced", "topic": "transit"},
        ]

        conflicts = skill.detect_conflicts(claims)
        assert len(conflicts) == 0

    def test_generate_summary(self, skill):
        """Test generating analysis summary."""
        result = AnalysisResult(
            mode=AnalysisMode.ACTOR,
            actors=[ActorStats("Test Actor", "test/Cp", claim_count=5)],
            statistics={"total_edges": 100},
        )

        summary = skill.generate_summary(result)

        assert "Actor Analysis" in summary
        assert "Test Actor" in summary


class TestFeedbackSkill:
    """Tests for FeedbackSkill."""

    @pytest.fixture
    def skill(self):
        return FeedbackSkill()

    def test_skill_name(self, skill):
        """Test skill name."""
        assert skill.SKILL_NAME == "feedback"

    def test_get_tools(self, skill):
        """Test getting tool list."""
        tools = skill.get_tools()

        assert "mcp__graphbrain__apply_feedback" in tools
        assert "mcp__graphbrain__get_pending_reviews" in tools

    def test_decide_action_human_approved(self, skill):
        """Test that human approval always applies."""
        item = ReviewItem(
            review_id="rev1",
            predicate="test",
            original_class=None,
            suggested_class="claim",
            confidence=0.5,  # Low confidence
        )

        decision = skill.decide_action(item, human_approved=True)
        assert decision == FeedbackDecision.APPLY

    def test_decide_action_high_confidence(self, skill):
        """Test high confidence auto-applies."""
        item = ReviewItem(
            review_id="rev1",
            predicate="test",
            original_class=None,
            suggested_class="claim",
            confidence=0.98,
        )

        decision = skill.decide_action(item)
        assert decision == FeedbackDecision.APPLY

    def test_decide_action_low_confidence(self, skill):
        """Test low confidence defers."""
        item = ReviewItem(
            review_id="rev1",
            predicate="test",
            original_class=None,
            suggested_class="claim",
            confidence=0.5,
        )

        decision = skill.decide_action(item)
        assert decision == FeedbackDecision.DEFER

    def test_format_review_item(self, skill):
        """Test formatting a review item."""
        item = skill.format_review_item(
            predicate="announce",
            suggested_class="claim",
            confidence=0.75,
            similar_predicates=[
                {"predicate": "declare", "class_id": "claim", "similarity": 0.9},
            ],
        )

        assert item.predicate == "announce"
        assert item.suggested_class == "claim"
        assert item.confidence == 0.75
        assert "similar" in item.evidence

    def test_process_batch(self, skill):
        """Test processing a batch of reviews."""
        items = [
            ReviewItem("r1", "pred1", None, "claim", 0.9),
            ReviewItem("r2", "pred2", None, "action", 0.8),
            ReviewItem("r3", "pred3", None, "conflict", 0.7),
        ]

        decisions = {
            "r1": FeedbackDecision.APPLY,
            "r2": FeedbackDecision.REJECT,
            "r3": FeedbackDecision.DEFER,
        }

        session = skill.process_batch(items, decisions)

        assert session.processed == 3
        assert session.applied == 1
        assert session.rejected == 1
        assert session.deferred == 1
        assert session.classes_updated.get("claim") == 1

    def test_identify_improvement_opportunities(self, skill):
        """Test identifying improvement opportunities."""
        stats = {
            "unclassified_high_freq": 15,
            "avg_confidence": 0.6,
            "by_class": {"claim": 100, "action": 10},
            "total_reviews": 20,
            "rejected": 8,
        }

        opportunities = skill.identify_improvement_opportunities(stats)

        assert len(opportunities) >= 2  # At least low confidence and imbalance

    def test_generate_session_summary(self, skill):
        """Test generating session summary."""
        session = FeedbackSession(
            processed=10,
            applied=5,
            rejected=2,
            deferred=3,
            classes_updated={"claim": 3, "action": 2},
        )

        summary = skill.generate_session_summary(session)

        assert "Processed: 10" in summary
        assert "Applied: 5" in summary
        assert "claim: +3" in summary
