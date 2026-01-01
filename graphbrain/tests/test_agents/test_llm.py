"""Tests for LLM skills and models."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import ValidationError


def run_async(coro):
    """Run async coroutine in sync test."""
    return asyncio.get_event_loop().run_until_complete(coro)

from graphbrain.agents.llm.models import (
    PredicateCategory,
    PredicateClassification,
    BatchPredicateResult,
    EntityType,
    EntityClassification,
    BatchEntityResult,
)
from graphbrain.agents.llm.providers.base import (
    LLMProvider,
    LLMError,
    LLMValidationError,
    LLMRateLimitError,
)
from graphbrain.agents.skills.llm_classification import LLMClassificationSkill
from graphbrain.agents.skills.llm_entity_typing import LLMEntityTypingSkill


class TestPredicateCategory:
    """Tests for PredicateCategory enum."""

    def test_all_categories_defined(self):
        """Test all expected categories exist."""
        expected = [
            "claim", "conflict", "action", "cognition",
            "emotion", "movement", "possession", "perception", "unknown"
        ]
        actual = [c.value for c in PredicateCategory]
        assert set(expected) == set(actual)

    def test_category_string_values(self):
        """Test categories are string enums."""
        assert PredicateCategory.CLAIM.value == "claim"
        assert PredicateCategory.CONFLICT.value == "conflict"


class TestPredicateClassification:
    """Tests for PredicateClassification model."""

    def test_valid_classification(self):
        """Test creating a valid classification."""
        result = PredicateClassification(
            lemma="announce",
            category=PredicateCategory.CLAIM,
            confidence=0.95,
            reasoning="Speech act verb",
            similar_predicates=["declare", "state"],
        )

        assert result.lemma == "announce"
        assert result.category == PredicateCategory.CLAIM
        assert result.confidence == 0.95
        assert len(result.similar_predicates) == 2

    def test_confidence_validation(self):
        """Test confidence must be 0.0-1.0."""
        with pytest.raises(ValidationError):
            PredicateClassification(
                lemma="test",
                category=PredicateCategory.ACTION,
                confidence=1.5,  # Invalid
                reasoning="test",
            )

    def test_reasoning_max_length(self):
        """Test reasoning has max length."""
        long_reasoning = "x" * 600  # Over 500 chars
        with pytest.raises(ValidationError):
            PredicateClassification(
                lemma="test",
                category=PredicateCategory.ACTION,
                confidence=0.8,
                reasoning=long_reasoning,
            )

    def test_similar_predicates_max_length(self):
        """Test similar_predicates list max 5."""
        with pytest.raises(ValidationError):
            PredicateClassification(
                lemma="test",
                category=PredicateCategory.ACTION,
                confidence=0.8,
                reasoning="test",
                similar_predicates=["a", "b", "c", "d", "e", "f"],  # 6 items
            )

    def test_string_category_accepted(self):
        """Test category can be provided as string."""
        result = PredicateClassification(
            lemma="test",
            category="claim",
            confidence=0.8,
            reasoning="test",
        )
        assert result.category == PredicateCategory.CLAIM


class TestEntityType:
    """Tests for EntityType enum."""

    def test_all_types_defined(self):
        """Test all expected entity types exist."""
        expected = ["person", "organization", "location", "group", "event", "unknown"]
        actual = [t.value for t in EntityType]
        assert set(expected) == set(actual)


class TestEntityClassification:
    """Tests for EntityClassification model."""

    def test_valid_classification(self):
        """Test creating a valid entity classification."""
        result = EntityClassification(
            entity="Seattle City Council",
            entity_type=EntityType.ORGANIZATION,
            confidence=0.92,
            reasoning="Contains 'Council' indicating government body",
            subtypes=["government", "legislative"],
        )

        assert result.entity == "Seattle City Council"
        assert result.entity_type == EntityType.ORGANIZATION
        assert result.confidence == 0.92
        assert "government" in result.subtypes

    def test_string_entity_type_accepted(self):
        """Test entity_type can be provided as string."""
        result = EntityClassification(
            entity="John Smith",
            entity_type="person",
            confidence=0.9,
            reasoning="First+last name pattern",
        )
        assert result.entity_type == EntityType.PERSON

    def test_subtypes_default_empty(self):
        """Test subtypes defaults to empty list."""
        result = EntityClassification(
            entity="Test",
            entity_type=EntityType.UNKNOWN,
            confidence=0.5,
            reasoning="Unknown entity",
        )
        assert result.subtypes == []


class TestBatchResults:
    """Tests for batch result models."""

    def test_batch_predicate_result(self):
        """Test BatchPredicateResult."""
        classifications = [
            PredicateClassification(
                lemma="say",
                category=PredicateCategory.CLAIM,
                confidence=0.95,
                reasoning="Speech act",
            ),
            PredicateClassification(
                lemma="attack",
                category=PredicateCategory.CONFLICT,
                confidence=0.88,
                reasoning="Conflict verb",
            ),
        ]
        result = BatchPredicateResult(
            classifications=classifications,
            unclassified=["unknown_verb"],
        )

        assert len(result.classifications) == 2
        assert len(result.unclassified) == 1

    def test_batch_entity_result(self):
        """Test BatchEntityResult."""
        entities = [
            EntityClassification(
                entity="Seattle",
                entity_type=EntityType.LOCATION,
                confidence=0.9,
                reasoning="City name",
            ),
        ]
        result = BatchEntityResult(
            entities=entities,
            unclassified=[],
        )

        assert len(result.entities) == 1
        assert len(result.unclassified) == 0


class TestLLMErrors:
    """Tests for LLM error classes."""

    def test_llm_error(self):
        """Test base LLM error."""
        error = LLMError("Test error", provider="anthropic")
        assert str(error) == "Test error"
        assert error.provider == "anthropic"

    def test_llm_validation_error(self):
        """Test validation error."""
        error = LLMValidationError(
            "Invalid response",
            provider="anthropic",
            details={"field": "category"},
        )
        assert error.provider == "anthropic"
        assert error.details["field"] == "category"

    def test_llm_rate_limit_error(self):
        """Test rate limit error."""
        error = LLMRateLimitError(
            "Rate limited",
            provider="anthropic",
            details={"retry_after": 60},
        )
        assert error.details["retry_after"] == 60


class TestLLMClassificationSkill:
    """Tests for LLMClassificationSkill."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = MagicMock(spec=LLMProvider)
        provider.name = "mock_provider"
        return provider

    @pytest.fixture
    def skill(self, mock_provider):
        """Create skill with mock provider."""
        return LLMClassificationSkill(provider=mock_provider)

    def test_skill_name(self, skill):
        """Test skill name."""
        assert skill.SKILL_NAME == "llm_classification"

    def test_prompt_file(self, skill):
        """Test prompt file name."""
        assert skill.PROMPT_FILE == "llm_classification.md"

    def test_default_threshold(self, skill):
        """Test default confidence threshold."""
        assert skill.DEFAULT_CONFIDENCE_THRESHOLD == 0.75

    def test_get_tools(self, skill):
        """Test MCP tools list."""
        tools = skill.get_tools()
        assert "mcp__graphbrain__add_predicate_to_class" in tools
        assert "mcp__graphbrain__find_similar_predicates" in tools
        assert "mcp__graphbrain__list_semantic_classes" in tools
        assert "mcp__graphbrain__flag_for_review" in tools

    def test_category_seeds(self, skill):
        """Test seed predicates for categories."""
        seeds = skill.get_category_seeds()

        assert "claim" in seeds
        assert "say" in seeds["claim"]
        assert "announce" in seeds["claim"]

        assert "conflict" in seeds
        assert "attack" in seeds["conflict"]

    def test_classify_predicate_success(self, skill, mock_provider):
        """Test successful predicate classification."""
        expected = PredicateClassification(
            lemma="announce",
            category=PredicateCategory.CLAIM,
            confidence=0.92,
            reasoning="Speech act verb",
            similar_predicates=["declare"],
        )
        mock_provider.classify = AsyncMock(return_value=expected)

        result = run_async(skill.classify_predicate("announce"))

        assert result.success is True
        assert result.data.category == PredicateCategory.CLAIM
        assert result.confidence == 0.92

    def test_classify_predicate_error(self, skill, mock_provider):
        """Test classification error handling."""
        mock_provider.classify = AsyncMock(side_effect=Exception("API error"))

        result = run_async(skill.classify_predicate("test"))

        assert result.success is False
        assert "API error" in result.error

    def test_classify_batch_empty(self, skill, mock_provider):
        """Test batch classification with empty list."""
        result = run_async(skill.classify_batch([]))

        assert result.success is True
        assert len(result.data.classifications) == 0


class TestLLMEntityTypingSkill:
    """Tests for LLMEntityTypingSkill."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = MagicMock(spec=LLMProvider)
        provider.name = "mock_provider"
        return provider

    @pytest.fixture
    def skill(self, mock_provider):
        """Create skill with mock provider."""
        return LLMEntityTypingSkill(provider=mock_provider)

    def test_skill_name(self, skill):
        """Test skill name."""
        assert skill.SKILL_NAME == "llm_entity_typing"

    def test_prompt_file(self, skill):
        """Test prompt file name."""
        assert skill.PROMPT_FILE == "llm_entity_typing.md"

    def test_default_threshold(self, skill):
        """Test default confidence threshold is higher for entities."""
        assert skill.DEFAULT_CONFIDENCE_THRESHOLD == 0.8

    def test_get_tools(self, skill):
        """Test MCP tools list."""
        tools = skill.get_tools()
        assert "mcp__graphbrain__edges_with_root" in tools
        assert "mcp__graphbrain__search_edges" in tools
        assert "mcp__graphbrain__pattern_match" in tools
        assert "mcp__graphbrain__flag_for_review" in tools

    def test_type_indicators(self, skill):
        """Test type indicators dictionary."""
        indicators = skill.get_type_indicators()

        assert "person" in indicators
        assert "mayor" in indicators["person"]

        assert "organization" in indicators
        assert "council" in indicators["organization"]

        assert "location" in indicators
        assert "city" in indicators["location"]

    def test_type_entity_success(self, skill, mock_provider):
        """Test successful entity typing."""
        expected = EntityClassification(
            entity="Seattle City Council",
            entity_type=EntityType.ORGANIZATION,
            confidence=0.95,
            reasoning="Government body",
            subtypes=["government"],
        )
        mock_provider.classify = AsyncMock(return_value=expected)

        result = run_async(skill.type_entity("Seattle City Council"))

        assert result.success is True
        assert result.data.entity_type == EntityType.ORGANIZATION
        assert result.confidence == 0.95

    def test_type_entity_with_context(self, skill, mock_provider):
        """Test entity typing with context edges."""
        expected = EntityClassification(
            entity="Ballard",
            entity_type=EntityType.LOCATION,
            confidence=0.85,
            reasoning="Used with 'in' preposition",
            subtypes=["neighborhood"],
        )
        mock_provider.classify = AsyncMock(return_value=expected)

        result = run_async(skill.type_entity(
            "Ballard",
            context_edges=["(in/Br residents/Cc ballard/Cp)"],
        ))

        assert result.success is True
        assert result.data.entity_type == EntityType.LOCATION

    def test_type_batch_empty(self, skill, mock_provider):
        """Test batch typing with empty list."""
        result = run_async(skill.type_batch([]))

        assert result.success is True
        assert len(result.data.entities) == 0

    def test_type_from_hypergraph(self, skill, mock_provider):
        """Test typing with hypergraph context."""
        expected = EntityClassification(
            entity="SDOT",
            entity_type=EntityType.ORGANIZATION,
            confidence=0.88,
            reasoning="Acronym pattern, government agency",
            subtypes=["agency"],
        )
        mock_provider.classify = AsyncMock(return_value=expected)

        hg_context = {
            "edges": ["(announced/Pd sdot/Cp plan/Cc)"],
            "predicates": ["announce", "release"],
        }

        result = run_async(skill.type_from_hypergraph("SDOT", hg_context))

        assert result.success is True
        assert result.data.entity_type == EntityType.ORGANIZATION
        assert "agency" in result.data.subtypes
