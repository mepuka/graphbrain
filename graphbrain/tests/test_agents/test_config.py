"""Tests for agent configuration and factory."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from graphbrain.agents.config import (
    AgentType,
    ModelChoice,
    AgentDefinition,
    AGENT_DEFINITIONS,
    get_agent_definition,
    create_agent_options,
    create_multi_agent_options,
)


class TestAgentDefinition:
    """Tests for AgentDefinition dataclass."""

    def test_all_agent_types_defined(self):
        """Test that all agent types have definitions."""
        for agent_type in AgentType:
            assert agent_type in AGENT_DEFINITIONS

    def test_extraction_definition(self):
        """Test extraction agent definition."""
        definition = AGENT_DEFINITIONS[AgentType.EXTRACTION]

        assert definition.agent_type == AgentType.EXTRACTION
        assert definition.model == ModelChoice.SONNET
        assert "mcp__graphbrain__add_edge" in definition.tools
        assert "mcp__graphbrain__pattern_match" in definition.tools
        assert definition.prompt_file == "extraction.md"

    def test_query_definition(self):
        """Test query agent definition."""
        definition = AGENT_DEFINITIONS[AgentType.QUERY]

        assert definition.agent_type == AgentType.QUERY
        assert "mcp__graphbrain__search_edges" in definition.tools
        assert "mcp__graphbrain__hybrid_search" in definition.tools
        assert "mcp__graphbrain__hypergraph_stats" in definition.tools

    def test_classification_definition(self):
        """Test classification agent definition."""
        definition = AGENT_DEFINITIONS[AgentType.CLASSIFICATION]

        assert definition.agent_type == AgentType.CLASSIFICATION
        assert "mcp__graphbrain__classify_predicate" in definition.tools
        assert "mcp__graphbrain__discover_predicates" in definition.tools

    def test_analysis_uses_opus(self):
        """Test that analysis agent uses opus model."""
        definition = AGENT_DEFINITIONS[AgentType.ANALYSIS]

        assert definition.model == ModelChoice.OPUS

    def test_feedback_uses_haiku(self):
        """Test that feedback agent uses haiku model."""
        definition = AGENT_DEFINITIONS[AgentType.FEEDBACK]

        assert definition.model == ModelChoice.HAIKU

    def test_get_prompt_with_system_prompt(self):
        """Test getting prompt when system_prompt is set."""
        definition = AgentDefinition(
            agent_type=AgentType.EXTRACTION,
            description="Test",
            tools=[],
            system_prompt="Custom prompt"
        )

        prompt = definition.get_prompt()
        assert prompt == "Custom prompt"

    def test_get_prompt_fallback(self):
        """Test fallback prompt when file not found."""
        definition = AgentDefinition(
            agent_type=AgentType.EXTRACTION,
            description="Test agent",
            tools=[],
        )

        prompt = definition.get_prompt()
        assert "extraction" in prompt
        assert "graphbrain" in prompt


class TestGetAgentDefinition:
    """Tests for get_agent_definition function."""

    def test_get_extraction(self):
        """Test getting extraction agent definition."""
        definition = get_agent_definition(AgentType.EXTRACTION)
        assert definition.agent_type == AgentType.EXTRACTION

    def test_get_all_types(self):
        """Test getting all agent type definitions."""
        for agent_type in AgentType:
            definition = get_agent_definition(agent_type)
            assert definition.agent_type == agent_type


class TestCreateAgentOptions:
    """Tests for create_agent_options function."""

    def test_basic_options(self):
        """Test creating basic agent options."""
        options = create_agent_options(
            AgentType.EXTRACTION,
            "postgresql://localhost/test"
        )

        assert "mcp_servers" in options
        assert "allowed_tools" in options
        assert "system_prompt" in options
        assert "agents" in options

    def test_mcp_server_config(self):
        """Test MCP server configuration in options."""
        options = create_agent_options(
            AgentType.QUERY,
            "postgresql://localhost/graphbrain",
            domain="urbanist"
        )

        mcp_config = options["mcp_servers"]["graphbrain"]
        assert mcp_config["type"] == "sdk"
        assert mcp_config["config"]["db_connection"] == "postgresql://localhost/graphbrain"
        assert mcp_config["config"]["domain"] == "urbanist"

    def test_allowed_tools(self):
        """Test allowed tools in options."""
        options = create_agent_options(
            AgentType.CLASSIFICATION,
            "postgresql://localhost/test"
        )

        allowed = options["allowed_tools"]
        assert "mcp__graphbrain__classify_predicate" in allowed
        assert "mcp__graphbrain__discover_predicates" in allowed

    def test_extra_tools(self):
        """Test adding extra tools."""
        options = create_agent_options(
            AgentType.EXTRACTION,
            "postgresql://localhost/test",
            extra_tools=["custom_tool_1", "custom_tool_2"]
        )

        allowed = options["allowed_tools"]
        assert "custom_tool_1" in allowed
        assert "custom_tool_2" in allowed

    def test_system_prompt_structure(self):
        """Test system prompt structure."""
        options = create_agent_options(
            AgentType.ANALYSIS,
            "postgresql://localhost/test"
        )

        prompt = options["system_prompt"]
        assert prompt["type"] == "preset"
        assert prompt["preset"] == "claude_code"
        assert "append" in prompt

    def test_agent_definitions_included(self):
        """Test that agent definitions are included."""
        options = create_agent_options(
            AgentType.FEEDBACK,
            "postgresql://localhost/test"
        )

        agents = options["agents"]
        assert "feedback" in agents
        assert agents["feedback"]["model"] == "haiku"

    def test_kwargs_passed_through(self):
        """Test that extra kwargs are passed through."""
        options = create_agent_options(
            AgentType.EXTRACTION,
            "postgresql://localhost/test",
            permission_mode="acceptEdits",
            max_turns=10
        )

        assert options["permission_mode"] == "acceptEdits"
        assert options["max_turns"] == 10


class TestCreateMultiAgentOptions:
    """Tests for create_multi_agent_options function."""

    def test_all_agents_defined(self):
        """Test that all agent types are defined."""
        options = create_multi_agent_options(
            "postgresql://localhost/test",
            primary_agent=AgentType.EXTRACTION
        )

        agents = options["agents"]
        for agent_type in AgentType:
            assert agent_type.value in agents

    def test_all_tools_allowed(self):
        """Test that tools from all agents are allowed."""
        options = create_multi_agent_options(
            "postgresql://localhost/test"
        )

        allowed = options["allowed_tools"]

        # Check tools from different agents
        assert "mcp__graphbrain__add_edge" in allowed  # extraction
        assert "mcp__graphbrain__hybrid_search" in allowed  # query
        assert "mcp__graphbrain__classify_predicate" in allowed  # classification
        assert "mcp__graphbrain__hypergraph_stats" in allowed  # analysis
        assert "mcp__graphbrain__apply_feedback" in allowed  # feedback

    def test_primary_agent_prompt(self):
        """Test that primary agent's prompt is used."""
        options = create_multi_agent_options(
            "postgresql://localhost/test",
            primary_agent=AgentType.ANALYSIS
        )

        # Primary agent's prompt should be in system_prompt
        prompt = options["system_prompt"]["append"]
        assert prompt  # Should have some content


class TestModelChoice:
    """Tests for ModelChoice enum."""

    def test_all_model_choices(self):
        """Test that all expected models exist."""
        assert ModelChoice.SONNET.value == "sonnet"
        assert ModelChoice.OPUS.value == "opus"
        assert ModelChoice.HAIKU.value == "haiku"
