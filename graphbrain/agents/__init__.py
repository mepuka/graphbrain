"""
Graphbrain Agent SDK Integration.

This module provides integration with Claude's Agent SDK for building
knowledge extraction, query, classification, analysis, and feedback agents.

Example usage:
    from graphbrain.agents import create_agent_options, AgentType

    options = create_agent_options(
        agent_type=AgentType.EXTRACTION,
        db_connection="postgresql://localhost/graphbrain"
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query("Extract knowledge from: 'The mayor announced...'")
        async for message in client.receive_response():
            process_message(message)
"""

from graphbrain.agents.config import (
    AgentType,
    create_agent_options,
    get_agent_definition,
)
from graphbrain.agents.memory.session import AgentSession, SessionManager
from graphbrain.agents.memory.decisions import DecisionLog, DecisionLogger
from graphbrain.agents.learning import (
    ActiveLearningSampler,
    SamplingStrategy,
    SuggestionEngine,
)
from graphbrain.agents.metrics import (
    QualityDashboard,
    MetricsCollector,
)
from graphbrain.agents.llm import (
    PredicateCategory,
    PredicateClassification,
    EntityType,
    EntityClassification,
    LLMProvider,
    AnthropicProvider,
)
from graphbrain.agents.skills import (
    BaseSkill,
    SkillResult,
    LLMClassificationSkill,
    LLMEntityTypingSkill,
)

__all__ = [
    # Agent configuration
    "AgentType",
    "create_agent_options",
    "get_agent_definition",
    # Session management
    "AgentSession",
    "SessionManager",
    # Decision logging
    "DecisionLog",
    "DecisionLogger",
    # Active learning
    "ActiveLearningSampler",
    "SamplingStrategy",
    "SuggestionEngine",
    # Quality metrics
    "QualityDashboard",
    "MetricsCollector",
    # LLM models
    "PredicateCategory",
    "PredicateClassification",
    "EntityType",
    "EntityClassification",
    # LLM providers
    "LLMProvider",
    "AnthropicProvider",
    # Skills
    "BaseSkill",
    "SkillResult",
    "LLMClassificationSkill",
    "LLMEntityTypingSkill",
]
