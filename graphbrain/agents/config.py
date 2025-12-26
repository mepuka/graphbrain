"""
Agent configuration and factory.

Provides ClaudeAgentOptions creation for different agent types,
with proper MCP server setup and tool permissions.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    """Types of available agents."""
    EXTRACTION = "extraction"
    QUERY = "query"
    CLASSIFICATION = "classification"
    ANALYSIS = "analysis"
    FEEDBACK = "feedback"


class ModelChoice(str, Enum):
    """Available model choices."""
    SONNET = "sonnet"
    OPUS = "opus"
    HAIKU = "haiku"


@dataclass
class AgentDefinition:
    """
    Definition for an agent skill.

    Contains all configuration needed to instantiate an agent
    with the Claude Agent SDK.
    """
    agent_type: AgentType
    description: str
    tools: list[str]
    model: ModelChoice = ModelChoice.SONNET
    prompt_file: Optional[str] = None
    system_prompt: Optional[str] = None

    def get_prompt(self, docs_path: Optional[Path] = None) -> str:
        """
        Get the system prompt for this agent.

        Args:
            docs_path: Path to docs/llm/agent-prompts/

        Returns:
            The system prompt string
        """
        if self.system_prompt:
            return self.system_prompt

        if self.prompt_file and docs_path:
            prompt_path = docs_path / self.prompt_file
            if prompt_path.exists():
                return prompt_path.read_text()

        # Fallback to minimal prompt
        return f"You are a {self.agent_type.value} agent for graphbrain semantic hypergraphs."


# Agent definitions with their tool permissions
AGENT_DEFINITIONS: dict[AgentType, AgentDefinition] = {
    AgentType.EXTRACTION: AgentDefinition(
        agent_type=AgentType.EXTRACTION,
        description="Extract semantic relations from text into hyperedges",
        tools=[
            "mcp__graphbrain__add_edge",
            "mcp__graphbrain__get_edge",
            "mcp__graphbrain__pattern_match",
            "mcp__graphbrain__flag_for_review",
        ],
        model=ModelChoice.SONNET,
        prompt_file="extraction.md",
    ),
    AgentType.QUERY: AgentDefinition(
        agent_type=AgentType.QUERY,
        description="Query and explore the semantic hypergraph",
        tools=[
            "mcp__graphbrain__search_edges",
            "mcp__graphbrain__pattern_match",
            "mcp__graphbrain__hybrid_search",
            "mcp__graphbrain__bm25_search",
            "mcp__graphbrain__edges_with_root",
            "mcp__graphbrain__hypergraph_stats",
        ],
        model=ModelChoice.SONNET,
        prompt_file="query.md",
    ),
    AgentType.CLASSIFICATION: AgentDefinition(
        agent_type=AgentType.CLASSIFICATION,
        description="Classify predicates using semantic similarity and predicate banks",
        tools=[
            "mcp__graphbrain__classify_predicate",
            "mcp__graphbrain__classify_edge",
            "mcp__graphbrain__discover_predicates",
            "mcp__graphbrain__find_similar_predicates",
            "mcp__graphbrain__get_predicate_classes",
            "mcp__graphbrain__list_predicates_by_class",
            "mcp__graphbrain__flag_for_review",
        ],
        model=ModelChoice.SONNET,
        prompt_file="classification.md",
    ),
    AgentType.ANALYSIS: AgentDefinition(
        agent_type=AgentType.ANALYSIS,
        description="Analyze the knowledge graph for insights and patterns",
        tools=[
            "mcp__graphbrain__pattern_match",
            "mcp__graphbrain__hybrid_search",
            "mcp__graphbrain__hypergraph_stats",
            "mcp__graphbrain__list_semantic_classes",
            "mcp__graphbrain__classification_stats",
        ],
        model=ModelChoice.OPUS,  # More capable for analysis
        prompt_file="analysis.md",
    ),
    AgentType.FEEDBACK: AgentDefinition(
        agent_type=AgentType.FEEDBACK,
        description="Process human feedback to improve classification quality",
        tools=[
            "mcp__graphbrain__get_pending_reviews",
            "mcp__graphbrain__apply_feedback",
            "mcp__graphbrain__submit_feedback",
            "mcp__graphbrain__feedback_stats",
            "mcp__graphbrain__add_predicate_to_class",
        ],
        model=ModelChoice.HAIKU,  # Fast for review processing
        prompt_file="feedback.md",
    ),
}


def get_agent_definition(agent_type: AgentType) -> AgentDefinition:
    """
    Get the definition for an agent type.

    Args:
        agent_type: The type of agent

    Returns:
        The agent definition
    """
    return AGENT_DEFINITIONS[agent_type]


def create_agent_options(
    agent_type: AgentType,
    db_connection: str,
    domain: str = "default",
    docs_path: Optional[Path] = None,
    extra_tools: Optional[list[str]] = None,
    **kwargs
) -> dict:
    """
    Create ClaudeAgentOptions for an agent.

    This factory function creates the full configuration needed
    to instantiate a Claude Agent SDK client for a specific agent type.

    Args:
        agent_type: The type of agent to create
        db_connection: PostgreSQL connection string
        domain: Domain for classification context
        docs_path: Path to docs/llm/agent-prompts/ for loading prompts
        extra_tools: Additional tools to allow
        **kwargs: Additional options passed to ClaudeAgentOptions

    Returns:
        Dictionary of options for ClaudeAgentOptions

    Example:
        options = create_agent_options(
            AgentType.EXTRACTION,
            "postgresql://localhost/graphbrain"
        )

        # With Claude Agent SDK:
        # from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
        # sdk_options = ClaudeAgentOptions(**options)
        # async with ClaudeSDKClient(options=sdk_options) as client:
        #     ...
    """
    definition = get_agent_definition(agent_type)

    # Build tool list
    tools = list(definition.tools)
    if extra_tools:
        tools.extend(extra_tools)

    # Get system prompt
    prompt = definition.get_prompt(docs_path)

    # Build MCP server config
    # Note: The actual MCP server needs to be created separately
    # This provides the configuration structure
    mcp_config = {
        "graphbrain": {
            "type": "sdk",
            "name": "graphbrain",
            "config": {
                "db_connection": db_connection,
                "domain": domain,
            }
        }
    }

    # Build agent definition for subagent spawning
    agent_defs = {
        agent_type.value: {
            "description": definition.description,
            "prompt": prompt,
            "tools": tools,
            "model": definition.model.value,
        }
    }

    options = {
        "mcp_servers": mcp_config,
        "allowed_tools": tools,
        "system_prompt": {
            "type": "preset",
            "preset": "claude_code",
            "append": prompt,
        },
        "agents": agent_defs,
        **kwargs
    }

    logger.info(f"Created agent options for {agent_type.value} with {len(tools)} tools")
    return options


def create_multi_agent_options(
    db_connection: str,
    domain: str = "default",
    docs_path: Optional[Path] = None,
    primary_agent: AgentType = AgentType.EXTRACTION,
    **kwargs
) -> dict:
    """
    Create options with all agent types available as subagents.

    This allows the primary agent to spawn other agent types
    for specialized tasks.

    Args:
        db_connection: PostgreSQL connection string
        domain: Domain for classification context
        docs_path: Path to agent prompt files
        primary_agent: The primary agent type
        **kwargs: Additional options

    Returns:
        Dictionary of options for ClaudeAgentOptions
    """
    primary_def = get_agent_definition(primary_agent)

    # Collect all tools from all agents
    all_tools = set()
    agent_defs = {}

    for agent_type, definition in AGENT_DEFINITIONS.items():
        tools = list(definition.tools)
        all_tools.update(tools)

        agent_defs[agent_type.value] = {
            "description": definition.description,
            "prompt": definition.get_prompt(docs_path),
            "tools": tools,
            "model": definition.model.value,
        }

    # Build MCP server config
    mcp_config = {
        "graphbrain": {
            "type": "sdk",
            "name": "graphbrain",
            "config": {
                "db_connection": db_connection,
                "domain": domain,
            }
        }
    }

    options = {
        "mcp_servers": mcp_config,
        "allowed_tools": list(all_tools),
        "system_prompt": {
            "type": "preset",
            "preset": "claude_code",
            "append": primary_def.get_prompt(docs_path),
        },
        "agents": agent_defs,
        **kwargs
    }

    logger.info(
        f"Created multi-agent options with primary={primary_agent.value}, "
        f"{len(agent_defs)} agents, {len(all_tools)} total tools"
    )
    return options
