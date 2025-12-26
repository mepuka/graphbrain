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

__all__ = [
    "AgentType",
    "create_agent_options",
    "get_agent_definition",
    "AgentSession",
    "SessionManager",
    "DecisionLog",
    "DecisionLogger",
]
