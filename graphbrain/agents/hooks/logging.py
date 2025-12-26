"""
Logging hooks for agent audit trail.

Provides PostToolUse hooks that log all tool calls
for audit and debugging purposes.
"""

from typing import Any, Optional, Callable
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


async def log_tool_call(
    input_data: dict[str, Any],
    tool_use_id: Optional[str],
    context: Any
) -> dict[str, Any]:
    """
    PostToolUse hook to log all tool calls.

    Creates an audit trail of all agent tool usage
    for debugging and compliance.

    Args:
        input_data: Tool input and output
        tool_use_id: The tool use ID
        context: Hook context

    Returns:
        Empty dict (logging only, no modification)
    """
    tool_name = input_data.get("tool_name", "unknown")
    tool_input = input_data.get("tool_input", {})
    tool_output = input_data.get("tool_output", {})

    # Extract key info for logging
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "tool_use_id": tool_use_id,
        "tool_name": tool_name,
        "input_summary": _summarize_input(tool_input),
        "output_status": _extract_status(tool_output),
    }

    logger.info(f"Tool call: {json.dumps(log_entry)}")
    return {}


def _summarize_input(tool_input: dict) -> dict:
    """Summarize tool input for logging (avoid huge payloads)."""
    summary = {}

    for key, value in tool_input.items():
        if isinstance(value, str):
            # Truncate long strings
            summary[key] = value[:100] + "..." if len(value) > 100 else value
        elif isinstance(value, (int, float, bool)):
            summary[key] = value
        elif isinstance(value, list):
            summary[key] = f"[list of {len(value)} items]"
        elif isinstance(value, dict):
            summary[key] = f"{{dict with {len(value)} keys}}"
        else:
            summary[key] = str(type(value).__name__)

    return summary


def _extract_status(tool_output: dict) -> str:
    """Extract status from tool output."""
    if isinstance(tool_output, dict):
        return tool_output.get("status", "unknown")
    return "unknown"


def create_audit_hook(
    decision_logger: Optional[Any] = None,
    session_id: Optional[str] = None
) -> Callable:
    """
    Create an audit hook with decision logging.

    Creates a PostToolUse hook that logs to both the standard
    logger and optionally to the DecisionLogger for persistence.

    Args:
        decision_logger: Optional DecisionLogger instance
        session_id: Optional session ID for decision logging

    Returns:
        Async hook function
    """
    async def audit_hook(
        input_data: dict[str, Any],
        tool_use_id: Optional[str],
        context: Any
    ) -> dict[str, Any]:
        """Audit hook with optional decision logging."""

        # Always log to standard logger
        await log_tool_call(input_data, tool_use_id, context)

        # Optionally log to decision logger
        if decision_logger and session_id:
            try:
                from graphbrain.agents.memory.decisions import (
                    DecisionLog,
                    DecisionType,
                    DecisionOutcome,
                )

                tool_name = input_data.get("tool_name", "")
                tool_input = input_data.get("tool_input", {})
                tool_output = input_data.get("tool_output", {})

                # Map tool name to decision type
                decision_type = _map_tool_to_decision_type(tool_name)

                if decision_type:
                    # Determine outcome
                    status = tool_output.get("status", "success") if isinstance(tool_output, dict) else "success"
                    outcome = DecisionOutcome.SUCCESS if status == "success" else DecisionOutcome.FAILED

                    decision = DecisionLog.create(
                        session_id=session_id,
                        decision_type=decision_type,
                        input_params=tool_input,
                        output_data=tool_output if isinstance(tool_output, dict) else {},
                        outcome=outcome,
                        method=tool_name,
                    )

                    decision_logger.log(decision)
                    logger.debug(f"Logged decision {decision.decision_id}")

            except Exception as e:
                logger.warning(f"Failed to log decision: {e}")

        return {}

    return audit_hook


def _map_tool_to_decision_type(tool_name: str) -> Optional[Any]:
    """Map tool name to decision type."""
    from graphbrain.agents.memory.decisions import DecisionType

    mapping = {
        "mcp__graphbrain__add_edge": DecisionType.ADD_EDGE,
        "mcp__graphbrain__classify_predicate": DecisionType.CLASSIFY_PREDICATE,
        "mcp__graphbrain__classify_edge": DecisionType.CLASSIFY_EDGE,
        "mcp__graphbrain__apply_feedback": DecisionType.APPLY_FEEDBACK,
        "mcp__graphbrain__flag_for_review": DecisionType.FLAG_REVIEW,
        "mcp__graphbrain__pattern_match": DecisionType.PATTERN_MATCH,
        "mcp__graphbrain__search_edges": DecisionType.QUERY,
        "mcp__graphbrain__hybrid_search": DecisionType.QUERY,
        "mcp__graphbrain__bm25_search": DecisionType.QUERY,
    }

    return mapping.get(tool_name)


def create_logging_hooks(
    decision_logger: Optional[Any] = None,
    session_id: Optional[str] = None
) -> dict:
    """
    Create the standard set of logging hooks.

    Args:
        decision_logger: Optional DecisionLogger for persistence
        session_id: Optional session ID

    Returns:
        Dictionary of hooks for ClaudeAgentOptions
    """
    audit_hook = create_audit_hook(decision_logger, session_id)

    return {
        "PostToolUse": [
            {
                "hooks": [audit_hook],
                "timeout": 10000,
            }
        ]
    }
