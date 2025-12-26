"""
Validation hooks for agent tool calls.

Provides PreToolUse hooks that validate input before
tool execution, preventing invalid operations.
"""

from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


async def validate_edge_syntax(
    input_data: dict[str, Any],
    tool_use_id: Optional[str],
    context: Any
) -> dict[str, Any]:
    """
    PreToolUse hook to validate edge syntax before adding.

    Validates that edge strings are valid hyperedge notation
    before they're added to the database.

    Args:
        input_data: Tool input containing tool_name and tool_input
        tool_use_id: The tool use ID
        context: Hook context

    Returns:
        Hook result - empty for success, deny for invalid
    """
    tool_name = input_data.get("tool_name", "")

    # Only validate add_edge calls
    if tool_name != "mcp__graphbrain__add_edge":
        return {}

    edge_str = input_data.get("tool_input", {}).get("edge", "")

    if not edge_str:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "Edge parameter is required"
            }
        }

    # Validate syntax by attempting to parse
    try:
        from graphbrain import hedge
        hedge(edge_str)
        logger.debug(f"Edge syntax valid: {edge_str[:50]}...")
        return {}
    except Exception as e:
        logger.warning(f"Invalid edge syntax: {edge_str[:50]}... - {e}")
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"Invalid edge syntax: {e}"
            }
        }


async def validate_pattern_syntax(
    input_data: dict[str, Any],
    tool_use_id: Optional[str],
    context: Any
) -> dict[str, Any]:
    """
    PreToolUse hook to validate pattern syntax before matching.

    Validates that pattern strings are valid pattern notation
    before pattern matching operations.

    Args:
        input_data: Tool input
        tool_use_id: The tool use ID
        context: Hook context

    Returns:
        Hook result
    """
    tool_name = input_data.get("tool_name", "")

    # Only validate pattern_match calls
    if tool_name != "mcp__graphbrain__pattern_match":
        return {}

    pattern_str = input_data.get("tool_input", {}).get("pattern", "")

    if not pattern_str:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "Pattern parameter is required"
            }
        }

    # Validate pattern syntax
    try:
        from graphbrain import hedge
        hedge(pattern_str)
        logger.debug(f"Pattern syntax valid: {pattern_str[:50]}...")
        return {}
    except Exception as e:
        logger.warning(f"Invalid pattern syntax: {pattern_str[:50]}... - {e}")
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"Invalid pattern syntax: {e}"
            }
        }


async def validate_classification_params(
    input_data: dict[str, Any],
    tool_use_id: Optional[str],
    context: Any
) -> dict[str, Any]:
    """
    PreToolUse hook to validate classification parameters.

    Ensures classification thresholds are within valid ranges.

    Args:
        input_data: Tool input
        tool_use_id: The tool use ID
        context: Hook context

    Returns:
        Hook result
    """
    tool_name = input_data.get("tool_name", "")

    # Only validate classification calls
    if tool_name not in [
        "mcp__graphbrain__classify_predicate",
        "mcp__graphbrain__classify_edge"
    ]:
        return {}

    tool_input = input_data.get("tool_input", {})
    threshold = tool_input.get("threshold")

    if threshold is not None:
        if not isinstance(threshold, (int, float)):
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "Threshold must be a number"
                }
            }

        if not (0.0 <= threshold <= 1.0):
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "Threshold must be between 0.0 and 1.0"
                }
            }

    logger.debug(f"Classification params valid for {tool_name}")
    return {}


def create_validation_hooks() -> dict:
    """
    Create the standard set of validation hooks.

    Returns:
        Dictionary of hooks for ClaudeAgentOptions
    """
    return {
        "PreToolUse": [
            {
                "matcher": "mcp__graphbrain__add_edge",
                "hooks": [validate_edge_syntax],
                "timeout": 5000,
            },
            {
                "matcher": "mcp__graphbrain__pattern_match",
                "hooks": [validate_pattern_syntax],
                "timeout": 5000,
            },
            {
                "hooks": [validate_classification_params],
                "timeout": 5000,
            },
        ]
    }
