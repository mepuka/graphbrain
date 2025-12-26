"""
Agent hooks for validation and logging.

Provides PreToolUse and PostToolUse hooks for:
- Input validation (edge syntax, parameters)
- Audit logging (all tool calls)
- Safety checks (dangerous operations)
"""

from graphbrain.agents.hooks.validation import (
    validate_edge_syntax,
    validate_pattern_syntax,
    validate_classification_params,
)
from graphbrain.agents.hooks.logging import (
    log_tool_call,
    create_audit_hook,
)

__all__ = [
    "validate_edge_syntax",
    "validate_pattern_syntax",
    "validate_classification_params",
    "log_tool_call",
    "create_audit_hook",
]
