"""Tests for agent hooks."""

import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from graphbrain.agents.hooks.validation import (
    validate_edge_syntax,
    validate_pattern_syntax,
    validate_classification_params,
    create_validation_hooks,
)
from graphbrain.agents.hooks.logging import (
    log_tool_call,
    create_audit_hook,
    create_logging_hooks,
    _summarize_input,
    _extract_status,
)


def run_async(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestValidateEdgeSyntax:
    """Tests for edge syntax validation hook."""

    def test_non_edge_tool_passes(self):
        """Test that non-add_edge tools pass through."""
        input_data = {
            "tool_name": "mcp__graphbrain__search_edges",
            "tool_input": {"query": "test"}
        }

        result = run_async(validate_edge_syntax(input_data, None, None))
        assert result == {}

    def test_missing_edge_denied(self):
        """Test that missing edge parameter is denied."""
        input_data = {
            "tool_name": "mcp__graphbrain__add_edge",
            "tool_input": {}
        }

        result = run_async(validate_edge_syntax(input_data, None, None))
        assert "hookSpecificOutput" in result
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert "required" in result["hookSpecificOutput"]["permissionDecisionReason"]

    def test_valid_edge_passes(self):
        """Test that valid edge syntax passes."""
        input_data = {
            "tool_name": "mcp__graphbrain__add_edge",
            "tool_input": {"edge": "(is/Pd sky/C blue/C)"}
        }

        result = run_async(validate_edge_syntax(input_data, None, None))
        assert result == {}

    def test_invalid_edge_denied(self):
        """Test that invalid edge syntax is denied."""
        input_data = {
            "tool_name": "mcp__graphbrain__add_edge",
            "tool_input": {"edge": "(broken syntax"}
        }

        result = run_async(validate_edge_syntax(input_data, None, None))
        assert "hookSpecificOutput" in result
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


class TestValidatePatternSyntax:
    """Tests for pattern syntax validation hook."""

    def test_non_pattern_tool_passes(self):
        """Test that non-pattern_match tools pass through."""
        input_data = {
            "tool_name": "mcp__graphbrain__add_edge",
            "tool_input": {"edge": "test"}
        }

        result = run_async(validate_pattern_syntax(input_data, None, None))
        assert result == {}

    def test_missing_pattern_denied(self):
        """Test that missing pattern parameter is denied."""
        input_data = {
            "tool_name": "mcp__graphbrain__pattern_match",
            "tool_input": {}
        }

        result = run_async(validate_pattern_syntax(input_data, None, None))
        assert "hookSpecificOutput" in result
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_valid_pattern_passes(self):
        """Test that valid pattern syntax passes."""
        input_data = {
            "tool_name": "mcp__graphbrain__pattern_match",
            "tool_input": {"pattern": "(*/P * *)"}
        }

        result = run_async(validate_pattern_syntax(input_data, None, None))
        assert result == {}


class TestValidateClassificationParams:
    """Tests for classification parameter validation hook."""

    def test_non_classification_tool_passes(self):
        """Test that non-classification tools pass through."""
        input_data = {
            "tool_name": "mcp__graphbrain__add_edge",
            "tool_input": {}
        }

        result = run_async(validate_classification_params(input_data, None, None))
        assert result == {}

    def test_valid_threshold_passes(self):
        """Test that valid threshold passes."""
        input_data = {
            "tool_name": "mcp__graphbrain__classify_predicate",
            "tool_input": {"predicate": "test", "threshold": 0.8}
        }

        result = run_async(validate_classification_params(input_data, None, None))
        assert result == {}

    def test_no_threshold_passes(self):
        """Test that missing threshold passes (uses default)."""
        input_data = {
            "tool_name": "mcp__graphbrain__classify_predicate",
            "tool_input": {"predicate": "test"}
        }

        result = run_async(validate_classification_params(input_data, None, None))
        assert result == {}

    def test_invalid_threshold_type_denied(self):
        """Test that non-numeric threshold is denied."""
        input_data = {
            "tool_name": "mcp__graphbrain__classify_predicate",
            "tool_input": {"predicate": "test", "threshold": "high"}
        }

        result = run_async(validate_classification_params(input_data, None, None))
        assert "hookSpecificOutput" in result
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert "number" in result["hookSpecificOutput"]["permissionDecisionReason"]

    def test_threshold_out_of_range_denied(self):
        """Test that out-of-range threshold is denied."""
        input_data = {
            "tool_name": "mcp__graphbrain__classify_edge",
            "tool_input": {"edge": "test", "threshold": 1.5}
        }

        result = run_async(validate_classification_params(input_data, None, None))
        assert "hookSpecificOutput" in result
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


class TestCreateValidationHooks:
    """Tests for create_validation_hooks function."""

    def test_returns_pretooluse_hooks(self):
        """Test that PreToolUse hooks are returned."""
        hooks = create_validation_hooks()

        assert "PreToolUse" in hooks
        assert len(hooks["PreToolUse"]) >= 3  # edge, pattern, classification


class TestLogToolCall:
    """Tests for tool call logging hook."""

    def test_logs_tool_call(self):
        """Test that tool calls are logged."""
        input_data = {
            "tool_name": "mcp__graphbrain__add_edge",
            "tool_input": {"edge": "(test/P)"},
            "tool_output": {"status": "success"}
        }

        # Should not raise
        result = run_async(log_tool_call(input_data, "tool123", None))
        assert result == {}


class TestSummarizeInput:
    """Tests for input summarization."""

    def test_string_truncation(self):
        """Test that long strings are truncated."""
        input_data = {"text": "x" * 200}
        summary = _summarize_input(input_data)

        assert len(summary["text"]) <= 103  # 100 + "..."

    def test_preserves_primitives(self):
        """Test that primitives are preserved."""
        input_data = {"count": 42, "enabled": True, "rate": 0.5}
        summary = _summarize_input(input_data)

        assert summary["count"] == 42
        assert summary["enabled"] is True
        assert summary["rate"] == 0.5

    def test_summarizes_collections(self):
        """Test that collections are summarized."""
        input_data = {
            "items": [1, 2, 3, 4, 5],
            "config": {"a": 1, "b": 2}
        }
        summary = _summarize_input(input_data)

        assert "5 items" in summary["items"]
        assert "2 keys" in summary["config"]


class TestExtractStatus:
    """Tests for status extraction."""

    def test_extracts_status(self):
        """Test extracting status from output."""
        output = {"status": "success", "data": {}}
        assert _extract_status(output) == "success"

    def test_unknown_for_missing(self):
        """Test unknown for missing status."""
        output = {"data": {}}
        assert _extract_status(output) == "unknown"

    def test_unknown_for_non_dict(self):
        """Test unknown for non-dict output."""
        assert _extract_status("string") == "unknown"


class TestCreateAuditHook:
    """Tests for create_audit_hook function."""

    def test_creates_hook_without_logger(self):
        """Test creating hook without decision logger."""
        hook = create_audit_hook()
        assert callable(hook)

    def test_creates_hook_with_logger(self):
        """Test creating hook with decision logger."""
        mock_logger = MagicMock()
        hook = create_audit_hook(mock_logger, "sess_test")
        assert callable(hook)

    def test_hook_logs_decision(self):
        """Test that hook logs to decision logger."""
        mock_logger = MagicMock()
        mock_logger.log = MagicMock()

        hook = create_audit_hook(mock_logger, "sess_test")

        input_data = {
            "tool_name": "mcp__graphbrain__add_edge",
            "tool_input": {"edge": "(test/P)"},
            "tool_output": {"status": "success"}
        }

        run_async(hook(input_data, "tool123", None))

        # Should have attempted to log
        mock_logger.log.assert_called()


class TestCreateLoggingHooks:
    """Tests for create_logging_hooks function."""

    def test_returns_posttooluse_hooks(self):
        """Test that PostToolUse hooks are returned."""
        hooks = create_logging_hooks()

        assert "PostToolUse" in hooks
        assert len(hooks["PostToolUse"]) >= 1
