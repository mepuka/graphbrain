"""Tests for agent decision logging."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from graphbrain.agents.memory.decisions import (
    DecisionLog,
    DecisionType,
    DecisionOutcome,
    DecisionLogger,
)


class TestDecisionLog:
    """Tests for DecisionLog dataclass."""

    def test_create_decision(self):
        """Test creating a new decision log."""
        decision = DecisionLog.create(
            session_id="sess_abc123",
            decision_type=DecisionType.ADD_EDGE,
            input_edge="(is/Pd sky/C blue/C)",
            confidence=0.95,
        )

        assert decision.decision_id.startswith("dec_")
        assert len(decision.decision_id) == 16  # dec_ + 12 hex chars
        assert decision.session_id == "sess_abc123"
        assert decision.decision_type == DecisionType.ADD_EDGE
        assert decision.input_edge == "(is/Pd sky/C blue/C)"
        assert decision.confidence == 0.95
        assert decision.outcome == DecisionOutcome.SUCCESS

    def test_decision_defaults(self):
        """Test decision default values."""
        decision = DecisionLog.create(
            session_id="sess_xyz",
            decision_type=DecisionType.QUERY
        )

        assert decision.input_text is None
        assert decision.input_edge is None
        assert decision.input_pattern is None
        assert decision.input_params == {}
        assert decision.method == ""
        assert decision.intermediate_results == {}
        assert decision.output_class is None
        assert decision.output_edge is None
        assert decision.output_data == {}
        assert decision.reasoning == ""

    def test_classification_decision(self):
        """Test creating a classification decision."""
        decision = DecisionLog.create(
            session_id="sess_class",
            decision_type=DecisionType.CLASSIFY_PREDICATE,
            input_text="announce",
            method="predicate_bank",
            output_class="claim",
            confidence=0.92,
            reasoning="Seed predicate in claim class",
            intermediate_results={
                "similar": ["declare", "state"],
                "scores": [0.95, 0.88],
            }
        )

        assert decision.decision_type == DecisionType.CLASSIFY_PREDICATE
        assert decision.input_text == "announce"
        assert decision.method == "predicate_bank"
        assert decision.output_class == "claim"
        assert decision.confidence == 0.92
        assert decision.intermediate_results["similar"] == ["declare", "state"]

    def test_to_dict(self):
        """Test serialization to dictionary."""
        decision = DecisionLog.create(
            session_id="sess_serialize",
            decision_type=DecisionType.APPLY_FEEDBACK,
            confidence=0.99,
            output_data={"applied": True},
        )

        data = decision.to_dict()

        assert data["decision_id"] == decision.decision_id
        assert data["session_id"] == "sess_serialize"
        assert data["decision_type"] == "apply_feedback"
        assert data["confidence"] == 0.99
        assert data["output_data"]["applied"] is True
        assert "timestamp" in data

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        original = DecisionLog.create(
            session_id="sess_deserialize",
            decision_type=DecisionType.FLAG_REVIEW,
            input_text="uncertain_predicate",
            confidence=0.65,
            outcome=DecisionOutcome.FLAGGED,
        )

        data = original.to_dict()
        restored = DecisionLog.from_dict(data)

        assert restored.decision_id == original.decision_id
        assert restored.session_id == original.session_id
        assert restored.decision_type == original.decision_type
        assert restored.input_text == original.input_text
        assert restored.confidence == original.confidence
        assert restored.outcome == original.outcome

    def test_roundtrip_serialization(self):
        """Test that serialization is reversible."""
        decision = DecisionLog.create(
            session_id="sess_roundtrip",
            decision_type=DecisionType.CLASSIFY_EDGE,
            input_edge="(says/Pd mary/C hello/C)",
            input_params={"threshold": 0.8},
            method="hybrid",
            intermediate_results={"bm25": 0.7, "semantic": 0.9},
            output_class="claim",
            output_data={"edge_classifications": []},
            outcome=DecisionOutcome.SUCCESS,
            confidence=0.85,
            model_version="v1.0",
            tool_versions={"graphbrain": "1.0.0"},
            reasoning="High semantic similarity to claim class",
        )

        data = decision.to_dict()
        restored = DecisionLog.from_dict(data)

        assert restored.decision_id == decision.decision_id
        assert restored.input_params == decision.input_params
        assert restored.method == decision.method
        assert restored.intermediate_results == decision.intermediate_results
        assert restored.output_class == decision.output_class
        assert restored.reasoning == decision.reasoning


class TestDecisionLogger:
    """Tests for DecisionLogger with mocked PostgreSQL."""

    @pytest.fixture
    def mock_connection(self):
        """Create a mock PostgreSQL connection."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        return conn, cursor

    def test_ensure_table_called_on_init(self, mock_connection):
        """Test that table is created on initialization."""
        conn, cursor = mock_connection

        logger = DecisionLogger(conn)

        cursor.execute.assert_called()
        conn.commit.assert_called()

    def test_log_decision(self, mock_connection):
        """Test logging a decision."""
        conn, cursor = mock_connection
        logger = DecisionLogger(conn)

        decision = DecisionLog.create(
            session_id="sess_log",
            decision_type=DecisionType.ADD_EDGE,
            input_edge="(test/P edge/C)",
            confidence=0.9,
        )

        result = logger.log(decision)

        assert result == decision
        assert cursor.execute.call_count >= 2  # table + insert
        conn.commit.assert_called()

    def test_get_decision_found(self, mock_connection):
        """Test getting an existing decision."""
        conn, cursor = mock_connection

        # Mock the database row
        cursor.fetchone.return_value = (
            "dec_test123",
            "sess_test",
            "add_edge",
            datetime.utcnow(),
            {"text": None, "edge": "(test/P)", "pattern": None, "params": {}},
            {"class": None, "edge": "(test/P)", "data": {}, "intermediate": {}, "model_version": "", "tool_versions": {}},
            0.95,
            "success",
            "add_edge",
            "Test reasoning",
        )

        logger = DecisionLogger(conn)
        result = logger.get("dec_test123")

        assert result is not None
        assert result.decision_id == "dec_test123"
        assert result.confidence == 0.95

    def test_get_decision_not_found(self, mock_connection):
        """Test getting a non-existent decision."""
        conn, cursor = mock_connection
        cursor.fetchone.return_value = None

        logger = DecisionLogger(conn)
        result = logger.get("nonexistent")

        assert result is None

    def test_get_stats(self, mock_connection):
        """Test getting decision statistics."""
        conn, cursor = mock_connection

        # Mock stats query result
        cursor.fetchone.return_value = (100, 85, 5, 10, 0.82, 5, 4)
        # Mock type distribution
        cursor.fetchall.return_value = [
            ("add_edge", 50),
            ("classify_predicate", 30),
            ("query", 20),
        ]

        logger = DecisionLogger(conn)
        stats = logger.get_stats()

        assert stats["total"] == 100
        assert stats["successful"] == 85
        assert stats["failed"] == 5
        assert stats["flagged"] == 10
        assert stats["avg_confidence"] == 0.82
        assert stats["by_type"]["add_edge"] == 50

    def test_cleanup_old_decisions(self, mock_connection):
        """Test cleaning up old decisions."""
        conn, cursor = mock_connection
        cursor.rowcount = 50

        logger = DecisionLogger(conn)
        deleted = logger.cleanup_old_decisions(days=30)

        assert deleted == 50
        conn.commit.assert_called()


class TestDecisionType:
    """Tests for DecisionType enum."""

    def test_all_decision_types(self):
        """Test that all expected decision types exist."""
        assert DecisionType.ADD_EDGE.value == "add_edge"
        assert DecisionType.CLASSIFY_PREDICATE.value == "classify_predicate"
        assert DecisionType.CLASSIFY_EDGE.value == "classify_edge"
        assert DecisionType.APPLY_FEEDBACK.value == "apply_feedback"
        assert DecisionType.FLAG_REVIEW.value == "flag_review"
        assert DecisionType.QUERY.value == "query"
        assert DecisionType.PATTERN_MATCH.value == "pattern_match"


class TestDecisionOutcome:
    """Tests for DecisionOutcome enum."""

    def test_all_outcomes(self):
        """Test that all expected outcomes exist."""
        assert DecisionOutcome.SUCCESS.value == "success"
        assert DecisionOutcome.FAILED.value == "failed"
        assert DecisionOutcome.FLAGGED.value == "flagged"
        assert DecisionOutcome.SKIPPED.value == "skipped"
