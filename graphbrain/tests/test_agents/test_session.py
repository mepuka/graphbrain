"""Tests for agent session management."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from graphbrain.agents.memory.session import (
    AgentSession,
    AgentType,
    ReviewMode,
    SessionManager,
)


class TestAgentSession:
    """Tests for AgentSession dataclass."""

    def test_create_session(self):
        """Test creating a new session."""
        session = AgentSession.create(
            agent_type=AgentType.EXTRACTION,
            user_id="user123",
            domain="urbanist"
        )

        assert session.session_id.startswith("sess_")
        assert len(session.session_id) == 17  # sess_ + 12 hex chars
        assert session.agent_type == AgentType.EXTRACTION
        assert session.user_id == "user123"
        assert session.domain == "urbanist"
        assert session.confidence_threshold == 0.7
        assert session.review_mode == ReviewMode.FLAG

    def test_session_defaults(self):
        """Test session default values."""
        session = AgentSession.create(agent_type=AgentType.QUERY)

        assert session.user_id is None
        assert session.domain == "default"
        assert session.active_patterns == []
        assert session.recent_edges == []
        assert session.classification_cache == {}
        assert session.edges_added == 0
        assert session.classifications_made == 0

    def test_add_recent_edge(self):
        """Test adding edges to recent list."""
        session = AgentSession.create(
            agent_type=AgentType.EXTRACTION,
            max_recent_edges=3
        )

        session.add_recent_edge("edge1")
        session.add_recent_edge("edge2")
        session.add_recent_edge("edge3")
        session.add_recent_edge("edge4")

        # Should only keep last 3
        assert len(session.recent_edges) == 3
        assert session.recent_edges == ["edge2", "edge3", "edge4"]

    def test_cache_classification(self):
        """Test caching classification decisions."""
        session = AgentSession.create(agent_type=AgentType.CLASSIFICATION)

        session.cache_classification("announce", "claim")
        session.cache_classification("attack", "conflict")

        assert session.classification_cache["announce"] == "claim"
        assert session.classification_cache["attack"] == "conflict"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        session = AgentSession.create(
            agent_type=AgentType.ANALYSIS,
            user_id="user456",
            domain="news"
        )
        session.edges_added = 10
        session.cache_classification("said", "claim")

        data = session.to_dict()

        assert data["session_id"] == session.session_id
        assert data["agent_type"] == "analysis"
        assert data["user_id"] == "user456"
        assert data["domain"] == "news"
        assert data["edges_added"] == 10
        assert data["classification_cache"] == {"said": "claim"}
        assert "started_at" in data
        assert "last_activity" in data

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        original = AgentSession.create(
            agent_type=AgentType.FEEDBACK,
            user_id="user789"
        )
        original.edges_added = 5
        original.feedback_submitted = 3

        data = original.to_dict()
        restored = AgentSession.from_dict(data)

        assert restored.session_id == original.session_id
        assert restored.agent_type == original.agent_type
        assert restored.user_id == original.user_id
        assert restored.edges_added == original.edges_added
        assert restored.feedback_submitted == original.feedback_submitted

    def test_roundtrip_serialization(self):
        """Test that serialization is reversible."""
        session = AgentSession.create(
            agent_type=AgentType.EXTRACTION,
            user_id="test",
            domain="test_domain"
        )
        session.add_recent_edge("edge1")
        session.cache_classification("verb", "action")
        session.edges_added = 42
        session.confidence_threshold = 0.85
        session.review_mode = ReviewMode.STRICT

        data = session.to_dict()
        restored = AgentSession.from_dict(data)

        assert restored.session_id == session.session_id
        assert restored.recent_edges == session.recent_edges
        assert restored.classification_cache == session.classification_cache
        assert restored.edges_added == session.edges_added
        assert restored.confidence_threshold == session.confidence_threshold
        assert restored.review_mode == session.review_mode


class TestSessionManager:
    """Tests for SessionManager with mocked PostgreSQL."""

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

        manager = SessionManager(conn)

        cursor.execute.assert_called()
        conn.commit.assert_called()

    def test_create_session(self, mock_connection):
        """Test creating a session in the database."""
        conn, cursor = mock_connection
        manager = SessionManager(conn)

        session = AgentSession.create(
            agent_type=AgentType.EXTRACTION,
            user_id="user123"
        )

        result = manager.create(session)

        assert result == session
        # Should have called execute twice: once for table, once for insert
        assert cursor.execute.call_count >= 2

    def test_get_session_found(self, mock_connection):
        """Test getting an existing session."""
        conn, cursor = mock_connection

        original = AgentSession.create(agent_type=AgentType.QUERY)
        cursor.fetchone.return_value = (original.to_dict(),)

        manager = SessionManager(conn)
        result = manager.get(original.session_id)

        assert result is not None
        assert result.session_id == original.session_id

    def test_get_session_not_found(self, mock_connection):
        """Test getting a non-existent session."""
        conn, cursor = mock_connection
        cursor.fetchone.return_value = None

        manager = SessionManager(conn)
        result = manager.get("nonexistent_id")

        assert result is None

    def test_update_session(self, mock_connection):
        """Test updating a session."""
        conn, cursor = mock_connection
        manager = SessionManager(conn)

        session = AgentSession.create(agent_type=AgentType.CLASSIFICATION)
        session.edges_added = 100

        result = manager.update(session)

        assert result == session
        conn.commit.assert_called()

    def test_delete_session(self, mock_connection):
        """Test deleting a session."""
        conn, cursor = mock_connection
        cursor.rowcount = 1

        manager = SessionManager(conn)
        result = manager.delete("sess_abc123")

        assert result is True
        conn.commit.assert_called()

    def test_delete_session_not_found(self, mock_connection):
        """Test deleting a non-existent session."""
        conn, cursor = mock_connection
        cursor.rowcount = 0

        manager = SessionManager(conn)
        result = manager.delete("nonexistent")

        assert result is False


class TestAgentType:
    """Tests for AgentType enum."""

    def test_all_agent_types(self):
        """Test that all expected agent types exist."""
        assert AgentType.EXTRACTION.value == "extraction"
        assert AgentType.QUERY.value == "query"
        assert AgentType.CLASSIFICATION.value == "classification"
        assert AgentType.ANALYSIS.value == "analysis"
        assert AgentType.FEEDBACK.value == "feedback"

    def test_agent_type_is_string_enum(self):
        """Test that AgentType values are strings."""
        for agent_type in AgentType:
            assert isinstance(agent_type.value, str)


class TestReviewMode:
    """Tests for ReviewMode enum."""

    def test_all_review_modes(self):
        """Test that all expected review modes exist."""
        assert ReviewMode.FLAG.value == "flag"
        assert ReviewMode.AUTO.value == "auto"
        assert ReviewMode.STRICT.value == "strict"
