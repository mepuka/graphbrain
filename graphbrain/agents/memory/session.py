"""
Agent session management.

Provides session state persistence for agent interactions, enabling
context continuity across multiple exchanges.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum
import json
import uuid
import logging

logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    """Types of available agents."""
    EXTRACTION = "extraction"
    QUERY = "query"
    CLASSIFICATION = "classification"
    ANALYSIS = "analysis"
    FEEDBACK = "feedback"


class ReviewMode(str, Enum):
    """How to handle uncertain classifications."""
    FLAG = "flag"       # Flag for optional review
    AUTO = "auto"       # Auto-apply all
    STRICT = "strict"   # Require review for anything < threshold


@dataclass
class AgentSession:
    """
    Session state for agent interactions.

    Tracks context, preferences, and metrics across multiple exchanges
    within a single agent session.
    """
    session_id: str
    agent_type: AgentType
    user_id: Optional[str] = None

    # Context
    domain: str = "default"
    active_patterns: list[str] = field(default_factory=list)
    recent_edges: list[str] = field(default_factory=list)  # Last N edges
    classification_cache: dict[str, str] = field(default_factory=dict)  # predicate -> class

    # Preferences
    confidence_threshold: float = 0.7
    review_mode: ReviewMode = ReviewMode.FLAG
    max_recent_edges: int = 50

    # Metrics
    edges_added: int = 0
    classifications_made: int = 0
    feedback_submitted: int = 0
    queries_executed: int = 0

    # Timestamps
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def create(
        cls,
        agent_type: AgentType,
        user_id: Optional[str] = None,
        domain: str = "default",
        **kwargs
    ) -> "AgentSession":
        """Create a new session with generated ID."""
        return cls(
            session_id=f"sess_{uuid.uuid4().hex[:12]}",
            agent_type=agent_type,
            user_id=user_id,
            domain=domain,
            **kwargs
        )

    def add_recent_edge(self, edge_key: str) -> None:
        """Add an edge to recent edges, maintaining max size."""
        self.recent_edges.append(edge_key)
        if len(self.recent_edges) > self.max_recent_edges:
            self.recent_edges = self.recent_edges[-self.max_recent_edges:]
        self.last_activity = datetime.utcnow()

    def cache_classification(self, predicate: str, class_id: str) -> None:
        """Cache a classification decision."""
        self.classification_cache[predicate] = class_id
        self.last_activity = datetime.utcnow()

    def to_dict(self) -> dict:
        """Serialize session to dictionary."""
        return {
            "session_id": self.session_id,
            "agent_type": self.agent_type.value,
            "user_id": self.user_id,
            "domain": self.domain,
            "active_patterns": self.active_patterns,
            "recent_edges": self.recent_edges,
            "classification_cache": self.classification_cache,
            "confidence_threshold": self.confidence_threshold,
            "review_mode": self.review_mode.value,
            "max_recent_edges": self.max_recent_edges,
            "edges_added": self.edges_added,
            "classifications_made": self.classifications_made,
            "feedback_submitted": self.feedback_submitted,
            "queries_executed": self.queries_executed,
            "started_at": self.started_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentSession":
        """Deserialize session from dictionary."""
        return cls(
            session_id=data["session_id"],
            agent_type=AgentType(data["agent_type"]),
            user_id=data.get("user_id"),
            domain=data.get("domain", "default"),
            active_patterns=data.get("active_patterns", []),
            recent_edges=data.get("recent_edges", []),
            classification_cache=data.get("classification_cache", {}),
            confidence_threshold=data.get("confidence_threshold", 0.7),
            review_mode=ReviewMode(data.get("review_mode", "flag")),
            max_recent_edges=data.get("max_recent_edges", 50),
            edges_added=data.get("edges_added", 0),
            classifications_made=data.get("classifications_made", 0),
            feedback_submitted=data.get("feedback_submitted", 0),
            queries_executed=data.get("queries_executed", 0),
            started_at=datetime.fromisoformat(data["started_at"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
        )


class SessionManager:
    """
    Manages agent session persistence in PostgreSQL.

    Provides CRUD operations for session state, enabling sessions
    to survive across process restarts.
    """

    def __init__(self, connection):
        """
        Initialize session manager.

        Args:
            connection: PostgreSQL connection or connection pool
        """
        self.connection = connection
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create agent_sessions table if it doesn't exist."""
        create_sql = """
        CREATE TABLE IF NOT EXISTS agent_sessions (
            session_id TEXT PRIMARY KEY,
            agent_type TEXT NOT NULL,
            user_id TEXT,
            domain TEXT DEFAULT 'default',
            state JSONB NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_agent_type
            ON agent_sessions(agent_type);
        CREATE INDEX IF NOT EXISTS idx_sessions_user_id
            ON agent_sessions(user_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_domain
            ON agent_sessions(domain);
        """
        try:
            with self.connection.cursor() as cur:
                cur.execute(create_sql)
            self.connection.commit()
            logger.debug("agent_sessions table ensured")
        except Exception as e:
            logger.error(f"Failed to create agent_sessions table: {e}")
            self.connection.rollback()
            raise

    def create(self, session: AgentSession) -> AgentSession:
        """
        Create a new session.

        Args:
            session: The session to create

        Returns:
            The created session
        """
        insert_sql = """
        INSERT INTO agent_sessions (session_id, agent_type, user_id, domain, state)
        VALUES (%s, %s, %s, %s, %s)
        """
        try:
            with self.connection.cursor() as cur:
                cur.execute(insert_sql, (
                    session.session_id,
                    session.agent_type.value,
                    session.user_id,
                    session.domain,
                    json.dumps(session.to_dict())
                ))
            self.connection.commit()
            logger.info(f"Created session {session.session_id}")
            return session
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            self.connection.rollback()
            raise

    def get(self, session_id: str) -> Optional[AgentSession]:
        """
        Get a session by ID.

        Args:
            session_id: The session ID to retrieve

        Returns:
            The session if found, None otherwise
        """
        select_sql = "SELECT state FROM agent_sessions WHERE session_id = %s"
        try:
            with self.connection.cursor() as cur:
                cur.execute(select_sql, (session_id,))
                row = cur.fetchone()
                if row:
                    return AgentSession.from_dict(row[0])
                return None
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            raise

    def update(self, session: AgentSession) -> AgentSession:
        """
        Update an existing session.

        Args:
            session: The session to update

        Returns:
            The updated session
        """
        update_sql = """
        UPDATE agent_sessions
        SET state = %s, updated_at = NOW()
        WHERE session_id = %s
        """
        try:
            with self.connection.cursor() as cur:
                cur.execute(update_sql, (
                    json.dumps(session.to_dict()),
                    session.session_id
                ))
            self.connection.commit()
            logger.debug(f"Updated session {session.session_id}")
            return session
        except Exception as e:
            logger.error(f"Failed to update session: {e}")
            self.connection.rollback()
            raise

    def delete(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: The session ID to delete

        Returns:
            True if deleted, False if not found
        """
        delete_sql = "DELETE FROM agent_sessions WHERE session_id = %s"
        try:
            with self.connection.cursor() as cur:
                cur.execute(delete_sql, (session_id,))
                deleted = cur.rowcount > 0
            self.connection.commit()
            if deleted:
                logger.info(f"Deleted session {session_id}")
            return deleted
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            self.connection.rollback()
            raise

    def list_by_user(self, user_id: str, limit: int = 50) -> list[AgentSession]:
        """
        List sessions for a user.

        Args:
            user_id: The user ID to filter by
            limit: Maximum number of sessions to return

        Returns:
            List of sessions for the user
        """
        select_sql = """
        SELECT state FROM agent_sessions
        WHERE user_id = %s
        ORDER BY updated_at DESC
        LIMIT %s
        """
        try:
            with self.connection.cursor() as cur:
                cur.execute(select_sql, (user_id, limit))
                rows = cur.fetchall()
                return [AgentSession.from_dict(row[0]) for row in rows]
        except Exception as e:
            logger.error(f"Failed to list sessions for user {user_id}: {e}")
            raise

    def list_by_domain(
        self,
        domain: str,
        agent_type: Optional[AgentType] = None,
        limit: int = 50
    ) -> list[AgentSession]:
        """
        List sessions for a domain.

        Args:
            domain: The domain to filter by
            agent_type: Optional agent type filter
            limit: Maximum number of sessions to return

        Returns:
            List of sessions for the domain
        """
        if agent_type:
            select_sql = """
            SELECT state FROM agent_sessions
            WHERE domain = %s AND agent_type = %s
            ORDER BY updated_at DESC
            LIMIT %s
            """
            params = (domain, agent_type.value, limit)
        else:
            select_sql = """
            SELECT state FROM agent_sessions
            WHERE domain = %s
            ORDER BY updated_at DESC
            LIMIT %s
            """
            params = (domain, limit)

        try:
            with self.connection.cursor() as cur:
                cur.execute(select_sql, params)
                rows = cur.fetchall()
                return [AgentSession.from_dict(row[0]) for row in rows]
        except Exception as e:
            logger.error(f"Failed to list sessions for domain {domain}: {e}")
            raise

    def cleanup_old_sessions(self, hours: int = 24) -> int:
        """
        Delete sessions older than specified hours.

        Args:
            hours: Age threshold in hours

        Returns:
            Number of sessions deleted
        """
        delete_sql = """
        DELETE FROM agent_sessions
        WHERE updated_at < NOW() - INTERVAL '%s hours'
        """
        try:
            with self.connection.cursor() as cur:
                cur.execute(delete_sql, (hours,))
                deleted = cur.rowcount
            self.connection.commit()
            logger.info(f"Cleaned up {deleted} old sessions")
            return deleted
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")
            self.connection.rollback()
            raise
