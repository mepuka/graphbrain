"""
Agent decision logging.

Provides an audit trail for all agent decisions, enabling
reproducibility and quality analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
from enum import Enum
import json
import uuid
import logging

logger = logging.getLogger(__name__)


class DecisionType(str, Enum):
    """Types of decisions agents can make."""
    ADD_EDGE = "add_edge"
    CLASSIFY_PREDICATE = "classify_predicate"
    CLASSIFY_EDGE = "classify_edge"
    APPLY_FEEDBACK = "apply_feedback"
    FLAG_REVIEW = "flag_review"
    QUERY = "query"
    PATTERN_MATCH = "pattern_match"


class DecisionOutcome(str, Enum):
    """Outcome of a decision."""
    SUCCESS = "success"
    FAILED = "failed"
    FLAGGED = "flagged"
    SKIPPED = "skipped"


@dataclass
class DecisionLog:
    """
    Audit log entry for an agent decision.

    Captures full context for reproducibility:
    - What was the input?
    - What method was used?
    - What was the output?
    - How confident is the decision?
    """
    decision_id: str
    session_id: str
    decision_type: DecisionType
    timestamp: datetime

    # Input context
    input_text: Optional[str] = None
    input_edge: Optional[str] = None
    input_pattern: Optional[str] = None
    input_params: dict = field(default_factory=dict)

    # Processing
    method: str = ""  # e.g., "predicate_bank", "semantic", "pattern", "hybrid"
    intermediate_results: dict = field(default_factory=dict)

    # Output
    output_class: Optional[str] = None
    output_edge: Optional[str] = None
    output_data: dict = field(default_factory=dict)
    outcome: DecisionOutcome = DecisionOutcome.SUCCESS

    # Confidence and audit
    confidence: float = 1.0
    model_version: str = ""
    tool_versions: dict = field(default_factory=dict)
    reasoning: str = ""

    @classmethod
    def create(
        cls,
        session_id: str,
        decision_type: DecisionType,
        **kwargs
    ) -> "DecisionLog":
        """Create a new decision log entry."""
        return cls(
            decision_id=f"dec_{uuid.uuid4().hex[:12]}",
            session_id=session_id,
            decision_type=decision_type,
            timestamp=datetime.utcnow(),
            **kwargs
        )

    def to_dict(self) -> dict:
        """Serialize decision log to dictionary."""
        return {
            "decision_id": self.decision_id,
            "session_id": self.session_id,
            "decision_type": self.decision_type.value,
            "timestamp": self.timestamp.isoformat(),
            "input_text": self.input_text,
            "input_edge": self.input_edge,
            "input_pattern": self.input_pattern,
            "input_params": self.input_params,
            "method": self.method,
            "intermediate_results": self.intermediate_results,
            "output_class": self.output_class,
            "output_edge": self.output_edge,
            "output_data": self.output_data,
            "outcome": self.outcome.value,
            "confidence": self.confidence,
            "model_version": self.model_version,
            "tool_versions": self.tool_versions,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DecisionLog":
        """Deserialize decision log from dictionary."""
        return cls(
            decision_id=data["decision_id"],
            session_id=data["session_id"],
            decision_type=DecisionType(data["decision_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            input_text=data.get("input_text"),
            input_edge=data.get("input_edge"),
            input_pattern=data.get("input_pattern"),
            input_params=data.get("input_params", {}),
            method=data.get("method", ""),
            intermediate_results=data.get("intermediate_results", {}),
            output_class=data.get("output_class"),
            output_edge=data.get("output_edge"),
            output_data=data.get("output_data", {}),
            outcome=DecisionOutcome(data.get("outcome", "success")),
            confidence=data.get("confidence", 1.0),
            model_version=data.get("model_version", ""),
            tool_versions=data.get("tool_versions", {}),
            reasoning=data.get("reasoning", ""),
        )


class DecisionLogger:
    """
    Logs agent decisions to PostgreSQL for audit trail.

    Enables:
    - Full reproducibility of agent decisions
    - Quality analysis over time
    - Debugging of incorrect decisions
    - Compliance and audit requirements
    """

    def __init__(self, connection):
        """
        Initialize decision logger.

        Args:
            connection: PostgreSQL connection or connection pool
        """
        self.connection = connection
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create agent_decisions table if it doesn't exist."""
        create_sql = """
        CREATE TABLE IF NOT EXISTS agent_decisions (
            decision_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            decision_type TEXT NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            input_data JSONB,
            output_data JSONB,
            confidence REAL,
            outcome TEXT DEFAULT 'success',
            method TEXT,
            reasoning TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_decisions_session
            ON agent_decisions(session_id);
        CREATE INDEX IF NOT EXISTS idx_decisions_type
            ON agent_decisions(decision_type);
        CREATE INDEX IF NOT EXISTS idx_decisions_timestamp
            ON agent_decisions(timestamp);
        CREATE INDEX IF NOT EXISTS idx_decisions_outcome
            ON agent_decisions(outcome);
        CREATE INDEX IF NOT EXISTS idx_decisions_confidence
            ON agent_decisions(confidence);
        """
        try:
            with self.connection.cursor() as cur:
                cur.execute(create_sql)
            self.connection.commit()
            logger.debug("agent_decisions table ensured")
        except Exception as e:
            logger.error(f"Failed to create agent_decisions table: {e}")
            self.connection.rollback()
            raise

    def log(self, decision: DecisionLog) -> DecisionLog:
        """
        Log a decision.

        Args:
            decision: The decision to log

        Returns:
            The logged decision
        """
        insert_sql = """
        INSERT INTO agent_decisions (
            decision_id, session_id, decision_type, timestamp,
            input_data, output_data, confidence, outcome, method, reasoning
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        # Combine input fields into single JSONB
        input_data = {
            "text": decision.input_text,
            "edge": decision.input_edge,
            "pattern": decision.input_pattern,
            "params": decision.input_params,
        }

        # Combine output fields into single JSONB
        output_data = {
            "class": decision.output_class,
            "edge": decision.output_edge,
            "data": decision.output_data,
            "intermediate": decision.intermediate_results,
            "model_version": decision.model_version,
            "tool_versions": decision.tool_versions,
        }

        try:
            with self.connection.cursor() as cur:
                cur.execute(insert_sql, (
                    decision.decision_id,
                    decision.session_id,
                    decision.decision_type.value,
                    decision.timestamp,
                    json.dumps(input_data),
                    json.dumps(output_data),
                    decision.confidence,
                    decision.outcome.value,
                    decision.method,
                    decision.reasoning,
                ))
            self.connection.commit()
            logger.debug(f"Logged decision {decision.decision_id}")
            return decision
        except Exception as e:
            logger.error(f"Failed to log decision: {e}")
            self.connection.rollback()
            raise

    def get(self, decision_id: str) -> Optional[DecisionLog]:
        """
        Get a decision by ID.

        Args:
            decision_id: The decision ID to retrieve

        Returns:
            The decision if found, None otherwise
        """
        select_sql = """
        SELECT decision_id, session_id, decision_type, timestamp,
               input_data, output_data, confidence, outcome, method, reasoning
        FROM agent_decisions
        WHERE decision_id = %s
        """
        try:
            with self.connection.cursor() as cur:
                cur.execute(select_sql, (decision_id,))
                row = cur.fetchone()
                if row:
                    return self._row_to_decision(row)
                return None
        except Exception as e:
            logger.error(f"Failed to get decision {decision_id}: {e}")
            raise

    def _row_to_decision(self, row: tuple) -> DecisionLog:
        """Convert database row to DecisionLog."""
        (decision_id, session_id, decision_type, timestamp,
         input_data, output_data, confidence, outcome, method, reasoning) = row

        return DecisionLog(
            decision_id=decision_id,
            session_id=session_id,
            decision_type=DecisionType(decision_type),
            timestamp=timestamp,
            input_text=input_data.get("text") if input_data else None,
            input_edge=input_data.get("edge") if input_data else None,
            input_pattern=input_data.get("pattern") if input_data else None,
            input_params=input_data.get("params", {}) if input_data else {},
            method=method or "",
            intermediate_results=output_data.get("intermediate", {}) if output_data else {},
            output_class=output_data.get("class") if output_data else None,
            output_edge=output_data.get("edge") if output_data else None,
            output_data=output_data.get("data", {}) if output_data else {},
            outcome=DecisionOutcome(outcome) if outcome else DecisionOutcome.SUCCESS,
            confidence=confidence or 1.0,
            model_version=output_data.get("model_version", "") if output_data else "",
            tool_versions=output_data.get("tool_versions", {}) if output_data else {},
            reasoning=reasoning or "",
        )

    def list_by_session(
        self,
        session_id: str,
        decision_type: Optional[DecisionType] = None,
        limit: int = 100
    ) -> list[DecisionLog]:
        """
        List decisions for a session.

        Args:
            session_id: The session ID to filter by
            decision_type: Optional decision type filter
            limit: Maximum number of decisions to return

        Returns:
            List of decisions for the session
        """
        if decision_type:
            select_sql = """
            SELECT decision_id, session_id, decision_type, timestamp,
                   input_data, output_data, confidence, outcome, method, reasoning
            FROM agent_decisions
            WHERE session_id = %s AND decision_type = %s
            ORDER BY timestamp DESC
            LIMIT %s
            """
            params = (session_id, decision_type.value, limit)
        else:
            select_sql = """
            SELECT decision_id, session_id, decision_type, timestamp,
                   input_data, output_data, confidence, outcome, method, reasoning
            FROM agent_decisions
            WHERE session_id = %s
            ORDER BY timestamp DESC
            LIMIT %s
            """
            params = (session_id, limit)

        try:
            with self.connection.cursor() as cur:
                cur.execute(select_sql, params)
                rows = cur.fetchall()
                return [self._row_to_decision(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to list decisions for session {session_id}: {e}")
            raise

    def get_stats(
        self,
        session_id: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> dict:
        """
        Get decision statistics.

        Args:
            session_id: Optional session filter
            since: Optional time filter

        Returns:
            Statistics dictionary
        """
        where_clauses = []
        params = []

        if session_id:
            where_clauses.append("session_id = %s")
            params.append(session_id)

        if since:
            where_clauses.append("timestamp >= %s")
            params.append(since)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        stats_sql = f"""
        SELECT
            COUNT(*) as total,
            COUNT(CASE WHEN outcome = 'success' THEN 1 END) as successful,
            COUNT(CASE WHEN outcome = 'failed' THEN 1 END) as failed,
            COUNT(CASE WHEN outcome = 'flagged' THEN 1 END) as flagged,
            AVG(confidence) as avg_confidence,
            COUNT(DISTINCT session_id) as sessions,
            COUNT(DISTINCT decision_type) as decision_types
        FROM agent_decisions
        {where_sql}
        """

        type_sql = f"""
        SELECT decision_type, COUNT(*) as count
        FROM agent_decisions
        {where_sql}
        GROUP BY decision_type
        ORDER BY count DESC
        """

        try:
            with self.connection.cursor() as cur:
                cur.execute(stats_sql, params)
                row = cur.fetchone()

                cur.execute(type_sql, params)
                type_rows = cur.fetchall()

            return {
                "total": row[0] or 0,
                "successful": row[1] or 0,
                "failed": row[2] or 0,
                "flagged": row[3] or 0,
                "avg_confidence": float(row[4]) if row[4] else 0.0,
                "sessions": row[5] or 0,
                "by_type": {r[0]: r[1] for r in type_rows},
            }
        except Exception as e:
            logger.error(f"Failed to get decision stats: {e}")
            raise

    def cleanup_old_decisions(self, days: int = 90) -> int:
        """
        Delete decisions older than specified days.

        Args:
            days: Age threshold in days

        Returns:
            Number of decisions deleted
        """
        delete_sql = """
        DELETE FROM agent_decisions
        WHERE timestamp < NOW() - INTERVAL '%s days'
        """
        try:
            with self.connection.cursor() as cur:
                cur.execute(delete_sql, (days,))
                deleted = cur.rowcount
            self.connection.commit()
            logger.info(f"Cleaned up {deleted} old decisions")
            return deleted
        except Exception as e:
            logger.error(f"Failed to cleanup old decisions: {e}")
            self.connection.rollback()
            raise
