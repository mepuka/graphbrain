"""
MCP tools for agent session and decision management.

Provides tools for:
- Creating and managing agent sessions
- Logging agent decisions
- Retrieving session state
"""

import logging
from typing import Optional
from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

from graphbrain.mcp.errors import (
    error_response,
    invalid_input_error,
    not_found_error,
    database_error,
    ErrorCode,
)

logger = logging.getLogger(__name__)


def register_agent_tools(server: FastMCP):
    """
    Register agent management tools with the MCP server.

    Args:
        server: The FastMCP server instance
    """

    def _get_connection():
        """Get database connection from MCP context."""
        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        hg = lifespan_data["hg"]
        return hg._conn

    @server.tool(name="create_agent_session")
    async def create_agent_session(
        agent_type: str,
        domain: str = "default",
        user_id: Optional[str] = None,
        confidence_threshold: float = 0.7,
    ) -> dict:
        """
        Create a new agent session for tracking state and decisions.

        Args:
            agent_type: Type of agent (extraction, query, classification, analysis, feedback)
            domain: Domain for classification context
            user_id: Optional user identifier
            confidence_threshold: Threshold for flagging uncertain results

        Returns:
            Session details including session_id
        """
        logger.info(f"Creating agent session: type={agent_type}, domain={domain}")

        valid_types = ["extraction", "query", "classification", "analysis", "feedback"]
        if agent_type not in valid_types:
            return invalid_input_error(
                f"Invalid agent_type. Must be one of: {valid_types}",
                {"agent_type": agent_type}
            )

        if not 0.0 <= confidence_threshold <= 1.0:
            return invalid_input_error(
                "confidence_threshold must be between 0.0 and 1.0",
                {"confidence_threshold": confidence_threshold}
            )

        try:
            from graphbrain.agents.memory.session import (
                AgentSession,
                AgentType,
                SessionManager,
            )

            conn = _get_connection()
            manager = SessionManager(conn)

            session = AgentSession.create(
                agent_type=AgentType(agent_type),
                user_id=user_id,
                domain=domain,
                confidence_threshold=confidence_threshold,
            )

            manager.create(session)

            logger.info(f"Created session {session.session_id}")
            return {
                "status": "success",
                "session_id": session.session_id,
                "agent_type": agent_type,
                "domain": domain,
                "confidence_threshold": confidence_threshold,
                "created_at": session.started_at.isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return database_error("create_agent_session", e)

    @server.tool(name="set_current_session")
    async def set_current_session(session_id: str) -> dict:
        """
        Set the current active session for automatic context injection.

        Once set, subsequent tool calls can access the current session
        without explicitly passing session_id. This enables automatic
        decision logging and session state tracking.

        Args:
            session_id: The session ID to make active

        Returns:
            Confirmation with session details
        """
        logger.info(f"Setting current session: {session_id}")

        try:
            from graphbrain.agents.memory.session import SessionManager

            conn = _get_connection()
            manager = SessionManager(conn)

            session = manager.get(session_id)
            if not session:
                return not_found_error("session", session_id)

            # Store in lifespan context
            ctx = server.get_context()
            lifespan_data = ctx.request_context.lifespan_context
            lifespan_data["current_session"] = session_id

            logger.info(f"Current session set to: {session_id}")
            return {
                "status": "success",
                "session_id": session_id,
                "agent_type": session.agent_type.value,
                "domain": session.domain,
                "message": "Session is now active. Subsequent calls will use this session automatically.",
            }

        except Exception as e:
            logger.error(f"Failed to set current session: {e}")
            return database_error("set_current_session", e)

    @server.tool(name="get_current_session")
    async def get_current_session() -> dict:
        """
        Get the currently active session.

        Returns the session_id that was set via set_current_session,
        or indicates if no session is currently active.

        Returns:
            Current session information or indication that none is set
        """
        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        current_session = lifespan_data.get("current_session")

        if not current_session:
            return {
                "status": "success",
                "session_id": None,
                "active": False,
                "message": "No session currently active. Use set_current_session to activate one.",
            }

        try:
            from graphbrain.agents.memory.session import SessionManager

            conn = _get_connection()
            manager = SessionManager(conn)

            session = manager.get(current_session)
            if not session:
                # Session was deleted, clear the reference
                lifespan_data["current_session"] = None
                return {
                    "status": "success",
                    "session_id": None,
                    "active": False,
                    "message": "Previously active session no longer exists.",
                }

            return {
                "status": "success",
                "session_id": current_session,
                "active": True,
                "agent_type": session.agent_type.value,
                "domain": session.domain,
                "edges_added": session.edges_added,
                "classifications_made": session.classifications_made,
            }

        except Exception as e:
            logger.error(f"Failed to get current session: {e}")
            return database_error("get_current_session", e)

    @server.tool(name="clear_current_session")
    async def clear_current_session() -> dict:
        """
        Clear the currently active session.

        After clearing, tools will no longer have automatic session context.

        Returns:
            Confirmation of session cleared
        """
        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        previous_session = lifespan_data.get("current_session")
        lifespan_data["current_session"] = None

        logger.info(f"Cleared current session (was: {previous_session})")
        return {
            "status": "success",
            "previous_session": previous_session,
            "message": "Session context cleared.",
        }

    @server.tool(name="get_session_state")
    async def get_session_state(session_id: str) -> dict:
        """
        Retrieve current session state.

        Args:
            session_id: The session ID to retrieve

        Returns:
            Full session state including recent edges and cache
        """
        logger.info(f"Getting session state: {session_id}")

        try:
            from graphbrain.agents.memory.session import SessionManager

            conn = _get_connection()
            manager = SessionManager(conn)

            session = manager.get(session_id)

            if not session:
                return not_found_error("session", session_id)

            return {
                "status": "success",
                "session": session.to_dict(),
            }

        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return database_error("get_session_state", e)

    @server.tool(name="update_session_state")
    async def update_session_state(
        session_id: Optional[str] = None,
        edges_added: Optional[int] = None,
        classifications_made: Optional[int] = None,
        recent_edge: Optional[str] = None,
        classification_cache: Optional[dict] = None,
    ) -> dict:
        """
        Update session state with new activity.

        Args:
            session_id: The session to update (optional, uses current session if not provided)
            edges_added: Increment edges_added counter
            classifications_made: Increment classifications counter
            recent_edge: Add edge to recent edges list
            classification_cache: Add entries to classification cache

        Returns:
            Updated session state
        """
        # Use current session if session_id not provided
        if session_id is None:
            ctx = server.get_context()
            lifespan_data = ctx.request_context.lifespan_context
            session_id = lifespan_data.get("current_session")
            if not session_id:
                return error_response(
                    ErrorCode.MISSING_PARAMETER,
                    "No session_id provided and no current session active."
                )

        logger.info(f"Updating session state: {session_id}")

        try:
            from graphbrain.agents.memory.session import SessionManager

            conn = _get_connection()
            manager = SessionManager(conn)

            session = manager.get(session_id)
            if not session:
                return not_found_error("session", session_id)

            if edges_added is not None:
                session.edges_added += edges_added

            if classifications_made is not None:
                session.classifications_made += classifications_made

            if recent_edge:
                session.add_recent_edge(recent_edge)

            if classification_cache:
                for pred, class_id in classification_cache.items():
                    session.cache_classification(pred, class_id)

            manager.update(session)

            return {
                "status": "success",
                "session_id": session_id,
                "edges_added": session.edges_added,
                "classifications_made": session.classifications_made,
                "recent_edges_count": len(session.recent_edges),
            }

        except Exception as e:
            logger.error(f"Failed to update session: {e}")
            return database_error("update_session_state", e)

    @server.tool(name="log_decision")
    async def log_decision(
        decision_type: str,
        input_data: dict,
        output_data: dict,
        session_id: Optional[str] = None,
        confidence: float = 1.0,
        method: str = "",
        reasoning: str = "",
    ) -> dict:
        """
        Log an agent decision for audit trail.

        Args:
            decision_type: Type of decision (add_edge, classify_predicate, etc.)
            input_data: Input to the decision
            output_data: Output/result of the decision
            session_id: The session making the decision (optional, uses current session if not provided)
            confidence: Confidence score
            method: Method used for the decision
            reasoning: Human-readable reasoning

        Returns:
            Decision log details
        """
        # Use current session if session_id not provided
        if session_id is None:
            ctx = server.get_context()
            lifespan_data = ctx.request_context.lifespan_context
            session_id = lifespan_data.get("current_session")
            if not session_id:
                return error_response(
                    ErrorCode.MISSING_PARAMETER,
                    "No session_id provided and no current session active. "
                    "Either provide session_id or call set_current_session first."
                )

        logger.info(f"Logging decision: session={session_id}, type={decision_type}")

        valid_types = [
            "add_edge", "classify_predicate", "classify_edge",
            "apply_feedback", "flag_review", "query", "pattern_match"
        ]
        if decision_type not in valid_types:
            return invalid_input_error(
                f"Invalid decision_type. Must be one of: {valid_types}",
                {"decision_type": decision_type}
            )

        try:
            from graphbrain.agents.memory.decisions import (
                DecisionLog,
                DecisionType,
                DecisionLogger,
            )

            conn = _get_connection()
            logger_inst = DecisionLogger(conn)

            decision = DecisionLog.create(
                session_id=session_id,
                decision_type=DecisionType(decision_type),
                input_params=input_data,
                output_data=output_data,
                confidence=confidence,
                method=method,
                reasoning=reasoning,
            )

            logger_inst.log(decision)

            logger.info(f"Logged decision {decision.decision_id}")
            return {
                "status": "success",
                "decision_id": decision.decision_id,
                "session_id": session_id,
                "decision_type": decision_type,
                "timestamp": decision.timestamp.isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to log decision: {e}")
            return database_error("log_decision", e)

    @server.tool(name="get_session_decisions")
    async def get_session_decisions(
        session_id: str,
        decision_type: Optional[str] = None,
        limit: int = 50,
    ) -> dict:
        """
        Get decisions for a session.

        Args:
            session_id: The session to query
            decision_type: Optional filter by decision type
            limit: Maximum number of decisions to return

        Returns:
            List of decisions
        """
        logger.info(f"Getting decisions: session={session_id}")

        try:
            from graphbrain.agents.memory.decisions import (
                DecisionType,
                DecisionLogger,
            )

            conn = _get_connection()
            logger_inst = DecisionLogger(conn)

            dt = DecisionType(decision_type) if decision_type else None
            decisions = logger_inst.list_by_session(session_id, dt, limit)

            return {
                "status": "success",
                "session_id": session_id,
                "count": len(decisions),
                "decisions": [d.to_dict() for d in decisions],
            }

        except Exception as e:
            logger.error(f"Failed to get decisions: {e}")
            return database_error("get_session_decisions", e)

    @server.tool(name="decision_stats")
    async def decision_stats(
        session_id: Optional[str] = None,
        since_hours: Optional[int] = None,
    ) -> dict:
        """
        Get decision statistics.

        Args:
            session_id: Optional filter by session
            since_hours: Optional filter by time (last N hours)

        Returns:
            Decision statistics
        """
        logger.info(f"Getting decision stats: session={session_id}")

        try:
            from graphbrain.agents.memory.decisions import DecisionLogger
            from datetime import timedelta

            conn = _get_connection()
            logger_inst = DecisionLogger(conn)

            since = None
            if since_hours:
                since = datetime.now(timezone.utc) - timedelta(hours=since_hours)

            stats = logger_inst.get_stats(session_id, since)

            return {
                "status": "success",
                **stats,
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return database_error("decision_stats", e)

    @server.tool(name="delete_session")
    async def delete_session(session_id: str) -> dict:
        """
        Delete a session and optionally its decisions.

        Args:
            session_id: The session to delete

        Returns:
            Deletion status
        """
        logger.info(f"Deleting session: {session_id}")

        try:
            from graphbrain.agents.memory.session import SessionManager

            conn = _get_connection()
            manager = SessionManager(conn)

            deleted = manager.delete(session_id)

            if not deleted:
                return not_found_error("session", session_id)

            return {
                "status": "success",
                "session_id": session_id,
                "deleted": True,
            }

        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            return database_error("delete_session", e)

    @server.tool(name="list_sessions")
    async def list_sessions(
        domain: Optional[str] = None,
        agent_type: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 50,
    ) -> dict:
        """
        List agent sessions.

        Args:
            domain: Filter by domain
            agent_type: Filter by agent type
            user_id: Filter by user
            limit: Maximum sessions to return

        Returns:
            List of session summaries
        """
        logger.info(f"Listing sessions: domain={domain}, type={agent_type}")

        try:
            from graphbrain.agents.memory.session import (
                AgentType,
                SessionManager,
            )

            conn = _get_connection()
            manager = SessionManager(conn)

            if user_id:
                sessions = manager.list_by_user(user_id, limit)
            elif domain:
                at = AgentType(agent_type) if agent_type else None
                sessions = manager.list_by_domain(domain, at, limit)
            else:
                # List all recent sessions (limited implementation)
                sessions = manager.list_by_domain("default", None, limit)

            return {
                "status": "success",
                "count": len(sessions),
                "sessions": [
                    {
                        "session_id": s.session_id,
                        "agent_type": s.agent_type.value,
                        "domain": s.domain,
                        "user_id": s.user_id,
                        "edges_added": s.edges_added,
                        "classifications_made": s.classifications_made,
                        "last_activity": s.last_activity.isoformat(),
                    }
                    for s in sessions
                ],
            }

        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return database_error("list_sessions", e)

    @server.tool(name="get_cached_classification")
    async def get_cached_classification(
        predicate: str,
        session_id: Optional[str] = None,
    ) -> dict:
        """
        Look up a predicate in the session's classification cache.

        The cache stores classifications made during the session to avoid
        redundant lookups. This is useful for checking if a predicate
        was already classified before making another API call.

        Args:
            predicate: The predicate lemma to look up
            session_id: Session to check (optional, uses current session if not provided)

        Returns:
            Classification info if cached, or cache miss indication
        """
        # Use current session if session_id not provided
        if session_id is None:
            ctx = server.get_context()
            lifespan_data = ctx.request_context.lifespan_context
            session_id = lifespan_data.get("current_session")
            if not session_id:
                return error_response(
                    ErrorCode.MISSING_PARAMETER,
                    "No session_id provided and no current session active."
                )

        try:
            from graphbrain.agents.memory.session import SessionManager

            conn = _get_connection()
            manager = SessionManager(conn)

            session = manager.get(session_id)
            if not session:
                return not_found_error("session", session_id)

            cached_class = session.classification_cache.get(predicate)

            if cached_class:
                return {
                    "status": "success",
                    "cache_hit": True,
                    "predicate": predicate,
                    "class_id": cached_class,
                    "session_id": session_id,
                }
            else:
                return {
                    "status": "success",
                    "cache_hit": False,
                    "predicate": predicate,
                    "session_id": session_id,
                    "message": "Predicate not in session cache",
                }

        except Exception as e:
            logger.error(f"Failed to get cached classification: {e}")
            return database_error("get_cached_classification", e)

    logger.info("Registered 9 agent management tools")
