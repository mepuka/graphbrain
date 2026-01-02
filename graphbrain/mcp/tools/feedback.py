"""Feedback tools for MCP server.

Provides tools for human-in-the-loop feedback on classifications.
"""

import logging
import uuid
from typing import Optional
from datetime import datetime

from mcp.server.fastmcp import FastMCP

from graphbrain.mcp.errors import (
    not_found_error,
)
from graphbrain.mcp.utils import to_isoformat

logger = logging.getLogger(__name__)


def register_feedback_tools(server: FastMCP):
    """Register feedback tools with the MCP server."""

    @server.tool(
        name="submit_feedback",
        description="""
Submit feedback on a classification decision.

Used for human-in-the-loop correction of classifications. Feedback
is stored for later processing and can be used to improve the
classification system.

Args:
    predicate: The predicate that was classified
    original_class: The original (possibly incorrect) classification
    correct_class: The correct classification
    confidence_adjustment: Optional adjustment to suggest (-1.0 to 1.0)
    reviewer_id: Optional identifier for the reviewer

Returns:
  - review_id: unique ID for tracking
  - status: "pending"
  - message: confirmation message
  - predicate: the predicate
  - correction: summary of the correction
""",
    )
    async def submit_feedback(
        predicate: str,
        original_class: str,
        correct_class: str,
        confidence_adjustment: Optional[float] = None,
        reviewer_id: Optional[str] = None,
    ) -> dict:
        """Submit classification feedback."""
        logger.debug(f"submit_feedback: predicate='{predicate}', {original_class} -> {correct_class}")

        from graphbrain.classification.models import ClassificationFeedback

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        repo = lifespan_data["repo"]

        # Generate review ID
        review_id = f"fb_{uuid.uuid4().hex[:8]}"

        feedback = ClassificationFeedback(
            review_id=review_id,
            predicate=predicate,
            original_class=original_class,
            correct_class=correct_class,
            confidence_adjustment=confidence_adjustment,
            reviewer_id=reviewer_id,
            status="pending",
            created_at=datetime.now(),
        )
        repo.save_feedback(feedback)

        logger.info(f"submit_feedback: created {review_id} for '{predicate}'")
        return {
            "status": "success",
            "review_id": review_id,
            "feedback_status": "pending",
            "message": "Feedback submitted successfully",
            "predicate": predicate,
            "correction": f"{original_class} -> {correct_class}",
        }

    @server.tool(
        name="get_pending_reviews",
        description="""
Get pending feedback items awaiting review.

Returns a list of feedback entries that haven't been applied yet.
Use this to review and apply corrections.

Args:
    limit: Maximum items to return (default 50)

Returns:
  - reviews: list of pending feedback items
  - total: number of reviews
""",
    )
    async def get_pending_reviews(
        limit: int = 50,
    ) -> dict:
        """Get pending feedback reviews."""
        logger.debug(f"get_pending_reviews: limit={limit}")

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        repo = lifespan_data["repo"]

        reviews = []
        for feedback in repo.get_pending_feedback(limit=limit):
            reviews.append({
                "review_id": feedback.review_id,
                "predicate": feedback.predicate,
                "original_class": feedback.original_class,
                "correct_class": feedback.correct_class,
                "confidence_adjustment": feedback.confidence_adjustment,
                "reviewer_id": feedback.reviewer_id,
                "created_at": to_isoformat(feedback.created_at),
            })

        logger.info(f"get_pending_reviews: found {len(reviews)} pending reviews")
        return {
            "status": "success",
            "reviews": reviews,
            "total": len(reviews),
        }

    @server.tool(
        name="apply_feedback",
        description="""
Apply a feedback correction to the classification system.

This moves the predicate from the original class to the correct class
and marks the feedback as applied.

Args:
    review_id: The feedback review ID to apply

Returns:
  - applied: true if successful
  - review_id: the review ID
  - predicate: the predicate that was corrected
  - new_class: the new class name
  - new_class_id: the new class ID
""",
    )
    async def apply_feedback(
        review_id: str,
    ) -> dict:
        """Apply feedback correction."""
        logger.debug(f"apply_feedback: review_id={review_id}")

        from graphbrain.classification.models import PredicateBankEntry

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        repo = lifespan_data["repo"]

        # Get the feedback entry
        pending = list(repo.get_pending_feedback(limit=1000))
        feedback = None
        for f in pending:
            if f.review_id == review_id:
                feedback = f
                break

        if not feedback:
            logger.warning(f"apply_feedback: review '{review_id}' not found or already applied")
            return not_found_error("feedback_review", review_id)

        # Get the correct class
        correct_class = repo.get_class_by_name(feedback.correct_class)
        if not correct_class:
            # Try to find by ID
            correct_class = repo.get_class(feedback.correct_class)

        if not correct_class:
            logger.warning(f"apply_feedback: target class '{feedback.correct_class}' not found")
            return not_found_error("semantic_class", feedback.correct_class)

        # Add predicate to correct class
        entry = PredicateBankEntry(
            class_id=correct_class.id,
            lemma=feedback.predicate,
            is_seed=False,  # Corrections are not seed predicates
            frequency=1,
            created_at=datetime.now(),
        )
        repo.save_predicate(entry)

        # Mark feedback as applied
        applied = repo.apply_feedback(review_id)

        logger.info(f"apply_feedback: applied {review_id}, moved '{feedback.predicate}' to '{correct_class.name}'")
        return {
            "status": "success",
            "applied": applied,
            "review_id": review_id,
            "predicate": feedback.predicate,
            "new_class": correct_class.name,
            "new_class_id": correct_class.id,
        }

    @server.tool(
        name="flag_for_review",
        description="""
Flag a classification for human review.

Creates a pending review entry for classifications that need
human verification. Use this when confidence is low or
classification is uncertain.

Args:
    predicate: The predicate to flag
    current_class: Optional current classification
    suggested_class: Optional suggested classification
    reviewer_id: Optional identifier for the reviewer

Returns:
  - review_id: unique ID for tracking this review
  - flag_status: "pending"
  - message: confirmation message
  - predicate: the flagged predicate
""",
    )
    async def flag_for_review(
        predicate: str,
        current_class: Optional[str] = None,
        suggested_class: Optional[str] = None,
        reviewer_id: Optional[str] = None,
    ) -> dict:
        """Flag for human review."""
        logger.debug(f"flag_for_review: predicate='{predicate}', current={current_class}, suggested={suggested_class}")

        from graphbrain.classification.models import ClassificationFeedback

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        repo = lifespan_data["repo"]

        # Generate review ID
        review_id = f"flag_{uuid.uuid4().hex[:8]}"

        feedback = ClassificationFeedback(
            review_id=review_id,
            predicate=predicate,
            original_class=current_class or "unclassified",
            correct_class=suggested_class or "unknown",
            confidence_adjustment=None,
            reviewer_id=reviewer_id,
            status="pending",
            created_at=datetime.now(),
        )
        repo.save_feedback(feedback)

        logger.info(f"flag_for_review: flagged '{predicate}' as {review_id}")
        return {
            "status": "success",
            "review_id": review_id,
            "flag_status": "pending",
            "message": "Flagged for review",
            "predicate": predicate,
        }

    @server.tool(
        name="feedback_stats",
        description="""
Get statistics about the feedback system.

Returns:
  - pending_feedback: count of pending feedback items
""",
    )
    async def feedback_stats() -> dict:
        """Get feedback statistics."""
        logger.debug("feedback_stats: computing statistics")

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        repo = lifespan_data["repo"]

        stats = repo.get_stats()
        pending = stats.get("pending_feedback", 0)

        logger.info(f"feedback_stats: {pending} pending")
        return {
            "status": "success",
            "pending_feedback": pending,
        }
