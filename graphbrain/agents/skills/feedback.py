"""
Feedback skill for human-in-the-loop improvement.

Manages the review queue and applies classification corrections.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum
from datetime import datetime, timezone
import logging

from graphbrain.agents.skills.base import BaseSkill, SkillResult

logger = logging.getLogger(__name__)


class ReviewStatus(str, Enum):
    """Status of a review item."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEFERRED = "deferred"


class FeedbackDecision(str, Enum):
    """Decision for a feedback item."""
    APPLY = "apply"           # Apply the suggested correction
    REJECT = "reject"         # Reject the suggestion
    DEFER = "defer"           # Defer for later review
    ESCALATE = "escalate"     # Escalate to expert


@dataclass
class ReviewItem:
    """An item in the review queue."""
    review_id: str
    predicate: str
    original_class: Optional[str]
    suggested_class: str
    confidence: float
    evidence: dict = field(default_factory=dict)
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    status: ReviewStatus = ReviewStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    notes: str = ""


@dataclass
class FeedbackSession:
    """Statistics for a feedback session."""
    processed: int = 0
    applied: int = 0
    rejected: int = 0
    deferred: int = 0
    escalated: int = 0
    classes_updated: dict[str, int] = field(default_factory=dict)
    flagged_issues: list[str] = field(default_factory=list)


class FeedbackSkill(BaseSkill):
    """
    Skill for managing human-in-the-loop feedback.

    Capabilities:
    - Process review queue
    - Apply classification corrections
    - Track quality metrics
    - Suggest active learning improvements
    """

    SKILL_NAME = "feedback"
    PROMPT_FILE = "feedback.md"
    DEFAULT_CONFIDENCE_THRESHOLD = 0.8

    # Threshold for auto-applying high-confidence feedback
    AUTO_APPLY_THRESHOLD = 0.95

    # MCP tools this skill uses
    TOOLS = [
        "mcp__graphbrain__get_pending_reviews",
        "mcp__graphbrain__apply_feedback",
        "mcp__graphbrain__submit_feedback",
        "mcp__graphbrain__feedback_stats",
        "mcp__graphbrain__add_predicate_to_class",
    ]

    def get_tools(self) -> list[str]:
        """Get the list of MCP tools this skill uses."""
        return self.TOOLS

    def decide_action(
        self,
        item: ReviewItem,
        human_approved: bool = False,
    ) -> FeedbackDecision:
        """
        Decide what action to take for a review item.

        Args:
            item: The review item
            human_approved: Whether human has approved

        Returns:
            FeedbackDecision
        """
        if human_approved:
            return FeedbackDecision.APPLY

        # Auto-apply if very high confidence
        if item.confidence >= self.AUTO_APPLY_THRESHOLD:
            return FeedbackDecision.APPLY

        # Check for conflicting evidence
        if item.alternatives:
            top_alt_score = item.alternatives[0][1] if item.alternatives else 0
            if top_alt_score > item.confidence * 0.9:
                # Close alternatives - needs human review
                return FeedbackDecision.DEFER

        # Moderate confidence - defer
        if item.confidence < self.DEFAULT_CONFIDENCE_THRESHOLD:
            return FeedbackDecision.DEFER

        return FeedbackDecision.DEFER

    def format_review_item(
        self,
        predicate: str,
        suggested_class: str,
        confidence: float,
        original_class: Optional[str] = None,
        similar_predicates: list[dict] = None,
        example_edges: list[str] = None,
    ) -> ReviewItem:
        """
        Format a review item for presentation.

        Args:
            predicate: The predicate being reviewed
            suggested_class: Suggested classification
            confidence: Classification confidence
            original_class: Current classification if any
            similar_predicates: Similar predicates with their classes
            example_edges: Example edges containing this predicate

        Returns:
            ReviewItem ready for review
        """
        evidence = {}
        alternatives = []

        if similar_predicates:
            evidence["similar"] = [
                {"predicate": p["predicate"], "class": p["class_id"], "similarity": p["similarity"]}
                for p in similar_predicates[:5]
            ]

            # Calculate alternatives from similar predicates
            class_scores: dict[str, list[float]] = {}
            for p in similar_predicates:
                class_id = p.get("class_id")
                if class_id and class_id != suggested_class:
                    if class_id not in class_scores:
                        class_scores[class_id] = []
                    class_scores[class_id].append(p.get("similarity", 0))

            for class_id, scores in class_scores.items():
                avg_score = sum(scores) / len(scores)
                alternatives.append((class_id, avg_score))

            alternatives.sort(key=lambda x: x[1], reverse=True)

        if example_edges:
            evidence["examples"] = example_edges[:3]

        return ReviewItem(
            review_id=f"rev_{predicate}_{int(datetime.now(timezone.utc).timestamp())}",
            predicate=predicate,
            original_class=original_class,
            suggested_class=suggested_class,
            confidence=confidence,
            evidence=evidence,
            alternatives=alternatives[:3],
        )

    def build_get_pending_params(
        self,
        limit: int = 20,
        sort_by: str = "confidence",
    ) -> dict:
        """Build parameters for get_pending_reviews MCP tool."""
        return {
            "limit": limit,
            "sort_by": sort_by,
        }

    def build_apply_feedback_params(
        self,
        review_id: str,
        decision: FeedbackDecision,
        notes: str = "",
    ) -> dict:
        """Build parameters for apply_feedback MCP tool."""
        return {
            "review_id": review_id,
            "approved": decision == FeedbackDecision.APPLY,
            "notes": notes,
        }

    def build_submit_feedback_params(
        self,
        predicate: str,
        original_class: str,
        correct_class: str,
        notes: str = "",
    ) -> dict:
        """Build parameters for submit_feedback MCP tool."""
        return {
            "predicate": predicate,
            "original_class": original_class,
            "correct_class": correct_class,
            "notes": notes,
        }

    def build_add_predicate_params(
        self,
        class_id: str,
        lemma: str,
        is_seed: bool = False,
    ) -> dict:
        """Build parameters for add_predicate_to_class MCP tool."""
        return {
            "class_id": class_id,
            "lemma": lemma,
            "is_seed": is_seed,
        }

    def process_batch(
        self,
        items: list[ReviewItem],
        decisions: dict[str, FeedbackDecision],
    ) -> FeedbackSession:
        """
        Process a batch of review items.

        Args:
            items: Review items to process
            decisions: Map of review_id to decision

        Returns:
            FeedbackSession with statistics
        """
        session = FeedbackSession()

        for item in items:
            decision = decisions.get(item.review_id, FeedbackDecision.DEFER)
            session.processed += 1

            if decision == FeedbackDecision.APPLY:
                session.applied += 1
                if item.suggested_class:
                    session.classes_updated[item.suggested_class] = \
                        session.classes_updated.get(item.suggested_class, 0) + 1
            elif decision == FeedbackDecision.REJECT:
                session.rejected += 1
            elif decision == FeedbackDecision.DEFER:
                session.deferred += 1
            elif decision == FeedbackDecision.ESCALATE:
                session.escalated += 1

        return session

    def identify_improvement_opportunities(
        self,
        stats: dict,
    ) -> list[str]:
        """
        Identify opportunities for improving classification.

        Args:
            stats: Feedback statistics

        Returns:
            List of improvement suggestions
        """
        opportunities = []

        # High frequency unclassified
        if stats.get("unclassified_high_freq", 0) > 10:
            opportunities.append(
                f"Found {stats['unclassified_high_freq']} high-frequency "
                "unclassified predicates - consider reviewing"
            )

        # Low average confidence
        avg_conf = stats.get("avg_confidence", 1.0)
        if avg_conf < 0.7:
            opportunities.append(
                f"Average confidence is low ({avg_conf:.2f}) - "
                "consider adding more seed predicates"
            )

        # Class imbalance
        class_counts = stats.get("by_class", {})
        if class_counts:
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            if max_count > min_count * 5:
                small_classes = [c for c, n in class_counts.items() if n < max_count / 3]
                if small_classes:
                    opportunities.append(
                        f"Class imbalance detected - consider adding seeds for: "
                        f"{', '.join(small_classes)}"
                    )

        # High rejection rate
        total = stats.get("total_reviews", 0)
        rejected = stats.get("rejected", 0)
        if total > 0 and rejected / total > 0.3:
            opportunities.append(
                f"High rejection rate ({rejected}/{total}) - "
                "review classification thresholds"
            )

        return opportunities

    def generate_session_summary(
        self,
        session: FeedbackSession,
    ) -> str:
        """
        Generate a summary of a feedback session.

        Args:
            session: The feedback session

        Returns:
            Markdown-formatted summary
        """
        lines = [
            "## Feedback Session Summary\n",
            f"Processed: {session.processed} reviews",
            f"- Applied: {session.applied}",
            f"- Rejected: {session.rejected}",
            f"- Deferred: {session.deferred}",
            f"- Escalated: {session.escalated}",
            "",
        ]

        if session.classes_updated:
            lines.append("### Classes Updated\n")
            for class_id, count in session.classes_updated.items():
                lines.append(f"- {class_id}: +{count} predicates")
            lines.append("")

        if session.flagged_issues:
            lines.append("### Flagged Issues\n")
            for issue in session.flagged_issues:
                lines.append(f"- {issue}")

        return "\n".join(lines)
