"""
Classification skill for predicate categorization.

Uses semantic similarity and predicate banks to classify predicates.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum
import logging

from graphbrain.agents.skills.base import BaseSkill, SkillResult

logger = logging.getLogger(__name__)


class ClassificationAction(str, Enum):
    """Action to take based on classification confidence."""
    AUTO_APPLY = "auto_apply"         # >= 0.9
    APPLY_WITH_LOG = "apply_with_log"  # 0.8-0.9
    FLAG_OPTIONAL = "flag_optional"    # 0.7-0.8
    REQUIRE_REVIEW = "require_review"  # 0.5-0.7
    REJECT = "reject"                  # < 0.5


@dataclass
class ClassificationResult:
    """Result from classifying a predicate or edge."""
    predicate: str
    suggested_class: Optional[str]
    confidence: float
    method: str  # predicate_bank, semantic, pattern, hybrid
    action: ClassificationAction
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    reasoning: str = ""


class ClassificationSkill(BaseSkill):
    """
    Skill for classifying predicates into semantic classes.

    Capabilities:
    - Classify predicates using predicate banks
    - Find semantically similar predicates
    - Discover unclassified predicates
    - Apply confidence thresholds
    - Flag uncertain classifications for review
    """

    SKILL_NAME = "classification"
    PROMPT_FILE = "classification.md"
    DEFAULT_CONFIDENCE_THRESHOLD = 0.8

    # Confidence thresholds for different actions
    AUTO_APPLY_THRESHOLD = 0.9
    APPLY_WITH_LOG_THRESHOLD = 0.8
    FLAG_OPTIONAL_THRESHOLD = 0.7
    REQUIRE_REVIEW_THRESHOLD = 0.5

    # MCP tools this skill uses
    TOOLS = [
        "mcp__graphbrain__classify_predicate",
        "mcp__graphbrain__classify_edge",
        "mcp__graphbrain__discover_predicates",
        "mcp__graphbrain__find_similar_predicates",
        "mcp__graphbrain__get_predicate_classes",
        "mcp__graphbrain__list_predicates_by_class",
        "mcp__graphbrain__flag_for_review",
    ]

    def get_tools(self) -> list[str]:
        """Get the list of MCP tools this skill uses."""
        return self.TOOLS

    def determine_action(self, confidence: float) -> ClassificationAction:
        """
        Determine what action to take based on confidence.

        Args:
            confidence: Classification confidence (0.0-1.0)

        Returns:
            ClassificationAction to take
        """
        if confidence >= self.AUTO_APPLY_THRESHOLD:
            return ClassificationAction.AUTO_APPLY
        elif confidence >= self.APPLY_WITH_LOG_THRESHOLD:
            return ClassificationAction.APPLY_WITH_LOG
        elif confidence >= self.FLAG_OPTIONAL_THRESHOLD:
            return ClassificationAction.FLAG_OPTIONAL
        elif confidence >= self.REQUIRE_REVIEW_THRESHOLD:
            return ClassificationAction.REQUIRE_REVIEW
        else:
            return ClassificationAction.REJECT

    def format_classification(
        self,
        predicate: str,
        suggested_class: Optional[str],
        confidence: float,
        method: str,
        alternatives: list[tuple[str, float]] = None,
        evidence: dict = None,
        reasoning: str = "",
    ) -> ClassificationResult:
        """
        Format a classification result.

        Args:
            predicate: The predicate being classified
            suggested_class: The suggested semantic class
            confidence: Classification confidence
            method: Method used (predicate_bank, semantic, etc.)
            alternatives: Alternative classifications with scores
            evidence: Supporting evidence
            reasoning: Human-readable reasoning

        Returns:
            ClassificationResult with action determined
        """
        action = self.determine_action(confidence)

        return ClassificationResult(
            predicate=predicate,
            suggested_class=suggested_class,
            confidence=confidence,
            method=method,
            action=action,
            alternatives=alternatives or [],
            evidence=evidence or {},
            reasoning=reasoning,
        )

    def build_classify_predicate_params(
        self,
        predicate: str,
        threshold: float = None,
    ) -> dict:
        """Build parameters for classify_predicate MCP tool."""
        params = {"predicate": predicate}
        if threshold is not None:
            params["threshold"] = threshold
        return params

    def build_classify_edge_params(
        self,
        edge: str,
        threshold: float = None,
    ) -> dict:
        """Build parameters for classify_edge MCP tool."""
        params = {"edge": edge}
        if threshold is not None:
            params["threshold"] = threshold
        return params

    def build_discover_params(
        self,
        min_frequency: int = 5,
        limit: int = 100,
    ) -> dict:
        """Build parameters for discover_predicates MCP tool."""
        return {
            "min_frequency": min_frequency,
            "limit": limit,
        }

    def build_find_similar_params(
        self,
        predicate: str,
        limit: int = 10,
    ) -> dict:
        """Build parameters for find_similar_predicates MCP tool."""
        return {
            "predicate": predicate,
            "limit": limit,
        }

    def build_flag_review_params(
        self,
        result: ClassificationResult,
    ) -> dict:
        """Build parameters for flag_for_review MCP tool."""
        notes = result.reasoning
        if result.alternatives:
            alt_str = ", ".join(f"{c}({s:.2f})" for c, s in result.alternatives[:3])
            notes += f"\nAlternatives: {alt_str}"

        return {
            "predicate": result.predicate,
            "suggested_class": result.suggested_class,
            "notes": notes,
        }

    def merge_similar_results(
        self,
        similar_predicates: list[dict],
    ) -> tuple[Optional[str], float, list[tuple[str, float]]]:
        """
        Merge results from similar predicates to suggest a class.

        Args:
            similar_predicates: List of {predicate, class_id, similarity} dicts

        Returns:
            Tuple of (suggested_class, confidence, alternatives)
        """
        if not similar_predicates:
            return None, 0.0, []

        # Group by class and calculate weighted scores
        class_scores: dict[str, list[float]] = {}

        for pred in similar_predicates:
            class_id = pred.get("class_id")
            similarity = pred.get("similarity", 0.0)

            if class_id:
                if class_id not in class_scores:
                    class_scores[class_id] = []
                class_scores[class_id].append(similarity)

        if not class_scores:
            return None, 0.0, []

        # Calculate average score per class
        class_avg = {
            class_id: sum(scores) / len(scores)
            for class_id, scores in class_scores.items()
        }

        # Sort by score
        sorted_classes = sorted(
            class_avg.items(),
            key=lambda x: x[1],
            reverse=True
        )

        suggested = sorted_classes[0][0]
        confidence = sorted_classes[0][1]
        alternatives = sorted_classes[1:4]  # Top 3 alternatives

        return suggested, confidence, alternatives

    def get_classification_summary(
        self,
        results: list[ClassificationResult],
    ) -> dict:
        """
        Summarize a batch of classification results.

        Args:
            results: List of classification results

        Returns:
            Summary dictionary
        """
        by_action = {action: 0 for action in ClassificationAction}
        by_class: dict[str, int] = {}
        total_confidence = 0.0

        for result in results:
            by_action[result.action] += 1
            if result.suggested_class:
                by_class[result.suggested_class] = by_class.get(result.suggested_class, 0) + 1
            total_confidence += result.confidence

        return {
            "total": len(results),
            "by_action": {a.value: c for a, c in by_action.items() if c > 0},
            "by_class": by_class,
            "avg_confidence": total_confidence / len(results) if results else 0.0,
            "auto_applied": by_action[ClassificationAction.AUTO_APPLY],
            "needs_review": (
                by_action[ClassificationAction.REQUIRE_REVIEW] +
                by_action[ClassificationAction.FLAG_OPTIONAL]
            ),
        }
