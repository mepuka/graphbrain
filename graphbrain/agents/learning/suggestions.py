"""Improvement suggestions engine for active learning.

Analyzes classification quality and generates actionable
suggestions for improving the system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


class SuggestionType(str, Enum):
    """Types of improvement suggestions."""
    ADD_SEED = "add_seed"                    # Add seed predicates to class
    REVIEW_BORDERLINE = "review_borderline"  # Review low-confidence cases
    MERGE_CLASSES = "merge_classes"          # Consider merging similar classes
    SPLIT_CLASS = "split_class"              # Consider splitting diverse class
    ADD_PATTERN = "add_pattern"              # Add pattern rule
    ADJUST_THRESHOLD = "adjust_threshold"    # Adjust confidence threshold
    EXPAND_COVERAGE = "expand_coverage"      # Add predicates to underrepresented class
    RESOLVE_CONFLICT = "resolve_conflict"    # Resolve conflicting classifications


class SuggestionPriority(str, Enum):
    """Priority levels for suggestions."""
    CRITICAL = "critical"  # Blocking issues
    HIGH = "high"          # Significant impact
    MEDIUM = "medium"      # Moderate impact
    LOW = "low"            # Nice to have


@dataclass
class ImprovementSuggestion:
    """A suggestion for improving classification quality."""
    suggestion_id: str
    suggestion_type: SuggestionType
    priority: SuggestionPriority
    title: str
    description: str
    affected_class: Optional[str] = None
    affected_predicates: list[str] = field(default_factory=list)
    expected_impact: str = ""
    action_items: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "suggestion_id": self.suggestion_id,
            "type": self.suggestion_type.value,
            "priority": self.priority.value,
            "title": self.title,
            "description": self.description,
            "affected_class": self.affected_class,
            "affected_predicates": self.affected_predicates,
            "expected_impact": self.expected_impact,
            "action_items": self.action_items,
            "evidence": self.evidence,
            "created_at": self.created_at.isoformat(),
        }


class SuggestionEngine:
    """
    Engine for generating improvement suggestions.

    Analyzes classification statistics and generates
    actionable suggestions for improving quality.
    """

    # Thresholds for triggering suggestions
    LOW_CONFIDENCE_THRESHOLD = 0.7
    HIGH_REJECTION_RATE = 0.3
    CLASS_IMBALANCE_RATIO = 5.0
    MIN_PREDICATES_PER_CLASS = 3
    CONFLICT_SIMILARITY_THRESHOLD = 0.8

    def __init__(self, backend: Any):
        """
        Initialize the suggestion engine.

        Args:
            backend: Classification backend for data access
        """
        self._backend = backend
        self._suggestion_counter = 0

    def _generate_id(self) -> str:
        """Generate a unique suggestion ID."""
        self._suggestion_counter += 1
        timestamp = int(datetime.now(timezone.utc).timestamp())
        return f"sug_{timestamp}_{self._suggestion_counter:04d}"

    def analyze(self) -> list[ImprovementSuggestion]:
        """
        Analyze classification state and generate suggestions.

        Returns:
            List of improvement suggestions sorted by priority
        """
        suggestions = []

        # Get current statistics
        stats = self._backend.get_stats()

        # Run analysis checks
        suggestions.extend(self._check_class_coverage(stats))
        suggestions.extend(self._check_confidence_distribution(stats))
        suggestions.extend(self._check_feedback_patterns(stats))
        suggestions.extend(self._check_class_balance(stats))
        suggestions.extend(self._check_predicate_conflicts())

        # Sort by priority
        priority_order = {
            SuggestionPriority.CRITICAL: 0,
            SuggestionPriority.HIGH: 1,
            SuggestionPriority.MEDIUM: 2,
            SuggestionPriority.LOW: 3,
        }
        suggestions.sort(key=lambda s: priority_order[s.priority])

        return suggestions

    def _check_class_coverage(self, stats: dict) -> list[ImprovementSuggestion]:
        """Check for classes with insufficient seed predicates."""
        suggestions = []

        for sem_class in self._backend.list_classes():
            predicates = list(self._backend.get_predicates_by_class(sem_class.id))
            seed_count = sum(1 for p in predicates if p.is_seed)

            if seed_count < self.MIN_PREDICATES_PER_CLASS:
                suggestions.append(ImprovementSuggestion(
                    suggestion_id=self._generate_id(),
                    suggestion_type=SuggestionType.ADD_SEED,
                    priority=SuggestionPriority.HIGH if seed_count == 0 else SuggestionPriority.MEDIUM,
                    title=f"Add seed predicates to '{sem_class.name}'",
                    description=(
                        f"Class '{sem_class.name}' has only {seed_count} seed predicates. "
                        f"Adding more seeds will improve classification accuracy."
                    ),
                    affected_class=sem_class.id,
                    expected_impact="Improved recall for this class",
                    action_items=[
                        f"Identify 3-5 representative predicates for '{sem_class.name}'",
                        "Add them as seed predicates using add_predicate_to_class",
                    ],
                    evidence={
                        "current_seeds": seed_count,
                        "total_predicates": len(predicates),
                        "recommended_minimum": self.MIN_PREDICATES_PER_CLASS,
                    },
                ))

        return suggestions

    def _check_confidence_distribution(self, stats: dict) -> list[ImprovementSuggestion]:
        """Check for low average confidence."""
        suggestions = []

        avg_confidence = stats.get("avg_confidence", 1.0)

        if avg_confidence < self.LOW_CONFIDENCE_THRESHOLD:
            suggestions.append(ImprovementSuggestion(
                suggestion_id=self._generate_id(),
                suggestion_type=SuggestionType.REVIEW_BORDERLINE,
                priority=SuggestionPriority.HIGH,
                title="Review low-confidence classifications",
                description=(
                    f"Average confidence is {avg_confidence:.2%}, below the threshold of "
                    f"{self.LOW_CONFIDENCE_THRESHOLD:.0%}. Many classifications may be incorrect."
                ),
                expected_impact="Improved precision through human review",
                action_items=[
                    "Use get_pending_reviews to find low-confidence cases",
                    "Review and correct misclassifications",
                    "Add corrected predicates as seeds for their correct classes",
                ],
                evidence={
                    "avg_confidence": avg_confidence,
                    "threshold": self.LOW_CONFIDENCE_THRESHOLD,
                },
            ))

        return suggestions

    def _check_feedback_patterns(self, stats: dict) -> list[ImprovementSuggestion]:
        """Analyze feedback for systematic issues."""
        suggestions = []

        pending_feedback = stats.get("pending_feedback", 0)
        total_reviewed = stats.get("total_reviewed", 0)
        rejected = stats.get("rejected_feedback", 0)

        # High pending feedback
        if pending_feedback > 20:
            suggestions.append(ImprovementSuggestion(
                suggestion_id=self._generate_id(),
                suggestion_type=SuggestionType.REVIEW_BORDERLINE,
                priority=SuggestionPriority.MEDIUM,
                title="Process pending feedback queue",
                description=(
                    f"There are {pending_feedback} pending feedback items. "
                    "Processing these will improve classification quality."
                ),
                expected_impact=f"~{pending_feedback} potential corrections",
                action_items=[
                    "Review pending feedback using get_pending_reviews",
                    "Apply valid corrections",
                    "Update predicate banks with new mappings",
                ],
                evidence={"pending_count": pending_feedback},
            ))

        # High rejection rate
        if total_reviewed > 0:
            rejection_rate = rejected / total_reviewed
            if rejection_rate > self.HIGH_REJECTION_RATE:
                suggestions.append(ImprovementSuggestion(
                    suggestion_id=self._generate_id(),
                    suggestion_type=SuggestionType.ADJUST_THRESHOLD,
                    priority=SuggestionPriority.HIGH,
                    title="Adjust classification threshold",
                    description=(
                        f"Rejection rate is {rejection_rate:.0%}, suggesting the auto-classify "
                        "threshold may be too aggressive."
                    ),
                    expected_impact="Reduced false positives",
                    action_items=[
                        "Increase confidence threshold for auto-classification",
                        "Consider adding more seed predicates",
                        "Review recent rejections for patterns",
                    ],
                    evidence={
                        "rejection_rate": rejection_rate,
                        "total_reviewed": total_reviewed,
                        "rejected": rejected,
                    },
                ))

        return suggestions

    def _check_class_balance(self, stats: dict) -> list[ImprovementSuggestion]:
        """Check for class imbalance issues."""
        suggestions = []

        class_counts = {}
        for sem_class in self._backend.list_classes():
            count = len(list(self._backend.get_predicates_by_class(sem_class.id)))
            class_counts[sem_class.id] = {
                "name": sem_class.name,
                "count": count,
            }

        if not class_counts:
            return suggestions

        counts = [c["count"] for c in class_counts.values()]
        max_count = max(counts)
        min_count = min(counts)

        if max_count > 0 and min_count > 0:
            ratio = max_count / min_count
            if ratio > self.CLASS_IMBALANCE_RATIO:
                small_classes = [
                    info["name"] for info in class_counts.values()
                    if info["count"] < max_count / self.CLASS_IMBALANCE_RATIO
                ]
                large_classes = [
                    info["name"] for info in class_counts.values()
                    if info["count"] == max_count
                ]

                suggestions.append(ImprovementSuggestion(
                    suggestion_id=self._generate_id(),
                    suggestion_type=SuggestionType.EXPAND_COVERAGE,
                    priority=SuggestionPriority.MEDIUM,
                    title="Address class imbalance",
                    description=(
                        f"Significant class imbalance detected (ratio {ratio:.1f}:1). "
                        f"Small classes: {', '.join(small_classes)}."
                    ),
                    affected_predicates=small_classes,
                    expected_impact="More balanced classification coverage",
                    action_items=[
                        f"Add more predicates to: {', '.join(small_classes)}",
                        "Consider if small classes should be merged with related classes",
                        "Review if large classes should be split",
                    ],
                    evidence={
                        "imbalance_ratio": ratio,
                        "class_counts": {k: v["count"] for k, v in class_counts.items()},
                    },
                ))

        return suggestions

    def _check_predicate_conflicts(self) -> list[ImprovementSuggestion]:
        """Check for predicates that appear in multiple classes."""
        suggestions = []
        conflicts = []

        # Find predicates in multiple classes
        predicate_classes: dict[str, list[str]] = {}

        for sem_class in self._backend.list_classes():
            for pred in self._backend.get_predicates_by_class(sem_class.id):
                if pred.lemma not in predicate_classes:
                    predicate_classes[pred.lemma] = []
                predicate_classes[pred.lemma].append(sem_class.id)

        for lemma, classes in predicate_classes.items():
            if len(classes) > 1:
                conflicts.append({
                    "predicate": lemma,
                    "classes": classes,
                })

        if conflicts:
            # Group by severity
            severe = [c for c in conflicts if len(c["classes"]) > 2]
            moderate = [c for c in conflicts if len(c["classes"]) == 2]

            if severe:
                suggestions.append(ImprovementSuggestion(
                    suggestion_id=self._generate_id(),
                    suggestion_type=SuggestionType.RESOLVE_CONFLICT,
                    priority=SuggestionPriority.HIGH,
                    title="Resolve multi-class predicate conflicts",
                    description=(
                        f"Found {len(severe)} predicates in 3+ classes. "
                        "These create classification ambiguity."
                    ),
                    affected_predicates=[c["predicate"] for c in severe],
                    expected_impact="Reduced classification conflicts",
                    action_items=[
                        "Review each conflicting predicate",
                        "Determine the correct primary class",
                        "Remove from incorrect classes",
                    ],
                    evidence={"conflicts": severe},
                ))

            if moderate:
                suggestions.append(ImprovementSuggestion(
                    suggestion_id=self._generate_id(),
                    suggestion_type=SuggestionType.RESOLVE_CONFLICT,
                    priority=SuggestionPriority.MEDIUM,
                    title="Review predicates in multiple classes",
                    description=(
                        f"Found {len(moderate)} predicates appearing in exactly 2 classes. "
                        "These may indicate class overlap or polysemy."
                    ),
                    affected_predicates=[c["predicate"] for c in moderate],
                    expected_impact="Clearer class boundaries",
                    action_items=[
                        "Determine if predicates are truly polysemous",
                        "Consider context-based classification",
                        "Or assign to primary class only",
                    ],
                    evidence={"conflicts": moderate},
                ))

        return suggestions

    def get_summary(self) -> dict:
        """
        Get a summary of the current suggestion state.

        Returns:
            Dictionary with counts by type and priority
        """
        suggestions = self.analyze()

        by_priority = {}
        by_type = {}

        for s in suggestions:
            by_priority[s.priority.value] = by_priority.get(s.priority.value, 0) + 1
            by_type[s.suggestion_type.value] = by_type.get(s.suggestion_type.value, 0) + 1

        return {
            "total": len(suggestions),
            "by_priority": by_priority,
            "by_type": by_type,
            "critical_count": by_priority.get("critical", 0),
            "high_count": by_priority.get("high", 0),
        }
