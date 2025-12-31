"""Metrics collectors for classification quality monitoring.

Collects and aggregates metrics from the classification system.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime, timezone
import logging
import statistics

logger = logging.getLogger(__name__)


@dataclass
class ClassificationMetrics:
    """Metrics about classification quality."""
    total_classifications: int = 0
    avg_confidence: float = 0.0
    confidence_std: float = 0.0
    high_confidence_count: int = 0  # >= 0.9
    medium_confidence_count: int = 0  # 0.7-0.9
    low_confidence_count: int = 0  # < 0.7
    by_method: dict[str, int] = field(default_factory=dict)
    by_class: dict[str, int] = field(default_factory=dict)


@dataclass
class FeedbackMetrics:
    """Metrics about the feedback system."""
    total_feedback: int = 0
    pending: int = 0
    applied: int = 0
    rejected: int = 0
    approval_rate: float = 0.0
    avg_time_to_apply: float = 0.0  # seconds
    by_class_correction: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class CoverageMetrics:
    """Metrics about classification coverage."""
    total_classes: int = 0
    total_predicates: int = 0
    avg_predicates_per_class: float = 0.0
    seed_predicates: int = 0
    discovered_predicates: int = 0
    total_patterns: int = 0
    classes_with_patterns: int = 0
    unclassified_estimate: int = 0


@dataclass
class AggregatedMetrics:
    """All metrics aggregated together."""
    classification: ClassificationMetrics
    feedback: FeedbackMetrics
    coverage: CoverageMetrics
    collected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    collection_duration_ms: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "classification": {
                "total": self.classification.total_classifications,
                "avg_confidence": self.classification.avg_confidence,
                "confidence_std": self.classification.confidence_std,
                "by_confidence": {
                    "high": self.classification.high_confidence_count,
                    "medium": self.classification.medium_confidence_count,
                    "low": self.classification.low_confidence_count,
                },
                "by_method": self.classification.by_method,
                "by_class": self.classification.by_class,
            },
            "feedback": {
                "total": self.feedback.total_feedback,
                "pending": self.feedback.pending,
                "applied": self.feedback.applied,
                "rejected": self.feedback.rejected,
                "approval_rate": self.feedback.approval_rate,
            },
            "coverage": {
                "classes": self.coverage.total_classes,
                "predicates": self.coverage.total_predicates,
                "avg_per_class": self.coverage.avg_predicates_per_class,
                "seeds": self.coverage.seed_predicates,
                "discovered": self.coverage.discovered_predicates,
                "patterns": self.coverage.total_patterns,
            },
            "collected_at": self.collected_at.isoformat(),
            "collection_duration_ms": self.collection_duration_ms,
        }


class MetricsCollector:
    """
    Collects metrics from the classification system.

    Provides comprehensive quality metrics for monitoring
    and improvement suggestions.
    """

    # Confidence thresholds
    HIGH_CONFIDENCE = 0.9
    MEDIUM_CONFIDENCE = 0.7

    def __init__(self, backend: Any):
        """
        Initialize the metrics collector.

        Args:
            backend: Classification backend for data access
        """
        self._backend = backend

    def collect_all(self) -> AggregatedMetrics:
        """
        Collect all metrics.

        Returns:
            AggregatedMetrics with all collected data
        """
        start = datetime.now(timezone.utc)

        classification = self._collect_classification_metrics()
        feedback = self._collect_feedback_metrics()
        coverage = self._collect_coverage_metrics()

        end = datetime.now(timezone.utc)
        duration_ms = (end - start).total_seconds() * 1000

        return AggregatedMetrics(
            classification=classification,
            feedback=feedback,
            coverage=coverage,
            collected_at=start,
            collection_duration_ms=duration_ms,
        )

    def _collect_classification_metrics(self) -> ClassificationMetrics:
        """Collect classification quality metrics."""
        metrics = ClassificationMetrics()
        confidences = []
        by_method: dict[str, int] = {}
        by_class: dict[str, int] = {}

        # Iterate over all classifications
        for sem_class in self._backend.list_classes():
            for classification in self._backend.get_edges_by_class(sem_class.id, limit=10000):
                metrics.total_classifications += 1
                confidences.append(classification.confidence)

                # Count by confidence level
                if classification.confidence >= self.HIGH_CONFIDENCE:
                    metrics.high_confidence_count += 1
                elif classification.confidence >= self.MEDIUM_CONFIDENCE:
                    metrics.medium_confidence_count += 1
                else:
                    metrics.low_confidence_count += 1

                # Count by method
                method = classification.method or "unknown"
                by_method[method] = by_method.get(method, 0) + 1

                # Count by class
                by_class[sem_class.id] = by_class.get(sem_class.id, 0) + 1

        # Calculate statistics
        if confidences:
            metrics.avg_confidence = statistics.mean(confidences)
            if len(confidences) > 1:
                metrics.confidence_std = statistics.stdev(confidences)

        metrics.by_method = by_method
        metrics.by_class = by_class

        return metrics

    def _collect_feedback_metrics(self) -> FeedbackMetrics:
        """Collect feedback system metrics."""
        metrics = FeedbackMetrics()

        # Get pending feedback
        pending_list = list(self._backend.get_pending_feedback(limit=10000))
        metrics.pending = len(pending_list)

        # Get overall stats from backend
        stats = self._backend.get_stats()
        metrics.total_feedback = stats.get("total_feedback", metrics.pending)
        metrics.applied = stats.get("applied_feedback", 0)
        metrics.rejected = stats.get("rejected_feedback", 0)

        # Calculate approval rate
        reviewed = metrics.applied + metrics.rejected
        if reviewed > 0:
            metrics.approval_rate = metrics.applied / reviewed

        return metrics

    def _collect_coverage_metrics(self) -> CoverageMetrics:
        """Collect coverage metrics."""
        metrics = CoverageMetrics()

        predicates_per_class = []

        for sem_class in self._backend.list_classes():
            metrics.total_classes += 1

            predicates = list(self._backend.get_predicates_by_class(sem_class.id))
            predicate_count = len(predicates)
            predicates_per_class.append(predicate_count)
            metrics.total_predicates += predicate_count

            for pred in predicates:
                if pred.is_seed:
                    metrics.seed_predicates += 1
                else:
                    metrics.discovered_predicates += 1

            patterns = list(self._backend.get_patterns_by_class(sem_class.id))
            if patterns:
                metrics.classes_with_patterns += 1
                metrics.total_patterns += len(patterns)

        # Calculate average
        if predicates_per_class:
            metrics.avg_predicates_per_class = statistics.mean(predicates_per_class)

        return metrics

    def get_confidence_histogram(
        self,
        bins: int = 10,
    ) -> list[dict]:
        """
        Get a histogram of confidence scores.

        Args:
            bins: Number of histogram bins

        Returns:
            List of {range: str, count: int} dictionaries
        """
        confidences = []

        for sem_class in self._backend.list_classes():
            for classification in self._backend.get_edges_by_class(sem_class.id, limit=10000):
                confidences.append(classification.confidence)

        if not confidences:
            return []

        # Build histogram
        bin_size = 1.0 / bins
        histogram = []

        for i in range(bins):
            low = i * bin_size
            high = (i + 1) * bin_size
            count = sum(1 for c in confidences if low <= c < high)
            if i == bins - 1:  # Include 1.0 in last bin
                count += sum(1 for c in confidences if c == 1.0)
            histogram.append({
                "range": f"{low:.1f}-{high:.1f}",
                "count": count,
            })

        return histogram

    def get_class_distribution(self) -> list[dict]:
        """
        Get distribution of predicates across classes.

        Returns:
            List of {class_id: str, name: str, count: int} sorted by count
        """
        distribution = []

        for sem_class in self._backend.list_classes():
            count = len(list(self._backend.get_predicates_by_class(sem_class.id)))
            distribution.append({
                "class_id": sem_class.id,
                "name": sem_class.name,
                "count": count,
            })

        distribution.sort(key=lambda x: x["count"], reverse=True)
        return distribution

    def get_method_effectiveness(self) -> dict[str, dict]:
        """
        Analyze effectiveness of different classification methods.

        Returns:
            Dictionary mapping method to {count, avg_confidence, approved_rate}
        """
        methods: dict[str, list[float]] = {}

        for sem_class in self._backend.list_classes():
            for classification in self._backend.get_edges_by_class(sem_class.id, limit=10000):
                method = classification.method or "unknown"
                if method not in methods:
                    methods[method] = []
                methods[method].append(classification.confidence)

        result = {}
        for method, confidences in methods.items():
            result[method] = {
                "count": len(confidences),
                "avg_confidence": statistics.mean(confidences) if confidences else 0,
                "min_confidence": min(confidences) if confidences else 0,
                "max_confidence": max(confidences) if confidences else 0,
            }

        return result
