"""Quality metrics dashboard for classification monitoring.

Provides a comprehensive dashboard view of classification quality
with trend analysis and actionable insights.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from datetime import datetime, timezone
import logging
import json

from graphbrain.agents.metrics.collectors import (
    MetricsCollector,
    AggregatedMetrics,
)

logger = logging.getLogger(__name__)


class TrendDirection(str, Enum):
    """Direction of metric trend."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"
    UNKNOWN = "unknown"


@dataclass
class MetricSnapshot:
    """A snapshot of a single metric over time."""
    name: str
    current_value: float
    previous_value: Optional[float] = None
    trend: TrendDirection = TrendDirection.UNKNOWN
    change_percent: float = 0.0
    unit: str = ""
    description: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "current": self.current_value,
            "previous": self.previous_value,
            "trend": self.trend.value,
            "change_percent": self.change_percent,
            "unit": self.unit,
            "description": self.description,
        }


@dataclass
class DashboardState:
    """Current state of the quality dashboard."""
    metrics: AggregatedMetrics
    snapshots: list[MetricSnapshot] = field(default_factory=list)
    health_score: float = 0.0
    health_status: str = "unknown"
    alerts: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class QualityDashboard:
    """
    Quality metrics dashboard for classification monitoring.

    Provides a unified view of classification quality with:
    - Real-time metrics
    - Trend analysis
    - Health scoring
    - Actionable recommendations
    """

    # Health score thresholds
    HEALTH_EXCELLENT = 0.9
    HEALTH_GOOD = 0.7
    HEALTH_FAIR = 0.5
    HEALTH_POOR = 0.3

    # Weights for health score calculation
    HEALTH_WEIGHTS = {
        "avg_confidence": 0.3,
        "coverage": 0.2,
        "feedback_processed": 0.2,
        "class_balance": 0.15,
        "seed_ratio": 0.15,
    }

    def __init__(
        self,
        backend: Any,
        history_file: Optional[str] = None,
    ):
        """
        Initialize the dashboard.

        Args:
            backend: Classification backend for data access
            history_file: Optional file path for storing metric history
        """
        self._backend = backend
        self._collector = MetricsCollector(backend)
        self._history_file = history_file
        self._previous_metrics: Optional[AggregatedMetrics] = None

        # Load previous metrics if available
        if history_file:
            self._load_history()

    def _load_history(self):
        """Load previous metrics from history file."""
        if not self._history_file:
            return

        try:
            with open(self._history_file, 'r') as f:
                data = json.load(f)
                # Reconstruct from dict - simplified
                self._previous_metrics = None  # TODO: full reconstruction
                logger.debug(f"Loaded history from {self._history_file}")
        except FileNotFoundError:
            logger.debug(f"No history file found at {self._history_file}")
        except Exception as e:
            logger.warning(f"Failed to load history: {e}")

    def _save_history(self, metrics: AggregatedMetrics):
        """Save current metrics to history file."""
        if not self._history_file:
            return

        try:
            with open(self._history_file, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)
            logger.debug(f"Saved history to {self._history_file}")
        except Exception as e:
            logger.warning(f"Failed to save history: {e}")

    def get_state(self) -> DashboardState:
        """
        Get the current dashboard state.

        Returns:
            DashboardState with all metrics and analysis
        """
        # Collect current metrics
        metrics = self._collector.collect_all()

        # Calculate snapshots with trends
        snapshots = self._calculate_snapshots(metrics)

        # Calculate health score
        health_score = self._calculate_health_score(metrics)
        health_status = self._get_health_status(health_score)

        # Generate alerts and recommendations
        alerts = self._generate_alerts(metrics)
        recommendations = self._generate_recommendations(metrics)

        # Save for next comparison
        self._previous_metrics = metrics
        self._save_history(metrics)

        return DashboardState(
            metrics=metrics,
            snapshots=snapshots,
            health_score=health_score,
            health_status=health_status,
            alerts=alerts,
            recommendations=recommendations,
        )

    def _calculate_snapshots(
        self,
        metrics: AggregatedMetrics,
    ) -> list[MetricSnapshot]:
        """Calculate metric snapshots with trend analysis."""
        snapshots = []

        # Classification metrics
        snapshots.append(MetricSnapshot(
            name="Classifications",
            current_value=metrics.classification.total_classifications,
            unit="edges",
            description="Total edges with classifications",
        ))

        snapshots.append(MetricSnapshot(
            name="Average Confidence",
            current_value=metrics.classification.avg_confidence * 100,
            unit="%",
            description="Average confidence of classifications",
        ))

        snapshots.append(MetricSnapshot(
            name="High Confidence",
            current_value=metrics.classification.high_confidence_count,
            unit="edges",
            description="Classifications with confidence >= 90%",
        ))

        # Feedback metrics
        snapshots.append(MetricSnapshot(
            name="Pending Reviews",
            current_value=metrics.feedback.pending,
            unit="items",
            description="Feedback items awaiting review",
        ))

        snapshots.append(MetricSnapshot(
            name="Approval Rate",
            current_value=metrics.feedback.approval_rate * 100,
            unit="%",
            description="Rate of feedback being applied",
        ))

        # Coverage metrics
        snapshots.append(MetricSnapshot(
            name="Semantic Classes",
            current_value=metrics.coverage.total_classes,
            unit="classes",
            description="Total semantic classes defined",
        ))

        snapshots.append(MetricSnapshot(
            name="Predicates",
            current_value=metrics.coverage.total_predicates,
            unit="predicates",
            description="Total predicates in banks",
        ))

        snapshots.append(MetricSnapshot(
            name="Patterns",
            current_value=metrics.coverage.total_patterns,
            unit="patterns",
            description="Classification patterns defined",
        ))

        # Calculate trends if we have previous data
        if self._previous_metrics:
            self._add_trends(snapshots, metrics, self._previous_metrics)

        return snapshots

    def _add_trends(
        self,
        snapshots: list[MetricSnapshot],
        current: AggregatedMetrics,
        previous: AggregatedMetrics,
    ):
        """Add trend information to snapshots."""
        # Map snapshot names to previous values
        prev_values = {
            "Classifications": previous.classification.total_classifications,
            "Average Confidence": previous.classification.avg_confidence * 100,
            "High Confidence": previous.classification.high_confidence_count,
            "Pending Reviews": previous.feedback.pending,
            "Approval Rate": previous.feedback.approval_rate * 100,
            "Semantic Classes": previous.coverage.total_classes,
            "Predicates": previous.coverage.total_predicates,
            "Patterns": previous.coverage.total_patterns,
        }

        for snapshot in snapshots:
            if snapshot.name in prev_values:
                prev = prev_values[snapshot.name]
                snapshot.previous_value = prev

                if prev > 0:
                    change = ((snapshot.current_value - prev) / prev) * 100
                    snapshot.change_percent = change

                    if change > 1:
                        snapshot.trend = TrendDirection.UP
                    elif change < -1:
                        snapshot.trend = TrendDirection.DOWN
                    else:
                        snapshot.trend = TrendDirection.STABLE
                else:
                    snapshot.trend = TrendDirection.STABLE

    def _calculate_health_score(self, metrics: AggregatedMetrics) -> float:
        """Calculate overall health score (0-1)."""
        scores = {}

        # Confidence score (target: 0.8+)
        scores["avg_confidence"] = min(1.0, metrics.classification.avg_confidence / 0.8)

        # Coverage score (based on predicates per class, target: 5+)
        if metrics.coverage.total_classes > 0:
            avg_per_class = metrics.coverage.avg_predicates_per_class
            scores["coverage"] = min(1.0, avg_per_class / 5.0)
        else:
            scores["coverage"] = 0.0

        # Feedback processing score (lower pending = better)
        total_feedback = metrics.feedback.pending + metrics.feedback.applied
        if total_feedback > 0:
            processed_ratio = metrics.feedback.applied / total_feedback
            scores["feedback_processed"] = processed_ratio
        else:
            scores["feedback_processed"] = 1.0  # No feedback = ok

        # Class balance score
        if metrics.coverage.total_predicates > 0 and metrics.coverage.total_classes > 0:
            expected_avg = metrics.coverage.total_predicates / metrics.coverage.total_classes
            # Measure how close we are to balanced distribution
            # This is simplified - could use Gini coefficient
            scores["class_balance"] = min(1.0, metrics.coverage.avg_predicates_per_class / max(1, expected_avg))
        else:
            scores["class_balance"] = 0.5

        # Seed ratio score (target: 20%+ seeds)
        if metrics.coverage.total_predicates > 0:
            seed_ratio = metrics.coverage.seed_predicates / metrics.coverage.total_predicates
            scores["seed_ratio"] = min(1.0, seed_ratio / 0.2)
        else:
            scores["seed_ratio"] = 0.0

        # Weighted average
        health = sum(
            scores[key] * self.HEALTH_WEIGHTS[key]
            for key in self.HEALTH_WEIGHTS
        )

        return health

    def _get_health_status(self, score: float) -> str:
        """Convert health score to status string."""
        if score >= self.HEALTH_EXCELLENT:
            return "excellent"
        elif score >= self.HEALTH_GOOD:
            return "good"
        elif score >= self.HEALTH_FAIR:
            return "fair"
        elif score >= self.HEALTH_POOR:
            return "poor"
        else:
            return "critical"

    def _generate_alerts(self, metrics: AggregatedMetrics) -> list[str]:
        """Generate alerts for concerning metrics."""
        alerts = []

        # Low confidence alert
        if metrics.classification.avg_confidence < 0.6:
            alerts.append(
                f"⚠️ Average confidence is low ({metrics.classification.avg_confidence:.0%})"
            )

        # High pending feedback
        if metrics.feedback.pending > 50:
            alerts.append(
                f"⚠️ {metrics.feedback.pending} feedback items pending review"
            )

        # Low approval rate
        if metrics.feedback.approval_rate < 0.5 and metrics.feedback.applied > 0:
            alerts.append(
                f"⚠️ Low approval rate ({metrics.feedback.approval_rate:.0%})"
            )

        # No patterns defined
        if metrics.coverage.total_patterns == 0 and metrics.coverage.total_classes > 0:
            alerts.append("⚠️ No classification patterns defined")

        # Too few predicates
        if metrics.coverage.avg_predicates_per_class < 2:
            alerts.append("⚠️ Average predicates per class is very low")

        return alerts

    def _generate_recommendations(self, metrics: AggregatedMetrics) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Process pending feedback
        if metrics.feedback.pending > 10:
            recommendations.append(
                "Process pending feedback to improve classification accuracy"
            )

        # Add more seeds
        if metrics.coverage.seed_predicates < metrics.coverage.total_classes * 3:
            recommendations.append(
                "Add more seed predicates (aim for 3+ per class)"
            )

        # Add patterns
        classes_without = metrics.coverage.total_classes - metrics.coverage.classes_with_patterns
        if classes_without > 0:
            recommendations.append(
                f"Add patterns to {classes_without} classes without pattern rules"
            )

        # Address low confidence
        if metrics.classification.low_confidence_count > metrics.classification.high_confidence_count:
            recommendations.append(
                "Review low-confidence classifications for quality improvement"
            )

        return recommendations

    def render_text(self) -> str:
        """
        Render dashboard as formatted text.

        Returns:
            Formatted text representation of the dashboard
        """
        state = self.get_state()

        lines = [
            "╔════════════════════════════════════════════════════════════════╗",
            "║           CLASSIFICATION QUALITY DASHBOARD                      ║",
            "╠════════════════════════════════════════════════════════════════╣",
            "",
        ]

        # Health score
        health_bar = self._render_health_bar(state.health_score)
        lines.extend([
            f"  Health Score: {state.health_score:.0%} [{health_bar}] {state.health_status.upper()}",
            "",
        ])

        # Key metrics
        lines.append("  ── Key Metrics ──────────────────────────────────────────────")
        for snapshot in state.snapshots[:6]:
            trend_icon = {"up": "↑", "down": "↓", "stable": "→", "unknown": " "}
            icon = trend_icon.get(snapshot.trend.value, " ")
            lines.append(
                f"  {snapshot.name:20} {snapshot.current_value:>8.0f} {snapshot.unit:10} {icon}"
            )

        # Alerts
        if state.alerts:
            lines.extend(["", "  ── Alerts ───────────────────────────────────────────────────"])
            for alert in state.alerts:
                lines.append(f"  {alert}")

        # Recommendations
        if state.recommendations:
            lines.extend(["", "  ── Recommendations ──────────────────────────────────────────"])
            for i, rec in enumerate(state.recommendations, 1):
                lines.append(f"  {i}. {rec}")

        lines.extend([
            "",
            "╚════════════════════════════════════════════════════════════════╝",
            f"  Last updated: {state.metrics.collected_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        ])

        return "\n".join(lines)

    def _render_health_bar(self, score: float, width: int = 20) -> str:
        """Render a visual health bar."""
        filled = int(score * width)
        empty = width - filled
        return "█" * filled + "░" * empty

    def to_dict(self) -> dict:
        """
        Get dashboard data as dictionary.

        Returns:
            Dictionary with all dashboard data
        """
        state = self.get_state()

        return {
            "health": {
                "score": state.health_score,
                "status": state.health_status,
            },
            "metrics": state.metrics.to_dict(),
            "snapshots": [s.to_dict() for s in state.snapshots],
            "alerts": state.alerts,
            "recommendations": state.recommendations,
        }
