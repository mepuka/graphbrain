"""Tests for metrics modules."""

import pytest
import tempfile
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from graphbrain.agents.metrics.collectors import (
    ClassificationMetrics,
    FeedbackMetrics,
    CoverageMetrics,
    AggregatedMetrics,
    MetricsCollector,
)
from graphbrain.agents.metrics.dashboard import (
    TrendDirection,
    MetricSnapshot,
    DashboardState,
    QualityDashboard,
)


class TestClassificationMetrics:
    """Tests for ClassificationMetrics dataclass."""

    def test_default_values(self):
        """Test default values."""
        metrics = ClassificationMetrics()

        assert metrics.total_classifications == 0
        assert metrics.avg_confidence == 0.0
        assert metrics.confidence_std == 0.0
        assert metrics.high_confidence_count == 0
        assert metrics.medium_confidence_count == 0
        assert metrics.low_confidence_count == 0
        assert metrics.by_method == {}
        assert metrics.by_class == {}

    def test_set_values(self):
        """Test setting values."""
        metrics = ClassificationMetrics(
            total_classifications=100,
            avg_confidence=0.85,
            high_confidence_count=50,
            by_method={"bank": 70, "semantic": 30},
        )

        assert metrics.total_classifications == 100
        assert metrics.avg_confidence == 0.85
        assert metrics.by_method["bank"] == 70


class TestFeedbackMetrics:
    """Tests for FeedbackMetrics dataclass."""

    def test_default_values(self):
        """Test default values."""
        metrics = FeedbackMetrics()

        assert metrics.total_feedback == 0
        assert metrics.pending == 0
        assert metrics.applied == 0
        assert metrics.rejected == 0
        assert metrics.approval_rate == 0.0
        assert metrics.by_class_correction == {}

    def test_set_values(self):
        """Test setting values."""
        metrics = FeedbackMetrics(
            total_feedback=50,
            pending=10,
            applied=30,
            rejected=10,
            approval_rate=0.75,
        )

        assert metrics.total_feedback == 50
        assert metrics.applied == 30
        assert metrics.approval_rate == 0.75


class TestCoverageMetrics:
    """Tests for CoverageMetrics dataclass."""

    def test_default_values(self):
        """Test default values."""
        metrics = CoverageMetrics()

        assert metrics.total_classes == 0
        assert metrics.total_predicates == 0
        assert metrics.avg_predicates_per_class == 0.0
        assert metrics.seed_predicates == 0
        assert metrics.discovered_predicates == 0
        assert metrics.total_patterns == 0
        assert metrics.classes_with_patterns == 0

    def test_set_values(self):
        """Test setting values."""
        metrics = CoverageMetrics(
            total_classes=5,
            total_predicates=50,
            avg_predicates_per_class=10.0,
            seed_predicates=15,
            discovered_predicates=35,
        )

        assert metrics.total_classes == 5
        assert metrics.avg_predicates_per_class == 10.0


class TestAggregatedMetrics:
    """Tests for AggregatedMetrics dataclass."""

    def test_create(self):
        """Test creating aggregated metrics."""
        classification = ClassificationMetrics(total_classifications=100)
        feedback = FeedbackMetrics(pending=5)
        coverage = CoverageMetrics(total_classes=3)

        metrics = AggregatedMetrics(
            classification=classification,
            feedback=feedback,
            coverage=coverage,
        )

        assert metrics.classification.total_classifications == 100
        assert metrics.feedback.pending == 5
        assert metrics.coverage.total_classes == 3

    def test_to_dict(self):
        """Test serialization to dict."""
        metrics = AggregatedMetrics(
            classification=ClassificationMetrics(
                total_classifications=100,
                avg_confidence=0.85,
                high_confidence_count=60,
                medium_confidence_count=30,
                low_confidence_count=10,
            ),
            feedback=FeedbackMetrics(
                pending=5,
                applied=20,
                rejected=3,
                approval_rate=0.87,
            ),
            coverage=CoverageMetrics(
                total_classes=4,
                total_predicates=40,
                seed_predicates=10,
            ),
        )

        d = metrics.to_dict()

        assert d["classification"]["total"] == 100
        assert d["classification"]["avg_confidence"] == 0.85
        assert d["classification"]["by_confidence"]["high"] == 60
        assert d["feedback"]["pending"] == 5
        assert d["coverage"]["classes"] == 4
        assert "collected_at" in d


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    @pytest.fixture
    def mock_backend(self):
        """Create mock classification backend."""
        backend = MagicMock()
        backend.list_classes.return_value = []
        backend.get_pending_feedback.return_value = []
        backend.get_stats.return_value = {}
        return backend

    @pytest.fixture
    def collector(self, mock_backend):
        """Create collector instance."""
        return MetricsCollector(mock_backend)

    def test_init(self, mock_backend):
        """Test initialization."""
        collector = MetricsCollector(mock_backend)
        assert collector._backend == mock_backend

    def test_collect_all_empty(self, collector):
        """Test collecting metrics from empty backend."""
        metrics = collector.collect_all()

        assert isinstance(metrics, AggregatedMetrics)
        assert metrics.classification.total_classifications == 0
        assert metrics.feedback.pending == 0
        assert metrics.coverage.total_classes == 0

    def test_collect_classification_metrics(self, mock_backend):
        """Test collecting classification metrics."""
        mock_class = MagicMock()
        mock_class.id = "claim"
        mock_backend.list_classes.return_value = [mock_class]

        # Create mock classifications
        classifications = [
            MagicMock(confidence=0.95, method="bank"),
            MagicMock(confidence=0.85, method="semantic"),
            MagicMock(confidence=0.60, method="semantic"),
        ]
        mock_backend.get_edges_by_class.return_value = classifications
        mock_backend.get_pending_feedback.return_value = []
        mock_backend.get_stats.return_value = {}
        mock_backend.get_predicates_by_class.return_value = []
        mock_backend.get_patterns_by_class.return_value = []

        collector = MetricsCollector(mock_backend)
        metrics = collector._collect_classification_metrics()

        assert metrics.total_classifications == 3
        assert metrics.high_confidence_count == 1  # 0.95
        assert metrics.medium_confidence_count == 1  # 0.85
        assert metrics.low_confidence_count == 1  # 0.60
        assert metrics.by_method["bank"] == 1
        assert metrics.by_method["semantic"] == 2

    def test_collect_feedback_metrics(self, mock_backend):
        """Test collecting feedback metrics."""
        pending = [MagicMock(), MagicMock(), MagicMock()]
        mock_backend.get_pending_feedback.return_value = pending
        mock_backend.get_stats.return_value = {
            "total_feedback": 50,
            "applied_feedback": 30,
            "rejected_feedback": 10,
        }

        collector = MetricsCollector(mock_backend)
        metrics = collector._collect_feedback_metrics()

        assert metrics.pending == 3
        assert metrics.total_feedback == 50
        assert metrics.applied == 30
        assert metrics.rejected == 10
        assert metrics.approval_rate == 0.75  # 30 / (30 + 10)

    def test_collect_coverage_metrics(self, mock_backend):
        """Test collecting coverage metrics."""
        mock_class = MagicMock()
        mock_class.id = "claim"
        mock_backend.list_classes.return_value = [mock_class]

        predicates = [
            MagicMock(is_seed=True),
            MagicMock(is_seed=True),
            MagicMock(is_seed=False),
            MagicMock(is_seed=False),
            MagicMock(is_seed=False),
        ]
        mock_backend.get_predicates_by_class.return_value = predicates

        patterns = [MagicMock(), MagicMock()]
        mock_backend.get_patterns_by_class.return_value = patterns
        mock_backend.get_pending_feedback.return_value = []
        mock_backend.get_stats.return_value = {}

        collector = MetricsCollector(mock_backend)
        metrics = collector._collect_coverage_metrics()

        assert metrics.total_classes == 1
        assert metrics.total_predicates == 5
        assert metrics.seed_predicates == 2
        assert metrics.discovered_predicates == 3
        assert metrics.total_patterns == 2
        assert metrics.classes_with_patterns == 1
        assert metrics.avg_predicates_per_class == 5.0

    def test_get_confidence_histogram(self, mock_backend):
        """Test getting confidence histogram."""
        mock_class = MagicMock()
        mock_class.id = "claim"
        mock_backend.list_classes.return_value = [mock_class]

        classifications = [
            MagicMock(confidence=0.05),
            MagicMock(confidence=0.15),
            MagicMock(confidence=0.55),
            MagicMock(confidence=0.75),
            MagicMock(confidence=0.95),
        ]
        mock_backend.get_edges_by_class.return_value = classifications

        collector = MetricsCollector(mock_backend)
        histogram = collector.get_confidence_histogram(bins=5)

        assert len(histogram) == 5
        # Check first bin 0.0-0.2 has 2 items
        assert histogram[0]["count"] == 2
        # Last bin 0.8-1.0 has 1 item
        assert histogram[-1]["count"] == 1

    def test_get_confidence_histogram_empty(self, collector):
        """Test histogram on empty data."""
        histogram = collector.get_confidence_histogram()
        assert histogram == []

    def test_get_class_distribution(self, mock_backend):
        """Test getting class distribution."""
        classes = [
            MagicMock(id="claim", name="Claim"),
            MagicMock(id="action", name="Action"),
        ]
        mock_backend.list_classes.return_value = classes
        mock_backend.get_predicates_by_class.side_effect = lambda c: (
            [MagicMock()] * 10 if c == "claim" else [MagicMock()] * 5
        )

        collector = MetricsCollector(mock_backend)
        distribution = collector.get_class_distribution()

        assert len(distribution) == 2
        # Sorted by count descending
        assert distribution[0]["class_id"] == "claim"
        assert distribution[0]["count"] == 10
        assert distribution[1]["class_id"] == "action"
        assert distribution[1]["count"] == 5

    def test_get_method_effectiveness(self, mock_backend):
        """Test getting method effectiveness."""
        mock_class = MagicMock()
        mock_class.id = "claim"
        mock_backend.list_classes.return_value = [mock_class]

        classifications = [
            MagicMock(confidence=0.95, method="bank"),
            MagicMock(confidence=0.90, method="bank"),
            MagicMock(confidence=0.70, method="semantic"),
        ]
        mock_backend.get_edges_by_class.return_value = classifications

        collector = MetricsCollector(mock_backend)
        effectiveness = collector.get_method_effectiveness()

        assert "bank" in effectiveness
        assert effectiveness["bank"]["count"] == 2
        assert effectiveness["bank"]["avg_confidence"] == 0.925
        assert effectiveness["semantic"]["count"] == 1
        assert effectiveness["semantic"]["avg_confidence"] == 0.70


class TestTrendDirection:
    """Tests for TrendDirection enum."""

    def test_all_directions_defined(self):
        """Test all directions are defined."""
        assert TrendDirection.UP == "up"
        assert TrendDirection.DOWN == "down"
        assert TrendDirection.STABLE == "stable"
        assert TrendDirection.UNKNOWN == "unknown"


class TestMetricSnapshot:
    """Tests for MetricSnapshot dataclass."""

    def test_create(self):
        """Test creating a snapshot."""
        snapshot = MetricSnapshot(
            name="Classifications",
            current_value=100,
            previous_value=90,
            trend=TrendDirection.UP,
            change_percent=11.1,
            unit="edges",
            description="Total classifications",
        )

        assert snapshot.name == "Classifications"
        assert snapshot.current_value == 100
        assert snapshot.trend == TrendDirection.UP

    def test_default_values(self):
        """Test default values."""
        snapshot = MetricSnapshot(name="Test", current_value=50)

        assert snapshot.previous_value is None
        assert snapshot.trend == TrendDirection.UNKNOWN
        assert snapshot.change_percent == 0.0
        assert snapshot.unit == ""
        assert snapshot.description == ""

    def test_to_dict(self):
        """Test serialization."""
        snapshot = MetricSnapshot(
            name="Test",
            current_value=100,
            trend=TrendDirection.UP,
        )

        d = snapshot.to_dict()

        assert d["name"] == "Test"
        assert d["current"] == 100
        assert d["trend"] == "up"


class TestQualityDashboard:
    """Tests for QualityDashboard."""

    @pytest.fixture
    def mock_backend(self):
        """Create mock classification backend."""
        backend = MagicMock()
        backend.list_classes.return_value = []
        backend.get_pending_feedback.return_value = []
        backend.get_stats.return_value = {}
        return backend

    @pytest.fixture
    def dashboard(self, mock_backend):
        """Create dashboard instance."""
        return QualityDashboard(mock_backend)

    def test_init(self, mock_backend):
        """Test initialization."""
        dashboard = QualityDashboard(mock_backend)

        assert dashboard._backend == mock_backend
        assert dashboard._collector is not None
        assert dashboard._previous_metrics is None

    def test_init_with_history_file(self, mock_backend):
        """Test initialization with history file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            dashboard = QualityDashboard(mock_backend, history_file=f.name)
            assert dashboard._history_file == f.name

    def test_get_state(self, dashboard):
        """Test getting dashboard state."""
        state = dashboard.get_state()

        assert isinstance(state, DashboardState)
        assert isinstance(state.metrics, AggregatedMetrics)
        assert isinstance(state.snapshots, list)
        assert isinstance(state.health_score, float)
        assert isinstance(state.health_status, str)
        assert isinstance(state.alerts, list)
        assert isinstance(state.recommendations, list)

    def test_calculate_health_score_empty(self, dashboard):
        """Test health score on empty data."""
        metrics = AggregatedMetrics(
            classification=ClassificationMetrics(),
            feedback=FeedbackMetrics(),
            coverage=CoverageMetrics(),
        )

        score = dashboard._calculate_health_score(metrics)

        assert 0.0 <= score <= 1.0

    def test_calculate_health_score_good(self, dashboard):
        """Test health score with good metrics."""
        metrics = AggregatedMetrics(
            classification=ClassificationMetrics(avg_confidence=0.9),
            feedback=FeedbackMetrics(pending=0, applied=100),
            coverage=CoverageMetrics(
                total_classes=5,
                total_predicates=50,
                avg_predicates_per_class=10.0,
                seed_predicates=15,
            ),
        )

        score = dashboard._calculate_health_score(metrics)

        assert score > 0.5  # Should be reasonably healthy

    def test_get_health_status_excellent(self, dashboard):
        """Test excellent health status."""
        status = dashboard._get_health_status(0.95)
        assert status == "excellent"

    def test_get_health_status_good(self, dashboard):
        """Test good health status."""
        status = dashboard._get_health_status(0.75)
        assert status == "good"

    def test_get_health_status_fair(self, dashboard):
        """Test fair health status."""
        status = dashboard._get_health_status(0.55)
        assert status == "fair"

    def test_get_health_status_poor(self, dashboard):
        """Test poor health status."""
        status = dashboard._get_health_status(0.35)
        assert status == "poor"

    def test_get_health_status_critical(self, dashboard):
        """Test critical health status."""
        status = dashboard._get_health_status(0.15)
        assert status == "critical"

    def test_generate_alerts_low_confidence(self, dashboard):
        """Test alert generation for low confidence."""
        metrics = AggregatedMetrics(
            classification=ClassificationMetrics(avg_confidence=0.5),
            feedback=FeedbackMetrics(),
            coverage=CoverageMetrics(),
        )

        alerts = dashboard._generate_alerts(metrics)

        assert any("confidence" in a.lower() for a in alerts)

    def test_generate_alerts_high_pending(self, dashboard):
        """Test alert generation for high pending feedback."""
        metrics = AggregatedMetrics(
            classification=ClassificationMetrics(avg_confidence=0.8),
            feedback=FeedbackMetrics(pending=100),
            coverage=CoverageMetrics(),
        )

        alerts = dashboard._generate_alerts(metrics)

        assert any("pending" in a.lower() for a in alerts)

    def test_generate_alerts_low_approval(self, dashboard):
        """Test alert generation for low approval rate."""
        metrics = AggregatedMetrics(
            classification=ClassificationMetrics(avg_confidence=0.8),
            feedback=FeedbackMetrics(approval_rate=0.3, applied=10, pending=0),
            coverage=CoverageMetrics(),
        )

        alerts = dashboard._generate_alerts(metrics)

        assert any("approval" in a.lower() for a in alerts)

    def test_generate_alerts_no_patterns(self, dashboard):
        """Test alert generation for missing patterns."""
        metrics = AggregatedMetrics(
            classification=ClassificationMetrics(avg_confidence=0.8),
            feedback=FeedbackMetrics(),
            coverage=CoverageMetrics(total_classes=5, total_patterns=0),
        )

        alerts = dashboard._generate_alerts(metrics)

        assert any("pattern" in a.lower() for a in alerts)

    def test_generate_recommendations_pending(self, dashboard):
        """Test recommendations for pending feedback."""
        metrics = AggregatedMetrics(
            classification=ClassificationMetrics(),
            feedback=FeedbackMetrics(pending=20),
            coverage=CoverageMetrics(),
        )

        recommendations = dashboard._generate_recommendations(metrics)

        assert any("pending" in r.lower() for r in recommendations)

    def test_generate_recommendations_seeds(self, dashboard):
        """Test recommendations for seed predicates."""
        metrics = AggregatedMetrics(
            classification=ClassificationMetrics(),
            feedback=FeedbackMetrics(),
            coverage=CoverageMetrics(total_classes=5, seed_predicates=5),
        )

        recommendations = dashboard._generate_recommendations(metrics)

        assert any("seed" in r.lower() for r in recommendations)

    def test_calculate_snapshots(self, dashboard):
        """Test calculating snapshots."""
        metrics = AggregatedMetrics(
            classification=ClassificationMetrics(
                total_classifications=100,
                avg_confidence=0.85,
                high_confidence_count=60,
            ),
            feedback=FeedbackMetrics(pending=5, approval_rate=0.9),
            coverage=CoverageMetrics(
                total_classes=4,
                total_predicates=40,
                total_patterns=10,
            ),
        )

        snapshots = dashboard._calculate_snapshots(metrics)

        assert len(snapshots) >= 8
        names = [s.name for s in snapshots]
        assert "Classifications" in names
        assert "Average Confidence" in names
        assert "Pending Reviews" in names

    def test_add_trends(self, dashboard):
        """Test adding trends to snapshots."""
        current = AggregatedMetrics(
            classification=ClassificationMetrics(total_classifications=110),
            feedback=FeedbackMetrics(),
            coverage=CoverageMetrics(),
        )
        previous = AggregatedMetrics(
            classification=ClassificationMetrics(total_classifications=100),
            feedback=FeedbackMetrics(),
            coverage=CoverageMetrics(),
        )

        snapshots = [
            MetricSnapshot(name="Classifications", current_value=110),
        ]

        dashboard._add_trends(snapshots, current, previous)

        assert snapshots[0].previous_value == 100
        assert snapshots[0].trend == TrendDirection.UP
        assert snapshots[0].change_percent == 10.0

    def test_render_health_bar(self, dashboard):
        """Test rendering health bar."""
        bar_full = dashboard._render_health_bar(1.0, width=10)
        bar_half = dashboard._render_health_bar(0.5, width=10)
        bar_empty = dashboard._render_health_bar(0.0, width=10)

        assert bar_full == "██████████"
        assert bar_half == "█████░░░░░"
        assert bar_empty == "░░░░░░░░░░"

    def test_render_text(self, dashboard):
        """Test rendering dashboard as text."""
        text = dashboard.render_text()

        assert "DASHBOARD" in text
        assert "Health Score" in text

    def test_to_dict(self, dashboard):
        """Test converting dashboard to dict."""
        d = dashboard.to_dict()

        assert "health" in d
        assert "metrics" in d
        assert "snapshots" in d
        assert "alerts" in d
        assert "recommendations" in d

    def test_save_and_load_history(self, mock_backend):
        """Test saving and loading history."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            history_file = f.name

        try:
            # Create dashboard and get state (saves history)
            dashboard = QualityDashboard(mock_backend, history_file=history_file)
            dashboard.get_state()

            # Verify history file was created
            with open(history_file, 'r') as f:
                data = json.load(f)
                assert "classification" in data
                assert "collected_at" in data

        finally:
            import os
            if os.path.exists(history_file):
                os.unlink(history_file)
