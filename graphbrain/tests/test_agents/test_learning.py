"""Tests for active learning modules."""

import pytest
import math
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from graphbrain.agents.learning.sampler import (
    SamplingStrategy,
    LearningCandidate,
    ActiveLearningSampler,
)
from graphbrain.agents.learning.suggestions import (
    SuggestionType,
    SuggestionPriority,
    ImprovementSuggestion,
    SuggestionEngine,
)


class TestSamplingStrategy:
    """Tests for SamplingStrategy enum."""

    def test_all_strategies_defined(self):
        """Test all sampling strategies are defined."""
        assert SamplingStrategy.UNCERTAINTY == "uncertainty"
        assert SamplingStrategy.DIVERSITY == "diversity"
        assert SamplingStrategy.HYBRID == "hybrid"
        assert SamplingStrategy.RANDOM == "random"
        assert SamplingStrategy.MARGIN == "margin"
        assert SamplingStrategy.ENTROPY == "entropy"

    def test_strategy_count(self):
        """Test number of strategies."""
        assert len(SamplingStrategy) == 6


class TestLearningCandidate:
    """Tests for LearningCandidate dataclass."""

    def test_create_candidate(self):
        """Test creating a learning candidate."""
        candidate = LearningCandidate(
            predicate="announce",
            current_class="claim",
            confidence=0.75,
            frequency=10,
        )

        assert candidate.predicate == "announce"
        assert candidate.current_class == "claim"
        assert candidate.confidence == 0.75
        assert candidate.frequency == 10

    def test_default_values(self):
        """Test default values for optional fields."""
        candidate = LearningCandidate(predicate="test")

        assert candidate.edge_key is None
        assert candidate.current_class is None
        assert candidate.suggested_class is None
        assert candidate.confidence == 0.0
        assert candidate.margin == 0.0
        assert candidate.entropy == 0.0
        assert candidate.frequency == 0
        assert candidate.example_edges == []
        assert candidate.class_distribution == {}
        assert candidate.informativeness_score == 0.0

    def test_created_at_set(self):
        """Test created_at is set to current time."""
        before = datetime.now(timezone.utc)
        candidate = LearningCandidate(predicate="test")
        after = datetime.now(timezone.utc)

        assert before <= candidate.created_at <= after


class TestActiveLearningSampler:
    """Tests for ActiveLearningSampler."""

    @pytest.fixture
    def mock_backend(self):
        """Create mock classification backend."""
        backend = MagicMock()
        backend.list_classes.return_value = []
        return backend

    @pytest.fixture
    def sampler(self, mock_backend):
        """Create sampler instance."""
        return ActiveLearningSampler(mock_backend)

    def test_init_default_strategy(self, mock_backend):
        """Test default initialization."""
        sampler = ActiveLearningSampler(mock_backend)

        assert sampler._strategy == SamplingStrategy.HYBRID
        assert sampler._threshold == 0.7
        assert sampler._weights == ActiveLearningSampler.DEFAULT_WEIGHTS

    def test_init_custom_strategy(self, mock_backend):
        """Test initialization with custom strategy."""
        sampler = ActiveLearningSampler(
            mock_backend,
            strategy=SamplingStrategy.ENTROPY,
            threshold=0.8,
        )

        assert sampler._strategy == SamplingStrategy.ENTROPY
        assert sampler._threshold == 0.8

    def test_init_custom_weights(self, mock_backend):
        """Test initialization with custom weights."""
        custom_weights = {"uncertainty": 0.5, "frequency": 0.5}
        sampler = ActiveLearningSampler(mock_backend, weights=custom_weights)

        assert sampler._weights == custom_weights

    def test_uncertainty_score_at_center(self, sampler):
        """Test uncertainty score at center (0.5)."""
        score = sampler._uncertainty_score(0.5)
        assert score == 1.0  # Maximum uncertainty at 0.5

    def test_uncertainty_score_at_extremes(self, sampler):
        """Test uncertainty score at extremes."""
        score_low = sampler._uncertainty_score(0.0)
        score_high = sampler._uncertainty_score(1.0)

        assert score_low == 0.0
        assert score_high == 0.0

    def test_uncertainty_score_symmetric(self, sampler):
        """Test uncertainty score is symmetric around 0.5."""
        score_below = sampler._uncertainty_score(0.3)
        score_above = sampler._uncertainty_score(0.7)

        assert abs(score_below - score_above) < 0.01

    def test_calculate_entropy_empty(self, sampler):
        """Test entropy calculation with empty list."""
        entropy = sampler._calculate_entropy([])
        assert entropy == 0.0

    def test_calculate_entropy_uniform(self, sampler):
        """Test entropy calculation with uniform distribution."""
        # Uniform distribution has max entropy
        entropy = sampler._calculate_entropy([0.25, 0.25, 0.25, 0.25])
        assert abs(entropy - 1.0) < 0.01

    def test_calculate_entropy_peaked(self, sampler):
        """Test entropy calculation with peaked distribution."""
        # One dominant class has low entropy
        entropy = sampler._calculate_entropy([0.9, 0.05, 0.05])
        assert entropy < 0.5

    def test_calculate_entropy_single(self, sampler):
        """Test entropy calculation with single probability."""
        entropy = sampler._calculate_entropy([1.0])
        assert entropy == 0.0

    def test_calculate_margin_single(self, sampler):
        """Test margin calculation with single probability."""
        margin = sampler._calculate_margin([0.8])
        assert margin == 1.0

    def test_calculate_margin_two_close(self, sampler):
        """Test margin calculation with close probabilities."""
        margin = sampler._calculate_margin([0.51, 0.49])
        assert abs(margin - 0.02) < 0.001

    def test_calculate_margin_two_spread(self, sampler):
        """Test margin calculation with spread probabilities."""
        margin = sampler._calculate_margin([0.9, 0.1])
        assert margin == 0.8

    def test_score_candidates_random(self, mock_backend):
        """Test random scoring strategy."""
        sampler = ActiveLearningSampler(mock_backend, strategy=SamplingStrategy.RANDOM)
        candidates = [
            LearningCandidate("pred1", confidence=0.8),
            LearningCandidate("pred2", confidence=0.5),
        ]

        with patch("random.random", side_effect=[0.3, 0.7]):
            scored = sampler._score_candidates(candidates)

        assert scored[0].informativeness_score == 0.3
        assert scored[1].informativeness_score == 0.7

    def test_score_candidates_uncertainty(self, mock_backend):
        """Test uncertainty scoring strategy."""
        sampler = ActiveLearningSampler(mock_backend, strategy=SamplingStrategy.UNCERTAINTY)
        candidates = [
            LearningCandidate("pred1", confidence=0.5),  # Max uncertainty
            LearningCandidate("pred2", confidence=0.9),  # Low uncertainty
        ]

        scored = sampler._score_candidates(candidates)

        # 0.5 confidence should score higher than 0.9
        assert scored[0].informativeness_score > scored[1].informativeness_score

    def test_score_candidates_margin(self, mock_backend):
        """Test margin scoring strategy."""
        sampler = ActiveLearningSampler(mock_backend, strategy=SamplingStrategy.MARGIN)
        candidates = [
            LearningCandidate("pred1", margin=0.1),  # Low margin = high uncertainty
            LearningCandidate("pred2", margin=0.8),  # High margin = low uncertainty
        ]

        scored = sampler._score_candidates(candidates)

        # Low margin should score higher
        assert scored[0].informativeness_score > scored[1].informativeness_score

    def test_score_candidates_entropy(self, mock_backend):
        """Test entropy scoring strategy."""
        sampler = ActiveLearningSampler(mock_backend, strategy=SamplingStrategy.ENTROPY)
        candidates = [
            LearningCandidate("pred1", entropy=0.9),  # High entropy
            LearningCandidate("pred2", entropy=0.2),  # Low entropy
        ]

        scored = sampler._score_candidates(candidates)

        # High entropy should score higher
        assert scored[0].informativeness_score > scored[1].informativeness_score

    def test_score_diversity(self, mock_backend):
        """Test diversity scoring."""
        sampler = ActiveLearningSampler(mock_backend, strategy=SamplingStrategy.DIVERSITY)
        candidates = [
            LearningCandidate("pred1", current_class="claim"),
            LearningCandidate("pred2", current_class="claim"),
            LearningCandidate("pred3", current_class="claim"),
            LearningCandidate("pred4", current_class="rare"),  # Rare class
        ]

        scored = sampler._score_diversity(candidates)

        # Find the rare class candidate
        rare_candidate = next(c for c in scored if c.current_class == "rare")
        claim_candidate = next(c for c in scored if c.current_class == "claim")

        # Rare class should score higher
        assert rare_candidate.informativeness_score > claim_candidate.informativeness_score

    def test_score_diversity_unclassified(self, mock_backend):
        """Test diversity scoring for unclassified."""
        sampler = ActiveLearningSampler(mock_backend, strategy=SamplingStrategy.DIVERSITY)
        candidates = [
            LearningCandidate("pred1", current_class="claim"),
            LearningCandidate("pred2", current_class=None),  # Unclassified
        ]

        scored = sampler._score_diversity(candidates)

        # Unclassified should get highest score
        unclassified = next(c for c in scored if c.current_class is None)
        assert unclassified.informativeness_score == 1.0

    def test_score_hybrid(self, mock_backend):
        """Test hybrid scoring combines factors."""
        sampler = ActiveLearningSampler(mock_backend, strategy=SamplingStrategy.HYBRID)
        candidates = [
            LearningCandidate("pred1", confidence=0.5, frequency=100, margin=0.1, current_class="claim"),
            LearningCandidate("pred2", confidence=0.9, frequency=10, margin=0.9, current_class="claim"),
        ]

        scored = sampler._score_hybrid(candidates)

        # First candidate should score higher (more uncertain, higher frequency)
        assert scored[0].informativeness_score > scored[1].informativeness_score

    def test_get_candidates_empty(self, mock_backend):
        """Test getting candidates from empty backend."""
        mock_backend.list_classes.return_value = []
        sampler = ActiveLearningSampler(mock_backend)

        candidates = list(sampler.get_candidates())
        assert candidates == []

    def test_get_unclassified_predicates(self, mock_backend):
        """Test finding unclassified predicates."""
        mock_backend.find_predicate.side_effect = lambda p: iter([]) if p != "known" else iter([MagicMock()])
        sampler = ActiveLearningSampler(mock_backend)

        predicates = ["known", "unknown1", "unknown2"]
        unclassified = sampler.get_unclassified_predicates(predicates)

        assert len(unclassified) == 2
        assert unclassified[0].predicate == "unknown1"
        assert unclassified[0].informativeness_score == 1.0

    def test_get_unclassified_predicates_limit(self, mock_backend):
        """Test limit on unclassified predicates."""
        mock_backend.find_predicate.return_value = iter([])
        sampler = ActiveLearningSampler(mock_backend)

        predicates = ["p1", "p2", "p3", "p4", "p5"]
        unclassified = sampler.get_unclassified_predicates(predicates, limit=2)

        assert len(unclassified) == 2

    def test_suggest_batch_size_small(self, sampler):
        """Test batch size suggestion for small dataset."""
        size = sampler.suggest_batch_size(total_predicates=25)
        assert 5 <= size <= 20

    def test_suggest_batch_size_large(self, sampler):
        """Test batch size suggestion for large dataset."""
        size = sampler.suggest_batch_size(total_predicates=10000)
        assert size <= 100

    def test_suggest_batch_size_low_accuracy(self, sampler):
        """Test batch size increases with low accuracy."""
        size_high_acc = sampler.suggest_batch_size(total_predicates=100, current_accuracy=0.9)
        size_low_acc = sampler.suggest_batch_size(total_predicates=100, current_accuracy=0.5)

        assert size_low_acc > size_high_acc


class TestSuggestionType:
    """Tests for SuggestionType enum."""

    def test_all_types_defined(self):
        """Test all suggestion types are defined."""
        assert SuggestionType.ADD_SEED == "add_seed"
        assert SuggestionType.REVIEW_BORDERLINE == "review_borderline"
        assert SuggestionType.MERGE_CLASSES == "merge_classes"
        assert SuggestionType.SPLIT_CLASS == "split_class"
        assert SuggestionType.ADD_PATTERN == "add_pattern"
        assert SuggestionType.ADJUST_THRESHOLD == "adjust_threshold"
        assert SuggestionType.EXPAND_COVERAGE == "expand_coverage"
        assert SuggestionType.RESOLVE_CONFLICT == "resolve_conflict"


class TestSuggestionPriority:
    """Tests for SuggestionPriority enum."""

    def test_all_priorities_defined(self):
        """Test all priorities are defined."""
        assert SuggestionPriority.CRITICAL == "critical"
        assert SuggestionPriority.HIGH == "high"
        assert SuggestionPriority.MEDIUM == "medium"
        assert SuggestionPriority.LOW == "low"


class TestImprovementSuggestion:
    """Tests for ImprovementSuggestion dataclass."""

    def test_create_suggestion(self):
        """Test creating a suggestion."""
        suggestion = ImprovementSuggestion(
            suggestion_id="sug_001",
            suggestion_type=SuggestionType.ADD_SEED,
            priority=SuggestionPriority.HIGH,
            title="Add seed predicates",
            description="Add more seed predicates to improve coverage",
            affected_class="claim",
        )

        assert suggestion.suggestion_id == "sug_001"
        assert suggestion.suggestion_type == SuggestionType.ADD_SEED
        assert suggestion.priority == SuggestionPriority.HIGH
        assert suggestion.affected_class == "claim"

    def test_default_values(self):
        """Test default values."""
        suggestion = ImprovementSuggestion(
            suggestion_id="sug_001",
            suggestion_type=SuggestionType.ADD_SEED,
            priority=SuggestionPriority.LOW,
            title="Test",
            description="Test",
        )

        assert suggestion.affected_class is None
        assert suggestion.affected_predicates == []
        assert suggestion.expected_impact == ""
        assert suggestion.action_items == []
        assert suggestion.evidence == {}

    def test_to_dict(self):
        """Test serialization to dict."""
        suggestion = ImprovementSuggestion(
            suggestion_id="sug_001",
            suggestion_type=SuggestionType.ADD_SEED,
            priority=SuggestionPriority.HIGH,
            title="Test",
            description="Test description",
            affected_class="claim",
            action_items=["Do this", "Do that"],
        )

        d = suggestion.to_dict()

        assert d["suggestion_id"] == "sug_001"
        assert d["type"] == "add_seed"
        assert d["priority"] == "high"
        assert d["affected_class"] == "claim"
        assert len(d["action_items"]) == 2


class TestSuggestionEngine:
    """Tests for SuggestionEngine."""

    @pytest.fixture
    def mock_backend(self):
        """Create mock classification backend."""
        backend = MagicMock()
        backend.list_classes.return_value = []
        backend.get_stats.return_value = {}
        return backend

    @pytest.fixture
    def engine(self, mock_backend):
        """Create engine instance."""
        return SuggestionEngine(mock_backend)

    def test_init(self, mock_backend):
        """Test initialization."""
        engine = SuggestionEngine(mock_backend)
        assert engine._backend == mock_backend
        assert engine._suggestion_counter == 0

    def test_generate_id(self, engine):
        """Test ID generation."""
        id1 = engine._generate_id()
        id2 = engine._generate_id()

        assert id1.startswith("sug_")
        assert id2.startswith("sug_")
        assert id1 != id2

    def test_analyze_empty_backend(self, engine):
        """Test analysis on empty backend."""
        suggestions = engine.analyze()
        assert isinstance(suggestions, list)

    def test_check_class_coverage_insufficient_seeds(self, mock_backend):
        """Test detecting classes with insufficient seeds."""
        # Create a class with only 1 seed predicate
        mock_class = MagicMock()
        mock_class.id = "claim"
        mock_class.name = "Claim"
        mock_backend.list_classes.return_value = [mock_class]

        pred = MagicMock()
        pred.is_seed = True
        mock_backend.get_predicates_by_class.return_value = [pred]
        mock_backend.get_stats.return_value = {}

        engine = SuggestionEngine(mock_backend)
        suggestions = engine._check_class_coverage({})

        assert len(suggestions) > 0
        assert suggestions[0].suggestion_type == SuggestionType.ADD_SEED

    def test_check_class_coverage_no_seeds(self, mock_backend):
        """Test detecting classes with zero seeds."""
        mock_class = MagicMock()
        mock_class.id = "claim"
        mock_class.name = "Claim"
        mock_backend.list_classes.return_value = [mock_class]
        mock_backend.get_predicates_by_class.return_value = []
        mock_backend.get_stats.return_value = {}

        engine = SuggestionEngine(mock_backend)
        suggestions = engine._check_class_coverage({})

        assert len(suggestions) > 0
        assert suggestions[0].priority == SuggestionPriority.HIGH

    def test_check_confidence_distribution_low(self, engine):
        """Test detecting low average confidence."""
        stats = {"avg_confidence": 0.5}
        suggestions = engine._check_confidence_distribution(stats)

        assert len(suggestions) > 0
        assert suggestions[0].suggestion_type == SuggestionType.REVIEW_BORDERLINE

    def test_check_confidence_distribution_ok(self, engine):
        """Test no suggestion when confidence is ok."""
        stats = {"avg_confidence": 0.85}
        suggestions = engine._check_confidence_distribution(stats)

        assert len(suggestions) == 0

    def test_check_feedback_patterns_high_pending(self, engine):
        """Test detecting high pending feedback."""
        stats = {"pending_feedback": 50}
        suggestions = engine._check_feedback_patterns(stats)

        assert len(suggestions) > 0

    def test_check_feedback_patterns_high_rejection(self, engine):
        """Test detecting high rejection rate."""
        stats = {
            "pending_feedback": 0,
            "total_reviewed": 100,
            "rejected_feedback": 50,  # 50% rejection
        }
        suggestions = engine._check_feedback_patterns(stats)

        assert len(suggestions) > 0
        assert any(s.suggestion_type == SuggestionType.ADJUST_THRESHOLD for s in suggestions)

    def test_check_class_balance_imbalanced(self, mock_backend):
        """Test detecting class imbalance."""
        mock_class1 = MagicMock()
        mock_class1.id = "claim"
        mock_class1.name = "Claim"
        mock_class2 = MagicMock()
        mock_class2.id = "action"
        mock_class2.name = "Action"

        mock_backend.list_classes.return_value = [mock_class1, mock_class2]
        mock_backend.get_predicates_by_class.side_effect = lambda c: (
            [MagicMock()] * 100 if c == "claim" else [MagicMock()]
        )
        mock_backend.get_stats.return_value = {}

        engine = SuggestionEngine(mock_backend)
        suggestions = engine._check_class_balance({})

        assert len(suggestions) > 0
        assert suggestions[0].suggestion_type == SuggestionType.EXPAND_COVERAGE

    def test_check_predicate_conflicts(self, mock_backend):
        """Test detecting predicate conflicts."""
        mock_class1 = MagicMock()
        mock_class1.id = "claim"
        mock_class1.name = "Claim"
        mock_class2 = MagicMock()
        mock_class2.id = "action"
        mock_class2.name = "Action"

        mock_backend.list_classes.return_value = [mock_class1, mock_class2]

        # Same predicate in both classes
        pred = MagicMock()
        pred.lemma = "announce"
        mock_backend.get_predicates_by_class.side_effect = lambda c: [pred]
        mock_backend.get_stats.return_value = {}

        engine = SuggestionEngine(mock_backend)
        suggestions = engine._check_predicate_conflicts()

        assert len(suggestions) > 0
        assert any(s.suggestion_type == SuggestionType.RESOLVE_CONFLICT for s in suggestions)

    def test_analyze_sorts_by_priority(self, mock_backend):
        """Test that analyze sorts by priority."""
        # Set up mixed priorities
        mock_class = MagicMock()
        mock_class.id = "claim"
        mock_class.name = "Claim"
        mock_backend.list_classes.return_value = [mock_class]
        mock_backend.get_predicates_by_class.return_value = []
        mock_backend.get_patterns_by_class.return_value = []
        mock_backend.get_stats.return_value = {
            "avg_confidence": 0.4,  # Low confidence (HIGH priority)
            "pending_feedback": 5,  # Some pending (MEDIUM priority)
        }

        engine = SuggestionEngine(mock_backend)
        suggestions = engine.analyze()

        if len(suggestions) >= 2:
            # Higher priorities should come first
            priority_order = {
                SuggestionPriority.CRITICAL: 0,
                SuggestionPriority.HIGH: 1,
                SuggestionPriority.MEDIUM: 2,
                SuggestionPriority.LOW: 3,
            }
            for i in range(len(suggestions) - 1):
                assert priority_order[suggestions[i].priority] <= priority_order[suggestions[i+1].priority]

    def test_get_summary(self, mock_backend):
        """Test getting suggestion summary."""
        mock_backend.list_classes.return_value = []
        mock_backend.get_stats.return_value = {}

        engine = SuggestionEngine(mock_backend)
        summary = engine.get_summary()

        assert "total" in summary
        assert "by_priority" in summary
        assert "by_type" in summary
        assert "critical_count" in summary
        assert "high_count" in summary
