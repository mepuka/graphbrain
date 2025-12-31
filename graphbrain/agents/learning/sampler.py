"""Active learning sampler for intelligent sample selection.

Implements uncertainty sampling and other strategies to identify
the most informative samples for human labeling.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Optional, Callable, Any
from datetime import datetime, timezone
import logging
import math

logger = logging.getLogger(__name__)


class SamplingStrategy(str, Enum):
    """Sampling strategies for active learning."""
    UNCERTAINTY = "uncertainty"  # Select samples with confidence near threshold
    DIVERSITY = "diversity"      # Select diverse samples across classes
    HYBRID = "hybrid"            # Combine uncertainty and diversity
    RANDOM = "random"            # Random baseline
    MARGIN = "margin"            # Smallest margin between top-2 classes
    ENTROPY = "entropy"          # Maximum entropy across all classes


@dataclass
class LearningCandidate:
    """A candidate for active learning review."""
    predicate: str
    edge_key: Optional[str] = None
    current_class: Optional[str] = None
    suggested_class: Optional[str] = None
    confidence: float = 0.0
    margin: float = 0.0           # Difference between top-2 class confidences
    entropy: float = 0.0          # Entropy across all class probabilities
    frequency: int = 0            # How often this predicate appears
    example_edges: list[str] = field(default_factory=list)
    class_distribution: dict[str, float] = field(default_factory=dict)
    informativeness_score: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ActiveLearningSampler:
    """
    Sampler for selecting informative samples for human review.

    Uses various active learning strategies to identify samples
    that would most improve the classification model.
    """

    # Confidence threshold where uncertainty is maximized
    UNCERTAINTY_CENTER = 0.5

    # Weights for hybrid scoring
    DEFAULT_WEIGHTS = {
        "uncertainty": 0.4,
        "frequency": 0.3,
        "margin": 0.2,
        "diversity": 0.1,
    }

    def __init__(
        self,
        backend: Any,
        strategy: SamplingStrategy = SamplingStrategy.HYBRID,
        threshold: float = 0.7,
        weights: Optional[dict[str, float]] = None,
    ):
        """
        Initialize the sampler.

        Args:
            backend: Classification backend for data access
            strategy: Sampling strategy to use
            threshold: Confidence threshold for auto-classification
            weights: Custom weights for hybrid scoring
        """
        self._backend = backend
        self._strategy = strategy
        self._threshold = threshold
        self._weights = weights or self.DEFAULT_WEIGHTS

    def get_candidates(
        self,
        limit: int = 20,
        domain: Optional[str] = None,
        min_frequency: int = 1,
    ) -> Iterator[LearningCandidate]:
        """
        Get candidate samples for human review.

        Args:
            limit: Maximum candidates to return
            domain: Optional domain filter
            min_frequency: Minimum frequency for candidates

        Yields:
            LearningCandidate objects sorted by informativeness
        """
        # Get all predicates with their classification info
        candidates = self._collect_candidates(domain, min_frequency)

        # Score candidates based on strategy
        scored = self._score_candidates(candidates)

        # Sort by informativeness and yield top candidates
        scored.sort(key=lambda c: c.informativeness_score, reverse=True)

        for candidate in scored[:limit]:
            yield candidate

    def _collect_candidates(
        self,
        domain: Optional[str],
        min_frequency: int,
    ) -> list[LearningCandidate]:
        """Collect candidate predicates from the backend."""
        candidates = []

        # Get predicates from predicate banks
        seen_predicates = set()

        for sem_class in self._backend.list_classes(domain=domain):
            for pred in self._backend.get_predicates_by_class(sem_class.id):
                if pred.lemma in seen_predicates:
                    continue
                if pred.frequency < min_frequency:
                    continue

                seen_predicates.add(pred.lemma)

                # Check if this predicate has low confidence or is disputed
                all_classes = list(self._backend.find_predicate(pred.lemma))
                class_dist = {}
                for entry, cls in all_classes:
                    score = entry.similarity_score or 0.5
                    class_dist[cls.id] = max(class_dist.get(cls.id, 0), score)

                # Calculate entropy and margin
                if class_dist:
                    probs = list(class_dist.values())
                    total = sum(probs)
                    if total > 0:
                        probs = [p / total for p in probs]
                    entropy = self._calculate_entropy(probs)
                    margin = self._calculate_margin(probs)
                else:
                    entropy = 0.0
                    margin = 1.0

                candidate = LearningCandidate(
                    predicate=pred.lemma,
                    current_class=sem_class.id,
                    confidence=pred.similarity_score or 0.5,
                    frequency=pred.frequency or 1,
                    margin=margin,
                    entropy=entropy,
                    class_distribution=class_dist,
                )
                candidates.append(candidate)

        return candidates

    def _score_candidates(
        self,
        candidates: list[LearningCandidate],
    ) -> list[LearningCandidate]:
        """Score candidates based on the sampling strategy."""
        if self._strategy == SamplingStrategy.RANDOM:
            import random
            for c in candidates:
                c.informativeness_score = random.random()
            return candidates

        elif self._strategy == SamplingStrategy.UNCERTAINTY:
            for c in candidates:
                c.informativeness_score = self._uncertainty_score(c.confidence)
            return candidates

        elif self._strategy == SamplingStrategy.MARGIN:
            for c in candidates:
                # Lower margin = more uncertain between classes
                c.informativeness_score = 1.0 - c.margin
            return candidates

        elif self._strategy == SamplingStrategy.ENTROPY:
            for c in candidates:
                c.informativeness_score = c.entropy
            return candidates

        elif self._strategy == SamplingStrategy.DIVERSITY:
            return self._score_diversity(candidates)

        else:  # HYBRID
            return self._score_hybrid(candidates)

    def _uncertainty_score(self, confidence: float) -> float:
        """
        Calculate uncertainty score.

        Highest when confidence is at the threshold boundary.
        """
        # Distance from uncertainty center (0.5)
        distance = abs(confidence - self.UNCERTAINTY_CENTER)
        # Score is highest when distance is lowest
        return 1.0 - (distance * 2)  # Scale to [0, 1]

    def _calculate_entropy(self, probs: list[float]) -> float:
        """Calculate Shannon entropy of probability distribution."""
        if not probs:
            return 0.0
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log2(p)
        # Normalize by max entropy
        max_entropy = math.log2(len(probs)) if len(probs) > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _calculate_margin(self, probs: list[float]) -> float:
        """Calculate margin between top-2 probabilities."""
        if len(probs) < 2:
            return 1.0
        sorted_probs = sorted(probs, reverse=True)
        return sorted_probs[0] - sorted_probs[1]

    def _score_diversity(
        self,
        candidates: list[LearningCandidate],
    ) -> list[LearningCandidate]:
        """Score candidates to maximize class diversity."""
        # Count current class representation
        class_counts: dict[str, int] = {}
        for c in candidates:
            if c.current_class:
                class_counts[c.current_class] = class_counts.get(c.current_class, 0) + 1

        # Score inversely proportional to class representation
        total = sum(class_counts.values()) or 1
        for c in candidates:
            if c.current_class and c.current_class in class_counts:
                # Rarer classes get higher scores
                c.informativeness_score = 1.0 - (class_counts[c.current_class] / total)
            else:
                c.informativeness_score = 1.0  # Unclassified gets highest

        return candidates

    def _score_hybrid(
        self,
        candidates: list[LearningCandidate],
    ) -> list[LearningCandidate]:
        """Score candidates using weighted combination of factors."""
        # Normalize frequency scores
        max_freq = max((c.frequency for c in candidates), default=1)

        # Calculate class representation for diversity
        class_counts: dict[str, int] = {}
        for c in candidates:
            if c.current_class:
                class_counts[c.current_class] = class_counts.get(c.current_class, 0) + 1
        total_classes = sum(class_counts.values()) or 1

        for c in candidates:
            # Uncertainty component
            uncertainty = self._uncertainty_score(c.confidence)

            # Frequency component (higher frequency = more impact)
            frequency = c.frequency / max_freq if max_freq > 0 else 0

            # Margin component (lower margin = more uncertain)
            margin_score = 1.0 - c.margin

            # Diversity component
            if c.current_class and c.current_class in class_counts:
                diversity = 1.0 - (class_counts[c.current_class] / total_classes)
            else:
                diversity = 1.0

            # Weighted combination
            c.informativeness_score = (
                self._weights.get("uncertainty", 0.4) * uncertainty +
                self._weights.get("frequency", 0.3) * frequency +
                self._weights.get("margin", 0.2) * margin_score +
                self._weights.get("diversity", 0.1) * diversity
            )

        return candidates

    def get_unclassified_predicates(
        self,
        predicates: list[str],
        limit: int = 20,
    ) -> list[LearningCandidate]:
        """
        Find predicates that aren't in any predicate bank.

        Args:
            predicates: List of predicate lemmas to check
            limit: Maximum to return

        Returns:
            List of unclassified predicates as candidates
        """
        unclassified = []

        for pred in predicates:
            matches = list(self._backend.find_predicate(pred))
            if not matches:
                candidate = LearningCandidate(
                    predicate=pred,
                    confidence=0.0,
                    informativeness_score=1.0,  # Highest priority
                )
                unclassified.append(candidate)
                if len(unclassified) >= limit:
                    break

        return unclassified

    def suggest_batch_size(
        self,
        total_predicates: int,
        current_accuracy: float = 0.7,
    ) -> int:
        """
        Suggest optimal batch size for next labeling round.

        Uses heuristics based on current accuracy and data size.

        Args:
            total_predicates: Total number of predicates
            current_accuracy: Estimated current accuracy

        Returns:
            Suggested batch size
        """
        # Base batch size scales with sqrt of total
        base = int(math.sqrt(total_predicates))

        # Adjust based on accuracy (lower accuracy = more samples needed)
        accuracy_factor = 1.0 + (1.0 - current_accuracy)

        suggested = int(base * accuracy_factor)

        # Clamp to reasonable range
        return max(5, min(100, suggested))
