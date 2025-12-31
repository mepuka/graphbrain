"""Active learning module for intelligent sample selection.

Provides uncertainty sampling and other strategies to select
the most informative samples for human review.
"""

from graphbrain.agents.learning.sampler import (
    ActiveLearningSampler,
    SamplingStrategy,
    LearningCandidate,
)
from graphbrain.agents.learning.suggestions import (
    SuggestionEngine,
    ImprovementSuggestion,
    SuggestionType,
)

__all__ = [
    "ActiveLearningSampler",
    "SamplingStrategy",
    "LearningCandidate",
    "SuggestionEngine",
    "ImprovementSuggestion",
    "SuggestionType",
]
