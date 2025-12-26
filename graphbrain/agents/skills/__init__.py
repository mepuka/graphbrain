"""
Agent skill implementations.

Each skill module provides:
- System prompt loading
- Helper methods for common operations
- Tool result processing
"""

from graphbrain.agents.skills.extraction import ExtractionSkill
from graphbrain.agents.skills.query import QuerySkill
from graphbrain.agents.skills.classification import ClassificationSkill
from graphbrain.agents.skills.analysis import AnalysisSkill
from graphbrain.agents.skills.feedback import FeedbackSkill

__all__ = [
    "ExtractionSkill",
    "QuerySkill",
    "ClassificationSkill",
    "AnalysisSkill",
    "FeedbackSkill",
]
