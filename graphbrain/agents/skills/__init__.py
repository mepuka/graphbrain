"""
Agent skill implementations.

Each skill module provides:
- System prompt loading
- Helper methods for common operations
- Tool result processing
"""

from graphbrain.agents.skills.base import BaseSkill, SkillResult
from graphbrain.agents.skills.extraction import ExtractionSkill
from graphbrain.agents.skills.query import QuerySkill
from graphbrain.agents.skills.classification import ClassificationSkill
from graphbrain.agents.skills.analysis import AnalysisSkill
from graphbrain.agents.skills.feedback import FeedbackSkill
from graphbrain.agents.skills.llm_classification import LLMClassificationSkill
from graphbrain.agents.skills.llm_entity_typing import LLMEntityTypingSkill

__all__ = [
    # Base
    "BaseSkill",
    "SkillResult",
    # Core skills
    "ExtractionSkill",
    "QuerySkill",
    "ClassificationSkill",
    "AnalysisSkill",
    "FeedbackSkill",
    # LLM-powered skills
    "LLMClassificationSkill",
    "LLMEntityTypingSkill",
]
