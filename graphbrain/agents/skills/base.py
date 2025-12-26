"""
Base class for agent skills.

Provides common functionality for all skill implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class SkillResult:
    """Result from a skill operation."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class BaseSkill(ABC):
    """
    Base class for agent skills.

    Provides common functionality:
    - Prompt loading from markdown files
    - Confidence threshold handling
    - Result formatting
    """

    # Override in subclasses
    SKILL_NAME: str = "base"
    PROMPT_FILE: str = "base.md"
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.7

    def __init__(
        self,
        confidence_threshold: float = None,
        docs_path: Optional[Path] = None,
    ):
        """
        Initialize the skill.

        Args:
            confidence_threshold: Threshold for flagging uncertain results
            docs_path: Path to docs/llm/agent-prompts/
        """
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.DEFAULT_CONFIDENCE_THRESHOLD
        )
        self.docs_path = docs_path or self._find_docs_path()
        self._prompt_cache: Optional[str] = None

    def _find_docs_path(self) -> Optional[Path]:
        """Find the docs/llm/agent-prompts/ directory."""
        # Try relative to this file
        current = Path(__file__).parent
        for _ in range(5):  # Go up to 5 levels
            candidate = current / "docs" / "llm" / "agent-prompts"
            if candidate.exists():
                return candidate
            current = current.parent

        # Try from cwd
        cwd_candidate = Path.cwd() / "docs" / "llm" / "agent-prompts"
        if cwd_candidate.exists():
            return cwd_candidate

        return None

    def get_prompt(self) -> str:
        """
        Get the system prompt for this skill.

        Returns:
            The system prompt string
        """
        if self._prompt_cache:
            return self._prompt_cache

        if self.docs_path:
            prompt_file = self.docs_path / self.PROMPT_FILE
            if prompt_file.exists():
                self._prompt_cache = prompt_file.read_text()
                return self._prompt_cache

        # Fallback to minimal prompt
        self._prompt_cache = self._get_fallback_prompt()
        return self._prompt_cache

    def _get_fallback_prompt(self) -> str:
        """Get a minimal fallback prompt."""
        return f"You are a {self.SKILL_NAME} agent for graphbrain semantic hypergraphs."

    def should_flag_for_review(self, confidence: float) -> bool:
        """Check if a result should be flagged for human review."""
        return confidence < self.confidence_threshold

    def format_result(
        self,
        success: bool,
        data: Any = None,
        error: str = None,
        confidence: float = 1.0,
        **metadata
    ) -> SkillResult:
        """Format a skill operation result."""
        return SkillResult(
            success=success,
            data=data,
            error=error,
            confidence=confidence,
            metadata=metadata,
        )

    @abstractmethod
    def get_tools(self) -> list[str]:
        """Get the list of MCP tools this skill uses."""
        pass
