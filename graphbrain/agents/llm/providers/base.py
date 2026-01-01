"""Abstract base class for LLM providers.

Defines the interface that all LLM providers must implement
for classification tasks.
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Type
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)


class LLMProvider(ABC):
    """Abstract base for LLM providers.

    Providers implement structured classification using their
    respective APIs (Anthropic, OpenAI, Ollama, etc.)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging/debugging."""
        pass

    @abstractmethod
    async def classify(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Run classification and return structured result.

        Args:
            prompt: The classification prompt
            response_model: Pydantic model class for response
            system_prompt: Optional system prompt override

        Returns:
            Validated instance of response_model

        Raises:
            ValueError: If LLM response doesn't match schema
            LLMError: If provider API call fails
        """
        pass

    @abstractmethod
    async def classify_batch(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Batch classification for multiple items.

        Args:
            prompt: The batch classification prompt
            response_model: Pydantic model class for batch response
            system_prompt: Optional system prompt override

        Returns:
            Validated instance of response_model (batch type)
        """
        pass


class LLMError(Exception):
    """Base exception for LLM provider errors."""

    def __init__(self, message: str, provider: str, details: dict = None):
        super().__init__(message)
        self.provider = provider
        self.details = details or {}


class LLMRateLimitError(LLMError):
    """Rate limit exceeded."""
    pass


class LLMValidationError(LLMError):
    """Response validation failed."""
    pass
