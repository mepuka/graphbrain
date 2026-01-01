"""LLM provider layer for agent skills.

Provides a thin abstraction over LLM providers (Anthropic, OpenAI, Ollama)
for structured classification tasks.
"""

from graphbrain.agents.llm.models import (
    PredicateCategory,
    PredicateClassification,
    BatchPredicateResult,
    EntityType,
    EntityClassification,
    BatchEntityResult,
)
from graphbrain.agents.llm.providers import (
    LLMProvider,
    AnthropicProvider,
)

__all__ = [
    # Models
    "PredicateCategory",
    "PredicateClassification",
    "BatchPredicateResult",
    "EntityType",
    "EntityClassification",
    "BatchEntityResult",
    # Providers
    "LLMProvider",
    "AnthropicProvider",
]
