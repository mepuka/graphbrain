"""Anthropic Claude provider using the Agent SDK.

Implements LLM classification using Claude's structured output
capabilities via the Agent SDK.
"""

import logging
from typing import Type, TypeVar

from pydantic import BaseModel

from graphbrain.agents.llm.providers.base import (
    LLMProvider,
    LLMError,
    LLMValidationError,
)

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

# Check for Agent SDK availability
try:
    from claude_agent_sdk import query, ClaudeAgentOptions
    _AGENT_SDK_AVAILABLE = True
except ImportError:
    _AGENT_SDK_AVAILABLE = False
    logger.debug("Claude Agent SDK not available")

# Fallback: Check for direct Anthropic SDK
try:
    import anthropic
    _ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    _ANTHROPIC_SDK_AVAILABLE = False
    logger.debug("Anthropic SDK not available")


class AnthropicProvider(LLMProvider):
    """Claude provider using Agent SDK with structured outputs.

    Uses the Agent SDK's output_format feature to guarantee
    JSON responses matching the provided Pydantic schema.

    Falls back to direct Anthropic API if Agent SDK not available.
    """

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        mcp_servers: dict | None = None,
        max_retries: int = 3,
    ):
        """Initialize the Anthropic provider.

        Args:
            model: Model name (default: claude-sonnet-4-20250514)
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
            mcp_servers: Optional MCP server configurations
            max_retries: Max retries on validation failure
        """
        self._model = model or self.DEFAULT_MODEL
        self._api_key = api_key
        self._mcp_servers = mcp_servers or {}
        self._max_retries = max_retries

        # Validate availability
        if not _AGENT_SDK_AVAILABLE and not _ANTHROPIC_SDK_AVAILABLE:
            raise ImportError(
                "Neither claude_agent_sdk nor anthropic SDK is available. "
                "Install with: pip install anthropic"
            )

        # Initialize appropriate client
        if _AGENT_SDK_AVAILABLE:
            self._use_agent_sdk = True
            logger.info("Using Claude Agent SDK for structured outputs")
        else:
            self._use_agent_sdk = False
            # Use AsyncAnthropic for async compatibility
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
            logger.info("Using direct Anthropic API (async)")

    @property
    def name(self) -> str:
        return "anthropic"

    async def classify(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Run classification using Claude with structured output."""
        if self._use_agent_sdk:
            return await self._classify_with_agent_sdk(
                prompt, response_model, system_prompt
            )
        else:
            return await self._classify_with_anthropic_sdk(
                prompt, response_model, system_prompt
            )

    async def classify_batch(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Batch classification - same as classify, schema handles batch."""
        return await self.classify(prompt, response_model, system_prompt)

    async def _classify_with_agent_sdk(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Use Agent SDK with output_format for structured responses."""
        options = ClaudeAgentOptions(
            model=self._model,
            mcp_servers=self._mcp_servers,
            output_format={
                "type": "json_schema",
                "schema": response_model.model_json_schema()
            }
        )

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        for attempt in range(self._max_retries):
            try:
                async for message in query(prompt=full_prompt, options=options):
                    if hasattr(message, 'structured_output') and message.structured_output:
                        return response_model.model_validate(message.structured_output)

                raise LLMError(
                    "No structured output received",
                    provider=self.name,
                    details={"prompt": prompt[:100]}
                )

            except Exception as e:
                if attempt < self._max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying...")
                    continue
                raise LLMValidationError(
                    f"Failed after {self._max_retries} attempts: {e}",
                    provider=self.name,
                    details={"model": self._model}
                )

    async def _classify_with_anthropic_sdk(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Fallback: Use direct Anthropic API with tool_use for structured output."""
        import json

        # Build tool definition from Pydantic schema
        tool_schema = response_model.model_json_schema()
        tool_name = f"return_{response_model.__name__.lower()}"

        tools = [{
            "name": tool_name,
            "description": f"Return structured {response_model.__name__}",
            "input_schema": tool_schema
        }]

        messages = [{"role": "user", "content": prompt}]

        for attempt in range(self._max_retries):
            try:
                response = await self._client.messages.create(
                    model=self._model,
                    max_tokens=4096,
                    system=system_prompt or "",
                    messages=messages,
                    tools=tools,
                    tool_choice={"type": "tool", "name": tool_name}
                )

                # Extract tool use result
                for block in response.content:
                    if block.type == "tool_use" and block.name == tool_name:
                        return response_model.model_validate(block.input)

                raise LLMError(
                    "No tool use in response",
                    provider=self.name,
                    details={"response": str(response.content)[:200]}
                )

            except anthropic.RateLimitError as e:
                from graphbrain.agents.llm.providers.base import LLMRateLimitError
                raise LLMRateLimitError(
                    str(e),
                    provider=self.name,
                    details={"retry_after": getattr(e, 'retry_after', None)}
                )

            except Exception as e:
                if attempt < self._max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying...")
                    continue
                raise LLMValidationError(
                    f"Failed after {self._max_retries} attempts: {e}",
                    provider=self.name,
                    details={"model": self._model}
                )
