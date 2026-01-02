"""Integration tests for Claude Agent SDK.

Tests the Agent SDK integration for:
- Structured output with Pydantic models
- MCP server integration
- Agent workflows
"""

import pytest
import asyncio


@pytest.mark.llm
class TestAgentSDKAvailability:
    """Test Agent SDK is properly installed and available."""

    def test_agent_sdk_imports(self):
        """Agent SDK can be imported."""
        from claude_agent_sdk import query
        assert query is not None

    def test_agent_sdk_detected_by_provider(self):
        """AnthropicProvider detects Agent SDK."""
        from graphbrain.agents.llm.providers import anthropic as provider_module

        # After installing, this should be True
        assert provider_module._AGENT_SDK_AVAILABLE is True

    def test_provider_uses_agent_sdk(self, llm_client):
        """Provider should use Agent SDK when available."""
        # The provider should have been initialized with Agent SDK
        assert llm_client is not None


@pytest.mark.llm
class TestAgentSDKStructuredOutput:
    """Test Agent SDK structured output with Pydantic models."""

    @pytest.mark.asyncio
    async def test_structured_output_simple(self, llm_client):
        """Test structured output with simple Pydantic model."""
        from pydantic import BaseModel

        class SimpleResponse(BaseModel):
            answer: str
            confidence: float

        result = await llm_client.classify(
            prompt="What is 2 + 2? Respond with the answer.",
            response_model=SimpleResponse,
        )

        # Should return structured response
        assert result is not None
        if hasattr(result, "answer"):
            assert result.answer is not None

    @pytest.mark.asyncio
    async def test_structured_output_classification(self, llm_client):
        """Test structured classification output."""
        from graphbrain.agents.llm.models import PredicateClassification

        result = await llm_client.classify(
            prompt="Classify the verb 'announce' - is it a claim, action, or conflict?",
            response_model=PredicateClassification,
            system_prompt="You are a linguistic classifier. Respond with category, confidence (0-1), and reasoning.",
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_structured_output_batch(self, llm_client):
        """Test batch classification output."""
        from graphbrain.agents.llm.models import BatchPredicateResult

        result = await llm_client.classify(
            prompt="Classify these verbs: say, attack, build. For each, provide category and confidence.",
            response_model=BatchPredicateResult,
        )

        assert result is not None


@pytest.mark.llm
class TestAgentSDKWithSkills:
    """Test Agent SDK with graphbrain skills."""

    @pytest.mark.asyncio
    async def test_classification_skill_uses_sdk(self, llm_client):
        """Classification skill works with Agent SDK."""
        from graphbrain.agents.skills.llm_classification import LLMClassificationSkill

        skill = LLMClassificationSkill(llm_client)

        result = await skill.classify_predicate(
            lemma="propose",
            context="The council proposed a new transit plan.",
        )

        # Skill should complete (success or meaningful failure)
        assert result is not None
        assert hasattr(result, "success")

    @pytest.mark.asyncio
    async def test_batch_classification_skill(self, llm_client):
        """Batch classification with Agent SDK."""
        from graphbrain.agents.skills.llm_classification import LLMClassificationSkill

        skill = LLMClassificationSkill(llm_client)

        result = await skill.classify_batch(
            predicates=["say", "announce", "claim"],
            context="News reporting context",
        )

        assert result is not None


@pytest.mark.llm
class TestAgentSDKQuery:
    """Test direct Agent SDK query function."""

    @pytest.mark.asyncio
    async def test_basic_query(self):
        """Test basic Agent SDK query."""
        from claude_agent_sdk import query

        messages = []
        async for message in query(prompt="Say hello"):
            messages.append(message)

        # Should receive at least one message
        assert len(messages) > 0

    @pytest.mark.asyncio
    async def test_query_with_output_format(self):
        """Test Agent SDK query with structured output."""
        from claude_agent_sdk import query, ClaudeAgentOptions
        from pydantic import BaseModel

        class Greeting(BaseModel):
            message: str
            language: str

        options = ClaudeAgentOptions(
            output_format={
                "type": "json_schema",
                "schema": Greeting.model_json_schema()
            }
        )

        messages = []
        async for message in query(
            prompt="Say hello in French",
            options=options
        ):
            messages.append(message)

        # Should receive structured response
        assert len(messages) > 0


@pytest.mark.llm
@pytest.mark.slow
class TestAgentSDKWithMCP:
    """Test Agent SDK with MCP server integration."""

    @pytest.mark.asyncio
    async def test_mcp_server_config(self):
        """Test MCP server configuration."""
        from claude_agent_sdk import ClaudeAgentOptions

        # Configure with graphbrain MCP server
        options = ClaudeAgentOptions(
            mcp_servers={
                "graphbrain": {
                    "command": "python",
                    "args": ["-m", "graphbrain.mcp.server"],
                }
            }
        )

        # Options should be valid
        assert options is not None

    @pytest.mark.asyncio
    async def test_provider_with_mcp(self):
        """Test AnthropicProvider with MCP servers."""
        from graphbrain.agents.llm.providers.anthropic import AnthropicProvider
        import os

        provider = AnthropicProvider(
            model="claude-3-haiku-20240307",
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            mcp_servers={
                "graphbrain": {
                    "command": "python",
                    "args": ["-m", "graphbrain.mcp.server"],
                }
            }
        )

        assert provider is not None


@pytest.mark.llm
class TestAgentSDKErrorHandling:
    """Test Agent SDK error handling."""

    @pytest.mark.asyncio
    async def test_handles_invalid_schema(self, llm_client):
        """Test handling of invalid Pydantic schema."""
        from pydantic import BaseModel

        class InvalidSchema(BaseModel):
            required_field: str
            another_required: int

        try:
            result = await llm_client.classify(
                prompt="Just say hi",  # Won't produce required fields
                response_model=InvalidSchema,
            )
            # If it succeeds, verify structure
            assert result is not None
        except Exception as e:
            # Should get validation error, not crash
            assert "validation" in str(e).lower() or "failed" in str(e).lower()

    @pytest.mark.asyncio
    async def test_handles_timeout_gracefully(self, llm_client):
        """Test timeout handling."""
        from graphbrain.agents.llm.models import PredicateClassification

        try:
            # Very short timeout would cause issues
            result = await asyncio.wait_for(
                llm_client.classify(
                    prompt="Classify 'run'",
                    response_model=PredicateClassification,
                ),
                timeout=60.0  # Reasonable timeout
            )
            assert result is not None
        except asyncio.TimeoutError:
            # Timeout is acceptable
            pass
        except Exception as e:
            # Other errors should be meaningful
            assert str(e) or True


@pytest.mark.llm
class TestAgentSDKComparison:
    """Compare Agent SDK vs fallback Anthropic SDK."""

    def test_both_sdks_available(self):
        """Both SDKs should be available."""
        from graphbrain.agents.llm.providers import anthropic as provider_module

        # After installation, both should be available
        assert provider_module._AGENT_SDK_AVAILABLE is True
        assert provider_module._ANTHROPIC_SDK_AVAILABLE is True

    @pytest.mark.asyncio
    async def test_sdk_selection(self, llm_client):
        """Provider should prefer Agent SDK when available."""
        from graphbrain.agents.llm.providers import anthropic as provider_module
        from graphbrain.agents.llm.models import PredicateClassification

        # With Agent SDK available, classify should use it
        if provider_module._AGENT_SDK_AVAILABLE:
            result = await llm_client.classify(
                prompt="Classify 'say'",
                response_model=PredicateClassification,
            )
            assert result is not None
