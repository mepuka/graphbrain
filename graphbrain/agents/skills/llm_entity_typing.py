"""LLM-powered entity typing skill.

Uses Claude Agent SDK to classify proper nouns (entities) into
types like person, organization, location, group, event.
"""

import logging
from typing import Optional

from graphbrain.agents.skills.base import BaseSkill, SkillResult
from graphbrain.agents.llm.models import (
    EntityClassification,
    BatchEntityResult,
    EntityType,
)
from graphbrain.agents.llm.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class LLMEntityTypingSkill(BaseSkill):
    """LLM-powered entity typing using Claude Agent SDK.

    Classifies proper nouns into entity types:
    - person: Individual humans (Mayor Wilson, Dr. Smith)
    - organization: Companies, agencies (Seattle City Council, SDOT)
    - location: Geographic places (Seattle, Ballard, I-5)
    - group: Collective entities (residents, protesters, advocates)
    - event: Named occurrences (Election 2024, Transit Summit)
    """

    SKILL_NAME = "llm_entity_typing"
    PROMPT_FILE = "llm_entity_typing.md"
    DEFAULT_CONFIDENCE_THRESHOLD = 0.8  # Higher threshold for entities

    # MCP tools for context lookup
    TOOLS = [
        "mcp__graphbrain__edges_with_root",
        "mcp__graphbrain__search_edges",
        "mcp__graphbrain__pattern_match",
        "mcp__graphbrain__flag_for_review",
    ]

    # Common indicators for each entity type
    TYPE_INDICATORS = {
        EntityType.PERSON: [
            "mayor", "director", "ceo", "president", "dr.", "mr.", "ms.",
            "said", "announced", "believes"
        ],
        EntityType.ORGANIZATION: [
            "inc", "corp", "llc", "council", "department", "agency",
            "commission", "authority", "board"
        ],
        EntityType.LOCATION: [
            "city", "street", "avenue", "blvd", "district", "neighborhood",
            "county", "state", "region"
        ],
        EntityType.GROUP: [
            "residents", "protesters", "members", "voters", "advocates",
            "critics", "supporters", "opponents"
        ],
        EntityType.EVENT: [
            "conference", "election", "summit", "festival", "meeting",
            "hearing", "session"
        ],
    }

    def __init__(
        self,
        provider: LLMProvider,
        confidence_threshold: float = None,
        docs_path: str = None,
    ):
        """Initialize the entity typing skill.

        Args:
            provider: LLM provider instance
            confidence_threshold: Override default threshold
            docs_path: Custom path to prompt files
        """
        super().__init__(confidence_threshold, docs_path)
        self._provider = provider

    def get_tools(self) -> list[str]:
        """Get MCP tools this skill uses."""
        return self.TOOLS

    async def type_entity(
        self,
        entity: str,
        context_edges: list[str] = None,
        predicates: list[str] = None,
    ) -> SkillResult:
        """Type a single entity.

        Args:
            entity: The entity name to type
            context_edges: Optional edges containing this entity
            predicates: Optional predicates used with this entity

        Returns:
            SkillResult with EntityClassification data
        """
        prompt = self._build_typing_prompt(entity, context_edges, predicates)

        try:
            result = await self._provider.classify(
                prompt=prompt,
                response_model=EntityClassification,
                system_prompt=self.get_prompt(),
            )

            logger.info(
                f"Typed '{entity}' as {result.entity_type} "
                f"(confidence: {result.confidence:.2f})"
            )

            return self.format_result(
                success=True,
                data=result,
                confidence=result.confidence,
                metadata={
                    "method": "llm",
                    "provider": self._provider.name,
                    "entity_type": result.entity_type,
                    "subtypes": result.subtypes,
                    "reasoning": result.reasoning,
                }
            )

        except Exception as e:
            logger.error(f"Entity typing failed for '{entity}': {e}")
            return self.format_result(
                success=False,
                error=str(e),
                confidence=0.0,
                metadata={"method": "llm", "entity": entity}
            )

    async def type_batch(
        self,
        entities: list[str],
        context: str = "",
    ) -> SkillResult:
        """Type multiple entities in one call.

        Args:
            entities: List of entity names
            context: Optional shared context

        Returns:
            SkillResult with BatchEntityResult data
        """
        if not entities:
            return self.format_result(
                success=True,
                data=BatchEntityResult(entities=[], unclassified=[]),
                confidence=1.0,
                metadata={"method": "llm_batch", "count": 0}
            )

        prompt = self._build_batch_prompt(entities, context)

        try:
            result = await self._provider.classify_batch(
                prompt=prompt,
                response_model=BatchEntityResult,
                system_prompt=self.get_prompt(),
            )

            avg_conf = self._avg_confidence(result.entities)
            by_type = self._group_by_type(result.entities)

            logger.info(
                f"Batch typed {len(result.entities)} entities "
                f"(avg confidence: {avg_conf:.2f})"
            )

            return self.format_result(
                success=True,
                data=result,
                confidence=avg_conf,
                metadata={
                    "method": "llm_batch",
                    "provider": self._provider.name,
                    "count": len(result.entities),
                    "unclassified": len(result.unclassified),
                    "by_type": {k: len(v) for k, v in by_type.items()},
                }
            )

        except Exception as e:
            logger.error(f"Batch entity typing failed: {e}")
            return self.format_result(
                success=False,
                error=str(e),
                confidence=0.0,
                metadata={"method": "llm_batch", "count": len(entities)}
            )

    async def type_from_hypergraph(
        self,
        entity: str,
        hg_context: dict,
    ) -> SkillResult:
        """Type entity using rich hypergraph context.

        Uses edge patterns and predicates to make better typing decisions.

        Args:
            entity: Entity to type
            hg_context: Dict with 'edges' and 'predicates' lists

        Returns:
            SkillResult with EntityClassification data
        """
        edges = hg_context.get("edges", [])
        predicates = hg_context.get("predicates", [])

        prompt = self._build_hypergraph_prompt(entity, edges, predicates)

        try:
            result = await self._provider.classify(
                prompt=prompt,
                response_model=EntityClassification,
                system_prompt=self.get_prompt(),
            )

            return self.format_result(
                success=True,
                data=result,
                confidence=result.confidence,
                metadata={
                    "method": "llm_with_hg_context",
                    "provider": self._provider.name,
                    "edges_used": len(edges),
                    "predicates_used": len(predicates),
                }
            )

        except Exception as e:
            logger.error(f"Hypergraph-context typing failed for '{entity}': {e}")
            return self.format_result(
                success=False,
                error=str(e),
                confidence=0.0,
                metadata={"method": "llm_with_hg_context", "entity": entity}
            )

    def _build_typing_prompt(
        self,
        entity: str,
        context_edges: list[str] = None,
        predicates: list[str] = None,
    ) -> str:
        """Build prompt for single entity typing."""
        parts = [
            "Determine the entity type for this proper noun.",
            "",
            f"Entity: {entity}",
        ]

        if context_edges:
            parts.append("")
            parts.append("Context edges:")
            for edge in context_edges[:5]:
                parts.append(f"- {edge}")

        if predicates:
            parts.append("")
            parts.append(f"Predicates used with entity: {', '.join(predicates[:10])}")

        parts.extend([
            "",
            "Types: person, organization, location, group, event, unknown",
            "",
            "Return the type, confidence (0-1), reasoning, and any subtypes."
        ])

        return "\n".join(parts)

    def _build_batch_prompt(
        self,
        entities: list[str],
        context: str = "",
    ) -> str:
        """Build prompt for batch entity typing."""
        entity_list = "\n".join(f"- {e}" for e in entities)

        parts = [
            "Determine entity types for these proper nouns.",
            "",
            "Entities:",
            entity_list,
        ]

        if context:
            parts.extend(["", f"Context: {context}"])

        parts.extend([
            "",
            "Types: person, organization, location, group, event, unknown",
            "",
            "Return type classification for each entity. "
            "List any that couldn't be typed in 'unclassified'."
        ])

        return "\n".join(parts)

    def _build_hypergraph_prompt(
        self,
        entity: str,
        edges: list[str],
        predicates: list[str],
    ) -> str:
        """Build prompt using hypergraph context."""
        parts = [
            "Determine the entity type using hypergraph context.",
            "",
            f"Entity: {entity}",
        ]

        if edges:
            parts.append("")
            parts.append("Edges containing entity:")
            for edge in edges[:10]:
                parts.append(f"- {edge}")

        if predicates:
            parts.append("")
            parts.append(f"Predicates used with entity: {', '.join(predicates[:10])}")

        parts.extend([
            "",
            "Use the relationship patterns to infer whether this is a person, "
            "organization, location, group, or event.",
            "",
            "Return the type, confidence (0-1), reasoning, and any subtypes."
        ])

        return "\n".join(parts)

    def _avg_confidence(
        self,
        entities: list[EntityClassification],
    ) -> float:
        """Calculate average confidence across classifications."""
        if not entities:
            return 0.0
        return sum(e.confidence for e in entities) / len(entities)

    def _group_by_type(
        self,
        entities: list[EntityClassification],
    ) -> dict[str, list[EntityClassification]]:
        """Group entities by their type."""
        groups: dict[str, list] = {}
        for e in entities:
            etype = e.entity_type if isinstance(e.entity_type, str) else e.entity_type.value
            if etype not in groups:
                groups[etype] = []
            groups[etype].append(e)
        return groups

    def get_type_indicators(self) -> dict[str, list[str]]:
        """Get indicator terms for each entity type.

        Useful for heuristic pre-classification or validation.
        """
        return {
            k.value if isinstance(k, EntityType) else k: v
            for k, v in self.TYPE_INDICATORS.items()
        }
