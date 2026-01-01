"""LLM-powered predicate classification skill.

Uses Claude Agent SDK for semantic classification of predicates
into categories like claim, conflict, action, etc.
"""

import logging
from typing import Optional

from graphbrain.agents.skills.base import BaseSkill, SkillResult
from graphbrain.agents.llm.models import (
    PredicateClassification,
    BatchPredicateResult,
    PredicateCategory,
)
from graphbrain.agents.llm.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class LLMClassificationSkill(BaseSkill):
    """LLM-powered predicate classification using Claude Agent SDK.

    Classifies predicates into semantic categories:
    - claim: say, announce, declare
    - conflict: attack, blame, accuse
    - action: do, make, create
    - cognition: think, believe, know
    - emotion: love, hate, fear
    - movement: go, come, move
    - possession: have, own, give
    - perception: see, hear, feel
    """

    SKILL_NAME = "llm_classification"
    PROMPT_FILE = "llm_classification.md"
    DEFAULT_CONFIDENCE_THRESHOLD = 0.75

    # MCP tools for storing results and context lookup
    TOOLS = [
        "mcp__graphbrain__add_predicate_to_class",
        "mcp__graphbrain__find_similar_predicates",
        "mcp__graphbrain__list_semantic_classes",
        "mcp__graphbrain__flag_for_review",
    ]

    def __init__(
        self,
        provider: LLMProvider,
        confidence_threshold: float = None,
        docs_path: str = None,
    ):
        """Initialize the LLM classification skill.

        Args:
            provider: LLM provider instance (e.g., AnthropicProvider)
            confidence_threshold: Override default threshold
            docs_path: Custom path to prompt files
        """
        super().__init__(confidence_threshold, docs_path)
        self._provider = provider

    def get_tools(self) -> list[str]:
        """Get MCP tools this skill uses."""
        return self.TOOLS

    async def classify_predicate(
        self,
        lemma: str,
        context: str = "",
        examples: list[str] = None,
    ) -> SkillResult:
        """Classify a single predicate.

        Args:
            lemma: The predicate lemma to classify
            context: Optional context for disambiguation
            examples: Optional example edges using this predicate

        Returns:
            SkillResult with PredicateClassification data
        """
        prompt = self._build_classification_prompt(lemma, context, examples)

        try:
            result = await self._provider.classify(
                prompt=prompt,
                response_model=PredicateClassification,
                system_prompt=self.get_prompt(),
            )

            logger.info(
                f"Classified '{lemma}' as {result.category} "
                f"(confidence: {result.confidence:.2f})"
            )

            return self.format_result(
                success=True,
                data=result,
                confidence=result.confidence,
                metadata={
                    "method": "llm",
                    "provider": self._provider.name,
                    "reasoning": result.reasoning,
                    "similar": result.similar_predicates,
                }
            )

        except Exception as e:
            logger.error(f"Classification failed for '{lemma}': {e}")
            return self.format_result(
                success=False,
                error=str(e),
                confidence=0.0,
                metadata={"method": "llm", "lemma": lemma}
            )

    async def classify_batch(
        self,
        predicates: list[str],
        context: str = "",
    ) -> SkillResult:
        """Classify multiple predicates in one call.

        More efficient than individual calls for bulk classification.

        Args:
            predicates: List of predicate lemmas
            context: Optional shared context

        Returns:
            SkillResult with BatchPredicateResult data
        """
        if not predicates:
            return self.format_result(
                success=True,
                data=BatchPredicateResult(classifications=[], unclassified=[]),
                confidence=1.0,
                metadata={"method": "llm_batch", "count": 0}
            )

        prompt = self._build_batch_prompt(predicates, context)

        try:
            result = await self._provider.classify_batch(
                prompt=prompt,
                response_model=BatchPredicateResult,
                system_prompt=self.get_prompt(),
            )

            avg_conf = self._avg_confidence(result.classifications)
            logger.info(
                f"Batch classified {len(result.classifications)} predicates "
                f"(avg confidence: {avg_conf:.2f})"
            )

            return self.format_result(
                success=True,
                data=result,
                confidence=avg_conf,
                metadata={
                    "method": "llm_batch",
                    "provider": self._provider.name,
                    "count": len(result.classifications),
                    "unclassified": len(result.unclassified),
                    "by_category": self._count_by_category(result.classifications),
                }
            )

        except Exception as e:
            logger.error(f"Batch classification failed: {e}")
            return self.format_result(
                success=False,
                error=str(e),
                confidence=0.0,
                metadata={"method": "llm_batch", "count": len(predicates)}
            )

    def _build_classification_prompt(
        self,
        lemma: str,
        context: str = "",
        examples: list[str] = None,
    ) -> str:
        """Build prompt for single predicate classification."""
        parts = [
            "Classify this predicate into a semantic category.",
            "",
            f"Predicate: {lemma}",
        ]

        if context:
            parts.append(f"Context: {context}")

        if examples:
            parts.append("")
            parts.append("Example edges:")
            for ex in examples[:5]:
                parts.append(f"- {ex}")

        parts.extend([
            "",
            "Return the category, confidence (0-1), reasoning, and similar predicates."
        ])

        return "\n".join(parts)

    def _build_batch_prompt(
        self,
        predicates: list[str],
        context: str = "",
    ) -> str:
        """Build prompt for batch predicate classification."""
        pred_list = "\n".join(f"- {p}" for p in predicates)

        parts = [
            "Classify these predicates into semantic categories.",
            "",
            "Predicates:",
            pred_list,
        ]

        if context:
            parts.extend(["", f"Context: {context}"])

        parts.extend([
            "",
            "Return classifications for each predicate. "
            "List any that couldn't be classified in 'unclassified'."
        ])

        return "\n".join(parts)

    def _avg_confidence(
        self,
        classifications: list[PredicateClassification],
    ) -> float:
        """Calculate average confidence across classifications."""
        if not classifications:
            return 0.0
        return sum(c.confidence for c in classifications) / len(classifications)

    def _count_by_category(
        self,
        classifications: list[PredicateClassification],
    ) -> dict[str, int]:
        """Count classifications by category."""
        counts: dict[str, int] = {}
        for c in classifications:
            cat = c.category if isinstance(c.category, str) else c.category.value
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def get_category_seeds(self) -> dict[str, list[str]]:
        """Get seed predicates for each category.

        Useful for bootstrapping classification or validation.
        """
        return {
            PredicateCategory.CLAIM.value: [
                "say", "announce", "declare", "report", "claim", "state"
            ],
            PredicateCategory.CONFLICT.value: [
                "attack", "blame", "accuse", "condemn", "criticize", "oppose"
            ],
            PredicateCategory.ACTION.value: [
                "do", "make", "create", "build", "launch", "implement"
            ],
            PredicateCategory.COGNITION.value: [
                "think", "believe", "know", "understand", "consider", "expect"
            ],
            PredicateCategory.EMOTION.value: [
                "love", "hate", "fear", "enjoy", "worry", "appreciate"
            ],
            PredicateCategory.MOVEMENT.value: [
                "go", "come", "move", "travel", "arrive", "leave"
            ],
            PredicateCategory.POSSESSION.value: [
                "have", "own", "give", "take", "receive", "hold"
            ],
            PredicateCategory.PERCEPTION.value: [
                "see", "hear", "feel", "notice", "observe", "watch"
            ],
        }
