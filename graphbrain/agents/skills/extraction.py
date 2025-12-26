"""
Extraction skill for knowledge extraction from text.

Transforms natural language into semantic hyperedges.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
import logging

from graphbrain import hedge
from graphbrain.agents.skills.base import BaseSkill, SkillResult

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result from extracting knowledge from text."""
    edge: str
    original_text: str
    confidence: float
    needs_review: bool = False
    reasoning: str = ""
    metadata: dict = field(default_factory=dict)


class ExtractionSkill(BaseSkill):
    """
    Skill for extracting semantic hyperedges from natural language.

    Capabilities:
    - Parse text into sentences
    - Identify predicates, subjects, objects
    - Construct hyperedges in SH notation
    - Handle coreference and attribution
    - Flag uncertain extractions for review
    """

    SKILL_NAME = "extraction"
    PROMPT_FILE = "extraction.md"
    DEFAULT_CONFIDENCE_THRESHOLD = 0.7

    # MCP tools this skill uses
    TOOLS = [
        "mcp__graphbrain__add_edge",
        "mcp__graphbrain__get_edge",
        "mcp__graphbrain__pattern_match",
        "mcp__graphbrain__flag_for_review",
    ]

    def get_tools(self) -> list[str]:
        """Get the list of MCP tools this skill uses."""
        return self.TOOLS

    def validate_edge(self, edge_str: str) -> tuple[bool, Optional[str]]:
        """
        Validate an edge string.

        Args:
            edge_str: The edge string to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            hedge(edge_str)
            return True, None
        except Exception as e:
            return False, str(e)

    def format_extraction(
        self,
        edge_str: str,
        original_text: str,
        confidence: float,
        source_url: Optional[str] = None,
        reasoning: str = "",
    ) -> ExtractionResult:
        """
        Format an extraction result.

        Args:
            edge_str: The extracted edge in SH notation
            original_text: The original source text
            confidence: Confidence score (0.0-1.0)
            source_url: Optional source URL for attribution
            reasoning: Optional reasoning for the extraction

        Returns:
            ExtractionResult with all metadata
        """
        needs_review = self.should_flag_for_review(confidence)

        metadata = {}
        if source_url:
            metadata["source"] = source_url

        return ExtractionResult(
            edge=edge_str,
            original_text=original_text,
            confidence=confidence,
            needs_review=needs_review,
            reasoning=reasoning,
            metadata=metadata,
        )

    def build_add_edge_params(
        self,
        extraction: ExtractionResult,
        primary: bool = True,
    ) -> dict:
        """
        Build parameters for the add_edge MCP tool.

        Args:
            extraction: The extraction result
            primary: Whether to add as primary edge

        Returns:
            Dictionary of parameters for add_edge tool
        """
        params = {
            "edge": extraction.edge,
            "primary": primary,
        }

        if extraction.original_text:
            params["text"] = extraction.original_text

        return params

    def build_flag_review_params(
        self,
        extraction: ExtractionResult,
        suggested_class: Optional[str] = None,
    ) -> dict:
        """
        Build parameters for the flag_for_review MCP tool.

        Args:
            extraction: The extraction result to flag
            suggested_class: Optional suggested classification

        Returns:
            Dictionary of parameters for flag_for_review tool
        """
        # Extract predicate from edge for review
        try:
            edge = hedge(extraction.edge)
            if not edge.is_atom():
                predicate = str(edge[0].root()) if edge[0].is_atom() else str(edge[0])
            else:
                predicate = str(edge.root())
        except Exception:
            predicate = extraction.edge[:50]

        return {
            "predicate": predicate,
            "suggested_class": suggested_class,
            "notes": extraction.reasoning or f"Confidence: {extraction.confidence:.2f}",
        }

    def estimate_confidence(
        self,
        edge_str: str,
        has_subject: bool = True,
        has_predicate: bool = True,
        is_attributed: bool = False,
        parse_ambiguity: float = 0.0,
    ) -> float:
        """
        Estimate confidence for an extraction.

        Args:
            edge_str: The extracted edge
            has_subject: Whether a clear subject was identified
            has_predicate: Whether a clear predicate was identified
            is_attributed: Whether the claim is attributed to a source
            parse_ambiguity: Ambiguity score from parsing (0.0-1.0)

        Returns:
            Confidence score (0.0-1.0)
        """
        base_confidence = 0.9

        # Penalize missing components
        if not has_subject:
            base_confidence -= 0.2
        if not has_predicate:
            base_confidence -= 0.3

        # Boost for attribution
        if is_attributed:
            base_confidence += 0.05

        # Penalize for parse ambiguity
        base_confidence -= parse_ambiguity * 0.3

        # Validate edge syntax
        is_valid, _ = self.validate_edge(edge_str)
        if not is_valid:
            base_confidence -= 0.4

        return max(0.0, min(1.0, base_confidence))

    def get_extraction_patterns(self) -> dict[str, str]:
        """
        Get common extraction patterns.

        Returns:
            Dictionary mapping pattern names to SH patterns
        """
        return {
            # Claims with attribution
            "attributed_claim": "(*/Pd.{sr} SPEAKER/Cp CLAIM/*)",

            # Simple declarative
            "declarative": "(*/Pd.{sc} SUBJECT/C COMPLEMENT/C)",

            # Action with object
            "action_object": "(*/Pd.{so} SUBJECT/C OBJECT/C)",

            # Possessive
            "possessive": "('s/Bp POSSESSOR/C POSSESSED/C)",

            # Compound name
            "compound_name": "(+/B/. FIRST/Cp REST/*)",

            # Relation
            "relation": "(*/Br.{ma} MAIN/C AUX/C)",

            # Temporal
            "temporal": "(*/Tt CONDITION/*)",
        }
