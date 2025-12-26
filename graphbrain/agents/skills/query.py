"""
Query skill for exploring the semantic hypergraph.

Translates natural language questions into pattern searches.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
import re
import logging

from graphbrain.agents.skills.base import BaseSkill, SkillResult

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result from a query operation."""
    edges: list[str]
    total_count: int
    pattern_used: Optional[str] = None
    search_method: str = "hybrid"
    metadata: dict = field(default_factory=dict)


class QuerySkill(BaseSkill):
    """
    Skill for querying and exploring the semantic hypergraph.

    Capabilities:
    - Translate natural language to patterns
    - Execute hybrid BM25 + semantic search
    - Navigate entity relationships
    - Aggregate and rank results
    """

    SKILL_NAME = "query"
    PROMPT_FILE = "query.md"
    DEFAULT_CONFIDENCE_THRESHOLD = 0.7

    # MCP tools this skill uses
    TOOLS = [
        "mcp__graphbrain__search_edges",
        "mcp__graphbrain__pattern_match",
        "mcp__graphbrain__hybrid_search",
        "mcp__graphbrain__bm25_search",
        "mcp__graphbrain__edges_with_root",
        "mcp__graphbrain__hypergraph_stats",
    ]

    def get_tools(self) -> list[str]:
        """Get the list of MCP tools this skill uses."""
        return self.TOOLS

    def translate_question(self, question: str) -> dict:
        """
        Translate a natural language question to search parameters.

        Args:
            question: Natural language question

        Returns:
            Dictionary with search strategy and parameters
        """
        question_lower = question.lower()

        # Detect question type
        if any(q in question_lower for q in ["who said", "who claimed", "who announced"]):
            return self._translate_attribution_question(question)

        elif any(q in question_lower for q in ["what did", "what has", "what actions"]):
            return self._translate_action_question(question)

        elif any(q in question_lower for q in ["find claims", "claims about", "statements about"]):
            return self._translate_claims_question(question)

        elif any(q in question_lower for q in ["how is", "how are", "related to", "connection"]):
            return self._translate_relationship_question(question)

        elif any(q in question_lower for q in ["when", "what time", "what date"]):
            return self._translate_temporal_question(question)

        else:
            # Default to hybrid search
            return {
                "method": "hybrid_search",
                "params": {"query": question},
                "pattern": None,
            }

    def _translate_attribution_question(self, question: str) -> dict:
        """Translate 'Who said X?' type questions."""
        return {
            "method": "pattern_match",
            "params": {
                "pattern": "(*/Pd.{sr} SPEAKER/Cp *)",
                "strict": False,
            },
            "follow_up": "hybrid_search",
            "pattern": "(*/Pd.{sr} SPEAKER/Cp *)",
        }

    def _translate_action_question(self, question: str) -> dict:
        """Translate 'What did X do?' type questions."""
        # Try to extract the entity
        entity = self._extract_entity(question)

        if entity:
            return {
                "method": "edges_with_root",
                "params": {"root": entity},
                "filter": "subject_role",
                "pattern": f"(*/Pd.{{s}} {entity}/* ...)",
            }
        else:
            return {
                "method": "pattern_match",
                "params": {
                    "pattern": "(*/Pd.{so} SUBJECT/C OBJECT/C)",
                    "strict": False,
                },
                "pattern": "(*/Pd.{so} SUBJECT/C OBJECT/C)",
            }

    def _translate_claims_question(self, question: str) -> dict:
        """Translate 'Find claims about X' type questions."""
        # Extract topic
        topic = self._extract_topic(question)

        return {
            "method": "hybrid_search",
            "params": {
                "query": topic or question,
                "class_id": "claim",
            },
            "pattern": "(*/Pd.{sr} */Cp *)",
        }

    def _translate_relationship_question(self, question: str) -> dict:
        """Translate 'How is X related to Y?' type questions."""
        entities = self._extract_entities(question)

        if len(entities) >= 2:
            return {
                "method": "pattern_match",
                "params": {
                    "pattern": f"(* (atoms {entities[0]}/*) (atoms {entities[1]}/*))",
                    "strict": False,
                },
                "entities": entities,
            }
        else:
            return {
                "method": "hybrid_search",
                "params": {"query": question},
            }

    def _translate_temporal_question(self, question: str) -> dict:
        """Translate temporal questions."""
        return {
            "method": "pattern_match",
            "params": {
                "pattern": "(*/Tt *)",
                "strict": False,
            },
            "follow_up": "hybrid_search",
        }

    def _extract_entity(self, question: str) -> Optional[str]:
        """Extract a single entity from a question."""
        # Look for quoted entities
        quoted = re.findall(r'"([^"]+)"', question)
        if quoted:
            return quoted[0].lower().replace(" ", "_")

        # Look for capitalized words (proper nouns)
        words = question.split()
        for i, word in enumerate(words):
            # Skip question words and common words
            if word[0].isupper() and word.lower() not in [
                "who", "what", "when", "where", "why", "how",
                "the", "a", "an", "is", "are", "was", "were",
                "did", "does", "do", "has", "have", "had"
            ]:
                # Combine with following capitalized words
                entity_parts = [word]
                for j in range(i + 1, len(words)):
                    if words[j][0].isupper():
                        entity_parts.append(words[j])
                    else:
                        break
                return "_".join(entity_parts).lower()

        return None

    def _extract_entities(self, question: str) -> list[str]:
        """Extract multiple entities from a question."""
        entities = []

        # Look for quoted entities
        quoted = re.findall(r'"([^"]+)"', question)
        entities.extend([q.lower().replace(" ", "_") for q in quoted])

        # Look for capitalized sequences
        words = question.split()
        i = 0
        while i < len(words):
            word = words[i]
            if word[0].isupper() and word.lower() not in [
                "who", "what", "when", "where", "why", "how",
                "the", "a", "an", "is", "are", "find"
            ]:
                entity_parts = [word]
                j = i + 1
                while j < len(words) and words[j][0].isupper():
                    entity_parts.append(words[j])
                    j += 1
                entities.append("_".join(entity_parts).lower())
                i = j
            else:
                i += 1

        return entities

    def _extract_topic(self, question: str) -> Optional[str]:
        """Extract the topic from a claims question."""
        # Remove common prefixes
        cleaned = question.lower()
        for prefix in ["find claims about", "claims about", "statements about", "what about"]:
            if cleaned.startswith(prefix):
                return cleaned[len(prefix):].strip()

        # Look for 'about X' pattern
        match = re.search(r'about\s+(.+?)(?:\?|$)', cleaned)
        if match:
            return match.group(1).strip()

        return None

    def build_hybrid_search_params(
        self,
        query: str,
        class_id: Optional[str] = None,
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7,
        limit: int = 50,
    ) -> dict:
        """Build parameters for hybrid_search MCP tool."""
        params = {
            "query": query,
            "limit": limit,
        }

        if class_id:
            params["class_id"] = class_id

        params["bm25_weight"] = bm25_weight
        params["semantic_weight"] = semantic_weight

        return params

    def build_pattern_match_params(
        self,
        pattern: str,
        strict: bool = False,
        limit: int = 100,
    ) -> dict:
        """Build parameters for pattern_match MCP tool."""
        return {
            "pattern": pattern,
            "strict": strict,
            "limit": limit,
        }

    def get_query_patterns(self) -> dict[str, str]:
        """
        Get common query patterns.

        Returns:
            Dictionary mapping pattern names to SH patterns
        """
        return {
            # Find all claims by a speaker
            "claims_by_speaker": "(*/Pd.{sr} SPEAKER/Cp *)",

            # Find all actions by a subject
            "actions_by_subject": "(*/Pd.{s} SUBJECT/C ...)",

            # Find all mentions of a concept
            "concept_mentions": "(atoms CONCEPT/C)",

            # Find temporal statements
            "temporal": "(*/Tt *)",

            # Find conditional statements
            "conditional": "(*/Tc *)",

            # Find compound concepts
            "compounds": "(+/B/. * *)",

            # Find relations
            "relations": "(*/Br.{ma} * *)",
        }
