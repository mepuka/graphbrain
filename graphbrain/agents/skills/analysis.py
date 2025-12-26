"""
Analysis skill for extracting insights from the knowledge graph.

Provides actor analysis, claim extraction, and relationship mapping.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum
import logging

from graphbrain.agents.skills.base import BaseSkill, SkillResult

logger = logging.getLogger(__name__)


class AnalysisMode(str, Enum):
    """Types of analysis."""
    ACTOR = "actor"
    CLAIM = "claim"
    RELATIONSHIP = "relationship"
    CONFLICT = "conflict"
    TEMPORAL = "temporal"


@dataclass
class ActorStats:
    """Statistics for an actor."""
    name: str
    edge_key: str
    claim_count: int = 0
    action_count: int = 0
    mention_count: int = 0
    topics: dict[str, int] = field(default_factory=dict)
    top_edges: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Result from an analysis operation."""
    mode: AnalysisMode
    actors: list[ActorStats] = field(default_factory=list)
    claims: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    statistics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class AnalysisSkill(BaseSkill):
    """
    Skill for analyzing the knowledge graph.

    Capabilities:
    - Actor analysis (who makes claims)
    - Claim extraction (what is being said)
    - Relationship mapping (how concepts connect)
    - Conflict detection (opposing views)
    - Temporal analysis (changes over time)
    """

    SKILL_NAME = "analysis"
    PROMPT_FILE = "analysis.md"
    DEFAULT_CONFIDENCE_THRESHOLD = 0.7

    # MCP tools this skill uses
    TOOLS = [
        "mcp__graphbrain__pattern_match",
        "mcp__graphbrain__hybrid_search",
        "mcp__graphbrain__hypergraph_stats",
        "mcp__graphbrain__list_semantic_classes",
        "mcp__graphbrain__classification_stats",
    ]

    def get_tools(self) -> list[str]:
        """Get the list of MCP tools this skill uses."""
        return self.TOOLS

    def get_analysis_patterns(self, mode: AnalysisMode) -> list[str]:
        """
        Get patterns for a specific analysis mode.

        Args:
            mode: The analysis mode

        Returns:
            List of patterns to use
        """
        patterns = {
            AnalysisMode.ACTOR: [
                "(*/Pd.{s} ACTOR/Cp ...)",  # Actor as subject
                "(*/Pd.{sr} ACTOR/Cp *)",   # Actor making claims
            ],
            AnalysisMode.CLAIM: [
                "(*/Pd.{sr} */Cp CLAIM/*)",  # Attributed claims
                "(*/Pd.{sc} SUBJECT/C *)",   # Declarative statements
            ],
            AnalysisMode.RELATIONSHIP: [
                "(*/Br.{ma} MAIN/C AUX/C)",  # Relational builders
                "(*/Pd * * ...)",            # Any predicate
            ],
            AnalysisMode.CONFLICT: [
                "(*/Pd.{so} */Cp */C)",      # Subject-object predicates
            ],
            AnalysisMode.TEMPORAL: [
                "(*/Tt *)",                  # Temporal triggers
                "(*/Pd.{sox} * * */T)",      # Predicates with temporal spec
            ],
        }

        return patterns.get(mode, [])

    def build_actor_analysis_params(
        self,
        domain: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Build parameters for actor analysis.

        Args:
            domain: Optional domain filter
            limit: Maximum results per pattern

        Returns:
            List of parameter dicts for pattern_match calls
        """
        patterns = self.get_analysis_patterns(AnalysisMode.ACTOR)
        return [
            {"pattern": p, "strict": False, "limit": limit}
            for p in patterns
        ]

    def build_claim_analysis_params(
        self,
        topic: Optional[str] = None,
        class_id: str = "claim",
        limit: int = 100,
    ) -> dict:
        """
        Build parameters for claim analysis.

        Args:
            topic: Optional topic to filter by
            class_id: Semantic class for claims
            limit: Maximum results

        Returns:
            Parameters for hybrid_search
        """
        params = {
            "class_id": class_id,
            "limit": limit,
        }

        if topic:
            params["query"] = topic

        return params

    def aggregate_actors(
        self,
        edges: list[dict],
    ) -> list[ActorStats]:
        """
        Aggregate actor statistics from edge results.

        Args:
            edges: List of edge dictionaries with bindings

        Returns:
            List of ActorStats
        """
        actor_map: dict[str, ActorStats] = {}

        for edge in edges:
            bindings = edge.get("bindings", {})
            actor_key = bindings.get("ACTOR") or bindings.get("SPEAKER")

            if not actor_key:
                continue

            if actor_key not in actor_map:
                actor_map[actor_key] = ActorStats(
                    name=self._extract_name(actor_key),
                    edge_key=actor_key,
                )

            stats = actor_map[actor_key]
            stats.mention_count += 1

            # Categorize by edge type
            edge_str = edge.get("edge", "")
            if "claim" in edge.get("class", "").lower():
                stats.claim_count += 1
            else:
                stats.action_count += 1

            # Track topics
            topic = self._extract_topic(edge)
            if topic:
                stats.topics[topic] = stats.topics.get(topic, 0) + 1

            # Keep top edges
            if len(stats.top_edges) < 5:
                stats.top_edges.append(edge_str)

        # Sort by total mentions
        actors = sorted(
            actor_map.values(),
            key=lambda a: a.mention_count,
            reverse=True
        )

        return actors

    def _extract_name(self, edge_key: str) -> str:
        """Extract human-readable name from edge key."""
        # Remove type annotations
        if "/" in edge_key:
            name = edge_key.split("/")[0]
        else:
            name = edge_key

        # Convert underscores to spaces, title case
        return name.replace("_", " ").title()

    def _extract_topic(self, edge: dict) -> Optional[str]:
        """Extract topic from an edge."""
        # This is a simplified extraction
        # Full implementation would parse the edge structure
        bindings = edge.get("bindings", {})
        claim = bindings.get("CLAIM") or bindings.get("OBJECT")

        if claim and isinstance(claim, str):
            # Get first significant word
            words = claim.replace("/", " ").split()
            for word in words:
                if len(word) > 3 and not word[0].isupper():
                    continue
                if word.lower() not in ["the", "a", "an", "is", "was", "are"]:
                    return word.lower()

        return None

    def detect_conflicts(
        self,
        claims: list[dict],
        conflict_predicates: list[str] = None,
    ) -> list[dict]:
        """
        Detect conflicting claims.

        Args:
            claims: List of claim edges
            conflict_predicates: Predicates indicating conflict

        Returns:
            List of detected conflicts
        """
        if conflict_predicates is None:
            conflict_predicates = [
                "oppose", "disagree", "criticize", "reject",
                "contradict", "deny", "dispute", "challenge",
            ]

        conflicts = []

        # Group claims by topic
        by_topic: dict[str, list[dict]] = {}
        for claim in claims:
            topic = self._extract_topic(claim)
            if topic:
                if topic not in by_topic:
                    by_topic[topic] = []
                by_topic[topic].append(claim)

        # Look for opposing claims on same topic
        for topic, topic_claims in by_topic.items():
            if len(topic_claims) < 2:
                continue

            # Check for conflict predicates
            for i, claim1 in enumerate(topic_claims):
                for claim2 in topic_claims[i + 1:]:
                    pred1 = claim1.get("predicate", "")
                    pred2 = claim2.get("predicate", "")

                    # Check if predicates indicate conflict
                    if any(cp in pred1.lower() or cp in pred2.lower()
                           for cp in conflict_predicates):
                        conflicts.append({
                            "topic": topic,
                            "claim1": claim1,
                            "claim2": claim2,
                            "type": "predicate_conflict",
                        })

        return conflicts

    def generate_summary(
        self,
        result: AnalysisResult,
    ) -> str:
        """
        Generate a markdown summary of analysis results.

        Args:
            result: The analysis result

        Returns:
            Markdown-formatted summary
        """
        lines = [f"## {result.mode.value.title()} Analysis\n"]

        if result.actors:
            lines.append("### Key Actors\n")
            lines.append("| Actor | Claims | Actions | Topics |")
            lines.append("|-------|--------|---------|--------|")
            for actor in result.actors[:10]:
                topics = ", ".join(list(actor.topics.keys())[:3])
                lines.append(
                    f"| {actor.name} | {actor.claim_count} | "
                    f"{actor.action_count} | {topics} |"
                )
            lines.append("")

        if result.claims:
            lines.append(f"### Claims ({len(result.claims)} total)\n")
            for claim in result.claims[:5]:
                speaker = claim.get("speaker", "Unknown")
                text = claim.get("text", claim.get("edge", ""))[:100]
                lines.append(f"- **{speaker}**: {text}")
            lines.append("")

        if result.conflicts:
            lines.append(f"### Conflicts ({len(result.conflicts)} detected)\n")
            for conflict in result.conflicts[:5]:
                lines.append(f"- **{conflict['topic']}**: {conflict['type']}")
            lines.append("")

        if result.statistics:
            lines.append("### Statistics\n")
            for key, value in result.statistics.items():
                lines.append(f"- {key}: {value}")

        return "\n".join(lines)
