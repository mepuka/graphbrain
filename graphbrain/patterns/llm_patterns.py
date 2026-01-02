"""LLM-enhanced pattern matching and discovery.

This module provides:
1. Natural language → pattern query translation
2. Pattern explanation in natural language
3. Automatic pattern discovery from examples
4. Pattern suggestion based on user intent

Usage:
    from graphbrain.patterns.llm_patterns import (
        PatternDiscovery,
        NaturalLanguagePatterns,
        PatternExplainer
    )

    # Discover patterns from examples
    discovery = PatternDiscovery(hg)
    patterns = discovery.discover_from_examples(example_edges)

    # Natural language to pattern
    nl = NaturalLanguagePatterns(hg)
    pattern = nl.query_to_pattern("Find all sentences where someone says something")

    # Explain a pattern
    explainer = PatternExplainer()
    explanation = explainer.explain_pattern("(says/P SPEAKER CONTENT)")
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterator, Optional

from graphbrain import hedge
from graphbrain.hyperedge import Hyperedge
from graphbrain.hypergraph import Hypergraph

logger = logging.getLogger(__name__)


# Pattern syntax documentation for LLM prompts
PATTERN_SYNTAX_GUIDE = """
Graphbrain Pattern Syntax:

WILDCARDS:
- *     : Matches any edge (atom or non-atom)
- .     : Matches any atom only
- (*)   : Matches any non-atomic edge
- ...   : Open-ended, matches 0 or more additional items

TYPE SPECIFIERS:
- */P   : Any predicate (verb/action)
- */C   : Any concept (noun/entity)
- */M   : Any modifier (adjective/adverb)
- */B   : Any builder (preposition/conjunction)
- */T   : Any trigger (subordinating element)

VARIABLES (uppercase):
- VARNAME   : Captures matched value as variable
- */P.so    : Predicate with subject-object argroles

ARGROLES:
- (is/P.sc SUBJ COMP)     : Ordered - subject then complement
- (is/P.{sc} SUBJ COMP)   : Unordered - either order works

FUNCTIONS:
- (lemma ROOT)            : Match by lemma (is/was/been all match "be")
- (any PAT1 PAT2)         : OR - match any sub-pattern
- (not PATTERN)           : Negation - match if doesn't match
- (atoms ATOM1 ATOM2)     : Match atoms at any depth

EXAMPLES:
- (*/P * *)               : Any predicate with 2 arguments
- (says/P SPEAKER *)      : Says with captured speaker
- (lemma be/P) * *)       : Any form of "be" (is/was/been)
- (*/P.{so} SUBJ OBJ)     : Predicate with subject and object
"""


@dataclass
class PatternCandidate:
    """A discovered pattern candidate."""
    pattern: str
    frequency: int
    examples: list
    confidence: float
    description: str = ""


class PatternDiscovery:
    """Discover common patterns from a hypergraph corpus.

    Uses structural analysis and optional LLM assistance to find
    recurring patterns worth querying.
    """

    def __init__(self, hg: Hypergraph):
        self.hg = hg

    def discover_structural_patterns(
        self,
        max_depth: int = 2,
        min_frequency: int = 5
    ) -> list[PatternCandidate]:
        """Discover patterns by analyzing edge structures.

        Args:
            max_depth: How deep to analyze nested structures
            min_frequency: Minimum occurrences for a pattern

        Returns:
            List of discovered pattern candidates
        """
        logger.info("Discovering structural patterns...")

        pattern_counts = Counter()
        pattern_examples = {}

        for edge in self.hg.all():
            if edge.atom:
                continue

            # Generate pattern signature
            pattern = self._edge_to_pattern(edge, max_depth)
            pattern_counts[pattern] += 1

            if pattern not in pattern_examples:
                pattern_examples[pattern] = []
            if len(pattern_examples[pattern]) < 5:
                pattern_examples[pattern].append(edge)

        # Filter by frequency and create candidates
        candidates = []
        for pattern, count in pattern_counts.most_common():
            if count < min_frequency:
                continue

            candidates.append(PatternCandidate(
                pattern=pattern,
                frequency=count,
                examples=pattern_examples[pattern],
                confidence=min(1.0, count / 100),  # Simple confidence score
                description=self._describe_pattern(pattern)
            ))

        logger.info(f"Discovered {len(candidates)} structural patterns")
        return candidates

    def _edge_to_pattern(self, edge: Hyperedge, depth: int) -> str:
        """Convert an edge to a pattern signature."""
        if edge.atom:
            # Keep type but wildcard the root
            if hasattr(edge, 'type'):
                return f"*/{edge.type()}"
            return "."

        if depth == 0:
            return "(*)"

        # Recursively build pattern
        parts = []
        for i, sub in enumerate(edge):
            if i == 0:  # Connector - keep more detail
                if sub.atom and hasattr(sub, 'type'):
                    parts.append(f"*/{sub.type()}")
                else:
                    parts.append(self._edge_to_pattern(sub, depth - 1))
            else:  # Arguments - generalize more
                parts.append(self._edge_to_pattern(sub, depth - 1))

        return f"({' '.join(parts)})"

    def _describe_pattern(self, pattern: str) -> str:
        """Generate a human-readable description of a pattern."""
        descriptions = []

        # Check for predicate types
        if '*/Pd' in pattern:
            descriptions.append("action/event")
        elif '*/P' in pattern:
            descriptions.append("predicate")

        # Check for concepts
        if '*/Cp' in pattern:
            descriptions.append("with proper noun")
        elif '*/Cc' in pattern:
            descriptions.append("with common noun")
        elif '*/C' in pattern:
            descriptions.append("with concept")

        # Check for modifiers
        if '*/M' in pattern:
            descriptions.append("with modifier")

        # Check for builders
        if '*/Br' in pattern:
            descriptions.append("with relation")
        elif '*/B' in pattern:
            descriptions.append("with builder")

        if descriptions:
            return "Edge pattern: " + ", ".join(descriptions)
        return "General edge pattern"

    def discover_from_examples(
        self,
        examples: list[Hyperedge],
        generalization_level: int = 1
    ) -> list[PatternCandidate]:
        """Discover patterns from example edges.

        Args:
            examples: List of example edges to learn from
            generalization_level: How much to generalize (0=exact, 2=very general)

        Returns:
            Patterns that match the examples
        """
        if not examples:
            return []

        # Find common structure
        patterns = []
        for ex in examples:
            pattern = self._edge_to_pattern(ex, 3 - generalization_level)
            patterns.append(pattern)

        # Find most common pattern
        pattern_counts = Counter(patterns)
        most_common = pattern_counts.most_common(1)[0][0]

        return [PatternCandidate(
            pattern=most_common,
            frequency=pattern_counts[most_common],
            examples=examples[:5],
            confidence=pattern_counts[most_common] / len(examples),
            description=f"Pattern learned from {len(examples)} examples"
        )]

    def suggest_patterns_for_query(
        self,
        query_intent: str,
        llm_client=None
    ) -> list[str]:
        """Use LLM to suggest patterns for a natural language query.

        Args:
            query_intent: What the user wants to find
            llm_client: LLM client for generation

        Returns:
            List of suggested pattern strings
        """
        if llm_client is None:
            # Fallback to rule-based suggestions
            return self._rule_based_suggestions(query_intent)

        prompt = f"""Given this pattern syntax guide:
{PATTERN_SYNTAX_GUIDE}

Generate 3 graphbrain patterns to find: "{query_intent}"

Return only the patterns, one per line, in valid graphbrain syntax.
"""
        try:
            response = llm_client.complete(prompt)
            patterns = []
            for line in response.strip().split('\n'):
                line = line.strip()
                if line.startswith('(') or line.startswith('*') or line.startswith('.'):
                    patterns.append(line)
            return patterns[:3]
        except Exception as e:
            logger.error(f"LLM pattern suggestion failed: {e}")
            return self._rule_based_suggestions(query_intent)

    def _rule_based_suggestions(self, query_intent: str) -> list[str]:
        """Rule-based pattern suggestions without LLM."""
        query_lower = query_intent.lower()

        suggestions = []

        # Check for common intents
        if 'say' in query_lower or 'claim' in query_lower or 'statement' in query_lower:
            suggestions.append("(lemma say/P) SPEAKER *)")
            suggestions.append("(*/Pd.* */Cp *)")

        if 'conflict' in query_lower or 'attack' in query_lower or 'against' in query_lower:
            suggestions.append("(*/Pd.{so} ACTOR TARGET)")
            suggestions.append("(any (attack/P * *) (blame/P * *) (accuse/P * *))")

        if 'relationship' in query_lower or 'between' in query_lower:
            suggestions.append("(*/P.{so} ENTITY1 ENTITY2)")
            suggestions.append("(*/Br * *)")

        if 'person' in query_lower or 'who' in query_lower:
            suggestions.append("(*/P */Cp *)")

        if 'location' in query_lower or 'where' in query_lower or 'place' in query_lower:
            suggestions.append("(in/T (*/P *))")
            suggestions.append("(*/Br.* * */Cp)")

        if 'time' in query_lower or 'when' in query_lower:
            suggestions.append("(*/T (*/P *))")

        # Default
        if not suggestions:
            suggestions.append("(*/P * *)")
            suggestions.append("(*/Pd.* */C *)")

        return suggestions[:3]


class NaturalLanguagePatterns:
    """Convert between natural language and pattern syntax."""

    def __init__(self, hg: Hypergraph):
        self.hg = hg

    def query_to_pattern(
        self,
        natural_query: str,
        llm_client=None
    ) -> Optional[str]:
        """Convert a natural language query to a pattern.

        Args:
            natural_query: Natural language description of what to find
            llm_client: Optional LLM for better translation

        Returns:
            Pattern string or None if translation fails
        """
        if llm_client:
            return self._llm_translate(natural_query, llm_client)
        return self._rule_translate(natural_query)

    def _llm_translate(self, query: str, llm_client) -> Optional[str]:
        """Use LLM to translate query to pattern."""
        prompt = f"""Convert this natural language query to a graphbrain pattern:

Query: "{query}"

Pattern syntax guide:
{PATTERN_SYNTAX_GUIDE}

Return ONLY the pattern, nothing else. Example output: (*/P AGENT *)
"""
        try:
            response = llm_client.complete(prompt).strip()
            # Extract pattern from response
            match = re.search(r'\([^)]+\)', response)
            if match:
                return match.group()
            if response.startswith('*') or response.startswith('.'):
                return response
            return None
        except Exception as e:
            logger.error(f"LLM translation failed: {e}")
            return self._rule_translate(query)

    def _rule_translate(self, query: str) -> Optional[str]:
        """Rule-based translation without LLM."""
        query_lower = query.lower()

        # Simple keyword-based translation
        if 'who' in query_lower and 'say' in query_lower:
            return "(lemma say/P) SPEAKER *)"

        if 'all' in query_lower and 'predicate' in query_lower:
            return "(*/P * *)"

        if 'concept' in query_lower:
            return "*/C"

        if 'modifier' in query_lower:
            return "(*/M *)"

        if 'relationship' in query_lower:
            return "(*/P.{so} * *)"

        # Default: try to extract key words
        words = query_lower.split()
        for word in words:
            if word in ['says', 'say', 'said']:
                return "(lemma say/P) * *)"
            if word in ['loves', 'love', 'loved']:
                return "(lemma love/P) * *)"
            if word in ['is', 'was', 'are', 'were']:
                return "(lemma be/P) * *)"

        return "(*/P * *)"

    def pattern_to_description(self, pattern: str) -> str:
        """Convert a pattern to natural language description."""
        return PatternExplainer.explain(pattern)


class PatternExplainer:
    """Explain patterns in natural language."""

    @staticmethod
    def explain(pattern: str) -> str:
        """Generate a natural language explanation of a pattern.

        Args:
            pattern: Graphbrain pattern string

        Returns:
            Human-readable explanation
        """
        explanations = []

        # Parse the pattern
        if pattern.startswith('(') and pattern.endswith(')'):
            inner = pattern[1:-1]
            parts = inner.split()

            if parts:
                connector = parts[0]
                args = parts[1:]

                # Explain connector
                if connector.startswith('*/P'):
                    explanations.append("any predicate (action/verb)")
                elif connector.startswith('lemma'):
                    if len(parts) > 1:
                        explanations.append(f"any form of '{parts[1].split('/')[0]}'")
                elif '/' in connector:
                    root = connector.split('/')[0]
                    explanations.append(f"the action '{root}'")
                else:
                    explanations.append(connector)

                # Explain arguments
                for i, arg in enumerate(args):
                    if arg.isupper():
                        explanations.append(f"capturing as {arg}")
                    elif arg == '*':
                        explanations.append("with any argument")
                    elif arg == '.':
                        explanations.append("with any atom")
                    elif arg.startswith('*/C'):
                        explanations.append("with any concept")
                    elif arg.startswith('*/P'):
                        explanations.append("with any predicate")
                    elif arg == '...':
                        explanations.append("and possibly more")

        elif pattern == '*':
            return "Matches any edge"
        elif pattern == '.':
            return "Matches any atom"
        elif pattern.startswith('*/'):
            type_code = pattern[2:]
            type_names = {
                'P': 'predicate', 'C': 'concept', 'M': 'modifier',
                'B': 'builder', 'T': 'trigger', 'J': 'junction'
            }
            return f"Matches any {type_names.get(type_code[0], 'edge')} of type {type_code}"

        if explanations:
            return "Find: " + " ".join(explanations)
        return f"Pattern: {pattern}"

    @staticmethod
    def explain_match(
        edge: Hyperedge,
        pattern: str,
        variables: dict
    ) -> str:
        """Explain why an edge matched a pattern.

        Args:
            edge: The matched edge
            pattern: The pattern that matched
            variables: Captured variable bindings

        Returns:
            Explanation of the match
        """
        parts = [f"Edge '{edge}' matches pattern '{pattern}'"]

        if variables:
            parts.append("with bindings:")
            for var, val in variables.items():
                parts.append(f"  {var} = {val}")

        return "\n".join(parts)


# Domain-specific pattern templates
DOMAIN_PATTERNS = {
    "claims": {
        "patterns": [
            "(lemma say/Pd) SPEAKER CLAIM)",
            "(lemma claim/Pd) SPEAKER CLAIM)",
            "(lemma announce/Pd) SPEAKER ANNOUNCEMENT)",
            "(lemma state/Pd) SPEAKER STATEMENT)",
        ],
        "keywords": ["say", "claim", "state", "announce", "declare", "assert"],
        "description": "Speech acts and claims by actors",
    },
    "conflict": {
        "patterns": [
            "(lemma attack/Pd) ACTOR TARGET)",
            "(lemma criticize/Pd) CRITIC TARGET)",
            "(lemma oppose/Pd) OPPONENT PROPOSAL)",
            "(lemma blame/Pd) BLAMER TARGET)",
        ],
        "keywords": ["attack", "criticize", "oppose", "blame", "accuse", "condemn"],
        "description": "Conflict and criticism between actors",
    },
    "support": {
        "patterns": [
            "(lemma support/Pd) SUPPORTER PROPOSAL)",
            "(lemma endorse/Pd) ENDORSER CANDIDATE)",
            "(lemma approve/Pd) APPROVER PROPOSAL)",
            "(lemma back/Pd) BACKER PROPOSAL)",
        ],
        "keywords": ["support", "endorse", "approve", "back", "champion"],
        "description": "Support and endorsement relationships",
    },
    "policy": {
        "patterns": [
            "(*/Pd.* */Cp (*/Br policy/Cc *))",
            "(*/Pd.* */Cp (*/Br plan/Cc *))",
            "(*/Pd.* */Cp (*/Br proposal/Cc *))",
        ],
        "keywords": ["policy", "plan", "proposal", "legislation", "ordinance"],
        "description": "Policy-related actions and proposals",
    },
    "location": {
        "patterns": [
            "(in/Br * */Cp)",
            "(at/Br * */Cp)",
            "(*/P.* * (in/T */Cp))",
        ],
        "keywords": ["in", "at", "near", "around", "city", "neighborhood"],
        "description": "Location and geographic references",
    },
    "temporal": {
        "patterns": [
            "(*/T (*/P *))",
            "(*/P.* * (before/T *))",
            "(*/P.* * (after/T *))",
        ],
        "keywords": ["before", "after", "when", "during", "until"],
        "description": "Temporal relationships and timing",
    },
}


class AdaptivePatternMatcher:
    """Combines pattern matching with LLM-enhanced features."""

    def __init__(self, hg: Hypergraph, llm_client=None):
        self.hg = hg
        self.llm_client = llm_client
        self.discovery = PatternDiscovery(hg)
        self.nl_patterns = NaturalLanguagePatterns(hg)
        self.domain_patterns = DOMAIN_PATTERNS

    def search_natural(self, query: str, limit: int = 100) -> Iterator[Hyperedge]:
        """Search using natural language query.

        Args:
            query: Natural language description of what to find
            limit: Maximum results

        Returns:
            Matching edges
        """
        pattern_str = self.nl_patterns.query_to_pattern(query, self.llm_client)
        if pattern_str is None:
            logger.warning(f"Could not translate query: {query}")
            return iter([])

        logger.info(f"Translated '{query}' to pattern: {pattern_str}")

        try:
            pattern = hedge(pattern_str)
            count = 0
            for edge in self.hg.search(pattern):
                yield edge
                count += 1
                if count >= limit:
                    break
        except Exception as e:
            logger.error(f"Pattern search failed: {e}")

    def discover_and_explain(self, min_frequency: int = 5) -> list[dict]:
        """Discover patterns and explain them.

        Returns:
            List of {pattern, frequency, explanation, examples}
        """
        candidates = self.discovery.discover_structural_patterns(min_frequency=min_frequency)

        results = []
        for cand in candidates[:20]:  # Top 20
            results.append({
                'pattern': cand.pattern,
                'frequency': cand.frequency,
                'explanation': PatternExplainer.explain(cand.pattern),
                'examples': [str(ex) for ex in cand.examples[:3]]
            })

        return results

    def suggest_related_patterns(self, pattern: str) -> list[str]:
        """Suggest patterns related to the given one.

        Args:
            pattern: Base pattern

        Returns:
            List of related patterns
        """
        suggestions = []

        # More specific version
        if '*' in pattern:
            suggestions.append(pattern.replace('*', '*/C', 1))
            suggestions.append(pattern.replace('*', '*/P', 1))

        # More general version
        if '/P' in pattern:
            suggestions.append(pattern.replace('/P', '/P.*', 1))

        # Add variable capture
        if 'CAPTURE' not in pattern and '*' in pattern:
            suggestions.append(pattern.replace('*', 'CAPTURE', 1))

        return suggestions[:5]

    def search_semantic(
        self,
        query: str,
        limit: int = 100
    ) -> list[dict]:
        """Search using semantic understanding of query intent.

        Combines domain patterns with natural language understanding.

        Args:
            query: Natural language query like "claims about housing policy"
            limit: Maximum results

        Returns:
            List of {edge, pattern_used, domain, confidence}
        """
        query_lower = query.lower()
        results = []

        # Identify relevant domains
        matching_domains = []
        for domain, config in self.domain_patterns.items():
            # Check if any keyword matches
            if any(kw in query_lower for kw in config["keywords"]):
                matching_domains.append((domain, config))
            # Check if domain name matches
            elif domain in query_lower:
                matching_domains.append((domain, config))

        if not matching_domains:
            # Default to claims domain for statement-like queries
            if any(w in query_lower for w in ["who", "what", "said", "about"]):
                matching_domains.append(("claims", self.domain_patterns["claims"]))

        # Search using domain patterns
        seen_edges = set()
        for domain, config in matching_domains:
            for pattern_str in config["patterns"]:
                try:
                    pattern = hedge(pattern_str)
                    for edge in self.hg.search(pattern):
                        edge_str = str(edge)
                        if edge_str not in seen_edges:
                            seen_edges.add(edge_str)
                            results.append({
                                "edge": edge,
                                "pattern_used": pattern_str,
                                "domain": domain,
                                "confidence": 0.8,
                                "description": config["description"]
                            })
                            if len(results) >= limit:
                                return results
                except Exception as e:
                    logger.debug(f"Pattern search failed for {pattern_str}: {e}")

        return results

    def get_domain_patterns(self, domain: str) -> list[str]:
        """Get patterns for a specific domain.

        Args:
            domain: Domain name (claims, conflict, support, policy, etc.)

        Returns:
            List of pattern strings for that domain
        """
        if domain in self.domain_patterns:
            return self.domain_patterns[domain]["patterns"]
        return []

    def list_domains(self) -> list[dict]:
        """List available semantic domains and their descriptions.

        Returns:
            List of {name, description, keywords, pattern_count}
        """
        domains = []
        for name, config in self.domain_patterns.items():
            domains.append({
                "name": name,
                "description": config["description"],
                "keywords": config["keywords"],
                "pattern_count": len(config["patterns"])
            })
        return domains

    def explain_query(self, query: str) -> str:
        """Explain how a query will be interpreted.

        Args:
            query: Natural language query

        Returns:
            Explanation of query interpretation
        """
        query_lower = query.lower()
        parts = [f"Query: \"{query}\"", ""]

        # Identify domains
        matching_domains = []
        for domain, config in self.domain_patterns.items():
            matched_keywords = [kw for kw in config["keywords"] if kw in query_lower]
            if matched_keywords:
                matching_domains.append({
                    "domain": domain,
                    "matched": matched_keywords,
                    "description": config["description"]
                })

        if matching_domains:
            parts.append("Detected domains:")
            for d in matching_domains:
                parts.append(f"  - {d['domain']}: {d['description']}")
                parts.append(f"    Keywords matched: {', '.join(d['matched'])}")
        else:
            parts.append("No specific domain detected, using general search.")

        # Show patterns that will be used
        parts.append("")
        parts.append("Patterns to search:")
        for d in matching_domains:
            patterns = self.get_domain_patterns(d["domain"])
            for p in patterns[:2]:
                parts.append(f"  - {p}")

        return "\n".join(parts)
