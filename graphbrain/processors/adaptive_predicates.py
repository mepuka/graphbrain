"""Adaptive predicate discovery and classification for actor/claim/conflict extraction.

Instead of hardcoded predicate lists, this module:
1. Discovers predicates dynamically from the corpus
2. Uses semantic similarity to classify predicates
3. Optionally uses LLMs to expand/refine predicate categories
4. Builds domain-adaptive actor/conflict networks

Usage:
    from graphbrain.processors.adaptive_predicates import (
        PredicateAnalyzer,
        AdaptiveActors,
        AdaptiveConflicts,
        AdaptiveClaims
    )

    # Analyze corpus to discover predicates
    analyzer = PredicateAnalyzer(hg)
    analyzer.discover_predicates()

    # Get predicates by semantic category
    conflict_preds = analyzer.get_predicates_like(['attack', 'blame', 'accuse'])
    claim_preds = analyzer.get_predicates_like(['say', 'claim', 'announce'])
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterator, Optional

from graphbrain.hyperedge import Hyperedge
from graphbrain.hypergraph import Hypergraph
from graphbrain.processor import Processor
from graphbrain.utils.lemmas import deep_lemma

logger = logging.getLogger(__name__)

# Seed predicates for each category - used to bootstrap similarity matching
SEED_PREDICATES = {
    'claim': ['say', 'claim', 'state', 'assert', 'announce', 'declare', 'report'],
    'conflict': ['attack', 'blame', 'accuse', 'condemn', 'warn', 'threaten', 'criticize'],
    'action': ['do', 'make', 'create', 'build', 'develop', 'launch', 'start'],
    'cognition': ['think', 'believe', 'know', 'understand', 'realize', 'consider'],
    'emotion': ['love', 'hate', 'fear', 'enjoy', 'like', 'dislike', 'prefer'],
    'movement': ['go', 'come', 'move', 'travel', 'leave', 'arrive', 'return'],
    'possession': ['have', 'own', 'possess', 'hold', 'keep', 'give', 'take'],
    'perception': ['see', 'hear', 'watch', 'notice', 'observe', 'feel', 'sense'],
}


@dataclass
class PredicateInfo:
    """Information about a discovered predicate."""
    lemma: str
    frequency: int = 0
    categories: dict = field(default_factory=dict)  # category -> similarity score
    examples: list = field(default_factory=list)  # sample edges using this predicate
    actors: set = field(default_factory=set)  # entities using this predicate

    @property
    def primary_category(self) -> Optional[str]:
        """Get the highest-scoring category."""
        if not self.categories:
            return None
        return max(self.categories.items(), key=lambda x: x[1])[0]


class PredicateAnalyzer:
    """Analyzes a hypergraph corpus to discover and classify predicates.

    This replaces hardcoded predicate lists with dynamic discovery based on:
    1. Corpus frequency analysis
    2. Semantic similarity to seed predicates
    3. Optional LLM-based classification
    """

    def __init__(self, hg: Hypergraph, embedding_matcher=None):
        """Initialize the analyzer.

        Args:
            hg: Hypergraph to analyze
            embedding_matcher: Optional SentenceTransformerMatcher for similarity
        """
        self.hg = hg
        self._matcher = embedding_matcher
        self._predicates: dict[str, PredicateInfo] = {}
        self._category_embeddings: dict[str, list] = {}

    @property
    def matcher(self):
        """Lazy-load the embedding matcher."""
        if self._matcher is None:
            try:
                from graphbrain.semsim.interface import get_matcher, SemSimType
                self._matcher = get_matcher(SemSimType.FIX)
                logger.info("Loaded SentenceTransformerMatcher for predicate similarity")
            except Exception as e:
                logger.warning(f"Could not load embedding matcher: {e}")
        return self._matcher

    def discover_predicates(self, min_frequency: int = 2) -> dict[str, PredicateInfo]:
        """Discover all predicates in the hypergraph.

        Args:
            min_frequency: Minimum occurrences to include predicate

        Returns:
            Dictionary of predicate lemma -> PredicateInfo
        """
        logger.info("Discovering predicates from corpus...")

        predicate_counts = Counter()
        predicate_examples = {}
        predicate_actors = {}

        for edge in self.hg.all():
            if edge.atom:
                continue

            # Check if this is a predicate edge
            connector = edge[0]
            if not hasattr(connector, 'type'):
                continue

            conn_type = connector.type()
            if not conn_type.startswith('P'):
                continue

            # Get the lemma
            lemma_edge = deep_lemma(self.hg, connector, same_if_none=True)
            if lemma_edge is None:
                continue
            lemma = lemma_edge.root()

            # Count and collect examples
            predicate_counts[lemma] += 1

            if lemma not in predicate_examples:
                predicate_examples[lemma] = []
            if len(predicate_examples[lemma]) < 5:  # Keep up to 5 examples
                predicate_examples[lemma].append(edge)

            # Track actors (subjects)
            if lemma not in predicate_actors:
                predicate_actors[lemma] = set()

            # Extract subject if present (usually first arg after connector)
            if len(edge) > 1:
                subj = edge[1]
                if hasattr(subj, 'type') and subj.type().startswith('C'):
                    predicate_actors[lemma].add(str(subj))

        # Build predicate info
        for lemma, count in predicate_counts.items():
            if count >= min_frequency:
                self._predicates[lemma] = PredicateInfo(
                    lemma=lemma,
                    frequency=count,
                    examples=predicate_examples.get(lemma, []),
                    actors=predicate_actors.get(lemma, set())
                )

        logger.info(f"Discovered {len(self._predicates)} predicates (min_freq={min_frequency})")

        # Classify predicates using semantic similarity
        self._classify_predicates_by_similarity()

        return self._predicates

    def _classify_predicates_by_similarity(self):
        """Classify discovered predicates using embedding similarity."""
        if not self.matcher:
            logger.warning("No embedding matcher available, skipping classification")
            return

        logger.info("Classifying predicates by semantic similarity...")

        for lemma, info in self._predicates.items():
            # Compare to each category's seed predicates
            for category, seeds in SEED_PREDICATES.items():
                sims = self.matcher._similarities(cand_word=lemma, ref_words=seeds)
                if sims:
                    # Use max similarity to any seed word
                    max_sim = max(sims.values())
                    if max_sim > 0.5:  # Only store if reasonably similar
                        info.categories[category] = max_sim

    def get_predicates_by_category(
        self,
        category: str,
        threshold: float = 0.7,
        min_frequency: int = 1
    ) -> list[str]:
        """Get predicates belonging to a semantic category.

        Args:
            category: Category name (e.g., 'claim', 'conflict')
            threshold: Minimum similarity score
            min_frequency: Minimum corpus frequency

        Returns:
            List of predicate lemmas
        """
        results = []
        for lemma, info in self._predicates.items():
            if info.frequency < min_frequency:
                continue
            if category in info.categories and info.categories[category] >= threshold:
                results.append(lemma)

        # Sort by similarity score
        results.sort(key=lambda x: self._predicates[x].categories.get(category, 0), reverse=True)
        return results

    def get_predicates_like(
        self,
        seed_words: list[str],
        threshold: float = 0.7
    ) -> list[str]:
        """Find predicates semantically similar to seed words.

        Args:
            seed_words: List of example predicates
            threshold: Minimum similarity score

        Returns:
            List of similar predicate lemmas with scores
        """
        if not self.matcher:
            logger.warning("No embedding matcher, returning exact matches only")
            return [w for w in seed_words if w in self._predicates]

        results = []
        for lemma in self._predicates:
            sims = self.matcher._similarities(cand_word=lemma, ref_words=seed_words)
            if sims:
                max_sim = max(sims.values())
                if max_sim >= threshold:
                    results.append((lemma, max_sim))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results]

    def expand_with_llm(
        self,
        category: str,
        llm_client=None,
        prompt_template: str = None
    ) -> list[str]:
        """Use an LLM to expand predicate categories.

        Args:
            category: Category to expand
            llm_client: LLM client (e.g., OpenAI, Anthropic)
            prompt_template: Custom prompt template

        Returns:
            Expanded list of predicates
        """
        if llm_client is None:
            logger.warning("No LLM client provided, cannot expand predicates")
            return self.get_predicates_by_category(category)

        # Get current predicates for context
        current = self.get_predicates_by_category(category, threshold=0.6)
        seeds = SEED_PREDICATES.get(category, [])

        # Build prompt
        if prompt_template is None:
            prompt_template = """Given these seed verbs for the category "{category}":
{seeds}

And these discovered verbs from the corpus:
{current}

List 20 more English verbs that belong to this semantic category "{category}".
Focus on verbs that express similar meanings or actions.
Return only the verb lemmas, one per line."""

        prompt = prompt_template.format(
            category=category,
            seeds=', '.join(seeds),
            current=', '.join(current[:20])
        )

        # Call LLM
        try:
            response = llm_client.complete(prompt)
            new_predicates = [line.strip().lower() for line in response.split('\n') if line.strip()]

            # Validate against corpus
            validated = []
            for pred in new_predicates:
                if pred in self._predicates:
                    validated.append(pred)
                    # Update category score
                    self._predicates[pred].categories[category] = 0.8  # LLM-assigned

            logger.info(f"LLM expanded '{category}' with {len(validated)} new predicates")
            return list(set(current + validated))

        except Exception as e:
            logger.error(f"LLM expansion failed: {e}")
            return current

    def get_predicate_stats(self) -> dict:
        """Get statistics about discovered predicates."""
        stats = {
            'total_predicates': len(self._predicates),
            'total_occurrences': sum(p.frequency for p in self._predicates.values()),
            'by_category': {},
            'top_predicates': [],
            'unclassified': 0
        }

        # Count by category
        for category in SEED_PREDICATES:
            preds = self.get_predicates_by_category(category, threshold=0.6)
            stats['by_category'][category] = len(preds)

        # Top predicates
        sorted_preds = sorted(
            self._predicates.values(),
            key=lambda x: x.frequency,
            reverse=True
        )
        stats['top_predicates'] = [
            {'lemma': p.lemma, 'freq': p.frequency, 'category': p.primary_category}
            for p in sorted_preds[:20]
        ]

        # Unclassified
        stats['unclassified'] = sum(
            1 for p in self._predicates.values()
            if not p.categories
        )

        return stats


class AdaptiveProcessor(Processor):
    """Base class for adaptive processors using dynamic predicate discovery."""

    def __init__(self, hg: Hypergraph, analyzer: PredicateAnalyzer = None):
        super().__init__(hg)
        self._analyzer = analyzer

    @property
    def analyzer(self) -> PredicateAnalyzer:
        """Lazy-load the predicate analyzer."""
        if self._analyzer is None:
            self._analyzer = PredicateAnalyzer(self.hg)
            self._analyzer.discover_predicates()
        return self._analyzer


class AdaptiveActors(AdaptiveProcessor):
    """Extract actors using adaptive predicate discovery.

    Unlike the hardcoded actors.py, this discovers action predicates
    dynamically from the corpus using semantic similarity.
    """

    def __init__(self, hg: Hypergraph, analyzer: PredicateAnalyzer = None):
        super().__init__(hg, analyzer)
        self._action_predicates = None
        self._actor_counts = Counter()
        self._actor_predicates = {}

    @property
    def action_predicates(self) -> set[str]:
        """Get action predicates from the analyzer."""
        if self._action_predicates is None:
            # Combine multiple relevant categories
            action_preds = set(self.analyzer.get_predicates_by_category('action', threshold=0.6))
            claim_preds = set(self.analyzer.get_predicates_by_category('claim', threshold=0.6))
            conflict_preds = set(self.analyzer.get_predicates_by_category('conflict', threshold=0.6))

            self._action_predicates = action_preds | claim_preds | conflict_preds
            logger.info(f"Using {len(self._action_predicates)} action predicates for actor extraction")

        return self._action_predicates

    def process_edge(self, edge: Hyperedge):
        """Process an edge to extract actors."""
        if edge.atom or len(edge) < 2:
            return

        connector = edge[0]
        if not hasattr(connector, 'type'):
            return

        # Check if predicate type
        if not connector.type().startswith('P'):
            return

        # Get lemma
        lemma_edge = deep_lemma(self.hg, connector, same_if_none=True)
        if lemma_edge is None:
            return
        lemma = lemma_edge.root()

        # Check if this is an action predicate
        if lemma not in self.action_predicates:
            return

        # Extract subject (actor)
        subject = edge[1]
        if hasattr(subject, 'type') and subject.type().startswith('C'):
            actor_str = str(subject)
            self._actor_counts[actor_str] += 1

            if actor_str not in self._actor_predicates:
                self._actor_predicates[actor_str] = Counter()
            self._actor_predicates[actor_str][lemma] += 1

    def actors(self, min_count: int = 1) -> Iterator[tuple]:
        """Yield discovered actors with their action profiles.

        Returns:
            Tuples of (actor_edge, count, top_predicates)
        """
        from graphbrain import hedge

        for actor_str, count in self._actor_counts.most_common():
            if count < min_count:
                continue

            actor_edge = hedge(actor_str)
            top_preds = self._actor_predicates[actor_str].most_common(5)

            yield (actor_edge, count, top_preds)


class AdaptiveConflicts(AdaptiveProcessor):
    """Extract conflicts using adaptive predicate discovery."""

    def __init__(self, hg: Hypergraph, analyzer: PredicateAnalyzer = None):
        super().__init__(hg, analyzer)
        self._conflict_predicates = None
        self._conflicts = []

    @property
    def conflict_predicates(self) -> set[str]:
        """Get conflict predicates from the analyzer."""
        if self._conflict_predicates is None:
            self._conflict_predicates = set(
                self.analyzer.get_predicates_by_category('conflict', threshold=0.6)
            )
            logger.info(f"Using {len(self._conflict_predicates)} conflict predicates")
        return self._conflict_predicates

    def process_edge(self, edge: Hyperedge):
        """Process an edge to extract conflicts."""
        if edge.atom or len(edge) < 3:
            return

        connector = edge[0]
        if not hasattr(connector, 'type'):
            return

        if not connector.type().startswith('P'):
            return

        # Get lemma
        lemma_edge = deep_lemma(self.hg, connector, same_if_none=True)
        if lemma_edge is None:
            return
        lemma = lemma_edge.root()

        # Check if conflict predicate
        if lemma not in self.conflict_predicates:
            return

        # Extract subject and object
        subject = edge[1]
        obj = edge[2] if len(edge) > 2 else None

        # Both must be concepts
        if not (hasattr(subject, 'type') and subject.type().startswith('C')):
            return
        if obj and not (hasattr(obj, 'type') and obj.type().startswith('C')):
            return

        self._conflicts.append({
            'source': subject,
            'target': obj,
            'predicate': lemma,
            'edge': edge
        })

    def conflicts(self) -> Iterator[dict]:
        """Yield discovered conflicts."""
        yield from self._conflicts


class AdaptiveClaims(AdaptiveProcessor):
    """Extract claims/statements using adaptive predicate discovery."""

    def __init__(self, hg: Hypergraph, analyzer: PredicateAnalyzer = None):
        super().__init__(hg, analyzer)
        self._claim_predicates = None
        self._claims = []

    @property
    def claim_predicates(self) -> set[str]:
        """Get claim predicates from the analyzer."""
        if self._claim_predicates is None:
            self._claim_predicates = set(
                self.analyzer.get_predicates_by_category('claim', threshold=0.6)
            )
            logger.info(f"Using {len(self._claim_predicates)} claim predicates")
        return self._claim_predicates

    def process_edge(self, edge: Hyperedge):
        """Process an edge to extract claims."""
        if edge.atom or len(edge) < 3:
            return

        connector = edge[0]
        if not hasattr(connector, 'type'):
            return

        if not connector.type().startswith('P'):
            return

        # Get lemma
        lemma_edge = deep_lemma(self.hg, connector, same_if_none=True)
        if lemma_edge is None:
            return
        lemma = lemma_edge.root()

        # Check if claim predicate
        if lemma not in self.claim_predicates:
            return

        # Extract claimer and claim content
        claimer = edge[1]
        content = edge[2] if len(edge) > 2 else None

        # Claimer should be a concept
        if not (hasattr(claimer, 'type') and claimer.type().startswith('C')):
            return

        self._claims.append({
            'claimer': claimer,
            'content': content,
            'predicate': lemma,
            'edge': edge
        })

    def claims(self) -> Iterator[dict]:
        """Yield discovered claims."""
        yield from self._claims


def analyze_corpus_predicates(hg: Hypergraph) -> dict:
    """Convenience function to analyze predicates in a hypergraph.

    Returns comprehensive statistics about predicates in the corpus.
    """
    analyzer = PredicateAnalyzer(hg)
    analyzer.discover_predicates()
    return analyzer.get_predicate_stats()
