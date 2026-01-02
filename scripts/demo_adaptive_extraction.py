#!/usr/bin/env python3
"""Demo of adaptive actor/conflict extraction and LLM-enhanced pattern discovery.

This script demonstrates:
1. Dynamic predicate discovery from corpus
2. Semantic similarity-based predicate classification
3. Adaptive actor/conflict extraction
4. Pattern discovery and explanation
5. Natural language pattern queries

Usage:
    python scripts/demo_adaptive_extraction.py --db knowledge.db
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from graphbrain import hgraph

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def demo_predicate_discovery(hg):
    """Demonstrate adaptive predicate discovery."""
    from graphbrain.processors.adaptive_predicates import PredicateAnalyzer

    print("\n" + "="*70)
    print("ADAPTIVE PREDICATE DISCOVERY")
    print("="*70)

    # Create analyzer and discover predicates
    analyzer = PredicateAnalyzer(hg)
    predicates = analyzer.discover_predicates(min_frequency=2)

    # Show statistics
    stats = analyzer.get_predicate_stats()
    print(f"\nTotal predicates discovered: {stats['total_predicates']}")
    print(f"Total predicate occurrences: {stats['total_occurrences']}")

    print("\n--- Predicates by Category ---")
    for category, count in sorted(stats['by_category'].items(), key=lambda x: -x[1]):
        preds = analyzer.get_predicates_by_category(category, threshold=0.6)[:5]
        print(f"\n{category.upper()} ({count} predicates):")
        for pred in preds:
            info = predicates[pred]
            sim = info.categories.get(category, 0)
            print(f"  {pred:15} freq={info.frequency:3} sim={sim:.2f}")

    print("\n--- Top 15 Predicates ---")
    for item in stats['top_predicates'][:15]:
        cat = item['category'] or 'unclassified'
        print(f"  {item['lemma']:15} freq={item['freq']:4} category={cat}")

    return analyzer


def demo_adaptive_actors(hg, analyzer):
    """Demonstrate adaptive actor extraction."""
    from graphbrain.processors.adaptive_predicates import AdaptiveActors

    print("\n" + "="*70)
    print("ADAPTIVE ACTOR EXTRACTION")
    print("="*70)

    # Run adaptive actor extraction
    actors_processor = AdaptiveActors(hg, analyzer)
    actors_processor.run()

    print(f"\nUsing {len(actors_processor.action_predicates)} action predicates")
    print("Sample predicates:", list(actors_processor.action_predicates)[:10])

    print("\n--- Discovered Actors ---")
    actor_count = 0
    for actor, count, top_preds in actors_processor.actors(min_count=2):
        print(f"\n  Actor: {actor}")
        print(f"    Occurrences: {count}")
        print(f"    Actions: {', '.join([f'{p}({c})' for p, c in top_preds])}")
        actor_count += 1
        if actor_count >= 15:
            break


def demo_adaptive_conflicts(hg, analyzer):
    """Demonstrate adaptive conflict extraction."""
    from graphbrain.processors.adaptive_predicates import AdaptiveConflicts

    print("\n" + "="*70)
    print("ADAPTIVE CONFLICT EXTRACTION")
    print("="*70)

    # Run adaptive conflict extraction
    conflicts_processor = AdaptiveConflicts(hg, analyzer)
    conflicts_processor.run()

    print(f"\nUsing {len(conflicts_processor.conflict_predicates)} conflict predicates")
    print("Sample predicates:", list(conflicts_processor.conflict_predicates)[:10])

    print("\n--- Discovered Conflicts ---")
    conflict_count = 0
    for conflict in conflicts_processor.conflicts():
        print(f"\n  {conflict['source']} --{conflict['predicate']}--> {conflict['target']}")
        conflict_count += 1
        if conflict_count >= 10:
            break

    if conflict_count == 0:
        print("  No conflicts detected in corpus")


def demo_adaptive_claims(hg, analyzer):
    """Demonstrate adaptive claim extraction."""
    from graphbrain.processors.adaptive_predicates import AdaptiveClaims

    print("\n" + "="*70)
    print("ADAPTIVE CLAIM EXTRACTION")
    print("="*70)

    claims_processor = AdaptiveClaims(hg, analyzer)
    claims_processor.run()

    print(f"\nUsing {len(claims_processor.claim_predicates)} claim predicates")
    print("Sample predicates:", list(claims_processor.claim_predicates)[:10])

    print("\n--- Discovered Claims ---")
    claim_count = 0
    for claim in claims_processor.claims():
        print(f"\n  Claimer: {claim['claimer']}")
        print(f"    Verb: {claim['predicate']}")
        if claim['content']:
            content_str = str(claim['content'])[:60]
            print(f"    Content: {content_str}...")
        claim_count += 1
        if claim_count >= 10:
            break


def demo_pattern_discovery(hg):
    """Demonstrate pattern discovery and explanation."""
    from graphbrain.patterns.llm_patterns import (
        PatternDiscovery,
        PatternExplainer,
        NaturalLanguagePatterns
    )

    print("\n" + "="*70)
    print("PATTERN DISCOVERY & EXPLANATION")
    print("="*70)

    discovery = PatternDiscovery(hg)
    patterns = discovery.discover_structural_patterns(max_depth=2, min_frequency=10)

    print(f"\nDiscovered {len(patterns)} structural patterns")

    print("\n--- Top Patterns ---")
    for cand in patterns[:10]:
        print(f"\n  Pattern: {cand.pattern}")
        print(f"    Frequency: {cand.frequency}")
        print(f"    Explanation: {cand.description}")
        print(f"    Human: {PatternExplainer.explain(cand.pattern)}")

    # Demo natural language patterns
    print("\n--- Natural Language Pattern Queries ---")
    nl = NaturalLanguagePatterns(hg)

    queries = [
        "Find all statements where someone says something",
        "Find relationships between people",
        "Find all actions with concepts",
        "Find modifiers describing things"
    ]

    for query in queries:
        pattern = nl.query_to_pattern(query)
        print(f"\n  Query: '{query}'")
        print(f"  Pattern: {pattern}")
        print(f"  Explanation: {PatternExplainer.explain(pattern)}")


def demo_semantic_predicate_search(hg, analyzer):
    """Demonstrate semantic predicate search."""
    print("\n" + "="*70)
    print("SEMANTIC PREDICATE SEARCH")
    print("="*70)

    print("\n--- Finding predicates similar to seed words ---")

    seed_sets = [
        (['love', 'adore', 'cherish'], "Affection predicates"),
        (['attack', 'fight', 'battle'], "Conflict predicates"),
        (['think', 'believe', 'consider'], "Cognition predicates"),
        (['walk', 'run', 'move'], "Movement predicates"),
    ]

    for seeds, description in seed_sets:
        similar = analyzer.get_predicates_like(seeds, threshold=0.65)
        print(f"\n{description} (similar to {seeds}):")
        for pred in similar[:8]:
            info = analyzer._predicates.get(pred)
            if info:
                print(f"  {pred:15} freq={info.frequency}")


def demo_actor_network(hg, analyzer):
    """Build an actor interaction network."""
    from graphbrain.processors.adaptive_predicates import AdaptiveActors
    from collections import defaultdict

    print("\n" + "="*70)
    print("ACTOR INTERACTION NETWORK")
    print("="*70)

    # Get all actors
    actors_processor = AdaptiveActors(hg, analyzer)
    actors_processor.run()

    # Build actor -> predicates -> targets map
    actor_actions = defaultdict(lambda: defaultdict(list))

    for edge in hg.all():
        if edge.atom or len(edge) < 3:
            continue

        connector = edge[0]
        if not hasattr(connector, 'type') or not connector.type().startswith('P'):
            continue

        from graphbrain.utils.lemmas import deep_lemma
        lemma_edge = deep_lemma(hg, connector, same_if_none=True)
        if not lemma_edge:
            continue
        lemma = lemma_edge.root()

        if lemma not in actors_processor.action_predicates:
            continue

        subj = edge[1]
        obj = edge[2] if len(edge) > 2 else None

        if hasattr(subj, 'type') and subj.type().startswith('C'):
            if obj and hasattr(obj, 'type') and obj.type().startswith('C'):
                actor_actions[str(subj)][lemma].append(str(obj))

    # Print network
    print("\n--- Actor Action Network ---")
    for actor, actions in sorted(actor_actions.items(), key=lambda x: -sum(len(v) for v in x[1].values()))[:10]:
        print(f"\n{actor}:")
        for predicate, targets in sorted(actions.items(), key=lambda x: -len(x[1]))[:5]:
            unique_targets = list(set(targets))[:3]
            print(f"  --{predicate}--> {', '.join(unique_targets)}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Demo adaptive extraction features')
    parser.add_argument('--db', default='knowledge.db', help='Database path')
    parser.add_argument('--demo', default='all',
                       choices=['all', 'predicates', 'actors', 'conflicts', 'claims', 'patterns', 'network'],
                       help='Which demo to run')

    args = parser.parse_args()

    # Open hypergraph
    hg = hgraph(args.db)
    edge_count = sum(1 for _ in hg.all())
    print(f"Opened: {args.db}")
    print(f"Total edges: {edge_count}")

    if edge_count < 100:
        print("\nWarning: Small dataset. Run build_test_hypergraph.py first to add more data.")

    # Run demos
    analyzer = None

    if args.demo in ['all', 'predicates']:
        analyzer = demo_predicate_discovery(hg)

    if args.demo in ['all', 'actors']:
        if analyzer is None:
            from graphbrain.processors.adaptive_predicates import PredicateAnalyzer
            analyzer = PredicateAnalyzer(hg)
            analyzer.discover_predicates()
        demo_adaptive_actors(hg, analyzer)

    if args.demo in ['all', 'conflicts']:
        if analyzer is None:
            from graphbrain.processors.adaptive_predicates import PredicateAnalyzer
            analyzer = PredicateAnalyzer(hg)
            analyzer.discover_predicates()
        demo_adaptive_conflicts(hg, analyzer)

    if args.demo in ['all', 'claims']:
        if analyzer is None:
            from graphbrain.processors.adaptive_predicates import PredicateAnalyzer
            analyzer = PredicateAnalyzer(hg)
            analyzer.discover_predicates()
        demo_adaptive_claims(hg, analyzer)

    if args.demo in ['all', 'patterns']:
        demo_pattern_discovery(hg)

    if args.demo in ['all', 'network']:
        if analyzer is None:
            from graphbrain.processors.adaptive_predicates import PredicateAnalyzer
            analyzer = PredicateAnalyzer(hg)
            analyzer.discover_predicates()
        demo_actor_network(hg, analyzer)

    hg.close()
    print("\n" + "="*70)
    print("Demo complete!")


if __name__ == "__main__":
    main()
