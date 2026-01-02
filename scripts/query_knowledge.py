#!/usr/bin/env python3
"""Query and explore a graphbrain hypergraph with semantic features.

This script demonstrates graphbrain's powerful features:
1. Pattern matching with wildcards
2. Semantic similarity queries
3. Edge analysis and statistics
4. Natural language to hyperedge exploration
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from graphbrain import hgraph, hedge


def show_edge_details(hg, edge, indent=0):
    """Show detailed information about an edge."""
    prefix = "  " * indent
    print(f"{prefix}Edge: {edge}")

    # Get attributes
    attrs = hg.get_attributes(edge)
    if attrs:
        text = attrs.get('text', '')
        source = attrs.get('source', '')
        if text:
            print(f"{prefix}  Text: {text[:80]}...")
        if source:
            print(f"{prefix}  Source: {source}")

    # Show structure for non-atoms
    if not edge.atom:
        print(f"{prefix}  Type: {edge[0].type() if hasattr(edge[0], 'type') else 'N/A'}")
        print(f"{prefix}  Connector: {edge[0]}")
        print(f"{prefix}  Args: {len(edge) - 1}")


def find_edges_about(hg, concept: str, limit=10):
    """Find edges that mention a concept."""
    print(f"\n=== Edges mentioning '{concept}' ===")

    count = 0
    for edge in hg.all():
        # Check if concept appears in any atom
        for atom in edge.atoms():
            if concept.lower() in atom.root().lower():
                show_edge_details(hg, edge)
                count += 1
                if count >= limit:
                    return
                break

    if count == 0:
        print(f"  No edges found mentioning '{concept}'")


def find_actions(hg, limit=10):
    """Find edges that represent actions (predicates)."""
    print(f"\n=== Actions (Predicate Edges) ===")

    count = 0
    for edge in hg.all():
        if not edge.atom and len(edge) > 0:
            connector = edge[0]
            if hasattr(connector, 'type') and connector.type().startswith('Pd'):
                attrs = hg.get_attributes(edge)
                text = attrs.get('text', '') if attrs else ''

                print(f"\n  Action: {connector.root()}")
                print(f"    Edge: {str(edge)[:70]}...")
                if text:
                    print(f"    Text: {text[:60]}...")

                count += 1
                if count >= limit:
                    break


def find_similar_words(matcher, word: str, candidates: list):
    """Find which candidates are similar to a word."""
    print(f"\n=== Similar to '{word}' ===")

    sims = matcher._similarities(cand_word=word, ref_words=candidates)
    if sims:
        for ref, score in sorted(sims.items(), key=lambda x: -x[1]):
            bar = '#' * int(score * 20)
            similar = "YES" if score > 0.85 else "maybe" if score > 0.7 else "no"
            print(f"  {ref:15} {score:.3f} {bar} ({similar})")


def semantic_edge_search(hg, matcher, target_word: str, threshold=0.8):
    """Find edges with atoms semantically similar to target word."""
    print(f"\n=== Edges semantically related to '{target_word}' (threshold={threshold}) ===")

    found = 0
    for edge in hg.all():
        attrs = hg.get_attributes(edge)
        if not attrs or 'text' not in attrs:
            continue

        # Check atoms for semantic similarity
        for atom in edge.atoms():
            root = atom.root()
            if len(root) < 3:  # Skip short words
                continue

            sims = matcher._similarities(cand_word=target_word, ref_words=[root])
            if sims and sims.get(root, 0) > threshold:
                print(f"\n  Match: '{root}' (sim={sims[root]:.3f})")
                print(f"    Text: {attrs['text'][:70]}...")
                found += 1
                if found >= 5:
                    return
                break


def analyze_sources(hg):
    """Analyze edges by source."""
    print("\n=== Source Analysis ===")

    sources = {}
    for edge in hg.all():
        attrs = hg.get_attributes(edge)
        if attrs and 'source' in attrs:
            source = attrs['source']
            sources[source] = sources.get(source, 0) + 1

    for source, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count} edges")


def find_character_mentions(hg, characters: list):
    """Find how often characters are mentioned."""
    print("\n=== Character Analysis ===")

    char_counts = {c: 0 for c in characters}

    for edge in hg.all():
        for atom in edge.atoms():
            root = atom.root().lower()
            for char in characters:
                if char.lower() in root:
                    char_counts[char] += 1

    for char, count in sorted(char_counts.items(), key=lambda x: -x[1]):
        bar = '#' * min(count, 30)
        print(f"  {char:15} {count:4} {bar}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Query a graphbrain knowledge base')
    parser.add_argument('--db', default='knowledge.db', help='Database path')
    parser.add_argument('--concept', type=str, help='Find edges about a concept')
    parser.add_argument('--similar', type=str, help='Find words similar to this')
    parser.add_argument('--semsearch', type=str, help='Semantic edge search')

    args = parser.parse_args()

    # Open hypergraph
    hg = hgraph(args.db)

    all_edges = list(hg.all())
    print(f"Opened: {args.db}")
    print(f"Total edges: {len(all_edges)}")

    # Initialize semantic similarity
    try:
        from graphbrain.semsim.interface import get_matcher, SemSimType
        matcher = get_matcher(SemSimType.FIX)
        has_semsim = True
    except:
        has_semsim = False
        matcher = None

    if args.concept:
        find_edges_about(hg, args.concept)
    elif args.similar and has_semsim:
        # Find similar words in the corpus
        words = set()
        for edge in all_edges[:500]:  # Sample
            for atom in edge.atoms():
                root = atom.root()
                if len(root) >= 3 and root.isalpha():
                    words.add(root.lower())

        find_similar_words(matcher, args.similar, list(words)[:50])
    elif args.semsearch and has_semsim:
        semantic_edge_search(hg, matcher, args.semsearch)
    else:
        # Default: show overview
        print("\n" + "="*60)
        print("KNOWLEDGE BASE OVERVIEW")
        print("="*60)

        analyze_sources(hg)

        # Character analysis for fiction
        find_character_mentions(hg, [
            'alice', 'rabbit', 'queen', 'cat', 'hatter',  # Alice
            'holmes', 'watson', 'sherlock', 'irene',       # Sherlock
        ])

        find_actions(hg, limit=5)

        if has_semsim:
            print("\n" + "="*60)
            print("SEMANTIC SIMILARITY DEMOS")
            print("="*60)

            # Semantic demos
            find_similar_words(matcher, 'detective', ['investigator', 'policeman', 'rabbit', 'queen'])
            find_similar_words(matcher, 'wonderland', ['fantasy', 'dream', 'mystery', 'crime'])
            find_similar_words(matcher, 'mystery', ['puzzle', 'crime', 'wonder', 'adventure'])

            # Semantic edge search
            semantic_edge_search(hg, matcher, 'investigate', threshold=0.75)

    hg.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
