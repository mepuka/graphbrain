#!/usr/bin/env python3
"""Build a test hypergraph with real data to exercise graphbrain functionality.

This script:
1. Parses sample texts, books, and web URLs into hyperedges
2. Stores them in a SQLite hypergraph
3. Demonstrates querying and pattern matching
4. Tests semantic similarity features

Web URL support uses Jina Reader API for clean content extraction.
"""

import logging
import os
import sys
import urllib.request
from pathlib import Path

# Add parent to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphbrain import hgraph, hedge
from graphbrain.parsers import create_parser

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Sample texts for quick testing
SAMPLE_TEXTS = [
    "Apple released a new iPhone with improved artificial intelligence features.",
    "Google announced that their quantum computer solved a problem in minutes.",
    "Scientists discovered a new species of deep-sea fish in the Pacific Ocean.",
    "Researchers at MIT created a robot that can learn from human demonstrations.",
    "NASA launched a spacecraft to study asteroids near Earth.",
    "Paris is the capital of France and home to the Eiffel Tower.",
    "Shakespeare wrote many famous plays including Hamlet and Romeo and Juliet.",
    "The quick brown fox jumps over the lazy dog.",
    "Climate change is affecting weather patterns around the world.",
    "Artificial intelligence is transforming how we work and live.",
]

# Project Gutenberg books (public domain)
GUTENBERG_BOOKS = {
    # === FICTION ===
    'alice': {
        'url': 'https://www.gutenberg.org/cache/epub/11/pg11.txt',
        'title': "Alice's Adventures in Wonderland",
        'author': 'Lewis Carroll',
        'genre': 'fiction',
    },
    'pride': {
        'url': 'https://www.gutenberg.org/cache/epub/1342/pg1342.txt',
        'title': 'Pride and Prejudice',
        'author': 'Jane Austen',
        'genre': 'fiction',
    },
    'sherlock': {
        'url': 'https://www.gutenberg.org/cache/epub/1661/pg1661.txt',
        'title': 'The Adventures of Sherlock Holmes',
        'author': 'Arthur Conan Doyle',
        'genre': 'fiction',
    },
    'frankenstein': {
        'url': 'https://www.gutenberg.org/cache/epub/84/pg84.txt',
        'title': 'Frankenstein',
        'author': 'Mary Shelley',
        'genre': 'fiction',
    },
    'moby': {
        'url': 'https://www.gutenberg.org/cache/epub/2701/pg2701.txt',
        'title': 'Moby Dick',
        'author': 'Herman Melville',
        'genre': 'fiction',
    },
    'dracula': {
        'url': 'https://www.gutenberg.org/cache/epub/345/pg345.txt',
        'title': 'Dracula',
        'author': 'Bram Stoker',
        'genre': 'fiction',
    },
    'jekyll': {
        'url': 'https://www.gutenberg.org/cache/epub/43/pg43.txt',
        'title': 'Strange Case of Dr Jekyll and Mr Hyde',
        'author': 'Robert Louis Stevenson',
        'genre': 'fiction',
    },
    'gatsby': {
        'url': 'https://www.gutenberg.org/cache/epub/64317/pg64317.txt',
        'title': 'The Great Gatsby',
        'author': 'F. Scott Fitzgerald',
        'genre': 'fiction',
    },
    # === NON-FICTION ===
    'darwin': {
        'url': 'https://www.gutenberg.org/cache/epub/1228/pg1228.txt',
        'title': 'On the Origin of Species',
        'author': 'Charles Darwin',
        'genre': 'non-fiction',
    },
    'republic': {
        'url': 'https://www.gutenberg.org/cache/epub/1497/pg1497.txt',
        'title': 'The Republic',
        'author': 'Plato',
        'genre': 'non-fiction',
    },
    'prince': {
        'url': 'https://www.gutenberg.org/cache/epub/1232/pg1232.txt',
        'title': 'The Prince',
        'author': 'Niccolò Machiavelli',
        'genre': 'non-fiction',
    },
    'meditations': {
        'url': 'https://www.gutenberg.org/cache/epub/2680/pg2680.txt',
        'title': 'Meditations',
        'author': 'Marcus Aurelius',
        'genre': 'non-fiction',
    },
    'art_of_war': {
        'url': 'https://www.gutenberg.org/cache/epub/132/pg132.txt',
        'title': 'The Art of War',
        'author': 'Sun Tzu',
        'genre': 'non-fiction',
    },
    'wealth': {
        'url': 'https://www.gutenberg.org/cache/epub/3300/pg3300.txt',
        'title': 'An Inquiry into the Nature and Causes of the Wealth of Nations',
        'author': 'Adam Smith',
        'genre': 'non-fiction',
    },
}


def download_book(book_key: str, cache_dir: str = "data/books") -> str:
    """Download a book from Project Gutenberg."""
    if book_key not in GUTENBERG_BOOKS:
        raise ValueError(f"Unknown book: {book_key}. Available: {list(GUTENBERG_BOOKS.keys())}")

    book = GUTENBERG_BOOKS[book_key]
    cache_path = Path(cache_dir) / f"{book_key}.txt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        logger.info(f"Using cached: {book['title']}")
        return cache_path.read_text(encoding='utf-8')

    logger.info(f"Downloading: {book['title']} by {book['author']}...")
    try:
        with urllib.request.urlopen(book['url'], timeout=30) as response:
            text = response.read().decode('utf-8')
            cache_path.write_text(text, encoding='utf-8')
            logger.info(f"Downloaded and cached: {len(text)} characters")
            return text
    except Exception as e:
        logger.error(f"Failed to download: {e}")
        return ""


def extract_sentences_from_book(text: str, max_sentences: int = None) -> list[str]:
    """Extract clean sentences from a Project Gutenberg book.

    Args:
        text: The full book text
        max_sentences: Maximum sentences to extract. None = all sentences.
    """
    import re

    # Find start of actual content (after Gutenberg header)
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "End of Project Gutenberg",
    ]

    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            start_idx = text.find('\n', idx) + 1
            break

    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    content = text[start_idx:end_idx]

    # Split into sentences using more robust splitting
    # Handle abbreviations better
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', content)

    # Clean and filter sentences
    clean_sentences = []
    for sent in sentences:
        # Clean whitespace
        sent = ' '.join(sent.split())

        # Skip short or invalid sentences
        if len(sent) < 20 or len(sent) > 500:
            continue

        # Skip sentences that look like chapter headers, etc
        if sent.isupper() or sent.startswith('CHAPTER'):
            continue

        # Must have at least 3 words
        if len(sent.split()) < 3:
            continue

        # Must have at least one letter
        if not any(c.isalpha() for c in sent):
            continue

        clean_sentences.append(sent)

        if max_sentences and len(clean_sentences) >= max_sentences:
            break

    return clean_sentences


def fetch_web_url(url: str, max_sentences: int = None, api_key: str = None) -> tuple[str, list[str]]:
    """Fetch content from a web URL using Jina Reader.

    Args:
        url: The URL to fetch
        max_sentences: Maximum sentences to extract (None = all)
        api_key: Optional Jina API key (or set JINA_API_KEY env var)

    Returns:
        Tuple of (title, list of sentences)
    """
    try:
        from graphbrain.readers.jina import fetch_jina_content
    except ImportError:
        logger.error("Jina reader not available. Install requests if missing.")
        return "", []

    logger.info(f"Fetching URL via Jina Reader: {url}")

    try:
        content = fetch_jina_content(url, api_key=api_key)
        logger.info(f"Fetched: {content.title}")

        sentences = list(content.sentences())
        if max_sentences and max_sentences > 0:
            sentences = sentences[:max_sentences]

        logger.info(f"Extracted {len(sentences)} sentences")
        return content.title, sentences

    except Exception as e:
        logger.error(f"Failed to fetch URL: {e}")
        return "", []


def search_web(query: str, max_results: int = 3, max_sentences: int = 50, api_key: str = None) -> list[tuple[str, str, list[str]]]:
    """Search the web and extract content from top results.

    Args:
        query: Search query
        max_results: Maximum search results to process
        max_sentences: Max sentences per result
        api_key: Optional Jina API key (or set JINA_API_KEY env var)

    Returns:
        List of (url, title, sentences) tuples
    """
    try:
        from graphbrain.readers.jina import search_jina
    except ImportError:
        logger.error("Jina reader not available.")
        return []

    logger.info(f"Searching web for: {query}")

    try:
        results = search_jina(query, api_key=api_key)
        results = results[:max_results]

        extracted = []
        for content in results:
            sentences = list(content.sentences())
            if max_sentences and max_sentences > 0:
                sentences = sentences[:max_sentences]
            extracted.append((content.url, content.title, sentences))
            logger.info(f"  Found: {content.title[:50]}... ({len(sentences)} sentences)")

        return extracted

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []


def parse_text(parser, text: str) -> tuple:
    """Parse text and return (main_edge, metadata) or (None, None)."""
    try:
        result = parser.parse(text)

        if result and 'parses' in result and result['parses']:
            parse = result['parses'][0]
            main_edge = parse.get('main_edge')
            if main_edge:
                return main_edge, {
                    'text': text,
                    'atom2word': parse.get('atom2word', {}),
                }
    except Exception as e:
        logger.debug(f"Parse error: {e}")

    return None, None


def build_hypergraph(
    db_path: str = "knowledge.db",
    sources: list[str] = None,
    max_per_source: int = 50,
    jina_api_key: str = None
):
    """Build a hypergraph from various sources.

    Args:
        db_path: Path to SQLite database
        sources: List of sources - 'samples', book keys, URLs, or 'search:query'
        max_per_source: Max sentences to parse per source
        jina_api_key: Optional Jina API key for web fetching (or set JINA_API_KEY env var)
    """
    if sources is None:
        sources = ['samples']

    # Create hypergraph
    logger.info(f"Creating hypergraph at: {db_path}")
    hg = hgraph(db_path)

    # Create parser
    logger.info("Loading English parser...")
    parser = create_parser(lang='en', corefs=False)
    logger.info("Parser loaded")

    total_edges = 0
    total_failed = 0

    for source in sources:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing source: {source}")
        logger.info('='*60)

        if source == 'samples':
            sentences = SAMPLE_TEXTS[:max_per_source]
            source_name = 'sample_texts'
        elif source in GUTENBERG_BOOKS:
            book_text = download_book(source)
            if not book_text:
                continue
            # max_per_source=0 means unlimited
            limit = max_per_source if max_per_source > 0 else None
            sentences = extract_sentences_from_book(book_text, limit)
            source_name = f"gutenberg_{source}"
            logger.info(f"Extracted {len(sentences)} sentences (limit: {'unlimited' if limit is None else limit})")
        elif source.startswith('http://') or source.startswith('https://'):
            # Web URL - use Jina Reader
            limit = max_per_source if max_per_source > 0 else None
            title, sentences = fetch_web_url(source, limit, api_key=jina_api_key)
            if not sentences:
                continue
            source_name = f"web:{title[:30] if title else source[:30]}"
            logger.info(f"Extracted {len(sentences)} sentences from web")
        elif source.startswith('search:'):
            # Web search - use Jina Search
            query = source[7:]  # Remove 'search:' prefix
            limit = max_per_source if max_per_source > 0 else 50
            search_results = search_web(query, max_results=3, max_sentences=limit, api_key=jina_api_key)
            if not search_results:
                continue
            # Flatten sentences from all results
            sentences = []
            for url, title, sents in search_results:
                sentences.extend(sents)
            source_name = f"search:{query[:30]}"
            logger.info(f"Extracted {len(sentences)} sentences from {len(search_results)} search results")
        else:
            logger.warning(f"Unknown source: {source}")
            continue

        # Parse sentences
        for i, text in enumerate(sentences, 1):
            if i % 10 == 0:
                logger.info(f"  Progress: {i}/{len(sentences)}")

            main_edge, metadata = parse_text(parser, text)

            if main_edge:
                # Add edge
                hg.add(main_edge)

                # Store attributes
                hg.set_attribute(main_edge, 'text', text)
                hg.set_attribute(main_edge, 'source', source_name)

                total_edges += 1
                logger.debug(f"  Added: {str(main_edge)[:60]}...")
            else:
                total_failed += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY")
    logger.info(f"  Edges added: {total_edges}")
    logger.info(f"  Parse failures: {total_failed}")
    logger.info(f"  Success rate: {total_edges/(total_edges+total_failed)*100:.1f}%")

    return hg


def demonstrate_queries(hg):
    """Demonstrate querying the hypergraph."""
    logger.info(f"\n{'='*60}")
    logger.info("QUERYING THE HYPERGRAPH")
    logger.info('='*60)

    # Count edges
    all_edges = list(hg.all())
    logger.info(f"\nTotal edges: {len(all_edges)}")

    # Sample edges
    logger.info("\n--- Sample Edges ---")
    for edge in all_edges[:5]:
        attrs = hg.get_attributes(edge)
        text = attrs.get('text', '') if attrs else ''
        logger.info(f"\nEdge: {edge}")
        if text:
            logger.info(f"Text: {text[:80]}...")

    # Pattern matching
    logger.info("\n--- Pattern Matching ---")

    patterns = [
        ("All predicate edges", "(*/Pd.* * *)"),
        ("Edges with concepts", "(* */C *)"),
        ("Modifier patterns", "(*/M *)"),
    ]

    for desc, pattern_str in patterns:
        logger.info(f"\n{desc}: {pattern_str}")
        try:
            pattern = hedge(pattern_str)
            matches = list(hg.search(pattern))
            logger.info(f"  Found: {len(matches)} matches")
            for match in matches[:2]:
                logger.info(f"    - {str(match)[:70]}...")
        except Exception as e:
            logger.info(f"  Error: {e}")


def demonstrate_edge_analysis(hg):
    """Analyze edges in the hypergraph."""
    logger.info(f"\n{'='*60}")
    logger.info("EDGE ANALYSIS")
    logger.info('='*60)

    edges = list(hg.all())
    if not edges:
        logger.info("No edges to analyze")
        return

    # Analyze edge types
    type_counts = {}
    for edge in edges:
        if not edge.atom and len(edge) > 0:
            connector = edge[0]
            if hasattr(connector, 'type'):
                edge_type = connector.type()
                type_counts[edge_type] = type_counts.get(edge_type, 0) + 1

    logger.info("\n--- Edge Type Distribution ---")
    for edge_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {edge_type}: {count}")

    # Analyze atoms
    logger.info("\n--- Most Common Atoms ---")
    atom_counts = {}
    for edge in edges:
        for atom in edge.atoms():
            root = atom.root()
            atom_counts[root] = atom_counts.get(root, 0) + 1

    for atom, count in sorted(atom_counts.items(), key=lambda x: -x[1])[:15]:
        logger.info(f"  {atom}: {count}")


def demonstrate_semsim(hg):
    """Demonstrate semantic similarity."""
    logger.info(f"\n{'='*60}")
    logger.info("SEMANTIC SIMILARITY")
    logger.info('='*60)

    try:
        from graphbrain.semsim.interface import (
            get_matcher,
            SemSimType,
            SENTENCE_TRANSFORMER_AVAILABLE,
        )

        if not SENTENCE_TRANSFORMER_AVAILABLE:
            logger.info("sentence-transformers not available, skipping")
            return

        logger.info("Initializing FIX matcher...")
        matcher = get_matcher(SemSimType.FIX)
        logger.info(f"Using: {type(matcher).__name__}")

        # Test similarities
        test_cases = [
            ('alice', ['rabbit', 'queen', 'cat', 'door']),
            ('sherlock', ['detective', 'watson', 'crime', 'mystery']),
            ('discover', ['find', 'explore', 'learn', 'create']),
            ('intelligent', ['smart', 'clever', 'artificial', 'dumb']),
        ]

        logger.info("\n--- Word Similarity Tests ---")
        for word, refs in test_cases:
            # Get raw similarities
            sims = matcher._similarities(cand_word=word, ref_words=refs)
            if sims:
                logger.info(f"\n'{word}' similarities:")
                for ref, score in sorted(sims.items(), key=lambda x: -x[1]):
                    bar = '#' * int(score * 20)
                    logger.info(f"  {ref:12} {score:.3f} {bar}")

        # Cache info
        cache_info = matcher.cache_info()
        logger.info(f"\nCache: {cache_info['hits']} hits, {cache_info['misses']} misses")

    except Exception as e:
        logger.error(f"Semsim error: {e}")
        import traceback
        traceback.print_exc()


def interactive_query(hg):
    """Interactive query mode."""
    logger.info(f"\n{'='*60}")
    logger.info("INTERACTIVE MODE")
    logger.info("Enter patterns to search (or 'quit' to exit)")
    logger.info("Example patterns: (*/Pd.* * *), (* alice/C *)")
    logger.info('='*60)

    while True:
        try:
            pattern_str = input("\nPattern> ").strip()
            if pattern_str.lower() in ('quit', 'exit', 'q'):
                break
            if not pattern_str:
                continue

            pattern = hedge(pattern_str)
            matches = list(hg.search(pattern))
            logger.info(f"Found {len(matches)} matches:")
            for match in matches[:10]:
                attrs = hg.get_attributes(match)
                text = attrs.get('text', '') if attrs else ''
                logger.info(f"  {match}")
                if text:
                    logger.info(f"    -> {text[:60]}...")

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {e}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Build and explore a graphbrain hypergraph',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Source types:
  samples                       Built-in sample sentences
  alice, sherlock, pride, ...   Project Gutenberg books
  https://example.com/article   Web URL (uses Jina Reader)
  search:query terms            Web search (uses Jina Search)

Examples:
  # Parse a book
  python scripts/build_test_hypergraph.py --sources pride --max 100

  # Parse a web article
  python scripts/build_test_hypergraph.py --sources https://en.wikipedia.org/wiki/Knowledge_graph

  # Search and parse web results
  python scripts/build_test_hypergraph.py --sources "search:artificial intelligence history"

  # Combine multiple sources
  python scripts/build_test_hypergraph.py --sources samples https://example.com search:topic
        """
    )
    parser.add_argument('--db', default='knowledge.db', help='Database path')
    parser.add_argument('--sources', nargs='+', default=['samples'],
                       help='Sources to parse (see examples below)')
    parser.add_argument('--max', type=int, default=0, help='Max sentences per source (0=unlimited)')
    parser.add_argument('--fresh', action='store_true', help='Remove existing database')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive query mode')
    parser.add_argument('--skip-build', action='store_true', help='Skip building, just query')
    parser.add_argument('--jina-key', default=None, help='Jina API key for higher rate limits')

    args = parser.parse_args()

    # Fresh start if requested
    if args.fresh and os.path.exists(args.db):
        os.remove(args.db)
        logger.info(f"Removed existing database: {args.db}")

    # Build or open hypergraph
    if args.skip_build:
        if not os.path.exists(args.db):
            logger.error(f"Database not found: {args.db}")
            return
        hg = hgraph(args.db)
        logger.info(f"Opened existing database: {args.db}")
    else:
        hg = build_hypergraph(args.db, args.sources, args.max, jina_api_key=args.jina_key)

    # Demonstrate features
    demonstrate_queries(hg)
    demonstrate_edge_analysis(hg)
    demonstrate_semsim(hg)

    # Interactive mode
    if args.interactive:
        interactive_query(hg)

    # Close
    hg.close()
    logger.info(f"\nDatabase: {args.db}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
