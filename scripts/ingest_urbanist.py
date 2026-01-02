#!/usr/bin/env python3
"""Ingest pre-scraped Urbanist articles into Graphbrain hypergraph.

Uses the scraped JSON data (no network requests needed) and adds
comprehensive metadata attributes to each edge.

Usage:
    # Full ingestion from scraped data
    python scripts/ingest_urbanist.py --fresh

    # Limit articles for testing
    python scripts/ingest_urbanist.py --max-articles 50 --fresh

    # Interactive query mode
    python scripts/ingest_urbanist.py --skip-build -i
"""

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Add parent to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def extract_sentences(content: str, min_length: int = 20) -> list[str]:
    """Extract sentences from markdown content.

    Args:
        content: Markdown content from article
        min_length: Minimum sentence length to include

    Returns:
        List of sentences
    """
    # Clean up markdown artifacts
    text = content

    # Remove images first (before links)
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)

    # Remove markdown links but keep text
    text = re.sub(r'\[([^\]]*)\]\([^)]+\)', r'\1', text)

    # Remove any remaining bare URLs
    text = re.sub(r'https?://[^\s\)]+', '', text)

    # Remove headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Remove bold/italic
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)

    # Remove inline code
    text = re.sub(r'`[^`]+`', '', text)

    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)

    # Remove empty brackets/parens
    text = re.sub(r'\[\s*\]', '', text)
    text = re.sub(r'\(\s*\)', '', text)

    # Remove horizontal rules
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)

    # Split into paragraphs
    paragraphs = text.split('\n\n')

    sentences = []
    for para in paragraphs:
        para = para.strip()
        if not para or len(para) < min_length:
            continue

        # Skip if it looks like metadata or navigation
        if para.startswith('Posted') or para.startswith('Share') or para.startswith('Filed'):
            continue
        if 'Subscribe' in para and len(para) < 100:
            continue

        # Handle common abbreviations before splitting
        para = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Jr|Sr|Inc|Ltd|Corp|vs|etc|St|Ave|Blvd)\.',
                     r'\1<DOT>', para)

        # Split on sentence boundaries
        sent_list = re.split(r'(?<=[.!?])\s+', para)

        for sent in sent_list:
            sent = sent.replace('<DOT>', '.').strip()
            if len(sent) >= min_length and not sent.startswith('http'):
                sentences.append(sent)

    return sentences


def parse_published_time(time_str: str) -> dict:
    """Parse ISO timestamp into components.

    Args:
        time_str: ISO format timestamp like '2025-12-25T13:20:18-08:00'

    Returns:
        Dict with year, month, day, date components
    """
    if not time_str:
        return {}

    try:
        # Handle timezone in format
        dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        return {
            'year': str(dt.year),
            'month': f'{dt.month:02d}',
            'day': f'{dt.day:02d}',
            'date': dt.strftime('%Y-%m-%d'),
            'datetime': time_str,
        }
    except Exception:
        # Fallback to regex extraction
        match = re.match(r'(\d{4})-(\d{2})-(\d{2})', time_str)
        if match:
            year, month, day = match.groups()
            return {
                'year': year,
                'month': month,
                'day': day,
                'date': f'{year}-{month}-{day}',
            }
    return {}


def extract_slug_from_url(url: str) -> str:
    """Extract article slug from URL."""
    match = re.search(r'/\d{4}/\d{2}/\d{2}/([^/]+)/?$', url)
    return match.group(1) if match else ''


def categorize_article(title: str, content: str) -> list[str]:
    """Infer article categories from title and content.

    Returns list of category tags.
    """
    text = (title + ' ' + content[:2000]).lower()
    categories = []

    category_keywords = {
        'housing': ['housing', 'apartment', 'rent', 'zoning', 'density', 'affordable', 'homeless', 'shelter'],
        'transit': ['transit', 'light rail', 'bus', 'metro', 'sound transit', 'orca', 'streetcar', 'train'],
        'transportation': ['bike', 'bicycle', 'pedestrian', 'walk', 'traffic', 'parking', 'road', 'highway', 'sdot'],
        'politics': ['council', 'mayor', 'election', 'vote', 'candidate', 'legislature', 'governor', 'senate'],
        'development': ['development', 'construction', 'building', 'tower', 'project', 'permit'],
        'safety': ['safety', 'police', 'crime', 'violence', 'crash', 'accident', 'death'],
        'environment': ['climate', 'environment', 'tree', 'park', 'green', 'sustainability'],
    }

    for category, keywords in category_keywords.items():
        if any(kw in text for kw in keywords):
            categories.append(category)

    return categories if categories else ['general']


def process_article(article: dict, parser) -> list[dict]:
    """Process a single article and return parsed edges with metadata.

    Args:
        article: Article dict from scraped data
        parser: Graphbrain parser instance

    Returns:
        List of dicts with edge and metadata
    """
    if article.get('error'):
        return []

    content = article.get('content', '')
    if not content:
        return []

    # Extract sentences
    sentences = extract_sentences(content)
    if not sentences:
        return []

    # Parse time info
    time_info = parse_published_time(article.get('published_time', ''))

    # Extract categories
    categories = categorize_article(
        article.get('title', ''),
        content
    )

    # Build base metadata
    base_meta = {
        'source': article['url'],
        'title': article.get('title', ''),
        'slug': extract_slug_from_url(article['url']),
        'word_count': str(article.get('word_count', 0)),
        'categories': ','.join(categories),
    }

    # Add time metadata
    base_meta.update(time_info)

    # Add author if available
    if article.get('author'):
        base_meta['author'] = article['author']

    # Parse each sentence
    results = []
    for sent in sentences:
        try:
            result = parser.parse(sent)

            if result and 'parses' in result and result['parses']:
                parse = result['parses'][0]
                main_edge = parse.get('main_edge')

                if main_edge:
                    # Create metadata for this edge
                    meta = base_meta.copy()
                    meta['text'] = sent

                    results.append({
                        'edge': main_edge,
                        'meta': meta
                    })

        except Exception as e:
            logger.debug(f"Parse error: {e}")

    return results


def ingest_articles(
    db_path: str,
    json_path: str,
    max_articles: int = 0,
    max_sentences: int = 0,
    corefs: bool = False,
    workers: int = 1,
):
    """Ingest articles from JSON into hypergraph.

    Args:
        db_path: Path to SQLite database
        json_path: Path to scraped articles JSON
        max_articles: Max articles to process (0=all)
        max_sentences: Max sentences per article (0=all)
        corefs: Enable coreference resolution
        workers: Number of parallel workers (not used currently, sequential is more stable)
    """
    from graphbrain import hgraph
    from graphbrain.parsers import create_parser

    # Load articles
    logger.info(f"Loading articles from {json_path}")
    with open(json_path) as f:
        data = json.load(f)

    articles = [a for a in data['articles'] if not a.get('error')]
    if max_articles > 0:
        articles = articles[:max_articles]

    logger.info(f"Processing {len(articles)} articles")

    # Initialize hypergraph
    logger.info(f"Creating hypergraph: {db_path}")
    hg = hgraph(db_path)

    # Initialize parser
    logger.info("Loading English parser...")
    if corefs:
        logger.info("Coreference resolution ENABLED")
    parser = create_parser(lang='en', corefs=corefs)
    logger.info("Parser loaded")

    # Stats tracking
    stats = {
        'articles_processed': 0,
        'sentences_parsed': 0,
        'edges_added': 0,
        'parse_failures': 0,
    }

    # Process articles
    for i, article in enumerate(articles, 1):
        url = article['url']
        title = article.get('title', '')[:60]

        logger.info(f"[{i}/{len(articles)}] {title}...")

        try:
            results = process_article(article, parser)

            # Limit sentences if requested
            if max_sentences > 0:
                results = results[:max_sentences]

            # Add to hypergraph with metadata
            for item in results:
                edge = item['edge']
                meta = item['meta']

                hg.add(edge)

                # Set all metadata attributes
                for key, value in meta.items():
                    if value:  # Only set non-empty values
                        hg.set_attribute(edge, key, str(value))

                stats['edges_added'] += 1

            stats['articles_processed'] += 1
            stats['sentences_parsed'] += len(results)

            logger.info(f"  Added {len(results)} edges")

        except Exception as e:
            logger.error(f"  Error: {e}")
            stats['parse_failures'] += 1

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Articles processed: {stats['articles_processed']}")
    logger.info(f"  Sentences parsed: {stats['sentences_parsed']}")
    logger.info(f"  Edges added: {stats['edges_added']}")
    logger.info(f"  Parse failures: {stats['parse_failures']}")

    return hg, stats


def analyze_hypergraph(hg):
    """Analyze the hypergraph contents."""
    logger.info("\n" + "=" * 60)
    logger.info("HYPERGRAPH ANALYSIS")
    logger.info("=" * 60)

    edges = list(hg.all())
    logger.info(f"Total edges: {len(edges)}")

    # Sample some edges with metadata
    logger.info("\n--- Sample Edges with Metadata ---")
    for edge in edges[:5]:
        attrs = hg.get_attributes(edge)
        if attrs:
            logger.info(f"\n  Edge: {edge}")
            logger.info(f"  Text: {attrs.get('text', '')[:80]}...")
            logger.info(f"  Date: {attrs.get('date', 'N/A')}")
            logger.info(f"  Categories: {attrs.get('categories', 'N/A')}")

    # Edge type distribution
    type_counts = {}
    for edge in edges:
        if not edge.atom and len(edge) > 0:
            connector = edge[0]
            if hasattr(connector, 'type'):
                edge_type = connector.type()
                type_counts[edge_type] = type_counts.get(edge_type, 0) + 1

    logger.info("\n--- Edge Types ---")
    for edge_type, count in sorted(type_counts.items(), key=lambda x: -x[1])[:10]:
        logger.info(f"  {edge_type}: {count}")

    # Articles by year
    year_counts = {}
    for edge in edges:
        attrs = hg.get_attributes(edge)
        if attrs and 'year' in attrs:
            year = attrs['year']
            year_counts[year] = year_counts.get(year, 0) + 1

    if year_counts:
        logger.info("\n--- Edges by Year ---")
        for year, count in sorted(year_counts.items(), reverse=True):
            logger.info(f"  {year}: {count}")

    # Category distribution
    cat_counts = {}
    for edge in edges:
        attrs = hg.get_attributes(edge)
        if attrs and 'categories' in attrs:
            for cat in attrs['categories'].split(','):
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

    if cat_counts:
        logger.info("\n--- Edges by Category ---")
        for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {cat}: {count}")


def interactive_query(hg):
    """Interactive query mode."""
    from graphbrain import hedge

    logger.info("\n" + "=" * 60)
    logger.info("INTERACTIVE MODE")
    logger.info("Enter patterns to search, or 'quit' to exit")
    logger.info("Examples:")
    logger.info("  (*/Pd.* * *)           - All predicate edges")
    logger.info("  (* seattle/C *)        - Edges mentioning Seattle")
    logger.info("  (* housing/C *)        - Edges about housing")
    logger.info("=" * 60)

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
                date = attrs.get('date', '') if attrs else ''

                logger.info(f"\n  {match}")
                if text:
                    logger.info(f"  Text: {text[:100]}...")
                if date:
                    logger.info(f"  Date: {date}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Ingest Urbanist articles into Graphbrain hypergraph',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--db', default='urbanist.db',
                       help='Database path (default: urbanist.db)')
    parser.add_argument('--json', default='data/urbanist/urbanist_all_articles.json',
                       help='Path to scraped articles JSON')
    parser.add_argument('--max-articles', '-n', type=int, default=0,
                       help='Maximum articles to process (0=all)')
    parser.add_argument('--max-sentences', '-s', type=int, default=0,
                       help='Max sentences per article (0=all)')
    parser.add_argument('--fresh', action='store_true',
                       help='Remove existing database first')
    parser.add_argument('--skip-build', action='store_true',
                       help='Skip building, just analyze/query')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive query mode after building')
    parser.add_argument('--corefs', action='store_true',
                       help='Enable coreference resolution (slower)')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of parallel workers (default: 1)')

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
        from graphbrain import hgraph
        hg = hgraph(args.db)
        logger.info(f"Opened existing database: {args.db}")
    else:
        if not os.path.exists(args.json):
            logger.error(f"JSON file not found: {args.json}")
            logger.info("Run scripts/scrape_urbanist.py first to scrape articles")
            return

        hg, stats = ingest_articles(
            args.db,
            args.json,
            max_articles=args.max_articles,
            max_sentences=args.max_sentences,
            corefs=args.corefs,
            workers=args.workers,
        )

    if hg:
        analyze_hypergraph(hg)

        if args.interactive:
            interactive_query(hg)

        hg.close()
        logger.info(f"\nDatabase saved: {args.db}")


if __name__ == "__main__":
    main()
