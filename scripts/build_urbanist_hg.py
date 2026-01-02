#!/usr/bin/env python3
"""Build a hypergraph from The Urbanist articles.

Crawls paginated category pages, extracts article URLs, fetches content
using Jina Reader, and parses into a hypergraph database.

Usage:
    # Basic usage - fetch 100 articles from politics category
    python scripts/build_urbanist_hg.py

    # Custom options
    python scripts/build_urbanist_hg.py --category politics-and-government --max-articles 50
    python scripts/build_urbanist_hg.py --category housing --max-sentences 100

    # Query existing database
    python scripts/build_urbanist_hg.py --skip-build --interactive

Requires JINA_API_KEY in .env or environment for higher rate limits.
"""

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path

# Add parent to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphbrain import hgraph, hedge
from graphbrain.parsers import create_parser

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# The Urbanist categories
URBANIST_CATEGORIES = {
    'politics': 'politics-and-government',
    'housing': 'housing',
    'transportation': 'transportation',
    'land-use': 'land-use',
    'environment': 'environment',
    'news': 'news',
    'opinion': 'opinion',
}

BASE_URL = "https://www.theurbanist.org"


def get_category_url(category: str, page: int = 1) -> str:
    """Build URL for a category page."""
    cat_slug = URBANIST_CATEGORIES.get(category, category)
    if page == 1:
        return f"{BASE_URL}/category/{cat_slug}/"
    return f"{BASE_URL}/category/{cat_slug}/page/{page}/"


def extract_article_urls(content: str) -> list[str]:
    """Extract article URLs from page content."""
    # Match Urbanist article URLs: /YYYY/MM/DD/slug/
    pattern = r'https://www\.theurbanist\.org/\d{4}/\d{2}/\d{2}/[^/\s\)\]\>"]+/'
    urls = re.findall(pattern, content)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


def crawl_category(
    category: str,
    max_articles: int = 100,
    delay: float = 1.0
) -> list[str]:
    """Crawl category pages to collect article URLs.

    Args:
        category: Category name or slug
        max_articles: Maximum number of articles to collect
        delay: Delay between page requests (rate limiting)

    Returns:
        List of article URLs
    """
    from graphbrain.readers.jina import fetch_jina_content

    all_urls = []
    page = 1
    max_pages = (max_articles // 8) + 5  # ~8 articles per page + buffer

    logger.info(f"Crawling category: {category}")

    while len(all_urls) < max_articles and page <= max_pages:
        url = get_category_url(category, page)
        logger.info(f"  Fetching page {page}: {url}")

        try:
            content = fetch_jina_content(url, response_format='markdown')
            urls = extract_article_urls(content.content)

            if not urls:
                logger.info(f"  No more articles found on page {page}")
                break

            new_urls = [u for u in urls if u not in all_urls]
            all_urls.extend(new_urls)
            logger.info(f"  Found {len(new_urls)} new articles (total: {len(all_urls)})")

            page += 1

            if len(all_urls) >= max_articles:
                break

            time.sleep(delay)

        except Exception as e:
            logger.error(f"  Error fetching page {page}: {e}")
            break

    return all_urls[:max_articles]


def extract_metadata_from_url(url: str) -> dict:
    """Extract publication date from URL pattern /YYYY/MM/DD/slug/"""
    match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/([^/]+)/', url)
    if match:
        year, month, day, slug = match.groups()
        return {
            'year': year,
            'month': month,
            'day': day,
            'date': f"{year}-{month}-{day}",
            'slug': slug
        }
    return {}


def extract_author_from_content(content: str) -> str:
    """Try to extract author from article content."""
    # Common patterns in news articles
    patterns = [
        r'(?:By|Author:?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        r'Posted by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, content[:2000])  # Check first 2000 chars
        if match:
            return match.group(1)
    return ""


def fetch_article(url: str) -> dict:
    """Fetch a single article using Jina Reader.

    Returns:
        Dict with title, content, sentences, url, and metadata
    """
    from graphbrain.readers.jina import fetch_jina_content

    try:
        content = fetch_jina_content(url)
        sentences = list(content.sentences())

        # Extract metadata
        url_meta = extract_metadata_from_url(url)
        author = extract_author_from_content(content.content)

        return {
            'url': url,
            'title': content.title,
            'content': content.content,
            'sentences': sentences,
            'date': url_meta.get('date', ''),
            'year': url_meta.get('year', ''),
            'month': url_meta.get('month', ''),
            'slug': url_meta.get('slug', ''),
            'author': author,
            'success': True
        }
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return {
            'url': url,
            'success': False,
            'error': str(e)
        }


def build_hypergraph(
    db_path: str,
    category: str = 'politics',
    max_articles: int = 100,
    max_sentences_per_article: int = 0,
    crawl_delay: float = 1.0,
    fetch_delay: float = 0.5,
    corefs: bool = False
):
    """Build hypergraph from Urbanist articles.

    Args:
        db_path: Path to SQLite database
        category: Category to crawl
        max_articles: Maximum articles to process
        max_sentences_per_article: Max sentences per article (0=unlimited)
        crawl_delay: Delay between category page fetches
        fetch_delay: Delay between article fetches
    """
    # Step 1: Crawl category for article URLs
    logger.info("=" * 60)
    logger.info("STEP 1: Crawling category pages for article URLs")
    logger.info("=" * 60)

    article_urls = crawl_category(category, max_articles, crawl_delay)
    logger.info(f"Collected {len(article_urls)} article URLs")

    if not article_urls:
        logger.error("No articles found!")
        return None

    # Step 2: Create hypergraph and parser
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Initializing hypergraph and parser")
    logger.info("=" * 60)

    hg = hgraph(db_path)
    logger.info(f"Created hypergraph: {db_path}")

    logger.info("Loading English parser...")
    if corefs:
        logger.info("Coreference resolution ENABLED (slower but better entity linking)")
    parser = create_parser(lang='en', corefs=corefs)
    logger.info("Parser loaded")

    # Step 3: Fetch and parse articles
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Fetching and parsing articles")
    logger.info("=" * 60)

    stats = {
        'articles_fetched': 0,
        'articles_failed': 0,
        'sentences_parsed': 0,
        'edges_added': 0,
        'parse_failures': 0
    }

    for i, url in enumerate(article_urls, 1):
        logger.info(f"\n[{i}/{len(article_urls)}] {url}")

        # Fetch article
        article = fetch_article(url)

        if not article['success']:
            stats['articles_failed'] += 1
            continue

        stats['articles_fetched'] += 1
        logger.info(f"  Title: {article['title'][:60]}...")
        if article.get('date'):
            logger.info(f"  Date: {article['date']}")
        if article.get('author'):
            logger.info(f"  Author: {article['author']}")

        sentences = article['sentences']
        if max_sentences_per_article > 0:
            sentences = sentences[:max_sentences_per_article]

        logger.info(f"  Sentences: {len(sentences)}")

        # Parse sentences
        article_edges = 0
        for sent in sentences:
            try:
                result = parser.parse(sent)

                if result and 'parses' in result and result['parses']:
                    parse = result['parses'][0]
                    main_edge = parse.get('main_edge')

                    if main_edge:
                        hg.add(main_edge)
                        hg.set_attribute(main_edge, 'text', sent)
                        hg.set_attribute(main_edge, 'source', url)
                        hg.set_attribute(main_edge, 'title', article['title'])

                        # Store publication metadata
                        if article.get('date'):
                            hg.set_attribute(main_edge, 'date', article['date'])
                        if article.get('year'):
                            hg.set_attribute(main_edge, 'year', article['year'])
                        if article.get('author'):
                            hg.set_attribute(main_edge, 'author', article['author'])

                        stats['edges_added'] += 1
                        article_edges += 1

            except Exception as e:
                stats['parse_failures'] += 1
                logger.debug(f"  Parse error: {e}")

        stats['sentences_parsed'] += len(sentences)
        logger.info(f"  Edges added: {article_edges}")

        # Rate limiting
        if i < len(article_urls):
            time.sleep(fetch_delay)

    # Step 4: Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Articles fetched: {stats['articles_fetched']}")
    logger.info(f"  Articles failed: {stats['articles_failed']}")
    logger.info(f"  Sentences parsed: {stats['sentences_parsed']}")
    logger.info(f"  Edges added: {stats['edges_added']}")
    logger.info(f"  Parse failures: {stats['parse_failures']}")

    if stats['sentences_parsed'] > 0:
        success_rate = stats['edges_added'] / stats['sentences_parsed'] * 100
        logger.info(f"  Success rate: {success_rate:.1f}%")

    return hg


def analyze_hypergraph(hg):
    """Analyze the hypergraph contents."""
    logger.info("\n" + "=" * 60)
    logger.info("HYPERGRAPH ANALYSIS")
    logger.info("=" * 60)

    edges = list(hg.all())
    logger.info(f"\nTotal edges: {len(edges)}")

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

    # Most common atoms
    atom_counts = {}
    for edge in edges:
        for atom in edge.atoms():
            root = atom.root()
            atom_counts[root] = atom_counts.get(root, 0) + 1

    logger.info("\n--- Most Common Terms ---")
    for atom, count in sorted(atom_counts.items(), key=lambda x: -x[1])[:20]:
        if len(atom) > 2 and atom not in ['+', 'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'for', 'on', 'is', 'are', 'was', 'were']:
            logger.info(f"  {atom}: {count}")


def interactive_query(hg):
    """Interactive query mode."""
    logger.info("\n" + "=" * 60)
    logger.info("INTERACTIVE MODE")
    logger.info("Enter patterns to search, or 'quit' to exit")
    logger.info("Examples: (*/Pd.* * *), (* seattle/C *), (* housing/C *)")
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
                logger.info(f"\n  {match}")
                if text:
                    logger.info(f"  -> {text[:80]}...")

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Build hypergraph from The Urbanist articles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Categories:
  politics      Politics and Government
  housing       Housing
  transportation Transportation
  land-use      Land Use
  environment   Environment
  news          News
  opinion       Opinion

Examples:
  python scripts/build_urbanist_hg.py --max-articles 50
  python scripts/build_urbanist_hg.py --category housing --max-sentences 50
  python scripts/build_urbanist_hg.py --skip-build -i  # Query existing DB
        """
    )

    parser.add_argument('--db', default='urbanist.db',
                       help='Database path (default: urbanist.db)')
    parser.add_argument('--category', '-c', default='politics',
                       choices=list(URBANIST_CATEGORIES.keys()),
                       help='Category to crawl (default: politics)')
    parser.add_argument('--max-articles', '-n', type=int, default=500,
                       help='Maximum articles to fetch (default: 500)')
    parser.add_argument('--max-sentences', '-s', type=int, default=0,
                       help='Max sentences per article, 0=unlimited (default: 0)')
    parser.add_argument('--fresh', action='store_true',
                       help='Remove existing database first')
    parser.add_argument('--skip-build', action='store_true',
                       help='Skip building, just analyze/query')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive query mode after building')
    parser.add_argument('--crawl-delay', type=float, default=0.3,
                       help='Delay between category page fetches (default: 0.3s)')
    parser.add_argument('--fetch-delay', type=float, default=0.3,
                       help='Delay between article fetches (default: 0.3s)')
    parser.add_argument('--corefs', action='store_true',
                       help='Enable coreference resolution (slower but better entity linking)')

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
        hg = build_hypergraph(
            args.db,
            category=args.category,
            max_articles=args.max_articles,
            max_sentences_per_article=args.max_sentences,
            crawl_delay=args.crawl_delay,
            fetch_delay=args.fetch_delay,
            corefs=args.corefs
        )

    if hg:
        analyze_hypergraph(hg)

        if args.interactive:
            interactive_query(hg)

        hg.close()
        logger.info(f"\nDatabase saved: {args.db}")


if __name__ == "__main__":
    main()
