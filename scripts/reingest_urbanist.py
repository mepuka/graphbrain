#!/usr/bin/env python3
"""Re-ingest urbanist articles with coreference resolution.

Uses the source URLs stored in the existing database to fetch and re-parse
with corefs enabled for better entity linking.

Usage:
    python scripts/reingest_urbanist.py
    python scripts/reingest_urbanist.py --max-articles 50
    python scripts/reingest_urbanist.py --output urbanist_corefs.db
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from graphbrain import hgraph
from graphbrain.parsers import create_parser
from graphbrain.readers.jina import fetch_jina_content

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_source_urls(db_path: str) -> list[str]:
    """Extract unique source URLs from existing database."""
    hg = hgraph(db_path)
    sources = set()

    for edge in hg.all():
        attrs = hg.get_attributes(edge)
        if attrs and 'source' in attrs:
            sources.add(attrs['source'])

    hg.close()
    return sorted(sources)


def ingest_with_corefs(
    source_urls: list[str],
    output_db: str,
    max_articles: int = 0,
    fetch_delay: float = 0.3
):
    """Fetch and parse articles with coreference resolution.

    Args:
        source_urls: List of article URLs
        output_db: Output database path
        max_articles: Max articles (0 = all)
        fetch_delay: Delay between fetches
    """
    if max_articles > 0:
        source_urls = source_urls[:max_articles]

    logger.info(f"Will process {len(source_urls)} articles with coreference resolution")

    # Create new hypergraph
    if os.path.exists(output_db):
        os.remove(output_db)
        logger.info(f"Removed existing: {output_db}")

    hg = hgraph(output_db)

    # Create parser with corefs enabled
    logger.info("Loading parser with coreference resolution...")
    parser = create_parser(lang='en', corefs=True)
    logger.info("Parser loaded")

    stats = {
        'articles_fetched': 0,
        'articles_failed': 0,
        'sentences_parsed': 0,
        'edges_added': 0,
        'parse_failures': 0
    }

    for i, url in enumerate(source_urls, 1):
        logger.info(f"\n[{i}/{len(source_urls)}] {url}")

        try:
            content = fetch_jina_content(url)
            stats['articles_fetched'] += 1

            logger.info(f"  Title: {content.title[:60]}...")
            sentences = list(content.sentences())
            logger.info(f"  Sentences: {len(sentences)}")

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
                            hg.set_attribute(main_edge, 'title', content.title)

                            stats['edges_added'] += 1
                            article_edges += 1

                except Exception as e:
                    stats['parse_failures'] += 1
                    logger.debug(f"  Parse error: {e}")

            stats['sentences_parsed'] += len(sentences)
            logger.info(f"  Edges added: {article_edges}")

        except Exception as e:
            logger.error(f"  Failed to fetch: {e}")
            stats['articles_failed'] += 1

        if i < len(source_urls):
            time.sleep(fetch_delay)

    # Summary
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

    hg.close()
    logger.info(f"\nDatabase saved: {output_db}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Re-ingest urbanist articles with coreference resolution'
    )
    parser.add_argument('--source-db', default='urbanist.db',
                       help='Source database with URLs (default: urbanist.db)')
    parser.add_argument('--output', '-o', default='urbanist_corefs.db',
                       help='Output database (default: urbanist_corefs.db)')
    parser.add_argument('--max-articles', '-n', type=int, default=0,
                       help='Max articles (0 = all)')
    parser.add_argument('--fetch-delay', type=float, default=0.3,
                       help='Delay between fetches (default: 0.3)')

    args = parser.parse_args()

    if not os.path.exists(args.source_db):
        logger.error(f"Source database not found: {args.source_db}")
        return

    # Get source URLs
    logger.info(f"Extracting URLs from: {args.source_db}")
    urls = get_source_urls(args.source_db)
    logger.info(f"Found {len(urls)} source URLs")

    # Re-ingest with corefs
    ingest_with_corefs(
        urls,
        args.output,
        max_articles=args.max_articles,
        fetch_delay=args.fetch_delay
    )


if __name__ == "__main__":
    main()
