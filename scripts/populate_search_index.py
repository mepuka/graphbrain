#!/usr/bin/env python3
"""Populate the FTS search index from existing edge text attributes.

This script backfills the edge_text FTS5 table from text attributes
stored in the hypergraph. Run this once after ingestion to enable
BM25 and hybrid search.

Usage:
    python scripts/populate_search_index.py urbanist.db
"""

import argparse
import logging
from graphbrain import hgraph
from graphbrain.classification.search import get_search_backend

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def populate_index(db_path: str, batch_size: int = 1000):
    """Populate FTS index from edge text attributes.

    Args:
        db_path: Path to SQLite database
        batch_size: Log progress every N edges
    """
    import sqlite3

    # Enable WAL mode for concurrent access (in case MCP server is running)
    logger.info(f"Enabling WAL mode on: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.close()

    logger.info(f"Opening hypergraph: {db_path}")
    hg = hgraph(db_path)

    logger.info("Initializing search backend")
    searcher = get_search_backend(db_path)

    # Get stats before
    try:
        stats = searcher.get_stats()
        logger.info(f"Before: {stats.get('edges_with_text', 0)} edges indexed")
    except Exception as e:
        logger.warning(f"Could not get initial stats: {e}")

    indexed = 0
    skipped = 0

    logger.info("Iterating through all edges...")
    for edge in hg.all():
        # Get text attribute
        text = hg.get_str_attribute(edge, 'text')

        if text:
            edge_key = edge.to_str()
            try:
                searcher.add_text(edge_key, text)
                indexed += 1

                if indexed % batch_size == 0:
                    logger.info(f"Indexed {indexed} edges...")
            except Exception as e:
                logger.warning(f"Failed to index edge: {e}")
                skipped += 1
        else:
            skipped += 1

    # Final stats
    try:
        stats = searcher.get_stats()
        logger.info(f"After: {stats.get('edges_with_text', indexed)} edges indexed")
    except Exception:
        pass

    logger.info(f"Done! Indexed: {indexed}, Skipped (no text): {skipped}")

    hg.close()
    searcher.close()


def main():
    parser = argparse.ArgumentParser(
        description='Populate FTS search index from existing edge text attributes'
    )
    parser.add_argument('db_path', help='Path to SQLite database')
    parser.add_argument(
        '--batch-size', type=int, default=1000,
        help='Log progress every N edges (default: 1000)'
    )
    args = parser.parse_args()

    populate_index(args.db_path, args.batch_size)


if __name__ == "__main__":
    main()
