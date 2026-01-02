#!/usr/bin/env python3
"""Scrape The Urbanist articles using Jina Reader API with parallel downloads.

Usage:
    python scripts/scrape_urbanist.py --category politics-and-government --max-pages 5
    python scripts/scrape_urbanist.py --output urbanist_articles.json
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urljoin

import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Jina API configuration
JINA_READER_URL = "https://r.jina.ai"
JINA_API_KEY = os.environ.get("JINA_API_KEY")

# The Urbanist base URL
URBANIST_BASE = "https://www.theurbanist.org"


@dataclass
class ArticleMetadata:
    """Full metadata from Jina API response."""
    url: str
    title: str
    description: str
    content: str
    author: Optional[str] = None
    published_time: Optional[str] = None
    modified_time: Optional[str] = None
    site_name: Optional[str] = None
    image: Optional[str] = None
    favicon: Optional[str] = None
    language: Optional[str] = None
    word_count: int = 0
    links: list = None
    raw_response: dict = None
    fetch_time: str = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.links is None:
            self.links = []
        if self.fetch_time is None:
            self.fetch_time = datetime.now(timezone.utc).isoformat()


def fetch_article_jina(url: str, api_key: str = None, timeout: int = 60) -> ArticleMetadata:
    """Fetch article using Jina Reader JSON API with full metadata.

    Args:
        url: Article URL to fetch
        api_key: Jina API key (uses env if not provided)
        timeout: Request timeout in seconds

    Returns:
        ArticleMetadata with all extracted data
    """
    api_key = api_key or JINA_API_KEY

    headers = {
        "User-Agent": "Graphbrain/1.0 (The Urbanist Scraper)",
        "Accept": "application/json",
        # Request all available metadata
        "X-With-Generated-Alt": "true",  # Generate alt text for images
        "X-With-Links-Summary": "true",  # Include extracted links
    }

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    reader_url = f"{JINA_READER_URL}/{url}"

    try:
        logger.debug(f"Fetching: {url}")
        response = requests.get(reader_url, headers=headers, timeout=timeout)

        if response.status_code == 429:
            logger.warning(f"Rate limited for {url}")
            return ArticleMetadata(
                url=url, title="", description="", content="",
                error="Rate limited"
            )

        response.raise_for_status()
        data = response.json()

        # Extract all metadata from Jina response
        article_data = data.get("data", {})

        # Calculate word count
        content = article_data.get("content", "")
        word_count = len(content.split()) if content else 0

        return ArticleMetadata(
            url=url,
            title=article_data.get("title", ""),
            description=article_data.get("description", ""),
            content=content,
            author=article_data.get("author"),
            published_time=article_data.get("publishedTime"),
            modified_time=article_data.get("modifiedTime"),
            site_name=article_data.get("siteName"),
            image=article_data.get("image"),
            favicon=article_data.get("favicon"),
            language=article_data.get("language"),
            word_count=word_count,
            links=article_data.get("links", []),
            raw_response=data
        )

    except requests.Timeout:
        logger.error(f"Timeout fetching {url}")
        return ArticleMetadata(
            url=url, title="", description="", content="",
            error=f"Timeout after {timeout}s"
        )
    except requests.RequestException as e:
        logger.error(f"Error fetching {url}: {e}")
        return ArticleMetadata(
            url=url, title="", description="", content="",
            error=str(e)
        )
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON from {url}: {e}")
        return ArticleMetadata(
            url=url, title="", description="", content="",
            error=f"Invalid JSON: {e}"
        )


def discover_articles_from_category(category_url: str, max_pages: int = 10) -> list[str]:
    """Discover article URLs from a category page.

    Fetches the category page and extracts article links.
    Handles pagination (page/2/, page/3/, etc.)

    Args:
        category_url: The category page URL
        max_pages: Maximum number of pagination pages to check

    Returns:
        List of article URLs
    """
    article_urls = set()
    prev_count = 0

    for page_num in range(1, max_pages + 1):
        if page_num == 1:
            page_url = category_url.rstrip('/')
        else:
            page_url = f"{category_url.rstrip('/')}/page/{page_num}/"

        logger.info(f"Discovering articles from page {page_num}: {page_url}")

        # Use Jina to fetch the page with links
        headers = {
            "User-Agent": "Graphbrain/1.0",
            "Accept": "application/json",
            "X-With-Links-Summary": "true",
        }

        if JINA_API_KEY:
            headers["Authorization"] = f"Bearer {JINA_API_KEY}"

        try:
            response = requests.get(
                f"{JINA_READER_URL}/{page_url}",
                headers=headers,
                timeout=60
            )

            if response.status_code == 404:
                logger.info(f"No more pages after page {page_num - 1}")
                break

            response.raise_for_status()
            data = response.json()

            content = data.get("data", {}).get("content", "")

            # Links is a dict with title:url format
            links = data.get("data", {}).get("links", {})
            if isinstance(links, dict):
                for title, href in links.items():
                    if is_article_url(href):
                        article_urls.add(href)

            # Extract article URLs directly from content using regex
            # Pattern: /YYYY/MM/DD/slug/
            urls_in_content = re.findall(
                r'https://www\.theurbanist\.org/\d{4}/\d{2}/\d{2}/[a-z0-9-]+/?',
                content
            )
            for url in urls_in_content:
                article_urls.add(url.rstrip('/') + '/')

            # Also extract from markdown content (links in [text](url) format)
            md_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
            for text, href in md_links:
                if is_article_url(href):
                    article_urls.add(href.rstrip('/') + '/')

            # Check if we found new articles on this page
            new_count = len(article_urls)
            if new_count == prev_count and page_num > 1:
                logger.info(f"No new articles found on page {page_num}, stopping")
                break

            logger.info(f"  Found {new_count - prev_count} new articles (total: {new_count})")
            prev_count = new_count

        except requests.RequestException as e:
            logger.error(f"Error fetching category page {page_url}: {e}")
            if page_num > 1:
                break
            raise

    logger.info(f"Discovered {len(article_urls)} total article URLs")

    return sorted(article_urls, reverse=True)  # Newest first


def is_article_url(url: str) -> bool:
    """Check if URL looks like an Urbanist article."""
    if not url:
        return False

    # Must be from The Urbanist
    if "theurbanist.org" not in url:
        return False

    # Skip category/tag/author pages
    skip_patterns = [
        '/category/', '/tag/', '/author/', '/page/',
        '/wp-content/', '/wp-includes/',
        '/about/', '/contact/', '/subscribe/', '/advertise/',
        '/privacy-policy/', '/terms-of-service/',
    ]
    for pattern in skip_patterns:
        if pattern in url:
            return False

    # Articles have date-like pattern: /2025/12/19/article-slug/
    article_pattern = r'/\d{4}/\d{2}/\d{2}/[a-z0-9-]+/?$'
    if re.search(article_pattern, url):
        return True

    return False


def scrape_articles_parallel(
    urls: list[str],
    max_workers: int = 5,
    api_key: str = None,
    delay: float = 0.5
) -> list[ArticleMetadata]:
    """Scrape multiple articles in parallel.

    Args:
        urls: List of article URLs
        max_workers: Number of parallel threads
        api_key: Jina API key
        delay: Small delay between batches to avoid rate limits

    Returns:
        List of ArticleMetadata objects
    """
    api_key = api_key or JINA_API_KEY
    results = []

    logger.info(f"Scraping {len(urls)} articles with {max_workers} workers")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_url = {
            executor.submit(fetch_article_jina, url, api_key): url
            for url in urls
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            completed += 1

            try:
                article = future.result()
                results.append(article)

                if article.error:
                    logger.warning(f"[{completed}/{len(urls)}] Error: {url} - {article.error}")
                else:
                    logger.info(f"[{completed}/{len(urls)}] Fetched: {article.title[:60]}...")

            except Exception as e:
                logger.error(f"[{completed}/{len(urls)}] Failed: {url} - {e}")
                results.append(ArticleMetadata(
                    url=url, title="", description="", content="",
                    error=str(e)
                ))

    # Sort by URL (which includes date) for consistent ordering
    results.sort(key=lambda a: a.url, reverse=True)

    return results


def save_articles(articles: list[ArticleMetadata], output_path: str, include_raw: bool = False):
    """Save articles to JSON file.

    Args:
        articles: List of ArticleMetadata
        output_path: Output JSON file path
        include_raw: Whether to include raw API response
    """
    output_data = {
        "metadata": {
            "source": "The Urbanist",
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "total_articles": len(articles),
            "successful": len([a for a in articles if not a.error]),
            "failed": len([a for a in articles if a.error]),
        },
        "articles": []
    }

    for article in articles:
        article_dict = asdict(article)
        if not include_raw:
            article_dict.pop("raw_response", None)
        output_data["articles"].append(article_dict)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(articles)} articles to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape The Urbanist articles using Jina Reader API"
    )
    parser.add_argument(
        "--category",
        default="politics-and-government",
        help="Category slug to scrape (default: politics-and-government)"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Maximum pagination pages to check (default: 10)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel download workers (default: 5)"
    )
    parser.add_argument(
        "--output",
        default="urbanist_articles.json",
        help="Output JSON file (default: urbanist_articles.json)"
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Include raw Jina API response in output"
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        help="Specific URLs to scrape (skips discovery)"
    )
    parser.add_argument(
        "--url-file",
        help="File containing URLs to scrape (one per line)"
    )
    parser.add_argument(
        "--api-key",
        help="Jina API key (or set JINA_API_KEY env var)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Use provided API key or env var
    api_key = args.api_key or JINA_API_KEY
    if not api_key:
        logger.warning("No JINA_API_KEY set - using free tier (20 req/min limit)")

    # Get article URLs
    if args.url_file:
        with open(args.url_file) as f:
            article_urls = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(article_urls)} URLs from {args.url_file}")
    elif args.urls:
        article_urls = args.urls
        logger.info(f"Using {len(article_urls)} provided URLs")
    else:
        category_url = f"{URBANIST_BASE}/category/{args.category}/"
        article_urls = discover_articles_from_category(category_url, args.max_pages)

    if not article_urls:
        logger.error("No articles found to scrape")
        return 1

    logger.info(f"Found {len(article_urls)} articles to scrape")

    # Scrape articles in parallel
    articles = scrape_articles_parallel(
        article_urls,
        max_workers=args.workers,
        api_key=api_key
    )

    # Save results
    save_articles(articles, args.output, include_raw=args.include_raw)

    # Print summary
    successful = [a for a in articles if not a.error]
    failed = [a for a in articles if a.error]

    print(f"\n{'='*60}")
    print(f"Scraping Complete!")
    print(f"{'='*60}")
    print(f"Total articles: {len(articles)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total words: {sum(a.word_count for a in successful):,}")
    print(f"Output: {args.output}")

    if failed:
        print(f"\nFailed articles:")
        for a in failed[:5]:
            print(f"  - {a.url}: {a.error}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")

    return 0


if __name__ == "__main__":
    sys.exit(main())
