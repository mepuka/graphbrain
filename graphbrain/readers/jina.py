"""Jina Reader-based web content extraction.

Uses Jina AI's Reader API (r.jina.ai) to extract clean, LLM-friendly
content from any web page. This handles JavaScript rendering, removes
clutter, and returns markdown text ready for parsing.

Usage:
    from graphbrain.readers.jina import JinaReader

    # Basic usage
    reader = JinaReader("https://example.com/article", hg=hg)
    reader.read()

    # With API key for higher rate limits
    reader = JinaReader(url, hg=hg, api_key="your_jina_api_key")
    reader.read()

    # Batch URLs
    from graphbrain.readers.jina import read_urls_with_jina
    read_urls_with_jina(["url1", "url2"], hg=hg)

API Documentation: https://jina.ai/reader/

Rate Limits:
    - Without API key: 20 requests/minute
    - With API key: 500+ requests/minute
    - Free tier includes 10M tokens
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Iterator, Optional
from urllib.parse import urlparse

import requests

from graphbrain.readers.reader import Reader

logger = logging.getLogger(__name__)

# Jina Reader API endpoints
JINA_READER_URL = "https://r.jina.ai"
JINA_SEARCH_URL = "https://s.jina.ai"

# Environment variable for API key
JINA_API_KEY_ENV = "JINA_API_KEY"

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use environment variables directly


def get_jina_api_key(api_key: str = None) -> Optional[str]:
    """Get Jina API key from argument, .env file, or environment.

    Priority:
    1. Explicit api_key argument
    2. JINA_API_KEY from .env file or environment variable
    3. None (uses free tier with rate limits)

    To use .env file, install python-dotenv:
        pip install python-dotenv
    """
    if api_key:
        return api_key
    return os.environ.get(JINA_API_KEY_ENV)

# Default timeout for requests
DEFAULT_TIMEOUT = 30

# Rate limit handling
DEFAULT_RETRY_DELAY = 2.0
MAX_RETRIES = 3


@dataclass
class JinaContent:
    """Extracted content from Jina Reader."""
    url: str
    title: str
    content: str
    description: str = ""
    links: list = None
    raw_response: dict = None

    def sentences(self) -> Iterator[str]:
        """Yield sentences from the content."""
        # Split on common sentence boundaries
        text = self.content

        # Clean up markdown artifacts
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Remove markdown links
        text = re.sub(r'#{1,6}\s*', '', text)  # Remove headers
        text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)  # Remove bold/italic
        text = re.sub(r'`[^`]+`', '', text)  # Remove inline code
        text = re.sub(r'```[\s\S]*?```', '', text)  # Remove code blocks

        # Split into paragraphs first
        paragraphs = text.split('\n\n')

        for para in paragraphs:
            para = para.strip()
            if not para or len(para) < 10:
                continue

            # Split paragraph into sentences
            # Handle common abbreviations
            para = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Jr|Sr|Inc|Ltd|Corp|vs|etc)\.',
                         r'\1<DOT>', para)

            sentences = re.split(r'(?<=[.!?])\s+', para)

            for sent in sentences:
                sent = sent.replace('<DOT>', '.').strip()
                if len(sent) >= 10:  # Skip very short fragments
                    yield sent


def fetch_jina_content(
    url: str,
    api_key: str = None,
    include_links: bool = False,
    response_format: str = "json",
    timeout: int = DEFAULT_TIMEOUT
) -> JinaContent:
    """Fetch clean content from a URL using Jina Reader.

    Args:
        url: The URL to fetch
        api_key: Optional Jina API key for higher rate limits.
                 If not provided, checks JINA_API_KEY environment variable.
        include_links: Whether to include extracted links
        response_format: "json" for structured data, "markdown" for raw text
        timeout: Request timeout in seconds

    Returns:
        JinaContent with extracted text and metadata

    Raises:
        requests.RequestException: On network errors
        ValueError: On invalid responses
    """
    # Get API key from argument or environment
    api_key = get_jina_api_key(api_key)

    headers = {
        "User-Agent": "Graphbrain/1.0"
    }

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if response_format == "json":
        headers["Accept"] = "application/json"

    if include_links:
        headers["X-With-Links-Summary"] = "true"

    reader_url = f"{JINA_READER_URL}/{url}"

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(reader_url, headers=headers, timeout=timeout)

            if response.status_code == 429:  # Rate limited
                wait_time = DEFAULT_RETRY_DELAY * (attempt + 1)
                logger.warning(f"Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()

            if response_format == "json":
                data = response.json()
                return JinaContent(
                    url=url,
                    title=data.get("data", {}).get("title", ""),
                    content=data.get("data", {}).get("content", ""),
                    description=data.get("data", {}).get("description", ""),
                    links=data.get("data", {}).get("links", []),
                    raw_response=data
                )
            else:
                # Parse markdown response
                text = response.text
                title = ""

                # Try to extract title from first heading
                title_match = re.match(r'^#\s+(.+)$', text, re.MULTILINE)
                if title_match:
                    title = title_match.group(1)

                return JinaContent(
                    url=url,
                    title=title,
                    content=text,
                    raw_response={"text": text}
                )

        except requests.Timeout:
            logger.warning(f"Timeout fetching {url}, attempt {attempt + 1}/{MAX_RETRIES}")
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(DEFAULT_RETRY_DELAY)

        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            raise

    raise RuntimeError(f"Failed to fetch {url} after {MAX_RETRIES} attempts")


def search_jina(
    query: str,
    api_key: str = None,
    timeout: int = DEFAULT_TIMEOUT
) -> list[JinaContent]:
    """Search the web using Jina Search and return content from top results.

    Args:
        query: Search query
        api_key: Optional Jina API key.
                 If not provided, checks JINA_API_KEY environment variable.
        timeout: Request timeout

    Returns:
        List of JinaContent from top search results
    """
    # Get API key from argument or environment
    api_key = get_jina_api_key(api_key)

    headers = {
        "User-Agent": "Graphbrain/1.0",
        "Accept": "application/json"
    }

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    search_url = f"{JINA_SEARCH_URL}/?q={query}"

    try:
        response = requests.get(search_url, headers=headers, timeout=timeout)
        response.raise_for_status()

        data = response.json()
        results = []

        for item in data.get("data", []):
            results.append(JinaContent(
                url=item.get("url", ""),
                title=item.get("title", ""),
                content=item.get("content", ""),
                description=item.get("description", ""),
                raw_response=item
            ))

        return results

    except requests.RequestException as e:
        logger.error(f"Search error: {e}")
        raise


class JinaReader(Reader):
    """Reader that uses Jina AI to extract content from web pages.

    Jina Reader handles:
    - JavaScript-rendered content
    - Cluttered pages (ads, navigation, etc.)
    - Various content types (articles, documentation, etc.)

    Returns clean markdown text ready for NLP parsing.
    """

    def __init__(
        self,
        url: str,
        hg=None,
        sequence=None,
        lang=None,
        corefs=False,
        parser=None,
        parser_class=None,
        infsrcs=False,
        api_key: str = None,
        include_links: bool = False,
        store_metadata: bool = True,
        max_sentences: int = 0,
        progress: bool = True
    ):
        """Initialize JinaReader.

        Args:
            url: URL to read content from
            hg: Hypergraph to add edges to
            sequence: Sequence name for edge tracking
            lang: Language code
            corefs: Enable coreference resolution
            parser: Pre-configured parser
            parser_class: Parser class to use
            infsrcs: Add inference sources
            api_key: Jina API key for higher rate limits
            include_links: Extract and store links from page
            store_metadata: Store title/description as edges
            max_sentences: Max sentences to parse (0 = unlimited)
            progress: Show progress bar
        """
        if sequence is None:
            sequence = url

        super().__init__(
            hg=hg,
            sequence=sequence,
            lang=lang,
            corefs=corefs,
            parser=parser,
            parser_class=parser_class,
            infsrcs=infsrcs
        )

        self.url = url
        self.api_key = api_key
        self.include_links = include_links
        self.store_metadata = store_metadata
        self.max_sentences = max_sentences
        self.show_progress = progress

        self._content: Optional[JinaContent] = None

    def fetch(self) -> JinaContent:
        """Fetch content from URL using Jina Reader."""
        logger.info(f"Fetching content from: {self.url}")

        self._content = fetch_jina_content(
            self.url,
            api_key=self.api_key,
            include_links=self.include_links
        )

        logger.info(f"Fetched: {self._content.title}")
        return self._content

    def read(self) -> dict:
        """Fetch and parse content, adding edges to hypergraph.

        Returns:
            Statistics about parsed content
        """
        from graphbrain import hedge

        if self._content is None:
            self.fetch()

        content = self._content
        stats = {
            'url': self.url,
            'title': content.title,
            'sentences': 0,
            'edges': 0,
            'failed': 0
        }

        # Collect all sentences
        sentences = list(content.sentences())
        if self.max_sentences > 0:
            sentences = sentences[:self.max_sentences]

        total = len(sentences)
        logger.info(f"Parsing {total} sentences from {content.title}")

        # Parse sentences
        if self.show_progress:
            try:
                import progressbar
                bar = progressbar.ProgressBar(max_value=total)
            except ImportError:
                bar = None
        else:
            bar = None

        for i, sentence in enumerate(sentences):
            stats['sentences'] += 1

            try:
                parse_result = self.parser.parse_and_add(
                    sentence,
                    self.hg,
                    sequence=self.sequence,
                    infsrcs=self.infsrcs
                )

                for parse in parse_result.get('parses', []):
                    edge = parse.get('main_edge')
                    if edge:
                        stats['edges'] += 1

            except Exception as e:
                logger.debug(f"Parse failed for: {sentence[:50]}... - {e}")
                stats['failed'] += 1

            if bar:
                bar.update(i + 1)

        if bar:
            bar.finish()

        # Store metadata as edges
        if self.store_metadata and self.hg:
            if content.title:
                try:
                    title_result = self.parser.parse_and_add(
                        content.title,
                        self.hg,
                        sequence=self.sequence,
                        infsrcs=self.infsrcs
                    )
                    for parse in title_result.get('parses', []):
                        edge = parse.get('main_edge')
                        if edge:
                            self.hg.add(hedge(['title', edge]))
                except Exception:
                    pass

            if content.description:
                try:
                    desc_result = self.parser.parse_and_add(
                        content.description,
                        self.hg,
                        sequence=self.sequence,
                        infsrcs=self.infsrcs
                    )
                    for parse in desc_result.get('parses', []):
                        edge = parse.get('main_edge')
                        if edge:
                            self.hg.add(hedge(['description', edge]))
                except Exception:
                    pass

        logger.info(f"Parsed {stats['edges']} edges from {stats['sentences']} sentences")
        return stats


class JinaSearchReader(Reader):
    """Reader that searches the web and parses top results.

    Uses Jina Search API to find relevant pages, then extracts
    and parses content from each result.
    """

    def __init__(
        self,
        query: str,
        hg=None,
        sequence=None,
        lang=None,
        corefs=False,
        parser=None,
        parser_class=None,
        infsrcs=False,
        api_key: str = None,
        max_results: int = 5,
        max_sentences_per_result: int = 100
    ):
        """Initialize JinaSearchReader.

        Args:
            query: Search query
            max_results: Maximum search results to process
            max_sentences_per_result: Max sentences per page
        """
        if sequence is None:
            sequence = f"search:{query}"

        super().__init__(
            hg=hg,
            sequence=sequence,
            lang=lang,
            corefs=corefs,
            parser=parser,
            parser_class=parser_class,
            infsrcs=infsrcs
        )

        self.query = query
        self.api_key = api_key
        self.max_results = max_results
        self.max_sentences_per_result = max_sentences_per_result

    def read(self) -> dict:
        """Search and parse results.

        Returns:
            Statistics about parsed content
        """
        logger.info(f"Searching for: {self.query}")

        results = search_jina(self.query, api_key=self.api_key)
        results = results[:self.max_results]

        stats = {
            'query': self.query,
            'results': len(results),
            'total_sentences': 0,
            'total_edges': 0,
            'by_url': []
        }

        for content in results:
            logger.info(f"Parsing: {content.title}")

            url_stats = {
                'url': content.url,
                'title': content.title,
                'sentences': 0,
                'edges': 0
            }

            sentences = list(content.sentences())
            if self.max_sentences_per_result > 0:
                sentences = sentences[:self.max_sentences_per_result]

            for sentence in sentences:
                url_stats['sentences'] += 1

                try:
                    parse_result = self.parser.parse_and_add(
                        sentence,
                        self.hg,
                        sequence=content.url,
                        infsrcs=self.infsrcs
                    )

                    for parse in parse_result.get('parses', []):
                        if parse.get('main_edge'):
                            url_stats['edges'] += 1

                except Exception:
                    pass

            stats['total_sentences'] += url_stats['sentences']
            stats['total_edges'] += url_stats['edges']
            stats['by_url'].append(url_stats)

        logger.info(f"Parsed {stats['total_edges']} edges from {len(results)} pages")
        return stats


def read_urls_with_jina(
    urls: list[str],
    hg,
    api_key: str = None,
    lang: str = None,
    corefs: bool = False,
    max_sentences: int = 0,
    delay: float = 1.0
) -> dict:
    """Convenience function to read multiple URLs.

    Args:
        urls: List of URLs to read
        hg: Hypergraph to add edges to
        api_key: Jina API key
        lang: Language code
        corefs: Enable coreference resolution
        max_sentences: Max sentences per URL (0 = unlimited)
        delay: Delay between requests (for rate limiting)

    Returns:
        Combined statistics
    """
    stats = {
        'urls': len(urls),
        'total_edges': 0,
        'by_url': []
    }

    for url in urls:
        logger.info(f"Processing: {url}")

        reader = JinaReader(
            url,
            hg=hg,
            api_key=api_key,
            lang=lang,
            corefs=corefs,
            max_sentences=max_sentences
        )

        try:
            url_stats = reader.read()
            stats['total_edges'] += url_stats['edges']
            stats['by_url'].append(url_stats)
        except Exception as e:
            logger.error(f"Failed to read {url}: {e}")
            stats['by_url'].append({'url': url, 'error': str(e)})

        if delay > 0:
            time.sleep(delay)

    return stats
