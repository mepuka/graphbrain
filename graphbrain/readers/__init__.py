"""Content readers for graphbrain.

Readers extract text from various sources and parse it into hyperedges.

Available readers:
    - Reader: Base class
    - URLReader: Generic URL content extraction (uses trafilatura)
    - WikipediaReader: Wikipedia article reader
    - JinaReader: Jina AI-powered web content extraction
    - JinaSearchReader: Search and parse web results
    - TxtReader: Plain text file reader
    - CSVReader: CSV file reader
    - DirReader: Directory of files reader
    - RedditReader: Reddit content reader
"""

from graphbrain.readers.reader import Reader
from graphbrain.readers.url import URLReader
from graphbrain.readers.wikipedia import WikipediaReader
from graphbrain.readers.jina import JinaReader, JinaSearchReader, fetch_jina_content
from graphbrain.readers.txt import TxtReader
from graphbrain.readers.csv import CsvReader
from graphbrain.readers.dir import DirReader
from graphbrain.readers.reddit import RedditReader

__all__ = [
    'Reader',
    'URLReader',
    'WikipediaReader',
    'JinaReader',
    'JinaSearchReader',
    'fetch_jina_content',
    'TxtReader',
    'CsvReader',
    'DirReader',
    'RedditReader',
]
