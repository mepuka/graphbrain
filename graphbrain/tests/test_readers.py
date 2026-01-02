"""Tests for reader modules."""

import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from graphbrain.readers.reader import Reader
from graphbrain.readers.txt import TxtReader
from graphbrain.readers.csv import CsvReader, file_lines, text_parts


class TestReader:
    """Tests for base Reader class."""

    def test_init_with_hg(self):
        """Test Reader initialization with hypergraph."""
        mock_hg = MagicMock()
        with patch('graphbrain.readers.reader.create_parser') as mock_create:
            mock_create.return_value = MagicMock()
            reader = Reader(hg=mock_hg, lang='en')
            assert reader.hg == mock_hg
            assert reader.lang == 'en'

    def test_init_with_parser(self):
        """Test Reader initialization with existing parser."""
        mock_parser = MagicMock()
        reader = Reader(parser=mock_parser)
        assert reader.parser == mock_parser

    def test_init_with_sequence(self):
        """Test Reader initialization with sequence."""
        with patch('graphbrain.readers.reader.create_parser') as mock_create:
            mock_create.return_value = MagicMock()
            reader = Reader(sequence=42, lang='en')
            assert reader.sequence == 42

    def test_init_infsrcs(self):
        """Test Reader initialization with infsrcs flag."""
        with patch('graphbrain.readers.reader.create_parser') as mock_create:
            mock_create.return_value = MagicMock()
            reader = Reader(infsrcs=True, lang='en')
            assert reader.infsrcs is True

    def test_read_base(self):
        """Test base read method does nothing."""
        with patch('graphbrain.readers.reader.create_parser') as mock_create:
            mock_create.return_value = MagicMock()
            reader = Reader(lang='en')
            result = reader.read()
            assert result is None


class TestTxtReader:
    """Tests for TxtReader class."""

    @pytest.fixture
    def temp_txt_file(self):
        """Create a temporary text file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is the first paragraph.\n")
            f.write("\n")  # Empty line
            f.write("This is the second paragraph.\n")
            f.write("This is the third paragraph.\n")
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    def test_init(self, temp_txt_file):
        """Test TxtReader initialization."""
        with patch('graphbrain.readers.reader.create_parser') as mock_create:
            mock_create.return_value = MagicMock()
            reader = TxtReader(temp_txt_file, lang='en')
            assert reader.infile == temp_txt_file

    def test_read_parses_paragraphs(self, temp_txt_file):
        """Test that read parses non-empty paragraphs."""
        mock_parser = MagicMock()
        mock_hg = MagicMock()

        reader = TxtReader(temp_txt_file, hg=mock_hg, parser=mock_parser)
        reader.read()

        # Should have parsed 3 paragraphs (empty lines are skipped)
        assert mock_parser.parse_and_add.call_count == 3

    def test_read_handles_empty_file(self):
        """Test reading an empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name

        try:
            mock_parser = MagicMock()
            reader = TxtReader(temp_path, parser=mock_parser)
            reader.read()
            # Should not call parse_and_add for empty file
            assert mock_parser.parse_and_add.call_count == 0
        finally:
            os.unlink(temp_path)

    def test_read_with_sequence(self, temp_txt_file):
        """Test reading with sequence number."""
        mock_parser = MagicMock()
        mock_hg = MagicMock()

        reader = TxtReader(temp_txt_file, hg=mock_hg, parser=mock_parser, sequence=5)
        reader.read()

        # Check sequence was passed to parse_and_add
        for call in mock_parser.parse_and_add.call_args_list:
            assert call.kwargs.get('sequence') == 5


class TestCsvReader:
    """Tests for CsvReader class."""

    @pytest.fixture
    def temp_csv_file(self):
        """Create a temporary CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,title,content\n")
            f.write("1,First Title,First content here.\n")
            f.write("2,Second Title,Second content here.\n")
            f.write("3,Third Title,Third content here.\n")
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    def test_init(self, temp_csv_file):
        """Test CsvReader initialization."""
        with patch('graphbrain.readers.reader.create_parser') as mock_create:
            mock_create.return_value = MagicMock()
            reader = CsvReader(temp_csv_file, column='title', lang='en')
            assert reader.infile == temp_csv_file
            assert reader.column == 'title'

    def test_read_parses_rows(self, temp_csv_file):
        """Test that read parses CSV rows."""
        mock_parser = MagicMock()
        mock_hg = MagicMock()

        reader = CsvReader(temp_csv_file, column='title', hg=mock_hg, parser=mock_parser)
        reader.read()

        # Should have parsed 3 rows
        assert mock_parser.parse_and_add.call_count >= 3

    def test_read_with_content_column(self, temp_csv_file):
        """Test reading content column."""
        mock_parser = MagicMock()
        mock_hg = MagicMock()

        reader = CsvReader(temp_csv_file, column='content', hg=mock_hg, parser=mock_parser)
        reader.read()

        # Check content was parsed
        assert mock_parser.parse_and_add.call_count >= 3


class TestCsvHelperFunctions:
    """Tests for CSV helper functions."""

    def test_file_lines(self):
        """Test file_lines counts correctly."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("line1\n")
            f.write("line2\n")
            f.write("line3\n")
            temp_path = f.name

        try:
            count = file_lines(temp_path)
            assert count == 3
        finally:
            os.unlink(temp_path)

    def test_file_lines_empty(self):
        """Test file_lines with empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("\n")  # Single newline
            temp_path = f.name

        try:
            count = file_lines(temp_path)
            assert count == 1
        finally:
            os.unlink(temp_path)

    def test_text_parts_simple(self):
        """Test text_parts with simple title."""
        parts = text_parts("Simple Title")
        assert parts == ["Simple Title"]

    def test_text_parts_with_pipe(self):
        """Test text_parts splits on pipe."""
        parts = text_parts("Part One | Part Two")
        assert "Part One" in parts
        assert "Part Two" in parts

    def test_text_parts_with_dash(self):
        """Test text_parts splits on dash separator."""
        parts = text_parts("Part One - Part Two")
        assert len(parts) == 2

    def test_text_parts_with_double_dash(self):
        """Test text_parts splits on double dash."""
        parts = text_parts("Part One -- Part Two")
        assert len(parts) == 2

    def test_text_parts_with_brackets(self):
        """Test text_parts handles brackets."""
        parts = text_parts("[TAG] Main Title")
        # Should extract TAG and Main Title
        assert len(parts) >= 1

    def test_text_parts_empty(self):
        """Test text_parts with empty string."""
        parts = text_parts("")
        assert parts == []

    def test_text_parts_whitespace_only(self):
        """Test text_parts with whitespace."""
        parts = text_parts("   ")
        # text_parts strips parts, so whitespace becomes empty string
        assert parts == [''] or parts == []


class TestReaderIntegration:
    """Integration tests for readers (require spacy)."""

    @pytest.fixture
    def temp_simple_txt(self):
        """Create simple text file for parsing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("The sky is blue.\n")
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    @pytest.mark.skipif(
        not os.environ.get('RUN_INTEGRATION_TESTS'),
        reason="Integration tests require spacy models"
    )
    def test_txt_reader_full_parse(self, temp_simple_txt):
        """Test full parsing with actual parser."""
        from graphbrain import hgraph

        hg = hgraph("test_reader_integration.db")
        reader = TxtReader(temp_simple_txt, hg=hg, lang='en')
        reader.read()

        # Should have added some edges
        edges = list(hg.all())
        assert len(edges) > 0
