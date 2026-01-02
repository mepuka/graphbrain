from typing import Any, Optional, Type

from graphbrain.parsers import create_parser, parser_lang
from graphbrain.parsers.parser import Parser


class Reader:
    """Base class for reading and parsing text from various sources."""

    def __init__(
        self,
        hg=None,
        sequence: Optional[str] = None,
        lang: Optional[str] = None,
        corefs: bool = False,
        parser: Optional[Parser] = None,
        parser_class: Optional[Type] = None,
        infsrcs: bool = False,
    ) -> None:
        """Initialize reader.

        Args:
            hg: Hypergraph to add parsed edges to.
            sequence: Optional sequence name for grouping edges.
            lang: Language code (e.g., 'en').
            corefs: Whether to resolve coreferences.
            parser: Pre-configured parser instance.
            parser_class: Parser class to instantiate.
            infsrcs: Whether to add inference source edges.
        """
        self.hg = hg
        self.sequence = sequence
        self.lang = lang
        self.infsrcs = infsrcs

        if parser_class:
            plang = parser_lang(parser_class)
            if lang:
                if lang != plang:
                    msg = 'specified language ({}) and parser language ({}) do not match'.format(lang, plang)
                    raise RuntimeError(msg)
            else:
                self.lang = plang

        if parser is None:
            self.parser = create_parser(lang=lang, parser_class=parser_class, lemmas=True, corefs=corefs)
        else:
            self.parser = parser

    def read(self) -> Optional[Any]:
        """Read and parse content. Override in subclasses."""
        pass
