import logging
from typing import Optional

from graphbrain.hyperedge import Hyperedge


logger = logging.getLogger(__name__)


class Processor:
    """Base class for hypergraph processors.

    Processors iterate over edges in a hypergraph and perform
    operations on them.
    """

    def __init__(self, hg, sequence: Optional[str] = None) -> None:
        """Initialize processor.

        Args:
            hg: Hypergraph to process.
            sequence: Optional sequence name to limit processing to.
        """
        self.hg = hg
        self.sequence = sequence

    def process_edge(self, edge: Hyperedge) -> None:
        """Process a single edge. Override in subclasses."""
        pass

    def on_end(self) -> None:
        """Called after all edges are processed. Override in subclasses."""
        pass

    def report(self) -> str:
        """Return a report string. Override in subclasses."""
        return ''

    def run(self) -> None:
        """Run the processor over all edges."""
        if self.sequence is None:
            for edge in self.hg.all():
                self.process_edge(edge)
        else:
            for edge in self.hg.sequence(self.sequence):
                self.process_edge(edge)
        self.on_end()
        logger.info(self.report())
