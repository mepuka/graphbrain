import logging

import progressbar

from graphbrain.readers.reader import Reader

logger = logging.getLogger(__name__)


class TxtReader(Reader):
    def __init__(self, infile, hg=None, sequence=None, lang=None, corefs=False, parser=None, parser_class=None,
                 infsrcs=False):
        super().__init__(hg=hg, sequence=sequence, lang=lang, corefs=corefs, parser=parser, parser_class=parser_class,
                         infsrcs=infsrcs)
        self.infile = infile

    def read(self):
        # read paragraphs
        paragraphs = []
        with open(self.infile, 'rt') as f:
            for line in f.readlines():
                paragraph = line.strip()
                if len(paragraph) > 0:
                    paragraphs.append(paragraph)

        with progressbar.ProgressBar(max_value=len(paragraphs)) as bar:
            for i, paragraph in enumerate(paragraphs):
                logger.debug('Processing paragraph: %s', paragraph[:100])
                try:
                    self.parser.parse_and_add(paragraph, self.hg, sequence=self.sequence, infsrcs=self.infsrcs)
                except RuntimeError as e:
                    logger.warning('Error parsing paragraph: %s', e)
                bar.update(i + 1)
