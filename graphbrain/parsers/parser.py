import logging
import re

import graphbrain.constants as const

from graphbrain import hedge
from graphbrain.hyperedge import UniqueAtom


logger = logging.getLogger(__name__)


def _contains_resolution(edge):
    if edge.atom:
        return False
    if str(edge[0]) == const.resolved_to_connector:
        return True
    return any(_contains_resolution(_edge) for _edge in edge)


class Parser(object):
    """Defines the common interface for parser objects.
    Parsers transofrm natural text into graphbrain hyperedges.
    """

    def __init__(self, lemmas=True, corefs=True, debug=False):
        self.lemmas = lemmas
        self.corefs = corefs
        self.debug = debug

        # to be created by derived classes
        self.lang = None

    def debug_msg(self, msg):
        if self.debug:
            logger.debug(msg)

    def parse(self, text):
        """Transforms the given text into hyperedges + aditional information.
        Returns a dictionary with two fields:

        -> parses: a sequence of dictionaries, with one dictionary for each
        sentence found in the text.

        -> inferred_edges: a sequence of edges inferred during by parsing
        process (e.g. genders, 'X is Y' relationships)

        Each sentence parse dictionary contains at least the following fields:

        -> main_edge: the hyperedge corresponding to the sentence.

        -> extra_edges: aditional edges, e.g. connecting atoms that appear
        in the main_edge to their lemmas.

        -> text: the string of natural language text corresponding to the
        main_edge, i.e.: the sentence itself.

        -> edges_text: a dictionary of all edges and subedges to their
        corresponding text.

        -> corefs: resolve coreferences.
        """
        # replace newlines with spaces
        clean_text = text.replace('\n', ' ').replace('\r', ' ')
        # remove repeated spaces
        clean_text = ' '.join(clean_text.split())
        parse_results = self._parse(clean_text)

        # coreference resolution
        if self.corefs:
            self._resolve_corefs(parse_results)
        else:
            for parse in parse_results['parses']:
                parse['resolved_corefs'] = parse['main_edge']

        return parse_results

    def _edge2txt_parts(self, edge, parse):
        if edge.not_atom and str(edge[0]) == const.resolved_to_connector:
            atoms = [UniqueAtom(atom) for atom in edge[1].all_atoms()]
            tokens = [parse['atom2token'][atom] for atom in atoms if atom in parse['atom2token']]
            return [(self._edge2text(edge[1], parse),
                     self._edge2text(edge[2], parse),
                     min(token.i for token in tokens))]
        elif _contains_resolution(edge):
            parts = []
            for _edge in edge:
                parts += self._edge2txt_parts(_edge, parse)
            return parts
        else:
            atoms = [UniqueAtom(atom) for atom in edge.all_atoms()]
            tokens = [parse['atom2token'][atom] for atom in atoms if atom in parse['atom2token']]
            txts = [token.text for token in tokens]
            pos = [token.i for token in tokens]
            return list(zip(txts, txts, pos))

    def _edge2text(self, edge, parse):
        if edge.not_atom and str(edge[0]) == const.possessive_builder:
            return self._poss2text(edge, parse)

        parts = self._edge2txt_parts(edge, parse)
        parts = sorted(parts, key=lambda x: x[2])

        prev_txt = None
        txt_parts = []
        sentence = str(parse['spacy_sentence'])
        for txt, _txt, _ in parts:
            if prev_txt is not None:
                res = re.search(r'{}(.*?){}'.format(re.escape(prev_txt), re.escape(txt)), sentence)
                if res:
                    sep = res.group(1)
                else:
                    sep = ' '
                if any(letter.isalnum() for letter in sep):
                    sep = ' '
                txt_parts.append(sep)
            txt_parts.append(_txt)
            prev_txt = txt
        return ''.join(txt_parts)

    def _set_edge_text(self, edge, reference_edge, hg, parse):
        if reference_edge.not_atom and str(reference_edge[0]) == const.resolved_to_connector:
            _reference_edge = reference_edge[2]
        else:
            _reference_edge = reference_edge
        text = self._edge2text(_reference_edge, parse)
        hg.set_attribute(edge, 'text', text)

        if edge.not_atom:
            for subedge, reference_subedge in zip(edge, _reference_edge):
                self._set_edge_text(subedge, reference_subedge, hg, parse)

    def parse_and_add(self, text, hg, sequence=None, infsrcs=False, max_text=None):
        """Parse text and add results to hypergraph.

        Args:
            text: Text to parse.
            hg: Hypergraph to add edges to.
            sequence: Optional sequence name to add edges to.
            infsrcs: Whether to add inference source edges.
            max_text: Maximum text length for coreference. None uses config default,
                     0 means no limit, -1 disables chunking.
        """
        from graphbrain.parsers.config import get_config

        # Determine max_text from config if not specified
        if max_text is None:
            config = get_config()
            max_text = config.parser.max_coref_text

        # Split large text blocks to avoid coreference resolution errors
        if self.corefs and 0 < max_text < len(text):
            parse_results = {'parses': [], 'inferred_edges': []}
            chunks = self._smart_chunk_text(text, max_text)
            for chunk in chunks:
                _parse_results = self.parse_and_add(chunk, hg=hg, sequence=sequence, infsrcs=infsrcs, max_text=-1)
                parse_results['parses'] += _parse_results['parses']
                parse_results['inferred_edges'] += _parse_results['inferred_edges']
            return parse_results

        parse_results = self.parse(text)
        edges = []
        for parse in parse_results['parses']:
            if parse['main_edge']:
                edges.append(parse['main_edge'])
            main_edge = parse['resolved_corefs']
            if self.corefs:
                unresolved_edge = parse['main_edge']
                reference_edge = parse['resolved_to']
            else:
                unresolved_edge = None
                reference_edge = parse['main_edge']
            # add main edge
            if main_edge:
                if sequence:
                    hg.add_to_sequence(sequence, main_edge)
                else:
                    hg.add(main_edge)
                # attach text to edge and subedges
                self._set_edge_text(main_edge, reference_edge, hg, parse)
                # attach token list and token position structure to edge
                self._set_edge_tokens(main_edge, hg, parse)
                if self.corefs:
                    if unresolved_edge != main_edge:
                        self._set_edge_text(unresolved_edge, unresolved_edge, hg, parse)
                        self._set_edge_tokens(unresolved_edge, hg, parse)
                    coref_res_edge = hedge((const.coref_res_connector, unresolved_edge, main_edge))
                    hg.add(coref_res_edge)
                # add extra edges
                for edge in parse['extra_edges']:
                    hg.add(edge)
        for edge in parse_results['inferred_edges']:
            hg.add(edge, count=True)
            if infsrcs:
                inference_srcs_edge = hedge([const.inference_srcs_connector, edge] + edges)
                hg.add(inference_srcs_edge)

        return parse_results

    def _smart_chunk_text(self, text, max_chars):
        """Split text into chunks that respect sentence boundaries.

        Creates chunks that:
        - Stay under max_chars limit
        - Break only at sentence boundaries
        - Group related sentences when possible

        Args:
            text: Text to chunk.
            max_chars: Maximum characters per chunk.

        Returns:
            List of text chunks.
        """
        from graphbrain.parsers.config import get_config

        sentences = self.sentences(text)

        if not sentences:
            return [text] if text.strip() else []

        config = get_config()
        overlap_chars = config.parser.chunk_overlap

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If single sentence exceeds max, include it alone
            if sentence_len > max_chars:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                chunks.append(sentence)
                continue

            # Check if adding this sentence would exceed limit
            if current_length + sentence_len + 1 > max_chars:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))

                    # Add overlap: include last sentence(s) in next chunk
                    if overlap_chars > 0 and current_chunk:
                        overlap_sentences = []
                        overlap_len = 0
                        for s in reversed(current_chunk):
                            if overlap_len + len(s) <= overlap_chars:
                                overlap_sentences.insert(0, s)
                                overlap_len += len(s) + 1
                            else:
                                break
                        current_chunk = overlap_sentences
                        current_length = overlap_len
                    else:
                        current_chunk = []
                        current_length = 0

            current_chunk.append(sentence)
            current_length += sentence_len + 1

        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def sentences(self, text):
        raise NotImplementedError()

    def atom_gender(self, atom):
        raise NotImplementedError()

    def atom_number(self, atom):
        raise NotImplementedError()

    def atom_person(self, atom):
        raise NotImplementedError()

    def atom_animacy(self, atom):
        raise NotImplementedError()

    def _parse_token(self, token, atom_type):
        raise NotImplementedError()

    def _parse(self, text):
        raise NotImplementedError()

    def _set_edge_tokens(self, edge, hg, parse):
        raise NotImplementedError()

    def _poss2text(self, edge, parse):
        raise NotImplementedError()

    def _resolve_corefs(self, parse_results):
        # do nothing if not implemented in derived classes
        for parse in parse_results['parses']:
            parse['resolved_corefs'] = parse['main_edge']
