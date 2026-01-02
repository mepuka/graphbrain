import logging
from collections import Counter
from typing import Optional, Set

import progressbar

from graphbrain import hedge

logger = logging.getLogger(__name__)
from graphbrain.utils.corefs import main_coref
from graphbrain.processor import Processor
from graphbrain.utils.concepts import has_proper_concept, strip_concept
from graphbrain.utils.lemmas import deep_lemma


# Fallback predicate set when semantic similarity is unavailable
CLAIM_PRED_LEMMAS = {'say', 'claim'}


def _subject_preposition(claim):
    subjects = claim.edges_with_argrole('s')
    if len(subjects) == 1:
        subject = strip_concept(subjects[0])
        if subject:
            atom = subject.atom_with_type('C')
            return atom.root()
    return None


def replace_subject(edge, new_subject):
    connector = edge[0]
    new_edge = list(edge)

    for pos, role in enumerate(connector.argroles()):
        if role == 's':
            new_edge[pos + 1] = new_subject
    return hedge(new_edge)


class Claims(Processor):
    """Extract claims from hypergraph edges.

    Uses PredicateAnalyzer for dynamic predicate discovery when available,
    falling back to hardcoded CLAIM_PRED_LEMMAS otherwise.
    """

    def __init__(self, hg, sequence=None, use_adaptive: bool = True,
                 similarity_threshold: float = 0.6):
        """Initialize the Claims processor.

        Args:
            hg: Hypergraph instance
            sequence: Optional sequence number
            use_adaptive: If True, use PredicateAnalyzer for dynamic discovery
            similarity_threshold: Threshold for semantic similarity matching
        """
        super().__init__(hg=hg, sequence=sequence)
        self.actors = set()
        self.female = None
        self.group = None
        self.male = None
        self.non_human = None
        self.female_counter = Counter()
        self.group_counter = Counter()
        self.male_counter = Counter()
        self.non_human_counter = Counter()
        self.claims = []
        self.anaphoras = 0
        self.use_adaptive = use_adaptive
        self.similarity_threshold = similarity_threshold
        self._claim_predicates: Optional[Set[str]] = None
        self._analyzer = None
        self._predicates_discovered = 0

    @property
    def claim_predicates(self) -> Set[str]:
        """Get claim predicates, using adaptive discovery if available."""
        if self._claim_predicates is not None:
            return self._claim_predicates

        if not self.use_adaptive:
            self._claim_predicates = CLAIM_PRED_LEMMAS
            return self._claim_predicates

        try:
            from graphbrain.processors.adaptive_predicates import PredicateAnalyzer

            if self._analyzer is None:
                self._analyzer = PredicateAnalyzer(self.hg)
                self._analyzer.discover_predicates(min_frequency=2)

            # Get claim predicates via semantic similarity
            claim_preds = set(self._analyzer.get_predicates_by_category(
                'claim', threshold=self.similarity_threshold))

            # Always include the seed predicates
            self._claim_predicates = claim_preds | CLAIM_PRED_LEMMAS
            self._predicates_discovered = len(claim_preds - CLAIM_PRED_LEMMAS)

            logger.info(
                f"Claim extraction using {len(self._claim_predicates)} predicates "
                f"({self._predicates_discovered} discovered via similarity)"
            )

        except Exception as e:
            logger.warning(f"Adaptive predicate discovery failed: {e}, using fallback")
            self._claim_predicates = CLAIM_PRED_LEMMAS

        return self._claim_predicates

    def _gender(self, actor):
        counts = (('female', self.female_counter[actor]),
                  ('group', self.group_counter[actor]),
                  ('male', self.male_counter[actor]),
                  ('non-human', self.non_human_counter[actor]))
        counts = sorted(counts, key=lambda x: x[1], reverse=True)
        if counts[0][1] > 0 and counts[0][1] > counts[1][1]:
            return counts[0][0]
        else:
            return None

    def _process_claim(self, actor, claim, edge):
        # gender detection
        prep = _subject_preposition(claim)
        if prep and set(edge[0].argroles()) == {'s', 'r'}:
            if prep == 'she':
                # print('she {}'.format(actor))
                self.female_counter[actor] += 1
            elif prep == 'they':
                # print('they {}'.format(actor))
                self.group_counter[actor] += 1
            elif prep == 'he':
                # print('he {}'.format(actor))
                self.male_counter[actor] += 1
            elif prep == 'it':
                # print('it {}'.format(actor))
                self.non_human_counter[actor] += 1

        # record claim
        self.claims.append({'actor': actor, 'claim': claim, 'edge': edge})

    def process_edge(self, edge):
        if edge.not_atom:
            ct = edge.connector_type()
            if ct[0] == 'P':
                pred = edge[0]
                lemma_edge = deep_lemma(self.hg, pred, same_if_none=True)
                if lemma_edge is None:
                    return
                if len(edge) > 2 and lemma_edge.root() in self.claim_predicates:
                    subjects = edge.edges_with_argrole('s')
                    claims = edge.edges_with_argrole('r')
                    if len(subjects) == 1 and len(claims) >= 1:
                        subject = strip_concept(subjects[0])
                        if subject and has_proper_concept(subject):
                            actor = main_coref(self.hg, subject)
                            self.actors.add(actor)
                            for claim in claims:
                                # if specificatin, claim is inside
                                if claim.mtype() == 'S':
                                    self._process_claim(actor, claim[1], edge)
                                else:
                                    self._process_claim(actor, claim, edge)

    def on_end(self):
        # assign genders
        self.female = set()
        self.group = set()
        self.male = set()
        self.non_human = set()

        logger.info('assigning genders')
        i = 0
        with progressbar.ProgressBar(max_value=len(self.actors)) as bar:
            for actor in self.actors:
                gender = self._gender(actor)
                if gender == 'female':
                    self.female.add(actor)
                elif gender == 'group':
                    self.group.add(actor)
                elif gender == 'male':
                    self.male.add(actor)
                elif gender == 'non-human':
                    self.non_human.add(actor)

                # write gender
                if gender:
                    gender_atom = '{}/P/.'.format(gender)
                    self.hg.add((gender_atom, actor))

                i += 1
                bar.update(i)

        # write claims
        logger.info('writing claims')
        i = 0
        with progressbar.ProgressBar(max_value=len(self.claims)) as bar:
            for claim_data in self.claims:
                actor = claim_data['actor']
                claim = claim_data['claim']
                edge = claim_data['edge']

                # anaphora resolution
                prep = _subject_preposition(claim)
                if prep:
                    resolve = False
                    if prep == 'she':
                        resolve = actor in self.female
                    elif prep == 'they':
                        resolve = actor in self.group
                    elif prep == 'he':
                        resolve = actor in self.male
                    elif prep == 'it':
                        resolve = actor in self.non_human

                    if resolve:
                        logger.debug('ANAPHORA resolution: actor=%s, before=%s', actor, claim)
                        claim = replace_subject(claim, actor)
                        logger.debug('ANAPHORA resolution: after=%s', claim)
                        self.anaphoras += 1

                # write claim
                self.hg.add(('claim/P/.', actor, claim, edge))

                i += 1
                bar.update(i)

    def report(self):
        rep_claims = 'claims: {}'.format(len(self.claims))
        rep_anaph = 'anaphora resolutions: {}'.format(self.anaphoras)
        counts = (len(self.female), len(self.group), len(self.male),
                  len(self.non_human))
        cs = tuple([str(x) for x in counts])
        rep_gen = 'female: {}; group: {}; male: {}; non-human: {}'.format(*cs)
        base = '; '.join((rep_claims, rep_anaph, rep_gen))
        if self._predicates_discovered > 0:
            return f'{base} (using {self._predicates_discovered} discovered predicates)'
        return base
