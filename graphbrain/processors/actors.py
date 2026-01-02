import logging
from collections import Counter
from typing import Optional, Set

from graphbrain.utils.corefs import main_coref

logger = logging.getLogger(__name__)
from graphbrain.processor import Processor
from graphbrain.utils.concepts import has_proper_concept, strip_concept
from graphbrain.utils.lemmas import deep_lemma


def is_actor(hg, edge):
    """Checks if the edge is a coreference to an actor."""
    if edge.mtype() == 'C':
        return hg.exists(('actor/P/.', main_coref(hg, edge)))
    else:
        return False


def find_actors(hg, edge):
    """Returns set of all coreferences to actors found in the edge."""
    _actors = set()
    if is_actor(hg, edge):
        _actors.add(main_coref(hg, edge))
    if edge.not_atom:
        for item in edge:
            _actors |= find_actors(hg, item)
    return _actors


def actors(hg):
    """"Returns an iterator over all actors."""
    return [edge[1] for edge in hg.search('(actor/P/. *)', strict=True)]


# Fallback predicate set when semantic similarity is unavailable
ACTOR_PRED_LEMMAS = {'say', 'claim', 'warn', 'kill', 'accuse', 'condemn',
                     'slam', 'arrest', 'clash', 'blame', 'want', 'call',
                     'tell'}


class Actors(Processor):
    """Extract actors from hypergraph edges.

    Uses PredicateAnalyzer for dynamic predicate discovery when available,
    falling back to hardcoded ACTOR_PRED_LEMMAS otherwise.
    """

    def __init__(self, hg, sequence=None, use_adaptive: bool = True,
                 similarity_threshold: float = 0.6):
        """Initialize the Actors processor.

        Args:
            hg: Hypergraph instance
            sequence: Optional sequence number
            use_adaptive: If True, use PredicateAnalyzer for dynamic discovery
            similarity_threshold: Threshold for semantic similarity matching
        """
        super().__init__(hg=hg, sequence=sequence)
        self.actor_counter = Counter()
        self.use_adaptive = use_adaptive
        self.similarity_threshold = similarity_threshold
        self._action_predicates: Optional[Set[str]] = None
        self._analyzer = None
        self._predicates_discovered = 0

    @property
    def action_predicates(self) -> Set[str]:
        """Get action predicates, using adaptive discovery if available."""
        if self._action_predicates is not None:
            return self._action_predicates

        if not self.use_adaptive:
            self._action_predicates = ACTOR_PRED_LEMMAS
            return self._action_predicates

        try:
            from graphbrain.processors.adaptive_predicates import PredicateAnalyzer

            if self._analyzer is None:
                self._analyzer = PredicateAnalyzer(self.hg)
                self._analyzer.discover_predicates(min_frequency=2)

            # Combine action, claim, and conflict predicates for actor detection
            action_preds = set(self._analyzer.get_predicates_by_category(
                'action', threshold=self.similarity_threshold))
            claim_preds = set(self._analyzer.get_predicates_by_category(
                'claim', threshold=self.similarity_threshold))
            conflict_preds = set(self._analyzer.get_predicates_by_category(
                'conflict', threshold=self.similarity_threshold))

            discovered = action_preds | claim_preds | conflict_preds

            # Always include the seed predicates
            self._action_predicates = discovered | ACTOR_PRED_LEMMAS
            self._predicates_discovered = len(discovered - ACTOR_PRED_LEMMAS)

            logger.info(
                f"Actor extraction using {len(self._action_predicates)} predicates "
                f"({self._predicates_discovered} discovered via similarity)"
            )

        except Exception as e:
            logger.warning(f"Adaptive predicate discovery failed: {e}, using fallback")
            self._action_predicates = ACTOR_PRED_LEMMAS

        return self._action_predicates

    def process_edge(self, edge):
        if edge.not_atom:
            ct = edge.connector_type()
            if ct[0] == 'P':
                subjects = edge.edges_with_argrole('s')
                if len(subjects) == 1:
                    subject = strip_concept(subjects[0])
                    if subject and has_proper_concept(subject):
                        pred = edge[0]
                        lemma_edge = deep_lemma(self.hg, pred, same_if_none=True)
                        if lemma_edge is None:
                            return
                        dlemma = lemma_edge.root()
                        if dlemma in self.action_predicates:
                            try:
                                actor = main_coref(self.hg, subject)
                                self.actor_counter[actor] += 1
                            except Exception as e:
                                logger.warning('Error processing actor: %s', e)

    def on_end(self):
        for actor in self.actor_counter:
            self.hg.add(('actor/P/.', actor))

    def report(self):
        base = f'actors found: {len(self.actor_counter)}'
        if self._predicates_discovered > 0:
            return f'{base} (using {self._predicates_discovered} discovered predicates)'
        return base
