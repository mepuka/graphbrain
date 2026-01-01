"""Structured claim extraction with modality analysis.

Extracts claims with additional metadata:
- certainty: definite, probable, possible
- source_type: direct, reported, inferred
- temporal: past, present, future
- polarity: positive, negative

Uses linguistic patterns to detect modality markers in claim edges.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Set, Dict, List, Tuple

from graphbrain.processor import Processor
from graphbrain.utils.corefs import main_coref
from graphbrain.utils.concepts import has_proper_concept, strip_concept
from graphbrain.utils.lemmas import deep_lemma

logger = logging.getLogger(__name__)


class Certainty(str, Enum):
    """Certainty level of a claim."""
    DEFINITE = "definite"      # is, does, will (factual)
    PROBABLE = "probable"      # should, would, likely
    POSSIBLE = "possible"      # may, might, could, perhaps


class SourceType(str, Enum):
    """How the claim was sourced."""
    DIRECT = "direct"          # Speaker's own statement
    REPORTED = "reported"      # Attribution to another source
    INFERRED = "inferred"      # Derived/implied claim


class Temporal(str, Enum):
    """Temporal reference of the claim."""
    PAST = "past"
    PRESENT = "present"
    FUTURE = "future"


class Polarity(str, Enum):
    """Polarity of the claim."""
    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass
class ClaimModality:
    """Modality metadata for a claim."""
    certainty: Certainty = Certainty.DEFINITE
    source_type: SourceType = SourceType.DIRECT
    temporal: Temporal = Temporal.PRESENT
    polarity: Polarity = Polarity.POSITIVE
    confidence: float = 0.8
    modality_markers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "certainty": self.certainty.value,
            "source_type": self.source_type.value,
            "temporal": self.temporal.value,
            "polarity": self.polarity.value,
            "confidence": self.confidence,
            "markers": self.modality_markers,
        }


# Linguistic patterns for modality detection

# Certainty markers
POSSIBLE_MARKERS = {
    "may", "might", "could", "perhaps", "possibly", "maybe",
    "uncertain", "unclear", "potential", "potentially",
}
PROBABLE_MARKERS = {
    "should", "would", "likely", "probably", "expected",
    "appears", "seems", "suggests", "indicates",
}
DEFINITE_MARKERS = {
    "is", "are", "was", "were", "will", "does", "do", "did",
    "confirms", "announced", "stated", "declared", "proved",
}

# Source type markers (predicates indicating reported speech)
REPORTED_PREDICATES = {
    "say", "said", "claim", "claimed", "report", "reported",
    "allege", "alleged", "according", "told", "tells",
    "announce", "announced", "state", "stated",
}
INFERRED_PREDICATES = {
    "suggest", "suggests", "imply", "implies", "indicate", "indicates",
    "appear", "appears", "seem", "seems", "likely",
}

# Temporal markers
PAST_MARKERS = {
    "yesterday", "ago", "last", "previous", "former", "was", "were",
    "did", "had", "used", "past",
}
FUTURE_MARKERS = {
    "tomorrow", "will", "shall", "going", "next", "upcoming",
    "future", "soon", "eventually", "later",
}
PRESENT_MARKERS = {
    "now", "today", "currently", "is", "are", "present",
    "ongoing", "active",
}

# Negation markers
NEGATION_MARKERS = {
    "not", "n't", "no", "never", "none", "neither", "nor",
    "nothing", "nobody", "nowhere", "without", "refuse",
    "deny", "denied", "reject", "rejected", "oppose", "opposed",
}


class StructuredClaims(Processor):
    """Extract claims with structured modality metadata.

    Enhances basic claim extraction with certainty, source type,
    temporal reference, and polarity analysis.
    """

    def __init__(
        self,
        hg,
        sequence=None,
        use_adaptive: bool = True,
        similarity_threshold: float = 0.6,
    ):
        """Initialize the StructuredClaims processor.

        Args:
            hg: Hypergraph instance
            sequence: Optional sequence number
            use_adaptive: If True, use PredicateAnalyzer for claim predicates
            similarity_threshold: Threshold for semantic similarity
        """
        super().__init__(hg=hg, sequence=sequence)
        self.use_adaptive = use_adaptive
        self.similarity_threshold = similarity_threshold

        # Claim storage
        self.claims: List[Dict] = []
        self.modality_stats = Counter()
        self._claim_predicates: Optional[Set[str]] = None
        self._analyzer = None

    @property
    def claim_predicates(self) -> Set[str]:
        """Get claim predicates, using adaptive discovery if available."""
        if self._claim_predicates is not None:
            return self._claim_predicates

        base_predicates = {"say", "claim", "announce", "state", "declare"}

        if not self.use_adaptive:
            self._claim_predicates = base_predicates
            return self._claim_predicates

        try:
            from graphbrain.processors.adaptive_predicates import PredicateAnalyzer

            if self._analyzer is None:
                self._analyzer = PredicateAnalyzer(self.hg)
                self._analyzer.discover_predicates(min_frequency=2)

            claim_preds = set(self._analyzer.get_predicates_by_category(
                'claim', threshold=self.similarity_threshold))
            self._claim_predicates = claim_preds | base_predicates

            logger.info(
                f"Structured claims using {len(self._claim_predicates)} predicates"
            )

        except Exception as e:
            logger.warning(f"Adaptive discovery failed: {e}")
            self._claim_predicates = base_predicates

        return self._claim_predicates

    def _extract_atoms_text(self, edge) -> Set[str]:
        """Extract text from all atoms in an edge."""
        texts = set()
        if edge.is_atom():
            texts.add(edge.root().lower())
        else:
            for atom in edge.all_atoms():
                texts.add(atom.root().lower())
        return texts

    def _detect_certainty(self, edge, predicate_lemma: str) -> Tuple[Certainty, List[str]]:
        """Detect certainty level from edge content."""
        atoms = self._extract_atoms_text(edge)
        markers = []

        # Check for possible markers
        possible_found = atoms & POSSIBLE_MARKERS
        if possible_found:
            markers.extend(possible_found)
            return Certainty.POSSIBLE, markers

        # Check for probable markers
        probable_found = atoms & PROBABLE_MARKERS
        if probable_found:
            markers.extend(probable_found)
            return Certainty.PROBABLE, markers

        # Check predicate itself
        if predicate_lemma in POSSIBLE_MARKERS:
            return Certainty.POSSIBLE, [predicate_lemma]
        if predicate_lemma in PROBABLE_MARKERS:
            return Certainty.PROBABLE, [predicate_lemma]

        # Default to definite
        return Certainty.DEFINITE, []

    def _detect_source_type(self, predicate_lemma: str, edge) -> Tuple[SourceType, List[str]]:
        """Detect source type from predicate and context."""
        markers = []

        # Check if predicate indicates reported speech
        if predicate_lemma in REPORTED_PREDICATES:
            return SourceType.REPORTED, [predicate_lemma]

        # Check if predicate indicates inference
        if predicate_lemma in INFERRED_PREDICATES:
            return SourceType.INFERRED, [predicate_lemma]

        # Check for "according to" pattern in atoms
        atoms = self._extract_atoms_text(edge)
        if "according" in atoms:
            return SourceType.REPORTED, ["according"]

        return SourceType.DIRECT, []

    def _detect_temporal(self, edge) -> Tuple[Temporal, List[str]]:
        """Detect temporal reference from edge content."""
        atoms = self._extract_atoms_text(edge)
        markers = []

        # Check for past markers
        past_found = atoms & PAST_MARKERS
        if past_found:
            markers.extend(past_found)
            return Temporal.PAST, markers

        # Check for future markers
        future_found = atoms & FUTURE_MARKERS
        if future_found:
            markers.extend(future_found)
            return Temporal.FUTURE, markers

        # Check verb features for tense
        if edge.not_atom:
            pred = edge[0]
            if pred.is_atom():
                features = pred.parts() if hasattr(pred, 'parts') else []
                # Check for past tense markers in atom features
                atom_str = str(pred)
                if ".<f" in atom_str or ".<p" in atom_str:
                    return Temporal.PAST, ["verb_past"]
                if ".<t" in atom_str:
                    return Temporal.FUTURE, ["verb_future"]

        return Temporal.PRESENT, []

    def _detect_polarity(self, edge) -> Tuple[Polarity, List[str]]:
        """Detect polarity (positive/negative) from edge content."""
        atoms = self._extract_atoms_text(edge)
        markers = []

        # Check for negation markers
        negation_found = atoms & NEGATION_MARKERS
        if negation_found:
            markers.extend(negation_found)
            return Polarity.NEGATIVE, markers

        # Check for negation in predicate structure
        if edge.not_atom:
            pred = edge[0]
            atom_str = str(pred)
            # Negation often marked in atom features
            if ".n" in atom_str:
                return Polarity.NEGATIVE, ["pred_negation"]

        return Polarity.POSITIVE, []

    def _analyze_modality(self, edge, predicate_lemma: str) -> ClaimModality:
        """Perform full modality analysis on an edge."""
        all_markers = []

        certainty, cert_markers = self._detect_certainty(edge, predicate_lemma)
        all_markers.extend(cert_markers)

        source_type, source_markers = self._detect_source_type(predicate_lemma, edge)
        all_markers.extend(source_markers)

        temporal, temp_markers = self._detect_temporal(edge)
        all_markers.extend(temp_markers)

        polarity, pol_markers = self._detect_polarity(edge)
        all_markers.extend(pol_markers)

        # Calculate confidence based on markers found
        marker_count = len(all_markers)
        confidence = min(0.95, 0.6 + marker_count * 0.1)

        return ClaimModality(
            certainty=certainty,
            source_type=source_type,
            temporal=temporal,
            polarity=polarity,
            confidence=confidence,
            modality_markers=all_markers,
        )

    def process_edge(self, edge):
        """Process edges to extract claims with modality."""
        if edge.not_atom:
            ct = edge.connector_type()
            if ct[0] == 'P':
                pred = edge[0]
                pred_lemma = deep_lemma(
                    self.hg, pred, same_if_none=True
                ).root()

                if (len(edge) > 2 and pred_lemma in self.claim_predicates):
                    subjects = edge.edges_with_argrole('s')
                    claim_edges = edge.edges_with_argrole('r')

                    if len(subjects) == 1 and len(claim_edges) >= 1:
                        subject = strip_concept(subjects[0])
                        if subject and has_proper_concept(subject):
                            try:
                                actor = main_coref(self.hg, subject)

                                for claim_edge in claim_edges:
                                    # Handle specification type
                                    if claim_edge.mtype() == 'S':
                                        claim_content = claim_edge[1]
                                    else:
                                        claim_content = claim_edge

                                    # Analyze modality
                                    modality = self._analyze_modality(
                                        claim_content, pred_lemma
                                    )

                                    self.claims.append({
                                        "actor": actor,
                                        "claim": claim_content,
                                        "edge": edge,
                                        "predicate": pred_lemma,
                                        "modality": modality,
                                    })

                                    # Update stats
                                    self.modality_stats[modality.certainty] += 1
                                    self.modality_stats[modality.source_type] += 1
                                    self.modality_stats[modality.temporal] += 1
                                    self.modality_stats[modality.polarity] += 1

                            except Exception as e:
                                logger.debug(f"Error processing claim: {e}")

    def on_end(self):
        """Write structured claims to hypergraph."""
        logger.info(f"Writing {len(self.claims)} structured claims")

        for claim_data in self.claims:
            actor = claim_data["actor"]
            claim = claim_data["claim"]
            edge = claim_data["edge"]
            modality = claim_data["modality"]

            # Store basic claim edge
            self.hg.add(('claim/P/.', actor, claim, edge))

            # Store modality attributes as separate edges
            # Certainty: (certainty/P/. claim certainty_value)
            certainty_atom = f"{modality.certainty.value}/Ca/."
            self.hg.add(('certainty/P/.', claim, certainty_atom))

            # Source type
            source_atom = f"{modality.source_type.value}/Ca/."
            self.hg.add(('source_type/P/.', claim, source_atom))

            # Temporal
            temporal_atom = f"{modality.temporal.value}/Ca/."
            self.hg.add(('temporal/P/.', claim, temporal_atom))

            # Polarity
            polarity_atom = f"{modality.polarity.value}/Ca/."
            self.hg.add(('polarity/P/.', claim, polarity_atom))

    def report(self) -> str:
        """Generate processing report."""
        parts = [f"structured claims: {len(self.claims)}"]

        # Certainty breakdown
        certainty_counts = [
            f"definite: {self.modality_stats.get(Certainty.DEFINITE, 0)}",
            f"probable: {self.modality_stats.get(Certainty.PROBABLE, 0)}",
            f"possible: {self.modality_stats.get(Certainty.POSSIBLE, 0)}",
        ]
        parts.append(f"certainty ({', '.join(certainty_counts)})")

        # Polarity breakdown
        pos = self.modality_stats.get(Polarity.POSITIVE, 0)
        neg = self.modality_stats.get(Polarity.NEGATIVE, 0)
        parts.append(f"polarity (positive: {pos}, negative: {neg})")

        return "; ".join(parts)
