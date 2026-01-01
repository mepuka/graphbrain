"""Entity type classification processor.

Classifies proper nouns (Cp atoms) into entity types:
- person: Individual humans (Mayor Wilson, Dr. Smith)
- organization: Companies, agencies (Seattle City Council, SDOT)
- location: Geographic places (Seattle, Ballard, I-5)
- group: Collective entities (residents, protesters, advocates)

Uses heuristic pattern matching first, with optional LLM fallback
for uncertain cases.
"""

import logging
import re
from collections import Counter
from enum import Enum
from typing import Optional, Set, Dict, List, Tuple

from graphbrain.processor import Processor
from graphbrain.utils.corefs import main_coref

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """Entity type categories."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    GROUP = "group"
    UNKNOWN = "unknown"


# Heuristic indicators for each entity type
TYPE_INDICATORS: Dict[EntityType, List[str]] = {
    EntityType.PERSON: [
        # Titles
        "mayor", "director", "ceo", "president", "chair", "chairman",
        "councilmember", "senator", "representative", "governor",
        "dr", "mr", "mrs", "ms", "prof", "professor",
        # Role words that often co-occur with persons
        "said", "announced", "believes", "argues", "stated", "told",
    ],
    EntityType.ORGANIZATION: [
        # Suffixes
        "inc", "corp", "llc", "ltd", "co", "company",
        # Government bodies
        "council", "department", "dept", "agency", "commission",
        "authority", "board", "office", "bureau", "administration",
        # Other org types
        "association", "foundation", "institute", "organization",
        "coalition", "alliance", "committee", "group",
        "university", "college", "school", "hospital",
    ],
    EntityType.LOCATION: [
        # Street types
        "street", "st", "avenue", "ave", "boulevard", "blvd",
        "road", "rd", "drive", "dr", "lane", "ln", "way",
        # Area types
        "city", "town", "county", "state", "district",
        "neighborhood", "region", "area", "zone",
        # Transit
        "station", "terminal", "airport", "port",
        # Geographic features
        "park", "lake", "river", "mountain", "bay", "island",
    ],
    EntityType.GROUP: [
        # Collective nouns
        "residents", "protesters", "demonstrators", "activists",
        "members", "voters", "advocates", "critics", "supporters",
        "opponents", "community", "citizens", "taxpayers",
        "workers", "employees", "staff", "team", "crew",
        "neighbors", "homeowners", "renters", "tenants",
    ],
}

# Patterns that strongly indicate person names
PERSON_NAME_PATTERNS = [
    # First + Last name
    r"^[A-Z][a-z]+ [A-Z][a-z]+$",
    # First Middle Last
    r"^[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+$",
    # Title + Name
    r"^(Mayor|Dr|Mr|Mrs|Ms|Prof|Sen|Rep|Gov)\b",
]

# Patterns that strongly indicate organizations
ORG_PATTERNS = [
    # Acronyms (all caps, 2-5 letters)
    r"^[A-Z]{2,5}$",
    # "X of Y" pattern common for orgs
    r"^(Department|Office|Bureau|Board|Commission) of",
    # "City/County of X"
    r"^(City|County|State) of [A-Z]",
]

# Patterns that indicate locations
LOCATION_PATTERNS = [
    # Interstate/Route
    r"^I-\d+$",
    r"^(US|State|County) (Route|Highway|Road) \d+$",
    r"^SR[ -]?\d+$",
    # Cardinal directions with names
    r"^(North|South|East|West|NE|NW|SE|SW) [A-Z]",
]


class EntityTypes(Processor):
    """Classify actors and proper nouns into entity types.

    Uses heuristic pattern matching with optional LLM fallback.
    Stores classifications as hyperedges: (entity_type/P/. entity)
    """

    def __init__(
        self,
        hg,
        sequence=None,
        use_llm: bool = False,
        llm_provider=None,
        confidence_threshold: float = 0.6,
    ):
        """Initialize the EntityTypes processor.

        Args:
            hg: Hypergraph instance
            sequence: Optional sequence number
            use_llm: If True, use LLM for uncertain classifications
            llm_provider: LLM provider instance (required if use_llm=True)
            confidence_threshold: Threshold for heuristic confidence
        """
        super().__init__(hg=hg, sequence=sequence)
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.confidence_threshold = confidence_threshold

        # Track classifications
        self.classifications: Dict[str, Tuple[EntityType, float]] = {}
        self.entity_contexts: Dict[str, Dict] = {}
        self._type_counter = Counter()
        self._uncertain_count = 0

    def _get_entity_text(self, entity) -> str:
        """Extract readable text from an entity edge."""
        if entity.is_atom():
            return entity.root().replace("_", " ")
        else:
            # Compound name - join parts from all atoms
            parts = []
            for atom in entity.all_atoms():
                if atom.type()[0] == 'C':
                    parts.append(atom.root().replace("_", " "))
            return " ".join(parts) if parts else str(entity)

    def _get_predicates_for_entity(self, entity) -> Set[str]:
        """Get predicates used with this entity."""
        predicates = set()
        try:
            # Find edges containing this entity
            for edge in self.hg.edges_with_root(entity.root() if entity.is_atom else str(entity)):
                if edge.not_atom and edge[0].type()[0] == 'P':
                    predicates.add(edge[0].root())
        except Exception as e:
            logger.debug(f"Error getting predicates for {entity}: {e}")
        return predicates

    def _classify_heuristic(
        self,
        entity_text: str,
        predicates: Set[str],
    ) -> Tuple[EntityType, float]:
        """Classify entity using heuristic patterns.

        Returns:
            Tuple of (EntityType, confidence)
        """
        text_lower = entity_text.lower()
        words = text_lower.split()
        word_set = set(words)
        last_word = words[-1] if words else ""
        scores: Dict[EntityType, float] = {t: 0.0 for t in EntityType}

        # Check indicator words first - gives context for pattern matching
        predicate_set = {p.lower() for p in predicates}
        has_location_indicator = False
        has_org_indicator = False
        has_group_indicator = False

        for entity_type, indicators in TYPE_INDICATORS.items():
            for indicator in indicators:
                indicator_lower = indicator.lower()
                # Check if indicator is a complete word in entity name
                if indicator_lower in word_set:
                    # Full word match - higher score if it's the last word
                    if indicator_lower == last_word:
                        scores[entity_type] += 0.45  # Strong signal
                    else:
                        scores[entity_type] += 0.3
                    if entity_type == EntityType.LOCATION:
                        has_location_indicator = True
                    elif entity_type == EntityType.ORGANIZATION:
                        has_org_indicator = True
                    elif entity_type == EntityType.GROUP:
                        has_group_indicator = True
                elif indicator_lower in text_lower:
                    # Substring match - only count if word boundary
                    pattern = r'\b' + re.escape(indicator_lower) + r'\b'
                    if re.search(pattern, text_lower):
                        scores[entity_type] += 0.25
                # Check in predicates (for person detection)
                if indicator_lower in predicate_set:
                    scores[entity_type] += 0.15

        # Check regex patterns - but context from indicators matters
        for pattern in PERSON_NAME_PATTERNS:
            if re.search(pattern, entity_text):
                # Don't apply person pattern if entity has location/org indicators
                if not has_location_indicator and not has_org_indicator and not has_group_indicator:
                    scores[EntityType.PERSON] += 0.4

        for pattern in ORG_PATTERNS:
            if re.search(pattern, entity_text):
                scores[EntityType.ORGANIZATION] += 0.4

        for pattern in LOCATION_PATTERNS:
            if re.search(pattern, entity_text):
                scores[EntityType.LOCATION] += 0.4

        # Find best match
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        # Normalize and apply minimum threshold
        if best_score < 0.2:
            return EntityType.UNKNOWN, 0.0

        # Calculate confidence based on margin over second best
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
        confidence = min(0.95, best_score + margin * 0.5)

        return best_type, confidence

    def process_edge(self, edge):
        """Process edges to find and classify entities."""
        if edge.not_atom:
            # Look for proper noun concepts
            for item in edge.all_atoms():
                if item.type() == 'Cp':
                    # Get the main coref for this entity
                    try:
                        entity = main_coref(self.hg, item)
                        entity_key = str(entity)

                        # Skip if already classified
                        if entity_key in self.classifications:
                            continue

                        # Get context
                        entity_text = self._get_entity_text(entity)
                        predicates = self._get_predicates_for_entity(entity)

                        # Store context for potential LLM use
                        self.entity_contexts[entity_key] = {
                            "text": entity_text,
                            "predicates": list(predicates),
                            "entity": entity,
                        }

                        # Classify using heuristics
                        entity_type, confidence = self._classify_heuristic(
                            entity_text, predicates
                        )

                        self.classifications[entity_key] = (entity_type, confidence)
                        self._type_counter[entity_type] += 1

                        if confidence < self.confidence_threshold:
                            self._uncertain_count += 1

                    except Exception as e:
                        logger.debug(f"Error processing entity {item}: {e}")

    async def _classify_with_llm(self, entity_key: str) -> Tuple[EntityType, float]:
        """Classify a single entity using LLM."""
        if not self.use_llm or not self.llm_provider:
            return EntityType.UNKNOWN, 0.0

        context = self.entity_contexts.get(entity_key)
        if not context:
            return EntityType.UNKNOWN, 0.0

        try:
            from graphbrain.agents.skills.llm_entity_typing import LLMEntityTypingSkill
            from graphbrain.agents.llm.models import EntityType as LLMEntityType

            skill = LLMEntityTypingSkill(provider=self.llm_provider)
            result = await skill.type_entity(
                entity=context["text"],
                predicates=context["predicates"],
            )

            if result.success:
                # Map LLM EntityType to our EntityType
                llm_type = result.data.entity_type
                if isinstance(llm_type, LLMEntityType):
                    llm_type = llm_type.value

                type_map = {
                    "person": EntityType.PERSON,
                    "organization": EntityType.ORGANIZATION,
                    "location": EntityType.LOCATION,
                    "group": EntityType.GROUP,
                    "event": EntityType.GROUP,  # Map event to group
                    "unknown": EntityType.UNKNOWN,
                }
                return type_map.get(llm_type, EntityType.UNKNOWN), result.confidence

        except Exception as e:
            logger.warning(f"LLM classification failed for {entity_key}: {e}")

        return EntityType.UNKNOWN, 0.0

    def on_end(self):
        """Write entity type classifications to hypergraph."""
        logger.info(f"Writing {len(self.classifications)} entity type classifications")

        for entity_key, (entity_type, confidence) in self.classifications.items():
            if entity_type == EntityType.UNKNOWN:
                continue

            context = self.entity_contexts.get(entity_key)
            if not context:
                continue

            entity = context["entity"]

            # Store as: (entity_type/P/. entity)
            type_atom = f"{entity_type.value}/P/."
            self.hg.add((type_atom, entity))

            logger.debug(
                f"Classified '{context['text']}' as {entity_type.value} "
                f"(confidence: {confidence:.2f})"
            )

    def report(self) -> str:
        """Generate classification report."""
        total = len(self.classifications)
        by_type = ", ".join(
            f"{t.value}: {c}" for t, c in self._type_counter.most_common()
        )
        uncertain = self._uncertain_count

        parts = [f"entities classified: {total}"]
        if by_type:
            parts.append(f"by type: {by_type}")
        if uncertain > 0:
            parts.append(f"uncertain: {uncertain}")

        return "; ".join(parts)
