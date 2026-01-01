"""Tests for EntityTypes processor."""

import pytest
from graphbrain import hgraph, hedge

from graphbrain.processors.entity_types import (
    EntityTypes,
    EntityType,
    TYPE_INDICATORS,
    PERSON_NAME_PATTERNS,
    ORG_PATTERNS,
    LOCATION_PATTERNS,
)


class TestEntityTypeEnum:
    """Test EntityType enum."""

    def test_all_types_defined(self):
        """Test all expected entity types exist."""
        expected = ["person", "organization", "location", "group", "unknown"]
        actual = [t.value for t in EntityType]
        assert set(expected) == set(actual)

    def test_string_enum(self):
        """Test EntityType is a string enum."""
        assert EntityType.PERSON.value == "person"
        assert EntityType.ORGANIZATION.value == "organization"
        assert str(EntityType.PERSON) == "EntityType.PERSON"


class TestTypeIndicators:
    """Test type indicator patterns."""

    def test_person_indicators(self):
        """Test person indicators include expected terms."""
        indicators = TYPE_INDICATORS[EntityType.PERSON]
        assert "mayor" in indicators
        assert "said" in indicators
        assert "president" in indicators

    def test_organization_indicators(self):
        """Test organization indicators."""
        indicators = TYPE_INDICATORS[EntityType.ORGANIZATION]
        assert "council" in indicators
        assert "department" in indicators
        assert "inc" in indicators

    def test_location_indicators(self):
        """Test location indicators."""
        indicators = TYPE_INDICATORS[EntityType.LOCATION]
        assert "city" in indicators
        assert "street" in indicators
        assert "neighborhood" in indicators

    def test_group_indicators(self):
        """Test group indicators."""
        indicators = TYPE_INDICATORS[EntityType.GROUP]
        assert "residents" in indicators
        assert "protesters" in indicators
        assert "community" in indicators


class TestEntityTypesProcessor:
    """Tests for EntityTypes processor."""

    @pytest.fixture
    def hg(self):
        """Create a test hypergraph."""
        return hgraph("test_entity_types.db")

    @pytest.fixture
    def processor(self, hg):
        """Create processor instance."""
        return EntityTypes(hg=hg)

    def test_init_defaults(self, processor):
        """Test default initialization."""
        assert processor.use_llm is False
        assert processor.llm_provider is None
        assert processor.confidence_threshold == 0.6
        assert len(processor.classifications) == 0

    def test_init_with_llm(self, hg):
        """Test initialization with LLM settings."""
        processor = EntityTypes(
            hg=hg,
            use_llm=True,
            confidence_threshold=0.8,
        )
        assert processor.use_llm is True
        assert processor.confidence_threshold == 0.8

    def test_get_entity_text_atom(self, processor):
        """Test extracting text from atom entity."""
        entity = hedge("john_smith/Cp")
        text = processor._get_entity_text(entity)
        assert text == "john smith"

    def test_get_entity_text_compound(self, processor):
        """Test extracting text from compound entity."""
        entity = hedge("(+/B/. john/Cp smith/Cp)")
        text = processor._get_entity_text(entity)
        assert "john" in text
        assert "smith" in text

    def test_classify_heuristic_person_name(self, processor):
        """Test heuristic classification of person names."""
        # Simple first + last name pattern
        entity_type, confidence = processor._classify_heuristic(
            "John Smith", set()
        )
        assert entity_type == EntityType.PERSON
        assert confidence > 0.3

    def test_classify_heuristic_person_title(self, processor):
        """Test heuristic classification with title."""
        entity_type, confidence = processor._classify_heuristic(
            "Mayor Wilson", set()
        )
        assert entity_type == EntityType.PERSON
        assert confidence > 0.3

    def test_classify_heuristic_organization_council(self, processor):
        """Test heuristic classification of councils."""
        entity_type, confidence = processor._classify_heuristic(
            "Seattle City Council", set()
        )
        assert entity_type == EntityType.ORGANIZATION
        assert confidence > 0.2

    def test_classify_heuristic_organization_acronym(self, processor):
        """Test heuristic classification of acronyms."""
        entity_type, confidence = processor._classify_heuristic(
            "SDOT", set()
        )
        assert entity_type == EntityType.ORGANIZATION
        assert confidence > 0.3

    def test_classify_heuristic_location_street(self, processor):
        """Test heuristic classification of streets."""
        entity_type, confidence = processor._classify_heuristic(
            "Main Street", set()
        )
        assert entity_type == EntityType.LOCATION
        assert confidence > 0.2

    def test_classify_heuristic_location_interstate(self, processor):
        """Test heuristic classification of interstates."""
        entity_type, confidence = processor._classify_heuristic(
            "I-5", set()
        )
        assert entity_type == EntityType.LOCATION
        assert confidence > 0.3

    def test_classify_heuristic_group_residents(self, processor):
        """Test heuristic classification of groups."""
        entity_type, confidence = processor._classify_heuristic(
            "Ballard residents", set()
        )
        assert entity_type == EntityType.GROUP
        assert confidence > 0.2

    def test_classify_heuristic_with_predicates(self, processor):
        """Test heuristic classification using predicates."""
        # Person-like predicates boost person classification
        entity_type, confidence = processor._classify_heuristic(
            "Wilson", {"said", "announced"}
        )
        assert entity_type == EntityType.PERSON
        assert confidence > 0.2

    def test_classify_heuristic_unknown(self, processor):
        """Test classification of ambiguous entities."""
        entity_type, confidence = processor._classify_heuristic(
            "xyz", set()
        )
        assert entity_type == EntityType.UNKNOWN
        assert confidence == 0.0

    def test_process_edge_finds_proper_nouns(self, hg, processor):
        """Test processing finds Cp atoms."""
        # Add an edge with a proper noun
        edge = hedge("(says/Pd.sr john_smith/Cp hello/C)")
        hg.add(edge)

        processor.process_edge(edge)

        # Should have classified john_smith
        assert len(processor.classifications) >= 1

    def test_report(self, processor):
        """Test report generation."""
        # Add some mock classifications
        processor.classifications["entity1"] = (EntityType.PERSON, 0.9)
        processor.classifications["entity2"] = (EntityType.ORGANIZATION, 0.8)
        processor._type_counter[EntityType.PERSON] = 1
        processor._type_counter[EntityType.ORGANIZATION] = 1

        report = processor.report()

        assert "entities classified: 2" in report
        assert "person" in report
        assert "organization" in report

    def test_report_with_uncertain(self, processor):
        """Test report includes uncertain count."""
        processor.classifications["entity1"] = (EntityType.PERSON, 0.5)
        processor._type_counter[EntityType.PERSON] = 1
        processor._uncertain_count = 1

        report = processor.report()

        assert "uncertain: 1" in report


class TestPatternMatching:
    """Test regex pattern matching."""

    def test_person_name_pattern_simple(self):
        """Test simple person name pattern."""
        import re
        pattern = PERSON_NAME_PATTERNS[0]  # First + Last
        assert re.search(pattern, "John Smith")
        assert re.search(pattern, "Mary Johnson")
        assert not re.search(pattern, "john smith")  # lowercase
        assert not re.search(pattern, "SDOT")

    def test_org_pattern_acronym(self):
        """Test organization acronym pattern."""
        import re
        pattern = ORG_PATTERNS[0]  # 2-5 letter acronyms
        assert re.search(pattern, "SDOT")
        assert re.search(pattern, "FBI")
        assert re.search(pattern, "NASA")
        assert not re.search(pattern, "Seattle")

    def test_location_pattern_interstate(self):
        """Test interstate pattern."""
        import re
        pattern = LOCATION_PATTERNS[0]  # I-XX
        assert re.search(pattern, "I-5")
        assert re.search(pattern, "I-90")
        assert re.search(pattern, "I-405")
        assert not re.search(pattern, "Interstate 5")
