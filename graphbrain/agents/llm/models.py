"""Pydantic models for LLM classification requests and responses.

These models define the structured output schemas that the LLM
must conform to, ensuring type-safe classification results.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ─── Predicate Classification ───────────────────────────────────────────────


class PredicateCategory(str, Enum):
    """Semantic categories for predicates."""
    CLAIM = "claim"           # say, announce, declare
    CONFLICT = "conflict"     # attack, blame, accuse
    ACTION = "action"         # do, make, create
    COGNITION = "cognition"   # think, believe, know
    EMOTION = "emotion"       # love, hate, fear
    MOVEMENT = "movement"     # go, come, move
    POSSESSION = "possession" # have, own, give
    PERCEPTION = "perception" # see, hear, feel
    UNKNOWN = "unknown"


class PredicateClassification(BaseModel):
    """LLM response for predicate classification."""
    lemma: str = Field(description="The predicate lemma being classified")
    category: PredicateCategory = Field(description="Semantic category")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Classification confidence score"
    )
    reasoning: str = Field(
        max_length=500,
        description="Brief explanation for the classification"
    )
    similar_predicates: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Similar predicates in the same category"
    )

    class Config:
        use_enum_values = True


class BatchPredicateResult(BaseModel):
    """Batch predicate classification response."""
    classifications: list[PredicateClassification] = Field(
        description="List of predicate classifications"
    )
    unclassified: list[str] = Field(
        default_factory=list,
        description="Predicates that couldn't be classified"
    )


# ─── Entity Typing ──────────────────────────────────────────────────────────


class EntityType(str, Enum):
    """Entity type categories."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    GROUP = "group"           # collective entities (protesters, residents)
    EVENT = "event"
    UNKNOWN = "unknown"


class EntityClassification(BaseModel):
    """LLM response for entity typing."""
    entity: str = Field(description="The entity being typed")
    entity_type: EntityType = Field(description="Entity type category")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Classification confidence score"
    )
    reasoning: str = Field(
        max_length=500,
        description="Brief explanation for the typing"
    )
    subtypes: list[str] = Field(
        default_factory=list,
        description="More specific subtypes (e.g., 'politician', 'government')"
    )

    class Config:
        use_enum_values = True


class BatchEntityResult(BaseModel):
    """Batch entity typing response."""
    entities: list[EntityClassification] = Field(
        description="List of entity classifications"
    )
    unclassified: list[str] = Field(
        default_factory=list,
        description="Entities that couldn't be typed"
    )


# ─── Request Models ─────────────────────────────────────────────────────────


class ClassificationRequest(BaseModel):
    """Request for predicate classification."""
    lemma: str = Field(description="Predicate lemma to classify")
    context: Optional[str] = Field(
        default=None,
        description="Optional context for disambiguation"
    )
    examples: list[str] = Field(
        default_factory=list,
        description="Example edges using this predicate"
    )


class EntityTypingRequest(BaseModel):
    """Request for entity typing."""
    entity: str = Field(description="Entity to type")
    context_edges: list[str] = Field(
        default_factory=list,
        description="Hypergraph edges containing this entity"
    )
    predicates: list[str] = Field(
        default_factory=list,
        description="Predicates used with this entity"
    )
