"""Embedding configuration for graphbrain.

Provides centralized configuration for:
- Embedding model selection
- Vector dimensions
- Distance metrics
- Index types (HNSW, IVFFlat)
- Batch processing settings
- Cache settings
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
from datetime import datetime


@dataclass
class EmbeddingConfig:
    """Configuration for embedding operations.

    Attributes:
        model_name: Name of the sentence transformer model to use.
            Default is 'intfloat/e5-base-v2' which produces 768-dim embeddings.
        dimensions: Embedding vector dimensionality. Must match the model output.
        distance_metric: Distance metric for similarity search.
            - 'cosine': Cosine distance (1 - cosine similarity), best for normalized vectors
            - 'l2': Euclidean (L2) distance
            - 'inner_product': Negative inner product (for maximum inner product search)
        index_type: Vector index type for PostgreSQL/pgvector.
            - 'hnsw': Hierarchical Navigable Small World (faster queries, more memory)
            - 'ivfflat': Inverted File with Flat compression (less memory, slower queries)
            - 'none': No index (exact search, slowest but most accurate)
        hnsw_m: HNSW max connections per layer. Higher = better recall, more memory.
            Typical values: 8-64. Default 16 is a good balance.
        hnsw_ef_construction: HNSW construction-time exploration factor.
            Higher = better index quality, slower build. Typical: 64-512.
        ivfflat_lists: Number of IVFFlat clusters. Typical: sqrt(n_vectors) to 4*sqrt(n_vectors).
            Default 100 works for ~10k-1M vectors.
        batch_size: Batch size for bulk embedding operations.
        cache_max_size: Maximum number of embeddings to cache in memory.
        normalize_embeddings: Whether to L2-normalize embeddings before storage.
            Recommended True for cosine similarity to enable inner product optimization.

    Example:
        >>> config = EmbeddingConfig()  # Use defaults
        >>> config = EmbeddingConfig(
        ...     model_name="sentence-transformers/all-MiniLM-L6-v2",
        ...     dimensions=384,
        ...     index_type="ivfflat"
        ... )
    """

    # Model settings
    model_name: str = "intfloat/e5-base-v2"
    dimensions: int = 768

    # Distance and indexing
    distance_metric: Literal["cosine", "l2", "inner_product"] = "cosine"
    index_type: Literal["ivfflat", "hnsw", "none"] = "hnsw"

    # HNSW parameters
    hnsw_m: int = 16  # max connections per layer
    hnsw_ef_construction: int = 64  # exploration factor during construction

    # IVFFlat parameters
    ivfflat_lists: int = 100  # number of clusters

    # Processing settings
    batch_size: int = 32
    cache_max_size: int = 10000
    normalize_embeddings: bool = True

    # Metadata
    created_at: Optional[datetime] = field(default_factory=datetime.now)

    def get_pgvector_ops_class(self) -> str:
        """Get the pgvector operator class for the configured distance metric.

        Returns:
            PostgreSQL operator class name for vector operations.
        """
        ops_map = {
            "cosine": "vector_cosine_ops",
            "l2": "vector_l2_ops",
            "inner_product": "vector_ip_ops",
        }
        return ops_map[self.distance_metric]

    def get_index_sql(self, table: str, column: str = "embedding") -> str:
        """Generate SQL for creating a vector index.

        Args:
            table: Table name
            column: Embedding column name

        Returns:
            SQL CREATE INDEX statement, or empty string if index_type is 'none'.
        """
        if self.index_type == "none":
            return ""

        ops_class = self.get_pgvector_ops_class()
        index_name = f"idx_{table}_{column}_{self.index_type}"

        if self.index_type == "hnsw":
            return f"""CREATE INDEX IF NOT EXISTS {index_name}
    ON {table} USING hnsw ({column} {ops_class})
    WITH (m = {self.hnsw_m}, ef_construction = {self.hnsw_ef_construction})"""
        elif self.index_type == "ivfflat":
            return f"""CREATE INDEX IF NOT EXISTS {index_name}
    ON {table} USING ivfflat ({column} {ops_class})
    WITH (lists = {self.ivfflat_lists})"""
        else:
            return ""

    def get_distance_operator(self) -> str:
        """Get the pgvector distance operator for queries.

        Returns:
            PostgreSQL operator for distance calculation.
        """
        ops_map = {
            "cosine": "<=>",  # cosine distance
            "l2": "<->",  # L2 distance
            "inner_product": "<#>",  # negative inner product
        }
        return ops_map[self.distance_metric]

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "distance_metric": self.distance_metric,
            "index_type": self.index_type,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef_construction": self.hnsw_ef_construction,
            "ivfflat_lists": self.ivfflat_lists,
            "batch_size": self.batch_size,
            "cache_max_size": self.cache_max_size,
            "normalize_embeddings": self.normalize_embeddings,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EmbeddingConfig":
        """Create config from dictionary."""
        return cls(
            model_name=data.get("model_name", cls.model_name),
            dimensions=data.get("dimensions", 768),
            distance_metric=data.get("distance_metric", "cosine"),
            index_type=data.get("index_type", "hnsw"),
            hnsw_m=data.get("hnsw_m", 16),
            hnsw_ef_construction=data.get("hnsw_ef_construction", 64),
            ivfflat_lists=data.get("ivfflat_lists", 100),
            batch_size=data.get("batch_size", 32),
            cache_max_size=data.get("cache_max_size", 10000),
            normalize_embeddings=data.get("normalize_embeddings", True),
        )


# Pre-configured profiles for common use cases
EMBEDDING_PROFILES = {
    "default": EmbeddingConfig(),

    "fast": EmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
        index_type="hnsw",
        hnsw_m=8,
        hnsw_ef_construction=32,
        batch_size=64,
    ),

    "quality": EmbeddingConfig(
        model_name="intfloat/e5-large-v2",
        dimensions=1024,
        index_type="hnsw",
        hnsw_m=32,
        hnsw_ef_construction=128,
        batch_size=16,
    ),

    "low_memory": EmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
        index_type="ivfflat",
        ivfflat_lists=50,
        batch_size=16,
        cache_max_size=1000,
    ),
}


def get_embedding_config(profile: str = "default") -> EmbeddingConfig:
    """Get a pre-configured embedding profile.

    Args:
        profile: Profile name ('default', 'fast', 'quality', 'low_memory')

    Returns:
        EmbeddingConfig instance

    Raises:
        ValueError: If profile name is not recognized
    """
    if profile not in EMBEDDING_PROFILES:
        available = ", ".join(EMBEDDING_PROFILES.keys())
        raise ValueError(f"Unknown profile '{profile}'. Available: {available}")
    return EMBEDDING_PROFILES[profile]
