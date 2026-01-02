"""Batch embedding operations for graphbrain.

Provides efficient bulk embedding generation and database updates.
"""

import logging
from typing import Iterator, Optional, Callable, Any

from graphbrain.embeddings.config import EmbeddingConfig

logger = logging.getLogger(__name__)

# Try to import numpy and sentence-transformers
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    np = None

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


class BatchEmbeddingProcessor:
    """Efficient batch processing for embedding generation and storage.

    This class provides:
    - Batched embedding generation to optimize GPU/CPU utilization
    - Bulk database updates for PostgreSQL and SQLite
    - Progress callbacks for long-running operations
    - Automatic normalization of embeddings

    Example:
        >>> config = EmbeddingConfig(batch_size=64)
        >>> processor = BatchEmbeddingProcessor(config)
        >>> texts = ["Hello world", "Another text", ...]
        >>> embeddings = processor.generate_embeddings(texts)
        >>> # Or with database update
        >>> processor.update_embeddings_bulk(conn, "edges", updates)
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        model: Optional[Any] = None,
    ):
        """Initialize batch processor.

        Args:
            config: Embedding configuration. Uses defaults if not provided.
            model: Pre-loaded SentenceTransformer model. If not provided,
                   will be loaded lazily on first use.
        """
        self.config = config or EmbeddingConfig()
        self._model = model
        self._model_loaded = model is not None

    def _get_model(self) -> "SentenceTransformer":
        """Get or lazily load the sentence transformer model."""
        if not self._model_loaded:
            if not _SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "sentence-transformers not available. "
                    "Install with: pip install sentence-transformers"
                )
            logger.info(f"Loading embedding model: {self.config.model_name}")
            self._model = SentenceTransformer(self.config.model_name)
            self._model_loaded = True
        return self._model

    def generate_embeddings(
        self,
        texts: list[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> "np.ndarray":
        """Generate embeddings for a list of texts in batches.

        Args:
            texts: List of text strings to embed
            batch_size: Override batch size from config
            show_progress: Show progress bar (requires tqdm)
            progress_callback: Optional callback(current, total) for progress

        Returns:
            numpy array of shape (len(texts), dimensions)

        Raises:
            ImportError: If sentence-transformers or numpy not available
        """
        if not _NUMPY_AVAILABLE:
            raise ImportError("numpy not available. Install with: pip install numpy")

        if not texts:
            return np.array([])

        model = self._get_model()
        batch_size = batch_size or self.config.batch_size

        all_embeddings = []
        total = len(texts)

        # Process in batches
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=show_progress and len(batch) > 10,
            )
            all_embeddings.append(batch_embeddings)

            if progress_callback:
                progress_callback(min(i + batch_size, total), total)

        # Concatenate all batches
        embeddings = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]

        logger.debug(f"Generated {len(embeddings)} embeddings with shape {embeddings.shape}")
        return embeddings

    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            Embedding as list of floats
        """
        embeddings = self.generate_embeddings([text])
        return embeddings[0].tolist()

    def update_embeddings_bulk_postgres(
        self,
        conn,
        table: str,
        updates: list[tuple[str, list[float]]],
        key_column: str = "edge_key",
        embedding_column: str = "embedding",
        batch_size: Optional[int] = None,
    ) -> int:
        """Bulk update embeddings in PostgreSQL.

        Uses efficient batch UPDATE with VALUES for performance.

        Args:
            conn: psycopg2 connection
            table: Table name
            updates: List of (key, embedding) tuples
            key_column: Name of the key column
            embedding_column: Name of the embedding column
            batch_size: Override batch size from config

        Returns:
            Number of rows updated
        """
        if not updates:
            return 0

        batch_size = batch_size or self.config.batch_size
        total_updated = 0

        with conn.cursor() as cur:
            # Process in batches
            for i in range(0, len(updates), batch_size):
                batch = updates[i:i + batch_size]

                # Build VALUES clause
                values_parts = []
                params = []
                for key, embedding in batch:
                    values_parts.append("(%s, %s::vector)")
                    params.extend([key, embedding])

                values_sql = ", ".join(values_parts)

                # Execute batch update
                cur.execute(
                    f"""
                    UPDATE {table} AS t
                    SET {embedding_column} = v.embedding,
                        updated_at = NOW()
                    FROM (VALUES {values_sql}) AS v(key, embedding)
                    WHERE t.{key_column} = v.key
                    """,
                    params
                )
                total_updated += cur.rowcount

            conn.commit()

        logger.info(f"Updated {total_updated} embeddings in {table}")
        return total_updated

    def update_embeddings_bulk_sqlite(
        self,
        conn,
        table: str,
        updates: list[tuple[str, list[float]]],
        key_column: str = "edge_key",
        embedding_column: str = "embedding",
        batch_size: Optional[int] = None,
    ) -> int:
        """Bulk update embeddings in SQLite.

        SQLite doesn't support batch UPDATE with VALUES, so we use
        executemany with individual updates.

        Args:
            conn: sqlite3 connection
            table: Table name
            updates: List of (key, embedding) tuples
            key_column: Name of the key column
            embedding_column: Name of the embedding column (stores as JSON)
            batch_size: Override batch size from config

        Returns:
            Number of rows updated
        """
        import json

        if not updates:
            return 0

        batch_size = batch_size or self.config.batch_size
        total_updated = 0

        # Prepare data (convert embeddings to JSON)
        update_data = [
            (json.dumps(embedding), key)
            for key, embedding in updates
        ]

        # Use executemany for efficiency
        cur = conn.cursor()
        try:
            for i in range(0, len(update_data), batch_size):
                batch = update_data[i:i + batch_size]
                cur.executemany(
                    f"UPDATE {table} SET {embedding_column} = ? WHERE {key_column} = ?",
                    batch
                )
                total_updated += cur.rowcount
            conn.commit()
        finally:
            cur.close()

        logger.info(f"Updated {total_updated} embeddings in {table}")
        return total_updated

    def generate_and_store_embeddings(
        self,
        conn,
        table: str,
        items: Iterator[tuple[str, str]],
        key_column: str = "edge_key",
        embedding_column: str = "embedding",
        backend: str = "postgres",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """Generate embeddings for texts and store them in database.

        This is a convenience method that combines generation and storage.

        Args:
            conn: Database connection (psycopg2 or sqlite3)
            table: Table name
            items: Iterator of (key, text) tuples
            key_column: Name of the key column
            embedding_column: Name of the embedding column
            backend: 'postgres' or 'sqlite'
            progress_callback: Optional callback(current, total) for progress

        Returns:
            Number of embeddings stored
        """
        # Collect items (we need to materialize for batching)
        items_list = list(items)
        if not items_list:
            return 0

        keys = [item[0] for item in items_list]
        texts = [item[1] for item in items_list]

        # Generate embeddings
        embeddings = self.generate_embeddings(
            texts,
            progress_callback=progress_callback
        )

        # Prepare updates
        updates = list(zip(keys, embeddings.tolist()))

        # Store in database
        if backend == "postgres":
            return self.update_embeddings_bulk_postgres(
                conn, table, updates, key_column, embedding_column
            )
        else:
            return self.update_embeddings_bulk_sqlite(
                conn, table, updates, key_column, embedding_column
            )

    def compute_similarity(
        self,
        embedding1: list[float],
        embedding2: list[float],
    ) -> float:
        """Compute similarity between two embeddings.

        Uses the distance metric from config and converts to similarity.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score (0-1 for cosine, varies for others)
        """
        if not _NUMPY_AVAILABLE:
            raise ImportError("numpy not available")

        v1 = np.array(embedding1)
        v2 = np.array(embedding2)

        if self.config.distance_metric == "cosine":
            # Cosine similarity
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 0 and norm2 > 0:
                return float(np.dot(v1, v2) / (norm1 * norm2))
            return 0.0
        elif self.config.distance_metric == "l2":
            # Convert L2 distance to similarity (using exponential decay)
            distance = np.linalg.norm(v1 - v2)
            return float(np.exp(-distance))
        elif self.config.distance_metric == "inner_product":
            # Inner product (assumes normalized vectors for similarity)
            return float(np.dot(v1, v2))
        else:
            raise ValueError(f"Unknown distance metric: {self.config.distance_metric}")
