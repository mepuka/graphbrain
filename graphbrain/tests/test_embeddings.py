"""Tests for embedding utilities.

Tests the embedding configuration, LRU cache, and batch processor.
"""

import pytest
import threading
import time


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from graphbrain.embeddings.config import EmbeddingConfig

        config = EmbeddingConfig()

        assert config.model_name == "intfloat/e5-base-v2"
        assert config.dimensions == 768
        assert config.distance_metric == "cosine"
        assert config.index_type == "hnsw"
        assert config.hnsw_m == 16
        assert config.hnsw_ef_construction == 64
        assert config.batch_size == 32
        assert config.cache_max_size == 10000

    def test_custom_config(self):
        """Test custom configuration values."""
        from graphbrain.embeddings.config import EmbeddingConfig

        config = EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            dimensions=384,
            distance_metric="l2",
            index_type="ivfflat",
            ivfflat_lists=50,
            batch_size=64,
        )

        assert config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.dimensions == 384
        assert config.distance_metric == "l2"
        assert config.index_type == "ivfflat"
        assert config.ivfflat_lists == 50

    def test_get_pgvector_ops_class(self):
        """Test pgvector operator class generation."""
        from graphbrain.embeddings.config import EmbeddingConfig

        config = EmbeddingConfig(distance_metric="cosine")
        assert config.get_pgvector_ops_class() == "vector_cosine_ops"

        config = EmbeddingConfig(distance_metric="l2")
        assert config.get_pgvector_ops_class() == "vector_l2_ops"

        config = EmbeddingConfig(distance_metric="inner_product")
        assert config.get_pgvector_ops_class() == "vector_ip_ops"

    def test_get_distance_operator(self):
        """Test distance operator generation."""
        from graphbrain.embeddings.config import EmbeddingConfig

        config = EmbeddingConfig(distance_metric="cosine")
        assert config.get_distance_operator() == "<=>"

        config = EmbeddingConfig(distance_metric="l2")
        assert config.get_distance_operator() == "<->"

        config = EmbeddingConfig(distance_metric="inner_product")
        assert config.get_distance_operator() == "<#>"

    def test_get_index_sql_hnsw(self):
        """Test HNSW index SQL generation."""
        from graphbrain.embeddings.config import EmbeddingConfig

        config = EmbeddingConfig(
            index_type="hnsw",
            distance_metric="cosine",
            hnsw_m=16,
            hnsw_ef_construction=64,
        )
        sql = config.get_index_sql("edges", "embedding")

        assert "USING hnsw" in sql
        assert "vector_cosine_ops" in sql
        assert "m = 16" in sql
        assert "ef_construction = 64" in sql

    def test_get_index_sql_ivfflat(self):
        """Test IVFFlat index SQL generation."""
        from graphbrain.embeddings.config import EmbeddingConfig

        config = EmbeddingConfig(
            index_type="ivfflat",
            distance_metric="l2",
            ivfflat_lists=100,
        )
        sql = config.get_index_sql("edges", "embedding")

        assert "USING ivfflat" in sql
        assert "vector_l2_ops" in sql
        assert "lists = 100" in sql

    def test_get_index_sql_none(self):
        """Test no index generation."""
        from graphbrain.embeddings.config import EmbeddingConfig

        config = EmbeddingConfig(index_type="none")
        sql = config.get_index_sql("edges", "embedding")

        assert sql == ""

    def test_to_dict_and_from_dict(self):
        """Test serialization roundtrip."""
        from graphbrain.embeddings.config import EmbeddingConfig

        config = EmbeddingConfig(
            model_name="test-model",
            dimensions=512,
            distance_metric="l2",
        )
        data = config.to_dict()
        restored = EmbeddingConfig.from_dict(data)

        assert restored.model_name == "test-model"
        assert restored.dimensions == 512
        assert restored.distance_metric == "l2"

    def test_get_embedding_config_profiles(self):
        """Test pre-configured profiles."""
        from graphbrain.embeddings.config import get_embedding_config

        default = get_embedding_config("default")
        assert default.model_name == "intfloat/e5-base-v2"

        fast = get_embedding_config("fast")
        assert fast.dimensions == 384
        assert fast.hnsw_m == 8

        quality = get_embedding_config("quality")
        assert quality.dimensions == 1024
        assert quality.hnsw_m == 32

        low_memory = get_embedding_config("low_memory")
        assert low_memory.index_type == "ivfflat"
        assert low_memory.cache_max_size == 1000

    def test_get_embedding_config_invalid_profile(self):
        """Test invalid profile raises ValueError."""
        from graphbrain.embeddings.config import get_embedding_config

        with pytest.raises(ValueError) as exc_info:
            get_embedding_config("nonexistent")

        assert "Unknown profile" in str(exc_info.value)


class TestLRUEmbeddingCache:
    """Tests for LRUEmbeddingCache."""

    def test_basic_operations(self):
        """Test basic get/put operations."""
        from graphbrain.embeddings.cache import LRUEmbeddingCache

        cache = LRUEmbeddingCache(max_size=100)

        # Initially empty
        assert len(cache) == 0
        assert cache.get("key1") is None

        # Put and get
        embedding = [0.1, 0.2, 0.3]
        cache.put("key1", embedding)
        assert len(cache) == 1
        assert cache.get("key1") == embedding

        # Contains
        assert "key1" in cache
        assert "key2" not in cache

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        from graphbrain.embeddings.cache import LRUEmbeddingCache

        cache = LRUEmbeddingCache(max_size=3)

        # Fill cache
        cache.put("key1", [1.0])
        cache.put("key2", [2.0])
        cache.put("key3", [3.0])
        assert len(cache) == 3

        # Add one more, should evict key1 (oldest)
        cache.put("key4", [4.0])
        assert len(cache) == 3
        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") is not None
        assert cache.get("key4") is not None

    def test_access_updates_lru_order(self):
        """Test that get() updates LRU order."""
        from graphbrain.embeddings.cache import LRUEmbeddingCache

        cache = LRUEmbeddingCache(max_size=3)

        cache.put("key1", [1.0])
        cache.put("key2", [2.0])
        cache.put("key3", [3.0])

        # Access key1 to make it recently used
        cache.get("key1")

        # Add key4, should evict key2 (oldest after key1 was accessed)
        cache.put("key4", [4.0])

        assert cache.get("key1") is not None  # Still present
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") is not None
        assert cache.get("key4") is not None

    def test_remove(self):
        """Test remove operation."""
        from graphbrain.embeddings.cache import LRUEmbeddingCache

        cache = LRUEmbeddingCache(max_size=100)

        cache.put("key1", [1.0])
        cache.put("key2", [2.0])

        assert cache.remove("key1") is True
        assert cache.get("key1") is None
        assert len(cache) == 1

        assert cache.remove("nonexistent") is False

    def test_clear(self):
        """Test clear operation."""
        from graphbrain.embeddings.cache import LRUEmbeddingCache

        cache = LRUEmbeddingCache(max_size=100)

        cache.put("key1", [1.0])
        cache.put("key2", [2.0])

        cache.clear()
        assert len(cache) == 0
        assert cache.get("key1") is None

    def test_stats(self):
        """Test cache statistics."""
        from graphbrain.embeddings.cache import LRUEmbeddingCache

        cache = LRUEmbeddingCache(max_size=100)

        cache.put("key1", [1.0])
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.get_stats()
        assert stats["size"] == 1
        assert stats["max_size"] == 100
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(66.67, rel=0.1)

    def test_load_bulk(self):
        """Test bulk loading."""
        from graphbrain.embeddings.cache import LRUEmbeddingCache

        cache = LRUEmbeddingCache(max_size=100)

        items = {
            "key1": [1.0, 1.1],
            "key2": [2.0, 2.1],
            "key3": [3.0, 3.1],
        }
        count = cache.load_bulk(items)

        assert count == 3
        assert len(cache) == 3
        assert cache.get("key1") == [1.0, 1.1]

    def test_load_bulk_with_eviction(self):
        """Test bulk loading with eviction."""
        from graphbrain.embeddings.cache import LRUEmbeddingCache

        cache = LRUEmbeddingCache(max_size=2)

        items = {
            "key1": [1.0],
            "key2": [2.0],
            "key3": [3.0],
        }
        cache.load_bulk(items)

        # Only 2 should remain
        assert len(cache) == 2

    def test_export(self):
        """Test export operation."""
        from graphbrain.embeddings.cache import LRUEmbeddingCache

        cache = LRUEmbeddingCache(max_size=100)

        cache.put("key1", [1.0])
        cache.put("key2", [2.0])

        exported = cache.export()
        assert exported == {"key1": [1.0], "key2": [2.0]}

    def test_thread_safety(self):
        """Test thread-safe operations."""
        from graphbrain.embeddings.cache import LRUEmbeddingCache

        cache = LRUEmbeddingCache(max_size=1000)
        errors = []

        def writer():
            try:
                for i in range(100):
                    cache.put(f"writer_{i}", [float(i)])
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(100):
                    cache.get(f"writer_{i % 50}")
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_resize_cache(self):
        """Test resizing the cache."""
        from graphbrain.embeddings.cache import LRUEmbeddingCache

        cache = LRUEmbeddingCache(max_size=10)

        for i in range(10):
            cache.put(f"key{i}", [float(i)])

        assert len(cache) == 10

        # Reduce size
        cache.max_size = 5
        assert len(cache) == 5

        # Increase size
        cache.max_size = 100
        for i in range(10, 20):
            cache.put(f"key{i}", [float(i)])
        assert len(cache) == 15


class TestBatchEmbeddingProcessor:
    """Tests for BatchEmbeddingProcessor (without loading actual models)."""

    def test_init_with_default_config(self):
        """Test initialization with default config."""
        from graphbrain.embeddings.batch import BatchEmbeddingProcessor
        from graphbrain.embeddings.config import EmbeddingConfig

        processor = BatchEmbeddingProcessor()
        assert processor.config.model_name == "intfloat/e5-base-v2"

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        from graphbrain.embeddings.batch import BatchEmbeddingProcessor
        from graphbrain.embeddings.config import EmbeddingConfig

        config = EmbeddingConfig(batch_size=64)
        processor = BatchEmbeddingProcessor(config=config)
        assert processor.config.batch_size == 64

    def test_compute_similarity_cosine(self):
        """Test cosine similarity computation."""
        pytest.importorskip("numpy")
        from graphbrain.embeddings.batch import BatchEmbeddingProcessor
        from graphbrain.embeddings.config import EmbeddingConfig

        config = EmbeddingConfig(distance_metric="cosine")
        processor = BatchEmbeddingProcessor(config=config)

        # Identical vectors should have similarity 1
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        assert processor.compute_similarity(v1, v2) == pytest.approx(1.0)

        # Orthogonal vectors should have similarity 0
        v3 = [0.0, 1.0, 0.0]
        assert processor.compute_similarity(v1, v3) == pytest.approx(0.0)

        # Opposite vectors should have similarity -1
        v4 = [-1.0, 0.0, 0.0]
        assert processor.compute_similarity(v1, v4) == pytest.approx(-1.0)

    def test_compute_similarity_l2(self):
        """Test L2 similarity computation."""
        pytest.importorskip("numpy")
        from graphbrain.embeddings.batch import BatchEmbeddingProcessor
        from graphbrain.embeddings.config import EmbeddingConfig

        config = EmbeddingConfig(distance_metric="l2")
        processor = BatchEmbeddingProcessor(config=config)

        # Identical vectors should have similarity 1 (exp(-0))
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        assert processor.compute_similarity(v1, v2) == pytest.approx(1.0)

        # Different vectors should have lower similarity
        v3 = [2.0, 0.0, 0.0]
        sim = processor.compute_similarity(v1, v3)
        assert sim < 1.0
        assert sim > 0.0

    def test_compute_similarity_inner_product(self):
        """Test inner product similarity computation."""
        pytest.importorskip("numpy")
        from graphbrain.embeddings.batch import BatchEmbeddingProcessor
        from graphbrain.embeddings.config import EmbeddingConfig

        config = EmbeddingConfig(distance_metric="inner_product")
        processor = BatchEmbeddingProcessor(config=config)

        # Normalized vectors with inner product
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        assert processor.compute_similarity(v1, v2) == pytest.approx(1.0)

        # Orthogonal
        v3 = [0.0, 1.0, 0.0]
        assert processor.compute_similarity(v1, v3) == pytest.approx(0.0)
