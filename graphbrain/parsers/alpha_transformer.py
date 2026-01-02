"""Transformer-based Alpha classifier for token-to-atom type prediction.

This module provides a modern alternative to the RandomForest-based Alpha classifier.
It uses sentence-transformers (MiniLM) to encode token context and a simple classifier
to predict atom types.

BENCHMARK RESULTS (December 2025):
==================================
Tested on atoms-en.csv dataset (6,936 samples, 80/20 train/test split):

| Metric            | RandomForest | Transformer (MiniLM) |
|-------------------|--------------|----------------------|
| Test Accuracy     | 96.40%       | 94.81%               |
| Train Time        | 0.35s        | 5.94s                |
| Predict Time/sample| 0.015ms     | 0.367ms              |

CONCLUSION: The original RandomForest classifier outperforms the transformer approach
for this specific task. This is because:

1. **Highly Informative Features**: The 5 hand-crafted syntactic features (tag, dep,
   hpos, hdep, pos_after) are extremely predictive for atom type classification.

2. **Information Loss**: Converting structured categorical features to text descriptions
   loses the discrete, combinatorial nature that RandomForest excels at modeling.

3. **Task Simplicity**: With only 7 target classes and clear syntactic patterns,
   the problem doesn't benefit from transformer's semantic understanding.

RECOMMENDATION: Keep the original RandomForest-based Alpha classifier. This module
is preserved for benchmarking and potential future use cases where semantic
understanding of novel syntactic patterns is more important than raw accuracy.

Usage:
    from graphbrain.parsers.alpha_transformer import AlphaTransformer

    # Load training data
    cases_str = open('atoms-en.csv').read()
    alpha = AlphaTransformer(cases_str)

    # Predict (same interface as Alpha)
    features = [('NNP', 'nsubj', 'VERB', 'ROOT', 'ADV'), ...]
    predictions = alpha.predict(features)
"""
from __future__ import annotations

import logging
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# Check if sentence-transformers is available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    SentenceTransformer = None

# Label mapping
ATOM_TYPES = ['C', 'M', 'P', 'B', 'T', 'J', 'X']
ATOM_TYPE_NAMES = {
    'C': 'concept',
    'M': 'modifier',
    'P': 'predicate',
    'B': 'builder',
    'T': 'trigger',
    'J': 'junction',
    'X': 'excluded'
}


class AlphaTransformer:
    """Transformer-based token classifier for semantic atom types.

    This is a drop-in replacement for the Alpha class that uses
    sentence-transformers for feature extraction instead of hand-crafted
    features with one-hot encoding.

    The classifier encodes each token's syntactic context as a natural
    language description, then uses a logistic regression classifier
    on top of the embeddings.
    """

    _MODEL_NAME = 'all-MiniLM-L6-v2'
    _EMBEDDING_CACHE_SIZE = 10000
    _MODEL_CACHE_DIR = Path.home() / '.graphbrain-data' / 'models' / 'alpha-transformer'

    def __init__(self, cases_str: str, use_cache: bool = True):
        """Initialize the transformer-based alpha classifier.

        Args:
            cases_str: Tab-separated training data (same format as Alpha).
            use_cache: Whether to cache/load trained model from disk.
        """
        if not SENTENCE_TRANSFORMER_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for AlphaTransformer. "
                "Install it with: pip install sentence-transformers"
            )

        self._model: Optional[SentenceTransformer] = None  # Lazy loaded
        self._clf: Optional[LogisticRegression] = None
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(ATOM_TYPES)
        self.empty = False

        # Try to load cached model
        cache_path = self._MODEL_CACHE_DIR / 'classifier.pkl'
        if use_cache and cache_path.exists():
            logger.info(f"Loading cached AlphaTransformer from {cache_path}")
            self._load_cached_model(cache_path)
        else:
            # Train from scratch
            self._train(cases_str)
            if use_cache:
                self._save_cached_model(cache_path)

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            logger.info(f"Loading sentence-transformer model: {self._MODEL_NAME}")
            self._model = SentenceTransformer(self._MODEL_NAME)
            logger.info("Model loaded successfully")
        return self._model

    def _train(self, cases_str: str):
        """Train the classifier on the provided training data."""
        logger.info("Training AlphaTransformer...")

        # Parse training data
        X_text = []
        y = []

        for line in cases_str.strip().split('\n'):
            sline = line.strip()
            if len(sline) > 0:
                row = sline.split('\t')
                if len(row) < 20:
                    continue

                label = row[0]
                token_text = row[1]
                tag = row[3]
                dep = row[4]
                head_pos = row[6]
                head_dep = row[8]
                pos_after = row[19]

                # Create natural language description of the token context
                description = self._create_description(
                    token_text, tag, dep, head_pos, head_dep, pos_after
                )

                X_text.append(description)
                y.append(label)

        if len(y) == 0:
            self.empty = True
            logger.warning("No training data provided, classifier will be empty")
            return

        logger.info(f"Encoding {len(X_text)} training samples...")

        # Get embeddings for all training samples
        X_embeddings = self.model.encode(
            X_text,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Train logistic regression classifier
        logger.info("Training classifier...")
        y_encoded = self._label_encoder.transform(y)

        self._clf = LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            random_state=777,
            n_jobs=-1
        )
        self._clf.fit(X_embeddings, y_encoded)

        # Compute training accuracy
        train_preds = self._clf.predict(X_embeddings)
        accuracy = np.mean(train_preds == y_encoded)
        logger.info(f"Training complete. Training accuracy: {accuracy:.2%}")

    def _create_description(
        self,
        token: str,
        tag: str,
        dep: str,
        head_pos: str,
        head_dep: str,
        pos_after: str
    ) -> str:
        """Create a natural language description of the token context.

        This description is then embedded by the sentence transformer.
        The format is designed to capture the key syntactic features
        in a way that leverages the model's pre-trained knowledge.
        """
        return f"{tag} {dep} head:{head_pos} {head_dep} next:{pos_after}"

    @lru_cache(maxsize=_EMBEDDING_CACHE_SIZE)
    def _get_embedding_cached(self, description: str) -> Tuple[float, ...]:
        """Get embedding for a description with caching."""
        embedding = self.model.encode(description, convert_to_numpy=True)
        return tuple(embedding.tolist())

    def predict(self, X: List[Tuple[str, str, str, str, str]]) -> Tuple[str, ...]:
        """Predict atom types for a list of token features.

        Args:
            X: List of tuples (tag, dep, hpos, hdep, pos_after)
               Same format as the original Alpha.predict()

        Returns:
            Tuple of predicted atom types (C, M, P, B, T, J, X)
        """
        if self.empty:
            return tuple('C' for _ in range(len(X)))

        if len(X) == 0:
            return ()

        # Create descriptions for each token
        descriptions = [
            self._create_description('', tag, dep, hpos, hdep, pos_after)
            for tag, dep, hpos, hdep, pos_after in X
        ]

        # Get unique descriptions to batch encode
        unique_descs = list(set(descriptions))

        # Batch encode all unique descriptions at once
        unique_embeddings = self.model.encode(
            unique_descs,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # Create mapping from description to embedding
        desc_to_emb = {desc: emb for desc, emb in zip(unique_descs, unique_embeddings)}

        # Build embeddings array in original order
        embeddings = np.array([desc_to_emb[desc] for desc in descriptions])

        # Predict
        pred_encoded = self._clf.predict(embeddings)
        predictions = self._label_encoder.inverse_transform(pred_encoded)

        return tuple(predictions)

    def predict_proba(self, X: List[Tuple[str, str, str, str, str]]) -> np.ndarray:
        """Predict probabilities for each atom type.

        Args:
            X: List of tuples (tag, dep, hpos, hdep, pos_after)

        Returns:
            Array of shape (n_samples, 7) with probabilities for each class.
        """
        if self.empty or len(X) == 0:
            return np.zeros((len(X), len(ATOM_TYPES)))

        descriptions = [
            self._create_description('', tag, dep, hpos, hdep, pos_after)
            for tag, dep, hpos, hdep, pos_after in X
        ]

        embeddings = np.array([
            list(self._get_embedding_cached(desc))
            for desc in descriptions
        ])

        return self._clf.predict_proba(embeddings)

    def _save_cached_model(self, path: Path):
        """Save the trained classifier to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'clf': self._clf,
                'label_encoder': self._label_encoder,
                'empty': self.empty
            }, f)
        logger.info(f"Saved AlphaTransformer to {path}")

    def _load_cached_model(self, path: Path):
        """Load a trained classifier from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self._clf = data['clf']
            self._label_encoder = data['label_encoder']
            self.empty = data['empty']

    def clear_cache(self):
        """Clear the embedding cache."""
        self._get_embedding_cached.cache_clear()

    def cache_info(self) -> dict:
        """Get cache statistics."""
        info = self._get_embedding_cached.cache_info()
        return {
            'hits': info.hits,
            'misses': info.misses,
            'hit_rate': info.hits / (info.hits + info.misses) if (info.hits + info.misses) > 0 else 0.0
        }


def benchmark_alpha_classifiers(cases_str: str, test_ratio: float = 0.2):
    """Benchmark AlphaTransformer against the original Alpha classifier.

    Args:
        cases_str: Training data string.
        test_ratio: Fraction of data to use for testing.

    Returns:
        Dictionary with benchmark results.
    """
    from graphbrain.parsers.alpha import Alpha
    import time

    # Split data
    lines = [l for l in cases_str.strip().split('\n') if l.strip()]
    np.random.seed(42)
    np.random.shuffle(lines)

    split_idx = int(len(lines) * (1 - test_ratio))
    train_lines = lines[:split_idx]
    test_lines = lines[split_idx:]

    train_str = '\n'.join(train_lines)

    # Parse test data
    test_X = []
    test_y = []
    for line in test_lines:
        row = line.split('\t')
        if len(row) >= 20:
            test_y.append(row[0])
            test_X.append((row[3], row[4], row[6], row[8], row[19]))

    results = {}

    # Benchmark original Alpha
    logger.info("Training original Alpha (RandomForest)...")
    start = time.time()
    alpha_rf = Alpha(train_str)
    rf_train_time = time.time() - start

    start = time.time()
    rf_preds = alpha_rf.predict(test_X)
    rf_pred_time = time.time() - start

    rf_accuracy = np.mean([p == t for p, t in zip(rf_preds, test_y)])
    results['random_forest'] = {
        'accuracy': rf_accuracy,
        'train_time': rf_train_time,
        'predict_time': rf_pred_time,
        'predict_time_per_sample': rf_pred_time / len(test_X) * 1000  # ms
    }

    # Benchmark AlphaTransformer
    logger.info("Training AlphaTransformer...")
    start = time.time()
    alpha_tf = AlphaTransformer(train_str, use_cache=False)
    tf_train_time = time.time() - start

    start = time.time()
    tf_preds = alpha_tf.predict(test_X)
    tf_pred_time = time.time() - start

    tf_accuracy = np.mean([p == t for p, t in zip(tf_preds, test_y)])
    results['transformer'] = {
        'accuracy': tf_accuracy,
        'train_time': tf_train_time,
        'predict_time': tf_pred_time,
        'predict_time_per_sample': tf_pred_time / len(test_X) * 1000  # ms
    }

    # Summary
    results['summary'] = {
        'test_samples': len(test_X),
        'accuracy_improvement': tf_accuracy - rf_accuracy,
        'speedup': rf_pred_time / tf_pred_time if tf_pred_time > 0 else float('inf')
    }

    return results
