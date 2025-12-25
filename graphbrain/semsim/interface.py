from __future__ import annotations

import logging
from enum import Enum
from typing import Type

from graphbrain.hyperedge import Hyperedge
from graphbrain.hypergraph import Hypergraph
from graphbrain.semsim.matcher.matcher import SemSimConfig, SemSimMatcher, SemSimType
from graphbrain.semsim.matcher.fixed_matcher import FixedEmbeddingMatcher
from graphbrain.semsim.matcher.context_matcher import ContextEmbeddingMatcher

# Try to import sentence-transformer matcher (optional dependency)
try:
    from graphbrain.semsim.matcher.sentence_transformer_matcher import (
        SentenceTransformerMatcher,
        SENTENCE_TRANSFORMER_AVAILABLE
    )
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    SentenceTransformerMatcher = None


logger = logging.getLogger(__name__)


class MatcherBackend(str, Enum):
    """Backend for FIX (fixed embedding) matcher."""
    AUTO = "AUTO"  # Prefer sentence-transformers, fallback to gensim
    SENTENCE_TRANSFORMER = "SENTENCE_TRANSFORMER"
    GENSIM = "GENSIM"  # Legacy Word2Vec


# Default configuration for FIX matcher
# - Uses sentence-transformers e5-base if available (smaller, better OOV handling)
# - Falls back to gensim Word2Vec if sentence-transformers not installed
DEFAULT_FIX_CONFIG_ST = SemSimConfig(
    model_name='intfloat/e5-base-v2',
    similarity_threshold=0.5
)

DEFAULT_FIX_CONFIG_GENSIM = SemSimConfig(
    model_name='word2vec-google-news-300',
    similarity_threshold=0.2
)


def get_default_fix_config() -> SemSimConfig:
    """Get the default FIX config based on available backends."""
    if SENTENCE_TRANSFORMER_AVAILABLE:
        return DEFAULT_FIX_CONFIG_ST
    return DEFAULT_FIX_CONFIG_GENSIM


DEFAULT_CONFIGS: dict[SemSimType, SemSimConfig] = {
    SemSimType.FIX: get_default_fix_config(),
    SemSimType.CTX: SemSimConfig(
        model_name='intfloat/e5-large-v2',
        similarity_threshold=0.65,
        embedding_prefix="query:"
    )
}

_matcher_type_mapping: dict[SemSimType, Type[SemSimMatcher]] = {
    SemSimType.FIX: FixedEmbeddingMatcher,
    SemSimType.CTX: ContextEmbeddingMatcher
}

_matchers: dict[SemSimType, SemSimMatcher] = {}

# Current backend preference for FIX matcher
_fix_backend: MatcherBackend = MatcherBackend.AUTO


def set_fix_backend(backend: MatcherBackend):
    """Set the backend preference for FIX matcher.

    Args:
        backend: MatcherBackend.AUTO (default), SENTENCE_TRANSFORMER, or GENSIM.

    Note: This must be called before init_matcher() or get_matcher() for FIX type.
    """
    global _fix_backend
    _fix_backend = backend
    logger.info(f"Set FIX matcher backend preference to: {backend.value}")


def _get_fix_matcher_class() -> Type[SemSimMatcher]:
    """Get the appropriate FIX matcher class based on backend preference."""
    global _fix_backend

    if _fix_backend == MatcherBackend.GENSIM:
        logger.info("Using gensim FixedEmbeddingMatcher (Word2Vec)")
        return FixedEmbeddingMatcher

    if _fix_backend == MatcherBackend.SENTENCE_TRANSFORMER:
        if not SENTENCE_TRANSFORMER_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for SENTENCE_TRANSFORMER backend. "
                "Install it with: pip install sentence-transformers"
            )
        logger.info("Using SentenceTransformerMatcher")
        return SentenceTransformerMatcher

    # AUTO: prefer sentence-transformers if available
    if SENTENCE_TRANSFORMER_AVAILABLE:
        logger.info("Using SentenceTransformerMatcher (AUTO backend)")
        return SentenceTransformerMatcher
    else:
        logger.info("Using gensim FixedEmbeddingMatcher (sentence-transformers not available)")
        return FixedEmbeddingMatcher


def init_matcher(matcher_type: SemSimType, config: SemSimConfig = None):
    """Initialize a semantic similarity matcher.

    Args:
        matcher_type: SemSimType.FIX or SemSimType.CTX
        config: Optional SemSimConfig. If not provided, uses defaults.
    """
    global _matchers

    if not config:
        config = DEFAULT_CONFIGS[matcher_type]
        logger.info(
            f"No SemSim config given, using default config "
            f"for SemSim matcher of type '{matcher_type}'"
        )

    # Get the appropriate matcher class
    if matcher_type == SemSimType.FIX:
        matcher_class = _get_fix_matcher_class()
    else:
        matcher_class = _matcher_type_mapping[matcher_type]

    _matchers[matcher_type] = matcher_class(config=config)
    logger.info(f"Initialized SemSim matcher for type '{matcher_type}': {config=}")


def get_matcher(matcher_type: SemSimType, config: SemSimConfig = None):
    global _matchers

    if config or matcher_type not in _matchers:
        init_matcher(matcher_type=matcher_type, config=config)

    return _matchers[matcher_type]


def semsim(
        semsim_type: str,
        threshold: float = None,
        cand_word: str = None,
        ref_words: list[str] = None,
        cand_edge: Hyperedge = None,
        cand_tok_pos: Hyperedge = None,
        ref_edges: list[Hyperedge] = None,
        ref_tok_poses: list[Hyperedge] = None,
        hg: Hypergraph = None
) -> bool:
    try:
        semsim_type: SemSimType = SemSimType(semsim_type)
    except ValueError:
        logger.error(f"Invalid SemSim model type given: '{semsim_type}")
        return False

    matcher = get_matcher(matcher_type=semsim_type)

    return matcher.similar(
        threshold=threshold,
        cand_word=cand_word,
        ref_words=ref_words,
        cand_edge=cand_edge,
        cand_tok_pos=cand_tok_pos,
        ref_edges=ref_edges,
        ref_tok_poses=ref_tok_poses,
        hg=hg
    )
