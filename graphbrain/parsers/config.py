"""Parser configuration system.

Provides centralized configuration for NLP parsers with support for:
- YAML/JSON configuration files
- Model fallback chains
- Development vs production modes
- Logging configuration
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for spaCy language models."""

    # Preferred model (highest quality)
    preferred: str = 'en_core_web_trf'

    # Fallback chain if preferred model is unavailable
    fallbacks: list[str] = field(default_factory=lambda: ['en_core_web_lg', 'en_core_web_md', 'en_core_web_sm'])

    # Coreference model (requires transformer model)
    coreference: str = 'en_coreference_web_trf'

    # Minimum model required for coreference resolution
    coreference_requires: str = 'en_core_web_trf'


@dataclass
class ParserConfig:
    """Configuration for the parser behavior."""

    # Beta stage mode: 'strict' or 'repair'
    beta: str = 'repair'

    # Whether to normalize hyperedges
    normalize: bool = True

    # Whether to perform post-processing
    post_process: bool = True

    # Whether to generate lemma edges
    lemmas: bool = False

    # Whether to perform coreference resolution
    corefs: bool = False

    # Maximum text length for coreference (0 = no limit, -1 = disable chunking)
    max_coref_text: int = 1500

    # Chunk overlap for long text processing
    chunk_overlap: int = 100

    # Beam search configuration
    # Use beam search instead of greedy parsing (more accurate but slower)
    use_beam_search: bool = False

    # Beam width for beam search parsing (higher = more candidates, slower)
    beam_width: int = 3

    # Maximum beam search iterations before fallback to greedy
    beam_max_iterations: int = 100


@dataclass
class LoggingConfig:
    """Configuration for logging behavior."""

    # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    level: str = 'INFO'

    # Whether to log model loading
    log_model_loading: bool = True

    # Whether to log parse timings
    log_timings: bool = False

    # Whether to log detailed parse steps (verbose)
    verbose: bool = False


@dataclass
class ParserSystemConfig:
    """Complete parser system configuration."""

    # Environment mode: 'development' or 'production'
    mode: str = 'production'

    # Model configuration
    models: ModelConfig = field(default_factory=ModelConfig)

    # Parser behavior configuration
    parser: ParserConfig = field(default_factory=ParserConfig)

    # Logging configuration
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_dict(cls, data: dict) -> 'ParserSystemConfig':
        """Create configuration from dictionary."""
        models_data = data.get('models', {})
        parser_data = data.get('parser', {})
        logging_data = data.get('logging', {})

        return cls(
            mode=data.get('mode', 'production'),
            models=ModelConfig(**models_data) if models_data else ModelConfig(),
            parser=ParserConfig(**parser_data) if parser_data else ParserConfig(),
            logging=LoggingConfig(**logging_data) if logging_data else LoggingConfig(),
        )

    @classmethod
    def from_yaml(cls, path: str) -> 'ParserSystemConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})

    @classmethod
    def from_json(cls, path: str) -> 'ParserSystemConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: Optional[str] = None) -> 'ParserSystemConfig':
        """Load configuration from file or environment.

        Searches for configuration in order:
        1. Explicit path parameter
        2. GRAPHBRAIN_CONFIG environment variable
        3. ./graphbrain.yaml or ./graphbrain.json in current directory
        4. ~/.graphbrain/config.yaml or config.json
        5. Default configuration
        """
        # Check explicit path
        if path:
            return cls._load_from_path(path)

        # Check environment variable
        env_path = os.environ.get('GRAPHBRAIN_CONFIG')
        if env_path and os.path.exists(env_path):
            logger.info(f"Loading parser config from GRAPHBRAIN_CONFIG: {env_path}")
            return cls._load_from_path(env_path)

        # Check current directory
        for filename in ['graphbrain.yaml', 'graphbrain.yml', 'graphbrain.json']:
            if os.path.exists(filename):
                logger.info(f"Loading parser config from: {filename}")
                return cls._load_from_path(filename)

        # Check home directory
        home_config = Path.home() / '.graphbrain'
        for filename in ['config.yaml', 'config.yml', 'config.json']:
            config_path = home_config / filename
            if config_path.exists():
                logger.info(f"Loading parser config from: {config_path}")
                return cls._load_from_path(str(config_path))

        # Return default configuration
        logger.debug("Using default parser configuration")
        return cls()

    @classmethod
    def _load_from_path(cls, path: str) -> 'ParserSystemConfig':
        """Load configuration from a specific path."""
        if path.endswith('.json'):
            return cls.from_json(path)
        else:
            return cls.from_yaml(path)

    @classmethod
    def development(cls) -> 'ParserSystemConfig':
        """Create a development-oriented configuration.

        Uses smaller models and more verbose logging.
        """
        return cls(
            mode='development',
            models=ModelConfig(
                preferred='en_core_web_sm',
                fallbacks=['en_core_web_md'],
            ),
            parser=ParserConfig(
                beta='repair',
                corefs=False,  # Disable corefs for faster dev iteration
            ),
            logging=LoggingConfig(
                level='DEBUG',
                log_model_loading=True,
                log_timings=True,
                verbose=True,
            ),
        )

    @classmethod
    def production(cls) -> 'ParserSystemConfig':
        """Create a production-oriented configuration.

        Uses highest quality models with minimal logging.
        """
        return cls(
            mode='production',
            models=ModelConfig(
                preferred='en_core_web_trf',
                fallbacks=['en_core_web_lg'],
            ),
            parser=ParserConfig(
                beta='repair',
                normalize=True,
                post_process=True,
            ),
            logging=LoggingConfig(
                level='WARNING',
                log_model_loading=False,
                log_timings=False,
                verbose=False,
            ),
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'mode': self.mode,
            'models': {
                'preferred': self.models.preferred,
                'fallbacks': self.models.fallbacks,
                'coreference': self.models.coreference,
                'coreference_requires': self.models.coreference_requires,
            },
            'parser': {
                'beta': self.parser.beta,
                'normalize': self.parser.normalize,
                'post_process': self.parser.post_process,
                'lemmas': self.parser.lemmas,
                'corefs': self.parser.corefs,
                'max_coref_text': self.parser.max_coref_text,
                'chunk_overlap': self.parser.chunk_overlap,
                'use_beam_search': self.parser.use_beam_search,
                'beam_width': self.parser.beam_width,
                'beam_max_iterations': self.parser.beam_max_iterations,
            },
            'logging': {
                'level': self.logging.level,
                'log_model_loading': self.logging.log_model_loading,
                'log_timings': self.logging.log_timings,
                'verbose': self.logging.verbose,
            },
        }

    def save_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def save_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Global configuration instance (lazily loaded)
_config: Optional[ParserSystemConfig] = None


def get_config() -> ParserSystemConfig:
    """Get the global parser configuration.

    Loads configuration on first access.
    """
    global _config
    if _config is None:
        _config = ParserSystemConfig.load()
    return _config


def set_config(config: ParserSystemConfig) -> None:
    """Set the global parser configuration."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration to trigger reload."""
    global _config
    _config = None
