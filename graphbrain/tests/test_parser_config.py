"""Tests for parser configuration system."""
import json
import tempfile
import unittest
import os

from graphbrain.parsers.config import (
    ParserSystemConfig, ModelConfig, ParserConfig, LoggingConfig,
    get_config, set_config, reset_config
)


class TestModelConfig(unittest.TestCase):
    """Tests for ModelConfig dataclass."""

    def test_default_values(self):
        """Default values are set correctly."""
        config = ModelConfig()
        self.assertEqual(config.preferred, 'en_core_web_trf')
        self.assertIn('en_core_web_lg', config.fallbacks)
        self.assertEqual(config.coreference, 'en_coreference_web_trf')

    def test_custom_values(self):
        """Custom values can be set."""
        config = ModelConfig(
            preferred='en_core_web_sm',
            fallbacks=['en_core_web_md']
        )
        self.assertEqual(config.preferred, 'en_core_web_sm')
        self.assertEqual(config.fallbacks, ['en_core_web_md'])


class TestParserConfig(unittest.TestCase):
    """Tests for ParserConfig dataclass."""

    def test_default_values(self):
        """Default values are set correctly."""
        config = ParserConfig()
        self.assertEqual(config.beta, 'repair')
        self.assertTrue(config.normalize)
        self.assertTrue(config.post_process)
        self.assertFalse(config.lemmas)
        self.assertFalse(config.corefs)


class TestParserSystemConfig(unittest.TestCase):
    """Tests for ParserSystemConfig."""

    def test_default_config(self):
        """Default configuration is production mode."""
        config = ParserSystemConfig()
        self.assertEqual(config.mode, 'production')

    def test_from_dict(self):
        """Configuration loads from dictionary."""
        data = {
            'mode': 'development',
            'models': {'preferred': 'en_core_web_sm'},
            'parser': {'beta': 'strict'},
            'logging': {'level': 'DEBUG'}
        }
        config = ParserSystemConfig.from_dict(data)
        self.assertEqual(config.mode, 'development')
        self.assertEqual(config.models.preferred, 'en_core_web_sm')
        self.assertEqual(config.parser.beta, 'strict')
        self.assertEqual(config.logging.level, 'DEBUG')

    def test_from_yaml(self):
        """Configuration loads from YAML file."""
        yaml_content = """
mode: development
models:
  preferred: en_core_web_sm
parser:
  beta: strict
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                config = ParserSystemConfig.from_yaml(f.name)
                self.assertEqual(config.mode, 'development')
                self.assertEqual(config.models.preferred, 'en_core_web_sm')
            finally:
                os.unlink(f.name)

    def test_from_json(self):
        """Configuration loads from JSON file."""
        json_content = {
            'mode': 'development',
            'models': {'preferred': 'en_core_web_lg'}
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_content, f)
            f.flush()
            try:
                config = ParserSystemConfig.from_json(f.name)
                self.assertEqual(config.mode, 'development')
                self.assertEqual(config.models.preferred, 'en_core_web_lg')
            finally:
                os.unlink(f.name)

    def test_development_preset(self):
        """Development preset uses smaller models."""
        config = ParserSystemConfig.development()
        self.assertEqual(config.mode, 'development')
        self.assertEqual(config.models.preferred, 'en_core_web_sm')
        self.assertEqual(config.logging.level, 'DEBUG')
        self.assertTrue(config.logging.verbose)

    def test_production_preset(self):
        """Production preset uses high-quality models."""
        config = ParserSystemConfig.production()
        self.assertEqual(config.mode, 'production')
        self.assertEqual(config.models.preferred, 'en_core_web_trf')
        self.assertEqual(config.logging.level, 'WARNING')

    def test_to_dict_roundtrip(self):
        """Configuration survives dict roundtrip."""
        original = ParserSystemConfig.development()
        data = original.to_dict()
        restored = ParserSystemConfig.from_dict(data)
        self.assertEqual(original.mode, restored.mode)
        self.assertEqual(original.models.preferred, restored.models.preferred)
        self.assertEqual(original.logging.level, restored.logging.level)

    def test_save_yaml(self):
        """Configuration saves to YAML file."""
        config = ParserSystemConfig.development()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            try:
                config.save_yaml(f.name)
                loaded = ParserSystemConfig.from_yaml(f.name)
                self.assertEqual(config.mode, loaded.mode)
            finally:
                os.unlink(f.name)

    def test_save_json(self):
        """Configuration saves to JSON file."""
        config = ParserSystemConfig.production()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            try:
                config.save_json(f.name)
                loaded = ParserSystemConfig.from_json(f.name)
                self.assertEqual(config.mode, loaded.mode)
            finally:
                os.unlink(f.name)


class TestGlobalConfig(unittest.TestCase):
    """Tests for global configuration management."""

    def setUp(self):
        reset_config()

    def tearDown(self):
        reset_config()

    def test_get_config_returns_default(self):
        """get_config returns default configuration."""
        config = get_config()
        self.assertIsInstance(config, ParserSystemConfig)

    def test_set_config(self):
        """set_config sets global configuration."""
        custom = ParserSystemConfig.development()
        set_config(custom)
        retrieved = get_config()
        self.assertEqual(retrieved.mode, 'development')

    def test_reset_config(self):
        """reset_config clears global configuration."""
        set_config(ParserSystemConfig.development())
        reset_config()
        # After reset, get_config should return a fresh default
        config = get_config()
        self.assertEqual(config.mode, 'production')


if __name__ == '__main__':
    unittest.main()
