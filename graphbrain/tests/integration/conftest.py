"""Shared fixtures for integration tests."""

import os
import pytest
from pathlib import Path

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from graphbrain import hgraph, hedge


# Test markers
def pytest_configure(config):
    config.addinivalue_line("markers", "llm: tests requiring LLM API access")
    config.addinivalue_line("markers", "slow: tests taking >5 seconds")
    config.addinivalue_line("markers", "urbanist: tests using urbanist dataset")
    config.addinivalue_line("markers", "asyncio: mark test as async")


@pytest.fixture(scope="session")
def anthropic_api_key():
    """Get Anthropic API key from environment."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set - set it to run LLM tests")
    return key


@pytest.fixture(scope="session")
def llm_client(anthropic_api_key):
    """Real Anthropic LLM client using Haiku for speed."""
    from graphbrain.agents.llm.providers.anthropic import AnthropicProvider

    return AnthropicProvider(
        api_key=anthropic_api_key,
        model="claude-3-haiku-20240307"
    )


@pytest.fixture(scope="function")
def test_hg(tmp_path):
    """Fresh test hypergraph for each test."""
    db_path = tmp_path / "test.db"
    hg = hgraph(str(db_path))
    yield hg
    hg.close()


@pytest.fixture(scope="function")
def populated_hg(test_hg):
    """Test hypergraph with sample edges."""
    edges = [
        "(says/Pd.sr mayor/Cp (will/Mv (build/Pd.so housing/Cc)))",
        "(announced/Pd.sr council/Cc (the/Md budget/Cc))",
        "(supports/Pd.sr harrell/Cp (the/Md (transit/Ma plan/Cc)))",
        "(criticized/Pd.sr (the/Md opposition/Cc) (the/Md policy/Cc))",
        "(is/Pd.sc seattle/Cp (a/Md city/Cc))",
        "(proposed/Pd.sr sdot/Cp (new/Ma (bike/Ma lanes/Cc)))",
    ]
    for edge_str in edges:
        test_hg.add(hedge(edge_str))
    return test_hg


@pytest.fixture(scope="session")
def urbanist_hg():
    """Real urbanist dataset - skips if not available."""
    db_path = Path("urbanist.db")
    if not db_path.exists():
        pytest.skip("urbanist.db not found - run scraper first")
    return hgraph(str(db_path))


@pytest.fixture(scope="session")
def parser():
    """English parser instance (shared for performance)."""
    from graphbrain.parsers import create_parser

    return create_parser(lang="en", lemmas=True, corefs=False)


@pytest.fixture(scope="function")
def classification_backend(tmp_path):
    """SQLite classification backend."""
    from graphbrain.classification.backends.sqlite import SqliteBackend

    db_path = tmp_path / "classification.db"
    backend = SqliteBackend(str(db_path))
    yield backend
    backend.close()
