#!/usr/bin/env python3
"""Test Agent SDK workflows with graphbrain MCP server.

This script tests the full agent workflow:
1. Starts the MCP server as a subprocess
2. Uses Claude Agent SDK to query the graphbrain tools
3. Tests various skill-based workflows

Usage:
    # Requires ANTHROPIC_API_KEY in environment or .env
    python scripts/test_agent_workflow.py

    # Test specific workflow
    python scripts/test_agent_workflow.py --workflow query
    python scripts/test_agent_workflow.py --workflow classify
    python scripts/test_agent_workflow.py --workflow analyze
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


# MCP server configuration - use existing .mcp.json
MCP_CONFIG_PATH = Path(__file__).parent.parent / ".mcp.json"

# Use full Claude CLI for MCP support
CLAUDE_CLI_PATH = "/opt/homebrew/opt/pnpm/bin/claude"


async def test_query_workflow():
    """Test the graphbrain-query skill workflow."""
    from claude_agent_sdk import query, ClaudeAgentOptions

    logger.info("=== Testing Query Workflow ===")

    prompt = """Use the graphbrain MCP tools to:
1. Get hypergraph statistics using hypergraph_stats
2. Search for edges about "housing policy" using bm25_search (limit 3)
3. Find edges with root "harrell" using edges_with_root (limit 3)

Report what you find in each step.
"""

    options = ClaudeAgentOptions(
        mcp_servers=MCP_CONFIG_PATH,
        max_turns=10,
        permission_mode="bypassPermissions",
        cwd=Path(__file__).parent.parent,  # Set working dir to project root
        cli_path=CLAUDE_CLI_PATH,  # Use full Claude CLI for MCP support
    )

    result_text = ""
    async for message in query(prompt=prompt, options=options):
        if hasattr(message, 'content'):
            result_text += str(message.content)

    logger.info(f"Query workflow result:\n{result_text[:1000]}...")
    return result_text


async def test_classify_workflow():
    """Test the graphbrain-classify skill workflow."""
    from claude_agent_sdk import query, ClaudeAgentOptions

    logger.info("=== Testing Classify Workflow ===")

    prompt = """Use the graphbrain MCP tools to:
1. List semantic classes using list_semantic_classes
2. Get classification stats using classification_stats
3. Try to classify the predicate "announce" using classify_predicate

Report what you find.
"""

    options = ClaudeAgentOptions(
        mcp_servers=MCP_CONFIG_PATH,
        max_turns=10,
        permission_mode="bypassPermissions",
        cwd=Path(__file__).parent.parent,
        cli_path=CLAUDE_CLI_PATH,
    )

    result_text = ""
    async for message in query(prompt=prompt, options=options):
        if hasattr(message, 'content'):
            result_text += str(message.content)

    logger.info(f"Classify workflow result:\n{result_text[:1000]}...")
    return result_text


async def test_analyze_workflow():
    """Test the graphbrain-analyze skill workflow."""
    from claude_agent_sdk import query, ClaudeAgentOptions

    logger.info("=== Testing Analyze Workflow ===")

    prompt = """Use the graphbrain MCP tools to:
1. Get graph statistics using get_graph_stats
2. Compute PageRank centrality using compute_centrality with algorithm="pagerank" and top_k=10
3. Find communities using find_communities with min_size=3

Report the key findings from each analysis.
"""

    options = ClaudeAgentOptions(
        mcp_servers=MCP_CONFIG_PATH,
        max_turns=15,
        permission_mode="bypassPermissions",
        cwd=Path(__file__).parent.parent,
        cli_path=CLAUDE_CLI_PATH,
    )

    result_text = ""
    async for message in query(prompt=prompt, options=options):
        if hasattr(message, 'content'):
            result_text += str(message.content)

    logger.info(f"Analyze workflow result:\n{result_text[:1000]}...")
    return result_text


async def test_learning_workflow():
    """Test the graphbrain-learning skill workflow."""
    from claude_agent_sdk import query, ClaudeAgentOptions

    logger.info("=== Testing Learning Workflow ===")

    prompt = """Use the graphbrain MCP tools to:
1. Get the quality dashboard using quality_dashboard
2. Get learning candidates using get_learning_candidates with limit=5 and strategy="uncertainty"
3. Get improvement suggestions using get_improvement_suggestions

Report what the quality metrics look like and what improvements are suggested.
"""

    options = ClaudeAgentOptions(
        mcp_servers=MCP_CONFIG_PATH,
        max_turns=10,
        permission_mode="bypassPermissions",
        cwd=Path(__file__).parent.parent,
        cli_path=CLAUDE_CLI_PATH,
    )

    result_text = ""
    async for message in query(prompt=prompt, options=options):
        if hasattr(message, 'content'):
            result_text += str(message.content)

    logger.info(f"Learning workflow result:\n{result_text[:1000]}...")
    return result_text


async def test_agent_session_workflow():
    """Test the graphbrain-agent skill workflow."""
    from claude_agent_sdk import query, ClaudeAgentOptions

    logger.info("=== Testing Agent Session Workflow ===")

    prompt = """Use the graphbrain MCP tools to:
1. Create an agent session using create_agent_session with agent_type="analysis" and description="Test session"
2. Log a decision using log_decision with appropriate parameters
3. Get the session state using get_session_state
4. List all sessions using list_sessions

Report the session details.
"""

    options = ClaudeAgentOptions(
        mcp_servers=MCP_CONFIG_PATH,
        max_turns=15,
        permission_mode="bypassPermissions",
        cwd=Path(__file__).parent.parent,
        cli_path=CLAUDE_CLI_PATH,
    )

    result_text = ""
    async for message in query(prompt=prompt, options=options):
        if hasattr(message, 'content'):
            result_text += str(message.content)

    logger.info(f"Agent session workflow result:\n{result_text[:1000]}...")
    return result_text


async def run_all_workflows():
    """Run all workflow tests."""
    results = {}

    workflows = [
        ("query", test_query_workflow),
        ("classify", test_classify_workflow),
        ("analyze", test_analyze_workflow),
        ("learning", test_learning_workflow),
        ("agent", test_agent_session_workflow),
    ]

    for name, test_func in workflows:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running {name} workflow...")
            logger.info('='*60)
            result = await test_func()
            results[name] = {"status": "success", "length": len(result)}
            logger.info(f"{name} workflow: SUCCESS")
        except Exception as e:
            results[name] = {"status": "error", "error": str(e)}
            logger.error(f"{name} workflow: FAILED - {e}")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("WORKFLOW TEST SUMMARY")
    logger.info("="*60)
    for name, result in results.items():
        status = "✓" if result["status"] == "success" else "✗"
        logger.info(f"  {status} {name}: {result['status']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test Agent SDK workflows")
    parser.add_argument(
        "--workflow", "-w",
        choices=["query", "classify", "analyze", "learning", "agent", "all"],
        default="all",
        help="Which workflow to test"
    )
    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not set. Add to .env or export it.")
        sys.exit(1)

    # Check database exists
    db_path = Path(__file__).parent.parent / "urbanist.db"
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        logger.error("Run: python scripts/ingest_urbanist.py --fresh")
        sys.exit(1)

    logger.info(f"Using database: {db_path}")

    # Run workflows
    if args.workflow == "all":
        asyncio.run(run_all_workflows())
    else:
        workflow_map = {
            "query": test_query_workflow,
            "classify": test_classify_workflow,
            "analyze": test_analyze_workflow,
            "learning": test_learning_workflow,
            "agent": test_agent_session_workflow,
        }
        asyncio.run(workflow_map[args.workflow]())


if __name__ == "__main__":
    main()
