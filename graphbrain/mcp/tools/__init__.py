"""MCP tools for graphbrain operations."""

from graphbrain.mcp.tools.hypergraph import register_hypergraph_tools
from graphbrain.mcp.tools.classification import register_classification_tools
from graphbrain.mcp.tools.semantic_classes import register_semantic_class_tools
from graphbrain.mcp.tools.predicates import register_predicate_tools
from graphbrain.mcp.tools.feedback import register_feedback_tools
from graphbrain.mcp.tools.algorithms import register_algorithm_tools

__all__ = [
    'register_hypergraph_tools',
    'register_classification_tools',
    'register_semantic_class_tools',
    'register_predicate_tools',
    'register_feedback_tools',
    'register_algorithm_tools',
]
