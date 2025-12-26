# Graphbrain Agent SDK Integration Design

> Phase 4 Implementation: Claude Agent SDK Integration for Knowledge Extraction and Analysis

## Executive Summary

This document defines the architecture for integrating Claude's Agent SDK with graphbrain's semantic hypergraph system. The integration creates a multi-agent system for:

- **Knowledge Extraction**: Transform natural language into semantic hyperedges
- **Query & Exploration**: Navigate the hypergraph with natural language
- **Classification**: Categorize predicates using hybrid search
- **Analysis**: Extract insights from the knowledge graph
- **Human Feedback**: Iterative refinement through human-in-the-loop

## SDK Integration Architecture

### Claude Agent SDK Overview

The SDK provides two interaction patterns:

| Pattern | Use Case | Features |
|---------|----------|----------|
| `query()` | Stateless, one-off tasks | Simple, no session management |
| `ClaudeSDKClient` | Stateful, multi-turn conversations | Session persistence, interrupts, hooks |

For graphbrain, we'll use **`ClaudeSDKClient`** for its session persistence and hooks capabilities.

### Integration Pattern

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Claude Agent SDK Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │ Extraction  │  │   Query     │  │  Classification             │  │
│  │   Agent     │  │   Agent     │  │     Agent                   │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────────┬──────────────┘  │
└─────────┼────────────────┼─────────────────────────┼────────────────┘
          │                │                         │
          ▼                ▼                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Graphbrain MCP Server (27 existing tools)               │
│  ┌───────────────┐  ┌─────────────────┐  ┌─────────────────────┐    │
│  │ Hypergraph    │  │ Classification  │  │ Semantic Class      │    │
│  │ Tools (6)     │  │ Tools (4)       │  │ Tools (7)           │    │
│  └───────────────┘  └─────────────────┘  └─────────────────────┘    │
│  ┌───────────────┐  ┌─────────────────┐                              │
│  │ Predicates    │  │ Feedback        │                              │
│  │ Tools (5)     │  │ Tools (5)       │                              │
│  └───────────────┘  └─────────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│               PostgreSQL (Unified Database)                          │
│  ┌───────────────┐  ┌─────────────────┐  ┌─────────────────────┐    │
│  │ edges +       │  │ semantic_classes│  │ classification_     │    │
│  │ embeddings    │  │ predicate_banks │  │ feedback            │    │
│  └───────────────┘  └─────────────────┘  └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

## Agent Skill Definitions

### 1. Knowledge Extraction Agent

**Purpose**: Parse text and extract semantic hyperedges

**SDK Configuration**:
```python
from claude_agent_sdk import ClaudeAgentOptions, AgentDefinition

extraction_agent = AgentDefinition(
    description="Extract semantic relations from text into hyperedges",
    prompt=EXTRACTION_SYSTEM_PROMPT,  # See below
    tools=[
        "mcp__graphbrain__add_edge",
        "mcp__graphbrain__get_edge",
        "mcp__graphbrain__pattern_match",
        "mcp__graphbrain__flag_for_review",
    ],
    model="sonnet"
)
```

**MCP Tools Used**:
- `add_edge` - Store extracted edges
- `get_edge` - Check for duplicates
- `pattern_match` - Validate structure
- `flag_for_review` - Mark uncertain extractions

**System Prompt** (stored in `docs/llm/agent-prompts/extraction.md`):
```
You are a knowledge extraction agent for graphbrain semantic hypergraphs.

<graphbrain-context>
[Core Knowledge block from docs/llm/05-system-prompts.md]
</graphbrain-context>

<sh-notation>
[Notation Reference block]
</sh-notation>

WORKFLOW:
1. Parse input text sentence by sentence
2. For each sentence, identify:
   - Main predicate (action/state) -> Type P
   - Arguments (subject, object, complements) -> Type C
   - Modifiers -> Type M
   - Source attribution if present
3. Construct hyperedge in SH notation
4. Check for existing similar edges via pattern_match
5. Add edge with source attribution
6. Flag uncertain extractions (confidence < 0.7) for review

CRITICAL RULES:
- Never fabricate entities not in source text
- Always include source attribution in edge attributes
- Use proper type annotations (Pd.sr for declarative with subject/recipient)
- Mark edges with confidence score
```

### 2. Query & Exploration Agent

**Purpose**: Natural language queries over the hypergraph

**SDK Configuration**:
```python
query_agent = AgentDefinition(
    description="Query and explore the semantic hypergraph",
    prompt=QUERY_SYSTEM_PROMPT,
    tools=[
        "mcp__graphbrain__search_edges",
        "mcp__graphbrain__pattern_match",
        "mcp__graphbrain__hybrid_search",
        "mcp__graphbrain__bm25_search",
        "mcp__graphbrain__edges_with_root",
        "mcp__graphbrain__hypergraph_stats",
    ],
    model="sonnet"
)
```

**Query Translation Patterns**:
```
"Who said X?" -> pattern_match("(*/Pd.{sr} SPEAKER/Cp *)")
"What did X do?" -> edges_with_root(X) + filter by subject role
"Find claims about Y" -> hybrid_search(Y, class_id="claim")
"How is X related to Y?" -> pattern_match with both entities
```

### 3. Classification Agent

**Purpose**: Categorize predicates and edges into semantic classes

**SDK Configuration**:
```python
classification_agent = AgentDefinition(
    description="Classify predicates using semantic similarity and predicate banks",
    prompt=CLASSIFICATION_SYSTEM_PROMPT,
    tools=[
        "mcp__graphbrain__classify_predicate",
        "mcp__graphbrain__classify_edge",
        "mcp__graphbrain__discover_predicates",
        "mcp__graphbrain__find_similar_predicates",
        "mcp__graphbrain__get_predicate_classes",
        "mcp__graphbrain__list_predicates_by_class",
        "mcp__graphbrain__flag_for_review",
    ],
    model="sonnet"
)
```

**Classification Workflow**:
1. For new predicates, check existing classifications
2. If unclassified, use semantic similarity
3. Apply confidence thresholds (see below)
4. Flag low-confidence for human review

### 4. Analysis Agent

**Purpose**: Extract insights and generate reports

**SDK Configuration**:
```python
analysis_agent = AgentDefinition(
    description="Analyze the knowledge graph for insights and patterns",
    prompt=ANALYSIS_SYSTEM_PROMPT,
    tools=[
        "mcp__graphbrain__pattern_match",
        "mcp__graphbrain__hybrid_search",
        "mcp__graphbrain__hypergraph_stats",
        "mcp__graphbrain__list_semantic_classes",
        "mcp__graphbrain__classification_stats",
    ],
    model="opus"  # More capable for analysis
)
```

**Analysis Modes**:
- Actor Analysis: Identify entities making claims
- Claim Analysis: Extract attributed statements
- Network Analysis: Find concept connections
- Conflict Detection: Identify opposing claims

### 5. Feedback Agent

**Purpose**: Manage human-in-the-loop improvement

**SDK Configuration**:
```python
feedback_agent = AgentDefinition(
    description="Process human feedback to improve classification quality",
    prompt=FEEDBACK_SYSTEM_PROMPT,
    tools=[
        "mcp__graphbrain__get_pending_reviews",
        "mcp__graphbrain__apply_feedback",
        "mcp__graphbrain__submit_feedback",
        "mcp__graphbrain__feedback_stats",
        "mcp__graphbrain__add_predicate_to_class",
    ],
    model="haiku"  # Fast for review processing
)
```

## Memory Architecture

### Session Memory (ClaudeSDKClient)

The SDK's session system provides conversation continuity:

```python
async with ClaudeSDKClient(options=options) as client:
    # First extraction
    await client.query("Extract knowledge from article about Seattle transit")
    # Follow-up - context preserved
    await client.query("Now classify the predicates you extracted")
```

### Working Memory (New Tables)

```sql
-- Agent session state
CREATE TABLE agent_sessions (
    session_id TEXT PRIMARY KEY,
    agent_type TEXT NOT NULL,  -- extraction, query, classification, analysis, feedback
    user_id TEXT,
    domain TEXT DEFAULT 'default',
    state JSONB NOT NULL,      -- Full session state
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Decision audit trail
CREATE TABLE agent_decisions (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES agent_sessions(session_id),
    decision_type TEXT NOT NULL,  -- classify, add_edge, apply_feedback
    input JSONB NOT NULL,
    output JSONB NOT NULL,
    confidence REAL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_decisions_session ON agent_decisions(session_id);
CREATE INDEX idx_decisions_type ON agent_decisions(decision_type);
```

### Long-Term Memory (Hypergraph)

Store agent decisions as edges for full auditability:

```
# Classification decision
(classified/Pd.sox agent/C predicate/C class/C confidence/Cn)

# Edge provenance
(extracted_from/Br edge/R source_url/C)

# Review chain
(reviewed/Pd.sox reviewer/C decision/C review_id/C)
```

## Confidence Thresholds

| Range | Meaning | Action |
|-------|---------|--------|
| 0.9-1.0 | High confidence | Auto-apply |
| 0.8-0.9 | Good confidence | Apply with logging |
| 0.7-0.8 | Moderate | Flag for optional review |
| 0.5-0.7 | Low | Require human review |
| < 0.5 | Very low | Reject or defer |

## Hooks for Safety

Use SDK hooks for validation and safety:

```python
from claude_agent_sdk import HookMatcher

async def validate_edge_syntax(input_data, tool_use_id, context):
    """PreToolUse hook to validate edge syntax before adding."""
    if input_data['tool_name'] == 'mcp__graphbrain__add_edge':
        edge_str = input_data['tool_input'].get('edge', '')
        # Validate syntax
        try:
            hedge(edge_str)
        except Exception as e:
            return {
                'hookSpecificOutput': {
                    'hookEventName': 'PreToolUse',
                    'permissionDecision': 'deny',
                    'permissionDecisionReason': f'Invalid edge syntax: {e}'
                }
            }
    return {}

async def log_all_decisions(input_data, tool_use_id, context):
    """PostToolUse hook to log all tool calls for audit."""
    # Log to agent_decisions table
    return {}

options = ClaudeAgentOptions(
    hooks={
        'PreToolUse': [
            HookMatcher(
                matcher='mcp__graphbrain__add_edge',
                hooks=[validate_edge_syntax]
            )
        ],
        'PostToolUse': [
            HookMatcher(hooks=[log_all_decisions])
        ]
    }
)
```

## Implementation

### New Files to Create

```
graphbrain/
├── agents/
│   ├── __init__.py           # Agent factory
│   ├── config.py             # ClaudeAgentOptions builder
│   ├── skills/
│   │   ├── __init__.py
│   │   ├── extraction.py     # Extraction agent setup
│   │   ├── query.py          # Query agent setup
│   │   ├── classification.py # Classification agent setup
│   │   ├── analysis.py       # Analysis agent setup
│   │   └── feedback.py       # Feedback agent setup
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── session.py        # Session management
│   │   └── decisions.py      # Decision logging
│   └── hooks/
│       ├── __init__.py
│       ├── validation.py     # Input validation hooks
│       └── logging.py        # Audit logging hooks
│
docs/llm/
├── agent-prompts/
│   ├── extraction.md         # Extraction agent prompt
│   ├── query.md              # Query agent prompt
│   ├── classification.md     # Classification agent prompt
│   ├── analysis.md           # Analysis agent prompt
│   └── feedback.md           # Feedback agent prompt
```

### MCP Server Extensions

Add new tools for agent-specific operations:

```python
# graphbrain/mcp/tools/agents.py

@server.tool(name="create_agent_session")
async def create_agent_session(
    agent_type: str,
    domain: str = "default",
    user_id: Optional[str] = None,
) -> dict:
    """Create a new agent session for tracking state and decisions."""

@server.tool(name="get_session_state")
async def get_session_state(session_id: str) -> dict:
    """Retrieve current session state."""

@server.tool(name="log_decision")
async def log_decision(
    session_id: str,
    decision_type: str,
    input_data: dict,
    output_data: dict,
    confidence: float,
) -> dict:
    """Log an agent decision for audit trail."""
```

### Main Entry Point

```python
# graphbrain/agents/__init__.py

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
from graphbrain.mcp import create_graphbrain_server

def create_graphbrain_agent(
    agent_type: str,
    db_connection: str,
    domain: str = "default",
) -> ClaudeAgentOptions:
    """Factory to create configured agent options."""

    # Create MCP server with PostgreSQL connection
    server = create_graphbrain_server(db_connection)

    # Get agent definition by type
    from graphbrain.agents.skills import get_agent_definition
    agent_def = get_agent_definition(agent_type)

    # Build options
    return ClaudeAgentOptions(
        mcp_servers={"graphbrain": server},
        allowed_tools=agent_def.tools,
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            "append": agent_def.prompt
        },
        agents={agent_type: agent_def},
        hooks=get_hooks_for_agent(agent_type),
    )

# Usage
async def main():
    options = create_graphbrain_agent(
        "extraction",
        "postgresql://localhost/graphbrain"
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query("Extract knowledge from: 'The mayor announced new transit plans.'")
        async for message in client.receive_response():
            process_message(message)
```

## Scientific Best Practices

### 1. Uncertainty Quantification

Every classification includes confidence score with reasoning:

```python
{
    "predicate": "announce",
    "class": "claim",
    "confidence": 0.92,
    "method": "predicate_bank",
    "evidence": ["similar: declare(0.95), state(0.88)", "frequency: 45 occurrences"]
}
```

### 2. Provenance Tracking

Every edge includes full provenance:

```python
edge_attributes = {
    "source": "jina:https://example.com/article",
    "extracted_by": "extraction_agent:sess_abc123",
    "extraction_method": "parser",
    "confidence": 0.85,
    "timestamp": "2025-12-26T10:30:00Z",
    "original_text": "The mayor announced..."
}
```

### 3. Reproducibility

Decision logging enables full reproducibility:

```python
@dataclass
class DecisionLog:
    session_id: str
    timestamp: datetime
    decision_type: str
    input_text: Optional[str]
    input_edge: Optional[str]
    method: str  # "predicate_bank", "semantic", "pattern"
    intermediate_results: dict
    output_class: Optional[str]
    confidence: float
    model_version: str
```

### 4. Iterative Refinement

Active learning loop for continuous improvement:

1. Agent classifies predicate with confidence C
2. If C < threshold, flag for review
3. Human provides feedback
4. Update predicate bank and/or patterns
5. Analyze feedback patterns for systematic improvements

## Testing Strategy

### Unit Tests

```python
# tests/test_agents/test_extraction.py

async def test_extraction_agent_parses_simple_sentence():
    """Extraction agent correctly parses simple declarative sentence."""

async def test_extraction_agent_flags_uncertain():
    """Low-confidence extractions are flagged for review."""

async def test_extraction_agent_prevents_duplicates():
    """Agent checks for existing edges before adding."""
```

### Integration Tests

```python
# tests/test_agents/test_workflows.py

async def test_full_extraction_to_classification_workflow():
    """End-to-end: extract -> classify -> review -> apply feedback."""

async def test_multi_agent_coordination():
    """Multiple agents can share hypergraph without conflicts."""
```

### Evaluation Benchmarks

- Extraction precision/recall on annotated corpus
- Classification accuracy by semantic class
- Query response relevance scoring
- Feedback loop convergence rate

## Implementation Phases

### Phase 4.1: Core Infrastructure (Week 1)

- [ ] Create `graphbrain/agents/` module structure
- [ ] Add session management tables to PostgreSQL
- [ ] Implement decision logging
- [ ] Create agent factory function

### Phase 4.2: Agent Skills (Week 2)

- [ ] Implement extraction agent with prompts
- [ ] Implement query agent with pattern translation
- [ ] Implement classification agent
- [ ] Add hooks for validation and logging

### Phase 4.3: Advanced Features (Week 3)

- [ ] Implement analysis agent
- [ ] Implement feedback agent
- [ ] Add active learning suggestions
- [ ] Create quality metrics dashboard

### Phase 4.4: Testing & Documentation (Week 4)

- [ ] Unit tests for all agents
- [ ] Integration tests for workflows
- [ ] Evaluation benchmarks
- [ ] User documentation

## Dependencies

```python
# pyproject.toml additions
[project.optional-dependencies]
agents = [
    "claude-agent-sdk>=0.1.0",
]
```

## Open Questions

1. **Session persistence**: Store sessions in PostgreSQL or use SDK's built-in persistence?
2. **Multi-user support**: How to handle concurrent users with separate sessions?
3. **Model selection**: Which models for which agents? (Currently: sonnet for most, opus for analysis, haiku for feedback)
4. **Rate limiting**: How to handle API rate limits in production?

---

*Generated: 2025-12-26*
*Status: Design Complete - Ready for Implementation*
