# Graphbrain LLM Documentation

> Optimized reference materials for LLM usage with Semantic Hypergraphs

## Contents

| Document | Purpose | Size |
|----------|---------|------|
| [01-core-concepts.md](01-core-concepts.md) | Foundation understanding of SH | ~3K tokens |
| [02-notation-reference.md](02-notation-reference.md) | Compact syntax reference | ~2K tokens |
| [03-pattern-matching.md](03-pattern-matching.md) | Query and extraction patterns | ~2K tokens |
| [04-api-patterns.md](04-api-patterns.md) | Python API usage examples | ~3K tokens |
| [05-system-prompts.md](05-system-prompts.md) | Composable prompt templates | ~3K tokens |
| [06-cheat-sheet.md](06-cheat-sheet.md) | Ultra-compact quick reference | ~500 tokens |

## Usage Guidelines

### For System Prompts

1. **Always include** `01-core-concepts.md` or the Core Knowledge block from `05-system-prompts.md`
2. **Add notation reference** if the task involves reading or writing SH notation
3. **Add pattern matching** if the task involves querying
4. **Add API patterns** if generating Python code
5. **Add MCP tools** if using tool-calling

### Recommended Combinations

| Task | Documents |
|------|-----------|
| General understanding | 01 only |
| Reading/interpreting edges | 01 + 02 |
| Building queries | 01 + 02 + 03 |
| Writing Python code | 01 + 04 |
| Tool-using agent | 01 + 05 (MCP block) |
| Full capability | All docs |

### Token Budget Management

- **Minimal context** (~500 tokens): Use 06-cheat-sheet.md
- **Basic context** (~3K tokens): Use 01-core-concepts.md
- **Working context** (~5K tokens): 01 + 02
- **Full context** (~13K tokens): All documents

### Integration Patterns

**Static inclusion**: Paste entire documents into system prompt

**Dynamic inclusion**: Load relevant sections based on task type

**Retrieval-augmented**: Index documents, retrieve relevant sections

## Key Principles

1. **Connector-first**: The first element of any edge defines the relationship type
2. **Type annotations matter**: `/Type.roles.features` encodes crucial semantic info
3. **Recursion is power**: Edges containing edges enable meta-knowledge
4. **Patterns are edges**: Query patterns are valid hyperedges
5. **Variables capture**: UPPERCASE atoms in patterns extract matched values
