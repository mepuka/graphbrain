# Graphbrain Core Concepts

> LLM-optimized reference for understanding Semantic Hypergraphs

## What is a Semantic Hypergraph?

A **Semantic Hypergraph (SH)** is a knowledge representation that bridges natural language and formal logic. It extends traditional graphs in two critical ways:

1. **N-ary relationships**: Edges connect any number of entities (not just pairs)
2. **Recursive structure**: Edges can contain other edges (relationships about relationships)

**Key insight**: Natural language is recursive ("Mary believes that John said that...") and expresses n-ary relations ("Alice gave Bob a book"). Traditional graphs cannot capture this; hypergraphs can.

## Hyperedge Structure

Every hyperedge follows one rule:
```
(connector arg1 arg2 ... argN)
```

- **Connector**: First element, defines HOW arguments relate
- **Arguments**: Can be atoms OR other hyperedges (recursion)

### Examples

```
(is/P berlin/C nice/C)           # Berlin is nice
(of/B capital/C germany/C)       # capital of Germany
(claims/P mary/C (is/P berlin/C nice/C))  # Mary claims Berlin is nice
```

The third example shows recursion: the fact "Berlin is nice" is itself an argument to "claims".

## The 8 Hyperedge Types

| Code | Type | Purpose | Example |
|------|------|---------|---------|
| **C** | Concept | Entities, things | `apple/C`, `berlin/Cp` |
| **P** | Predicate | Actions, states, relations | `(is/P sky/C blue/C)` |
| **M** | Modifier | Adjectives, adverbs | `(red/M apple/C)` |
| **B** | Builder | Compound concepts | `(of/B capital/C germany/C)` |
| **T** | Trigger | Conditions, specifiers | `(in/T 1994/C)` |
| **J** | Conjunction | Sequences, lists | `(and/J meat/C potatoes/C)` |
| **R** | Relation | (Inferred) Complete propositions | Result of P with arguments |
| **S** | Specifier | (Inferred) Relation modifiers | Result of T with arguments |

**Type inference**: R and S are never explicit; they're inferred from structure.

## Atom Anatomy

Atoms encode rich information in a compact string:

```
published/Pd.sox.<f-----/en
│         │  │   │       └── namespace (language: English)
│         │  │   └── verb features (tense, form, aspect, etc.)
│         │  └── argument roles (subject, object, specification)
│         └── subtype (declarative predicate)
└── root (lemma/word)
```

### Common Subtypes

**Concepts (C)**:
- `Cc` = common noun (apple)
- `Cp` = proper noun (Mary)
- `Cn` = number (27)
- `Ci` = pronoun (she)

**Predicates (P)**:
- `Pd` = declarative (states fact)
- `P?` = interrogative (question)
- `P!` = imperative (command)

**Modifiers (M)**:
- `Ma` = adjective (green)
- `Md` = determiner (the, a)
- `Mv` = verbal (will, have been)

## Argument Roles

Predicates specify what role each argument plays:

| Code | Role | Example |
|------|------|---------|
| s | subject | `(runs/Pd.s alice/C)` - Alice runs |
| o | object | `(reads/Pd.so alice/C book/C)` - Alice reads book |
| p | passive subject | `(written/Pd.p book/C)` - Book was written |
| c | complement | `(is/Pd.sc sky/C blue/C)` - Sky is blue |
| x | specification | `(runs/Pd.sx alice/C (in/T park/C))` |
| r | relative | For relative clauses |
| t | parataxis | Parenthetical remarks |

**Reading roles**: In `(says/Pd.sr mary/C hello/C)`:
- Position 1 (mary) = subject (s)
- Position 2 (hello) = recipient/relative (r)

## Key Operations

### Creating Edges
```python
from graphbrain import hedge, hgraph

edge = hedge('(is/P sky/C blue/C)')  # Parse string to edge
hg = hgraph('mydb.db')               # Open/create database
hg.add(edge)                          # Store edge
```

### Querying
```python
# Pattern search with wildcards
list(hg.search('(says/P * *)'))       # All "says" predicates

# Check existence
hg.exists(edge)                        # True/False

# Get neighborhood
list(hg.star(hedge('mary/C')))        # All edges containing mary
```

### Pattern Matching
```python
from graphbrain.patterns import match_pattern

pattern = hedge('(says/P.{sr} SPEAKER/C *)')
edge = hedge('(says/Pd.sr mary/Cp hello/C)')
match_pattern(edge, pattern)  # [{'SPEAKER': mary/Cp}]
```

## Special Relations

**Coreference** (same entity, different forms):
```
(coref/P/. turing/Cp (+/B/. alan/Cp turing/Cp))
```

**Taxonomy** (type hierarchy):
```
(type_of/P/. (black/M cat/C) cat/C)  # black cat IS-A cat
```

**Lemma** (word roots):
```
(lemma/P/. published/P publish/P)  # published -> publish
```

## Common Patterns

**Claim attribution** (who said what):
```
(says/Pd.sr SOURCE/C CLAIM/R)
(claims/Pd.sr SOURCE/C CLAIM/R)
(believes/Pd.sr SOURCE/C CLAIM/R)
```

**Concept modification**:
```
(MODIFIER/M CONCEPT/C)     # red apple
(of/B.ma MAIN/C AUX/C)     # city of Berlin
```

**Conditional/temporal**:
```
(when/Tt CONDITION/R)      # when X happens
(if/Tc CONDITION/R)        # if X is true
```

## Parser Usage

```python
from graphbrain.parsers import create_parser

parser = create_parser(lang='en')
result = parser.parse('Mary says that the sky is blue.')

main_edge = result['parses'][0]['main_edge']
# (says/Pd.sr mary/Cp (that/T (is/Pd.sc (the/Md sky/Cc) blue/Ca)))
```

**Parser options**:
- `lang='en'` - Language (currently English)
- `lemmas=True` - Generate lemma relations
- `corefs=True` - Resolve coreferences (she -> Mary)

## Key Principles for LLMs

1. **Connector-first**: Always read the first element to understand the relationship type
2. **Type matters**: The type annotation (after `/`) tells you what something IS
3. **Roles matter**: Argument roles (`.so`, `.sr`) tell you WHO does WHAT
4. **Recursion is power**: Nested edges let you represent meta-knowledge (claims about claims)
5. **Patterns are queries**: Use wildcards (`*`) and variables (`VAR/C`) to search and extract
