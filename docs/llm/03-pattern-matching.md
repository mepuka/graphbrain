# Pattern Matching Guide

> How to query and extract information from Semantic Hypergraphs

## Overview

Patterns are hyperedges with special placeholder atoms. They match edges in the hypergraph and optionally extract values.

**Two use cases**:
1. **Search**: Find edges matching a structure (`hg.search()`)
2. **Extract**: Match and capture specific parts (`match_pattern()`)

## Wildcards

### Basic Wildcards

| Pattern | Matches |
|---------|---------|
| `*` | Any edge (atomic or non-atomic) |
| `.` | Atoms only |
| `(*)` | Non-atomic edges only |

```python
# Find all predicates with any arguments
list(hg.search('(says/P * *)'))

# Find predicates where first arg is an atom
list(hg.search('(says/P . *)'))

# Find predicates where second arg is complex
list(hg.search('(says/P * (*))'))
```

### Typed Wildcards

Add type after wildcard to constrain:

| Pattern | Matches |
|---------|---------|
| `*/C` | Any concept |
| `*/P` | Any predicate |
| `*/Cp` | Any proper noun concept |
| `(*/P)` | Non-atomic predicates only |

```python
# Find edges with concept arguments
list(hg.search('(says/P */Cp */C)'))  # Proper noun says common noun
```

### Ellipsis

`...` matches zero or more additional arguments:

```python
# Match regardless of extra arguments
'(plays/P * * ...)'

# Matches:
# (plays/P alice/C chess/C)
# (plays/P alice/C chess/C (at/T club/C))
# (plays/P alice/C chess/C (at/T club/C) (with/T friends/C))
```

## Variables

Variables capture matched values. Any atom starting with uppercase is a variable:

```python
from graphbrain import hedge
from graphbrain.patterns import match_pattern

pattern = hedge('(says/P SPEAKER/C MESSAGE/*)')
edge = hedge('(says/Pd.sr mary/Cp hello/C)')

result = match_pattern(edge, pattern)
# [{'SPEAKER': mary/Cp, 'MESSAGE': hello/C}]
```

**Convention**: Use descriptive names like `SPEAKER`, `ACTION`, `TARGET`

## Argument Role Matching

### Strict Role Matching

Exact role sequence must match:

```python
'(is/P.sc * *)'   # First arg is subject, second is complement
```

### Flexible Role Matching (Curly Braces)

`{roles}` matches if those roles exist in any position:

```python
'(is/P.{sc} * *)'

# Matches both:
# (is/P.sc sky/C blue/C)    - subject first
# (is/P.cs blue/C sky/C)    - complement first
```

### Excluding Roles

`-` after roles excludes them:

```python
'(says/P.{sr}-x * *)'

# Matches: (says/P.sr mary/C hello/C)
# Rejects: (says/P.srx mary/C hello/C (loudly/M))
```

## Functional Patterns

Advanced patterns using function-like syntax.

### atoms

Match edges containing specific atoms at any depth:

```python
'(atoms says/P mary/C)'

# Matches:
# (says/P mary/C hello/C)
# (believes/P john/C (says/P mary/C hello/C))
# ((loudly/M says/P) mary/C hello/C)
```

### lemma

Match by lemma (requires hypergraph context):

```python
'(lemma be/P)'

# Matches: is/P, was/P, been/P, are/P, etc.

# Usage:
match_pattern(edge, pattern, hg=hg)  # Must pass hg
```

### var

Capture complex patterns as variables:

```python
'(var (atoms not/M */P) NEGATED_PRED)'

# Captures any predicate containing "not" modifier
```

## Search Modes

### Strict Search (Default)

Fast, uses database indexes. Requires exact structural match:

```python
list(hg.search('(says/Pd.sr * *)'))  # Must match Pd.sr exactly
```

### Non-Strict Search

Slower (scans all edges), but more flexible matching:

```python
list(hg.search('(says/P * *)', strict=False))

# Matches says/Pd.sr, says/Pd.so, says/P, etc.
# Type annotations in pattern are "minimum requirements"
```

**Use non-strict when**:
- Using `{role}` syntax
- Using `-role` exclusions
- Matching by base type without subtype
- Using functional patterns

## Match vs Search

| Method | Returns | Use Case |
|--------|---------|----------|
| `hg.search(pattern)` | Matching edges | Find edges |
| `hg.match(pattern)` | (edge, bindings) tuples | Extract values |
| `match_pattern(edge, pattern)` | Bindings list | Check single edge |

```python
# Search: just find edges
for edge in hg.search('(says/P * *)'):
    print(edge)

# Match: extract structured data
for edge, bindings in hg.match('(says/P.{sr} SPEAKER/C MESSAGE/*)', strict=False):
    print(f"{bindings['SPEAKER']} said: {bindings['MESSAGE']}")

# Direct match: check if one edge matches
bindings = match_pattern(my_edge, pattern)
if bindings:
    print("Matched!", bindings[0])
```

## Common Patterns

### Find all claims by someone

```python
'(*/Pd.{sr} SPEAKER/Cp CLAIM/*)'
# Captures: who said what
```

### Find concept modifications

```python
'(*/Ma CONCEPT/C)'
# Captures: adjective + noun pairs
```

### Find compound concepts

```python
'(of/B.{ma} MAIN/C AUX/C)'
# Captures: "X of Y" constructs
```

### Find temporal conditions

```python
'(when/Tt CONDITION/R)'
# Captures: "when X happens" clauses
```

### Find nested claims (beliefs about claims)

```python
'(*/Pd.{sr} BELIEVER/C (*/Pd.{sr} SPEAKER/C CLAIM/*))'
# Captures: X believes that Y said Z
```

## Pattern Composition

Build complex patterns incrementally:

```python
# Base pattern: any predicate with subject/object
base = '(*/Pd.{so} SUBJECT/C OBJECT/C)'

# Add temporal specification
with_time = '(*/Pd.{sox} SUBJECT/C OBJECT/C (*/Tt TIME/*))'

# Nested in a claim
claimed = f'(*/Pd.{{sr}} SOURCE/C {with_time})'
```

## Tips

1. **Start broad, narrow down**: Begin with `*`, add constraints as needed
2. **Use non-strict for semantic queries**: When you care about meaning, not exact form
3. **Variables are typed**: `SPEAKER/Cp` only matches proper nouns
4. **Check match return**: `match_pattern` returns `[]` on no match, not `None`
5. **Multiple bindings possible**: One pattern can match same edge multiple ways
6. **Patterns are edges**: They're valid hyperedges, can be stored in the graph
