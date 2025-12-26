# Graphbrain API Patterns

> Common operations and idiomatic usage patterns

## Setup

### Opening a Hypergraph

```python
from graphbrain import hgraph, hedge

# SQLite (default, recommended for most cases)
hg = hgraph('mydata.db')

# PostgreSQL (for larger datasets, concurrent access)
hg = hgraph('postgresql://user:pass@localhost/mydb')

# In-memory (for testing)
hg = hgraph('memory')
```

### Batch Operations

For adding many edges, use context manager for performance:

```python
from graphbrain import hopen

with hopen('mydata.db') as hg:
    for text in documents:
        edges = parse_document(text)
        for edge in edges:
            hg.add(edge)
```

## Edge Operations

### Creating Edges

```python
from graphbrain import hedge

# From string
edge = hedge('(is/Pd.sc sky/Cc blue/Ca)')

# From nested tuples/lists
edge = hedge(['is/Pd.sc', 'sky/Cc', 'blue/Ca'])

# Combining edges
connector = hedge('says/Pd.sr')
subject = hedge('mary/Cp')
claim = hedge('(is/Pd.sc sky/Cc blue/Ca)')
full_edge = hedge([connector, subject, claim])
```

### Edge Properties

```python
edge = hedge('(says/Pd.sr mary/Cp (is/Pd.sc sky/Cc blue/Ca))')

# Type checking
edge.is_atom()         # False
edge[1].is_atom()      # True (mary/Cp)

# Get type
edge.type()            # 'R' (relation - inferred)
edge[0].type()         # 'P' (predicate)
edge[1].type()         # 'C' (concept)

# Get connector
edge[0]                # says/Pd.sr

# Get arguments
edge[1:]               # (mary/Cp, (is/Pd.sc sky/Cc blue/Ca))

# Atom properties
atom = hedge('published/Pd.sox.<f-----/en')
atom.root()            # 'published'
atom.type()            # 'P'
atom.argroles()        # 'sox'
```

### Adding/Removing Edges

```python
# Add edge
hg.add(edge)                    # As primary
hg.add(edge, primary=False)     # As non-primary

# Check existence
hg.exists(edge)                 # True/False

# Check if primary
hg.is_primary(edge)             # True/False

# Remove edge
hg.remove(edge)

# Add with count (tracks occurrences)
hg.add(edge, count=True)
hg.get_int_attribute(edge, 'count')  # Returns occurrence count
```

## Querying

### Pattern Search

```python
# Find all matching edges
results = list(hg.search('(says/P * *)'))

# Non-strict search (more flexible)
results = list(hg.search('(says/P * *)', strict=False))

# With variable extraction
for edge, bindings_list in hg.match('(says/P.{sr} SPEAKER/C *)', strict=False):
    speaker = bindings_list[0]['SPEAKER']
    print(f"Found speaker: {speaker}")
```

### Neighborhood Queries

```python
# Star: edges containing this edge
for edge in hg.star(hedge('mary/Cp')):
    print(edge)

# Edges containing multiple edges
results = list(hg.edges_with_edges([
    hedge('says/P'),
    hedge('mary/Cp')
]))

# With root filter
results = list(hg.edges_with_edges(
    [hedge('says/P')],
    root='mary'
))
```

### Degrees

```python
# Direct connections
hg.degree('mary/Cp')        # Number of edges containing mary

# Including nested
hg.deep_degree('mary/Cp')   # Includes edges where mary appears at any depth
```

## Attributes

```python
# Set attributes
hg.set_attribute(edge, 'source', 'wikipedia')
hg.set_attribute(edge, 'confidence', 0.95)
hg.set_attribute(edge, 'count', 5)

# Get attributes
hg.get_str_attribute(edge, 'source')      # 'wikipedia'
hg.get_float_attribute(edge, 'confidence') # 0.95
hg.get_int_attribute(edge, 'count')        # 5

# Increment/decrement integers
hg.inc_attribute(edge, 'count')
hg.dec_attribute(edge, 'count')
```

## Parsing

### Basic Parsing

```python
from graphbrain.parsers import create_parser

# Create parser (takes 10-20s to load models)
parser = create_parser(lang='en')

# Parse text
result = parser.parse('Mary believes the sky is blue.')

# Get main edge
main_edge = result['parses'][0]['main_edge']
```

### Parser Options

```python
# With lemmas
parser = create_parser(lang='en', lemmas=True)
result = parser.parse('Mary published a paper.')
extra_edges = result['parses'][0]['extra_edges']
# Contains: (lemma/J/. published/P/en publish/P/en)

# With coreference resolution
parser = create_parser(lang='en', corefs=True)
result = parser.parse('Mary said she likes her dog.')
resolved = result['parses'][0]['resolved_corefs']
# Pronouns replaced with referents
```

### Parse Result Structure

```python
result = parser.parse('Einstein published relativity in 1905.')

# Access components
for parse in result['parses']:
    main_edge = parse['main_edge']      # Primary hyperedge
    extra = parse['extra_edges']         # Lemmas, inferences
    text = parse['text']                 # Original sentence
    failed = parse['failed']             # Parse success

    # Word-to-atom mapping
    for atom, (word, position) in parse['atom2word'].items():
        print(f"{atom} <- '{word}' (pos {position})")

# Cross-sentence inferences
for edge in result['inferred_edges']:
    print(f"Inferred: {edge}")
```

## Iteration

### All Edges

```python
# Iterate all edges
for edge in hg.all():
    if hg.is_primary(edge):
        process(edge)

# Count statistics
atoms = sum(1 for e in hg.all() if e.is_atom())
edges = sum(1 for e in hg.all() if not e.is_atom())
```

### Sequences

```python
# Add to named sequence
hg.add_to_sequence('chapter1', edge1)
hg.add_to_sequence('chapter1', edge2)

# Iterate sequence in order
for edge in hg.sequence('chapter1'):
    print(edge)

# List all sequences
for seq_name in hg.sequences():
    print(f"Sequence: {seq_name}")
```

## Pattern Matching (Direct)

```python
from graphbrain.patterns import match_pattern

edge = hedge('(says/Pd.sr mary/Cp hello/C)')
pattern = hedge('(says/P.{sr} SPEAKER/C MESSAGE/*)')

# Returns list of binding dicts (can match multiple ways)
bindings = match_pattern(edge, pattern)

if bindings:
    for binding in bindings:
        print(f"Speaker: {binding['SPEAKER']}")
        print(f"Message: {binding['MESSAGE']}")
else:
    print("No match")
```

## Common Workflows

### Extract All Claims

```python
def extract_claims(hg):
    """Extract who-said-what from hypergraph."""
    pattern = '(*/Pd.{sr} SPEAKER/Cp CLAIM/*)'

    for edge, bindings_list in hg.match(pattern, strict=False):
        for bindings in bindings_list:
            yield {
                'speaker': bindings['SPEAKER'].to_str(),
                'claim': bindings['CLAIM'].to_str(),
                'full_edge': edge.to_str()
            }
```

### Build Concept Network

```python
def get_related_concepts(hg, concept):
    """Find concepts directly related to given concept."""
    related = set()

    for edge in hg.star(hedge(concept)):
        for sub in edge:
            if sub.type() == 'C' and sub != hedge(concept):
                related.add(sub)

    return related
```

### Parse and Store Document

```python
def process_document(hg, parser, text, source_id):
    """Parse text and store with source attribution."""
    result = parser.parse(text)

    with hopen(hg.locator_string) as hg:
        for parse in result['parses']:
            if not parse['failed']:
                edge = parse['main_edge']
                hg.add(edge, primary=True)
                hg.set_attribute(edge, 'source', source_id)
                hg.set_attribute(edge, 'text', parse['text'])

                for extra in parse['extra_edges']:
                    hg.add(extra, primary=False)
```

## Error Handling

```python
from graphbrain import hedge

# Invalid edge syntax
try:
    edge = hedge('(broken syntax')
except Exception as e:
    print(f"Parse error: {e}")

# Check before operations
edge = hedge('(is/P sky/C blue/C)')
if hg.exists(edge):
    attrs = hg.get_attributes(edge)
else:
    print("Edge not in hypergraph")
```
