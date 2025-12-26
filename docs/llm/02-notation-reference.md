# SH Notation Quick Reference

> Compact reference for reading and writing Semantic Hypergraph notation

## Atom Structure

```
root/Type.roles.features/namespace.disambiguator
```

**Minimal**: `word/C`
**Full**: `published/Pd.sox.<f--3s-/en.1`

## Types (8 total)

| Code | Name | Use |
|------|------|-----|
| C | Concept | Nouns, entities, things |
| P | Predicate | Verbs, relations, actions |
| M | Modifier | Adjectives, adverbs, determiners |
| B | Builder | Compound nouns, possessives |
| T | Trigger | Conditions, time, place markers |
| J | Conjunction | And, or, lists |
| R | Relation | (inferred) Complete statement |
| S | Specifier | (inferred) Condition clause |

## Concept Subtypes (Cx)

| Code | Meaning | Example |
|------|---------|---------|
| Cc | common noun | `apple/Cc` |
| Cp | proper noun | `mary/Cp` |
| Cn | number | `42/Cn` |
| Ci | pronoun | `she/Ci` |
| Cw | interrogative | `who/Cw` |

## Predicate Subtypes (Px)

| Code | Meaning | Example |
|------|---------|---------|
| Pd | declarative | `is/Pd` (states fact) |
| P? | interrogative | `is/P?` (asks question) |
| P! | imperative | `go/P!` (gives command) |

## Modifier Subtypes (Mx)

| Code | Meaning | Example |
|------|---------|---------|
| Ma | adjective | `red/Ma` |
| Md | determiner | `the/Md`, `a/Md` |
| Mp | possessive | `my/Mp`, `her/Mp` |
| Mn | negation | `not/Mn` |
| Mv | verbal | `will/Mv`, `have/Mv` |
| M# | number (as modifier) | `three/M#` |

## Builder Subtypes (Bx)

| Code | Meaning | Example |
|------|---------|---------|
| Bp | possessive | `'s/Bp` |
| Br | relational | `of/Br`, `in/Br` |

## Trigger Subtypes (Tx)

| Code | Meaning | Example |
|------|---------|---------|
| Tt | temporal | `when/Tt` |
| Tl | local (place) | `where/Tl` |
| Tc | conditional | `if/Tc` |
| T> | causal | `because/T>` |
| T= | comparative | `like/T=` |

## Argument Roles (after type, e.g., Pd.sox)

| Code | Role | Position meaning |
|------|------|------------------|
| s | subject | Who/what does action |
| o | object | What action is done to |
| p | passive subject | Subject of passive verb |
| c | complement | Subject complement (is X) |
| x | specification | Extra info (time, place) |
| i | indirect object | Recipient |
| r | relative | Relative clause connection |
| t | parataxis | Parenthetical |
| j | interjection | Interjection |

**Reading example**: `says/Pd.sr`
- Position 1 = subject (s) = speaker
- Position 2 = relative/recipient (r) = what is said

## Builder Roles (after type, e.g., Br.ma)

| Code | Role |
|------|------|
| m | main concept |
| a | auxiliary concept |

**Example**: `(of/Br.ma capital/C germany/C)` = "capital of Germany"
- "capital" is main (m) - this IS a capital
- "germany" is auxiliary (a) - specifies which capital

## Verb Features (second subpart for predicates)

Position: `Pd.sr.|f--3s-` (after the dot following roles)

| Pos | Feature | Codes |
|-----|---------|-------|
| 1 | tense | `<` past, `\|` present, `>` future |
| 2 | form | `f` finite, `i` infinitive |
| 3 | aspect | `f` perfect, `g` progressive |
| 4 | mood | various |
| 5 | person | `1`, `2`, `3` |
| 6 | number | `s` singular, `p` plural |
| 7 | verb type | various |

**Example**: `|f--3s-` = present tense, finite, 3rd person singular

## Number (concepts)

After type: `apple/Cc.s`, `apples/Cc.p`
- `s` = singular
- `p` = plural

## Namespaces

After features: `sky/Cc.s/en`
- `/en` = English
- `/de` = German
- `/.` = Graphbrain internal

## Special Atoms

| Atom | Purpose |
|------|---------|
| `+/B/.` | Compound name builder: `(+/B/. alan/Cp turing/Cp)` |
| `:/J/.` | Generic conjunction |
| `coref/P/.` | Coreference marker |
| `lemma/P/.` | Lemma relation |
| `type_of/P/.` | Taxonomy relation |

## Reading Complex Edges

```
((first/M published/Pd.sox) einstein/Cp (of/Br.ma (the/Md theory/Cc) relativity/Cc) (in/Tt 1905/Cn))
```

**Parse**:
1. Outer connector: `(first/M published/Pd.sox)` - modified predicate "first published"
2. Role s (pos 1): `einstein/Cp` - subject (Einstein)
3. Role o (pos 2): `(of/Br.ma (the/Md theory/Cc) relativity/Cc)` - object (the theory of relativity)
4. Role x (pos 3): `(in/Tt 1905/Cn)` - specification (in 1905)

**English**: "Einstein first published the theory of relativity in 1905"

## Pattern Syntax

| Pattern | Matches |
|---------|---------|
| `*` | Any edge (atomic or not) |
| `.` | Any atom only |
| `(*)` | Any non-atomic edge only |
| `*/C` | Any concept |
| `*/P` | Any predicate |
| `...` | Zero or more additional arguments |
| `VAR/C` | Captures to variable VAR |
| `{so}` | Roles s and o in any order |
| `{so}-x` | Roles s,o required, x forbidden |

## Quick Examples

| Natural Language | SH Representation |
|-----------------|-------------------|
| The sky is blue | `(is/Pd.sc (the/Md sky/Cc) blue/Ca)` |
| Mary's book | `(poss/Bp mary/Cp book/Cc)` |
| Red apple | `(red/Ma apple/Cc)` |
| City of Berlin | `(of/Br.ma city/Cc berlin/Cp)` |
| Mary said hello | `(said/Pd.sr mary/Cp hello/C)` |
| John believes Mary is right | `(believes/Pd.sr john/Cp (is/Pd.sc mary/Cp right/Ca))` |
