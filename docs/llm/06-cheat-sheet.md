# Graphbrain Cheat Sheet

## Edge Structure
```
(connector arg1 arg2 ...)
```

## Atom Structure
```
root/Type.roles.features/namespace
```

## Types
- **C** = Concept (Cp=proper, Cc=common, Cn=number)
- **P** = Predicate (Pd=declarative, P?=question)
- **M** = Modifier (Ma=adj, Md=det, Mv=verbal)
- **B** = Builder (Br=relational, Bp=possessive)
- **T** = Trigger (Tt=temporal, Tc=conditional)
- **J** = Conjunction

## Argument Roles (Pd.sox)
- s=subject, o=object, c=complement
- x=specification, p=passive, r=relative

## Pattern Wildcards
- `*` = any edge
- `.` = atoms only
- `*/C` = any concept
- `...` = zero+ more args
- `VAR/C` = capture as variable
- `{so}` = roles in any order

## Quick API
```python
from graphbrain import hgraph, hedge
hg = hgraph('db.db')
edge = hedge('(is/P sky/C blue/C)')
hg.add(edge)
list(hg.search('(is/P * *)'))
```

## Examples
| English | SH |
|---------|-----|
| Sky is blue | `(is/Pd.sc sky/Cc blue/Ca)` |
| Mary said hi | `(said/Pd.sr mary/Cp hi/C)` |
| Capital of France | `(of/Br.ma capital/Cc france/Cp)` |
