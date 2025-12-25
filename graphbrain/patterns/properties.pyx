from typing import Set

# Base pattern functions (always available)
_BASE_FUNS: Set[str] = {
    'var',      # Variable binding
    'atoms',    # Match atoms in any order
    'lemma',    # Match by lemma
    'any',      # Match any of the sub-patterns (OR)
    'not',      # Negation - match if sub-pattern doesn't match
    'all',      # Universal - all sub-patterns must match
    'exists',   # Existential - at least one sub-pattern matches
    'depth',    # Depth constraint - match at specific nesting depth
}

# Try to import semsim functions (requires gensim)
try:
    from graphbrain.patterns.semsim.types import SEMSIM_FUNS
    SEMSIM_AVAILABLE = True
    FUNS: Set[str] = _BASE_FUNS | SEMSIM_FUNS.keys()
except ImportError:
    SEMSIM_AVAILABLE = False
    SEMSIM_FUNS = {}
    FUNS: Set[str] = _BASE_FUNS

def is_wildcard(atom):
    """Check if this atom defines a wildcard, i.e. if its root is a pattern matcher.
    (\*, ., ..., if it is surrounded by parenthesis or variable label starting with an uppercase letter)
    """
    if atom.atom:
        return atom.parens or atom[0][0] in {'*', '.'} or atom[0][0].isupper()
    else:
        return False


def is_fun_pattern(edge):
    if edge.atom:
        return False
    return str(edge[0]) in FUNS


def is_pattern(edge):
    """Check if this edge defines a pattern, i.e. if it includes at least
    one pattern matcher.

    Pattern matcher are:
    - '\*', '.', '(\*)', '...'
    - variables (atom label starting with an uppercase letter)
    - argument role matcher (unordered argument roles surrounded by curly brackets)
    - functional patterns (var, atoms, lemma, ...)
    """
    if edge.atom:
        return is_wildcard(edge) or '{' in edge.argroles()
    elif is_fun_pattern(edge):
        return True
    else:
        return any(is_pattern(item) for item in edge)


def is_unordered_pattern(edge):
    """Check if this edge defines an unordered pattern, i.e. if it includes at least
    one instance of unordered argument roles surrounded by curly brackets.
    """
    if edge.atom:
        return '{' in edge.argroles()
    else:
        return any(is_unordered_pattern(item) for item in edge)

def is_full_pattern(edge):
    """Check if every atom is a pattern matcher.

    Pattern matcher are:
    '\*', '.', '(\*)', '...', variables (atom label starting with an
    uppercase letter) and functional patterns.
    """
    if edge.atom:
        return is_pattern(edge)
    else:
        return all(is_pattern(item) for item in edge)