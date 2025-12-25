import itertools
import logging
from typing import TYPE_CHECKING, List, Tuple, Optional, Set

from graphbrain.hyperedge import Hyperedge, hedge
from graphbrain.patterns.atoms import _matches_atomic_pattern
from graphbrain.patterns.utils import _defun_pattern_argroles

if TYPE_CHECKING:
    from graphbrain.patterns.matcher import Matcher


logger = logging.getLogger(__name__)


# Threshold for using optimized matching (factorial grows very fast)
# 5! = 120, 6! = 720, 7! = 5040
PERMUTATION_THRESHOLD = 5


def _compute_compatibility_matrix(
        matcher: 'Matcher',
        eitems: List[Hyperedge],
        pitems: List[Hyperedge],
        curvars: dict
) -> List[List[bool]]:
    """Pre-compute which edge items can potentially match which pattern items.

    This creates a compatibility matrix where matrix[i][j] is True if
    eitems[i] could potentially match pitems[j].

    Args:
        matcher: The matcher instance.
        eitems: List of edge items to match.
        pitems: List of pattern items to match against.
        curvars: Current variable bindings.

    Returns:
        A 2D list where [i][j] indicates if eitems[i] can match pitems[j].
    """
    n_edge = len(eitems)
    n_pattern = len(pitems)
    matrix = [[False] * n_pattern for _ in range(n_edge)]

    for i, eitem in enumerate(eitems):
        for j, pitem in enumerate(pitems):
            # Quick structural check for atoms
            if pitem.atom and eitem.atom:
                # Use atomic pattern matching for quick compatibility check
                if _matches_atomic_pattern(eitem, pitem):
                    matrix[i][j] = True
            elif pitem.atom and eitem.not_atom:
                # Atom pattern can't match non-atom edge
                matrix[i][j] = False
            else:
                # For complex patterns, assume compatible (full match will verify)
                matrix[i][j] = True

    return matrix


def _find_valid_assignments(
        compat_matrix: List[List[bool]],
        n: int,
        assignment: Tuple[int, ...] = (),
        used_patterns: Set[int] = None
) -> List[Tuple[int, ...]]:
    """Find all valid assignments from compatibility matrix using backtracking.

    Uses early pruning to avoid exploring impossible branches.

    Args:
        compat_matrix: Compatibility matrix from _compute_compatibility_matrix.
        n: Number of items to assign.
        assignment: Current partial assignment (edge indices in pattern order).
        used_patterns: Set of already used pattern indices.

    Returns:
        List of valid assignments (tuples of edge indices for each pattern position).
    """
    if used_patterns is None:
        used_patterns = set()

    if len(assignment) == n:
        return [assignment]

    pattern_idx = len(assignment)
    n_edges = len(compat_matrix)
    results = []

    for edge_idx in range(n_edges):
        if edge_idx in [assignment[i] for i in range(len(assignment))]:
            continue  # Edge already used

        if compat_matrix[edge_idx][pattern_idx]:
            new_assignment = assignment + (edge_idx,)
            results.extend(_find_valid_assignments(
                compat_matrix, n, new_assignment, used_patterns
            ))

    return results


def _match_by_argroles_optimized(
        matcher: 'Matcher',
        edge: Hyperedge,
        pattern: Hyperedge,
        role_counts,
        min_vars,
        matched=(),
        curvars=None,
        tok_pos=None
):
    """Optimized argrole matching with early pruning.

    Uses compatibility matrix to prune impossible assignments before
    attempting expensive full matching.
    """
    if curvars is None:
        curvars = {}

    if len(role_counts) == 0:
        return [curvars]

    argrole, n = role_counts[0]

    # match connector
    if argrole == 'X':
        eitems = [edge[0]]
        pitems = [pattern[0]]
    # match any argrole
    elif argrole == '*':
        eitems = [e for e in edge if e not in matched]
        pitems = pattern[-n:]
    # match specific argrole
    else:
        eitems = edge.edges_with_argrole(argrole)
        pitems = _defun_pattern_argroles(pattern).edges_with_argrole(argrole)

    if len(eitems) < n:
        if len(curvars) >= min_vars:
            return [curvars]
        else:
            return []

    result = []

    # For small sets, use direct permutation (faster than computing compatibility)
    if len(eitems) <= PERMUTATION_THRESHOLD:
        return _match_by_argroles_direct(
            matcher, edge, pattern, eitems, pitems,
            role_counts, min_vars, matched, curvars, tok_pos, n
        )

    # For larger sets, use compatibility matrix for early pruning
    logger.debug(f"Using optimized matching for {len(eitems)} items")
    compat_matrix = _compute_compatibility_matrix(matcher, eitems, pitems, curvars)

    # Find valid assignments using backtracking with pruning
    valid_assignments = _find_valid_assignments(compat_matrix, n)

    if not valid_assignments:
        if len(curvars) >= min_vars:
            return [curvars]
        return []

    logger.debug(f"Pruned from {len(list(itertools.permutations(range(len(eitems)), n)))} "
                 f"to {len(valid_assignments)} assignments")

    if tok_pos:
        tok_pos_items = [tok_pos[i] for i, subedge in enumerate(edge) if subedge in eitems]

    for assignment in valid_assignments:
        perm = tuple(eitems[i] for i in assignment)

        if tok_pos:
            tok_pos_perm = tuple(tok_pos_items[i] for i in assignment)

        perm_result = [{}]
        for i, eitem in enumerate(perm):
            pitem = pitems[i]
            tok_pos_item = tok_pos_perm[i] if tok_pos else None
            item_result = []
            for variables in perm_result:
                item_result += matcher.match(
                    eitem,
                    pitem,
                    {**curvars, **variables},
                    tok_pos=tok_pos_item
                )
            perm_result = item_result
            if len(item_result) == 0:
                break

        for variables in perm_result:
            result += _match_by_argroles_optimized(
                matcher,
                edge,
                pattern,
                role_counts[1:],
                min_vars,
                matched + perm,
                {**curvars, **variables},
                tok_pos=tok_pos
            )

    return result


def _match_by_argroles_direct(
        matcher: 'Matcher',
        edge: Hyperedge,
        pattern: Hyperedge,
        eitems,
        pitems,
        role_counts,
        min_vars,
        matched,
        curvars,
        tok_pos,
        n
):
    """Direct permutation matching for small item counts."""
    result = []

    if tok_pos:
        tok_pos_items = [tok_pos[i] for i, subedge in enumerate(edge) if subedge in eitems]
        tok_pos_perms = tuple(itertools.permutations(tok_pos_items, r=n))

    for perm_n, perm in enumerate(tuple(itertools.permutations(eitems, r=n))):
        if tok_pos:
            tok_pos_perm = tok_pos_perms[perm_n]
        perm_result = [{}]
        for i, eitem in enumerate(perm):
            pitem = pitems[i]
            tok_pos_item = tok_pos_perm[i] if tok_pos else None
            item_result = []
            for variables in perm_result:
                item_result += matcher.match(
                    eitem,
                    pitem,
                    {**curvars, **variables},
                    tok_pos=tok_pos_item
                )
            perm_result = item_result
            if len(item_result) == 0:
                break

        for variables in perm_result:
            result += _match_by_argroles_optimized(
                matcher,
                edge,
                pattern,
                role_counts[1:],
                min_vars,
                matched + perm,
                {**curvars, **variables},
                tok_pos=tok_pos
            )

    return result


def _match_by_argroles(
        matcher: 'Matcher',
        edge: Hyperedge,
        pattern: Hyperedge,
        role_counts,
        min_vars,
        matched=(),
        curvars=None,
        tok_pos=None
):
    """Match edge arguments by their roles using optimized algorithm.

    This is the main entry point for argrole matching. It delegates to
    _match_by_argroles_optimized which uses early pruning for large sets.

    Args:
        matcher: The matcher instance.
        edge: Edge to match.
        pattern: Pattern to match against.
        role_counts: List of (role, count) tuples.
        min_vars: Minimum number of variables required for partial match.
        matched: Already matched edge items.
        curvars: Current variable bindings.
        tok_pos: Token positions for semantic similarity.

    Returns:
        List of variable binding dictionaries for successful matches.
    """
    return _match_by_argroles_optimized(
        matcher, edge, pattern, role_counts, min_vars,
        matched, curvars, tok_pos
    )


def edge2rolemap(edge):
    argroles = edge[0].argroles()
    if argroles[0] == '{':
        argroles = argroles[1:-1]
    args = list(zip(argroles, edge[1:]))
    rolemap = {}
    for role, subedge in args:
        if role not in rolemap:
            rolemap[role] = []
        rolemap[role].append(subedge)
    return rolemap


def rolemap2edge(pred, rm):
    roles = list(rm.keys())
    argroles = ''
    subedges = [pred]
    for role in roles:
        for arg in rm[role]:
            argroles += role
            subedges.append(arg)
    edge = hedge(subedges)
    return edge.replace_argroles(argroles)


def rolemap_pairings(rm1, rm2):
    roles = list(set(rm1.keys()) & set(rm2.keys()))
    role_counts = {}
    for role in roles:
        role_counts[role] = min(len(rm1[role]), len(rm2[role]))

    pairings = []
    for role in roles:
        role_pairings = []
        n = role_counts[role]
        for args1_combs in itertools.combinations(rm1[role], n):
            for args1 in itertools.permutations(args1_combs):
                for args2 in itertools.combinations(rm2[role], n):
                    role_pairings.append((args1, args2))
        pairings.append(role_pairings)

    for pairing in itertools.product(*pairings):
        rm1_ = {}
        rm2_ = {}
        for role, role_pairing in zip(roles, pairing):
            rm1_[role] = role_pairing[0]
            rm2_[role] = role_pairing[1]
        yield rm1_, rm2_
