from typing import Optional, Tuple

import graphbrain.constants as const
from graphbrain.hyperedge import Hyperedge


def lemma(hg, atom: Hyperedge, same_if_none: bool = False) -> Optional[Hyperedge]:
    """Return the lemma of the given atom if it exists.

    Args:
        hg: Hypergraph containing lemma edges.
        atom: Atom to find lemma for.
        same_if_none: If True, returns atom when lemma not found.
                     If False, returns None when lemma not found.

    Returns:
        Lemma atom, the original atom (if same_if_none), or None.
    """
    if atom.atom:
        satom = atom.simplify()
        for lemma_edge in hg.search((const.lemma_connector, satom, '*'), strict=True):
            return lemma_edge[2]

    if same_if_none:
        return atom

    return None


def deep_lemma(hg, edge: Hyperedge, same_if_none: bool = False) -> Optional[Hyperedge]:
    """Return lemma by recursively descending the edge.

    Finds the lemma of an atomic edge, or the lemma of the first atom
    found by recursively descending, always choosing the subedge
    immediately after the connector.

    Useful for finding the lemma of the central verb in a non-atomic
    predicate edge. For example:

    (not/A (is/A going/P)) -> go/P

    Args:
        hg: Hypergraph containing lemma edges.
        edge: Edge to find lemma for.
        same_if_none: If True, returns atom when lemma not found.

    Returns:
        Lemma atom, the original atom (if same_if_none), or None.
    """
    if edge.atom:
        return lemma(hg, edge, same_if_none)
    else:
        return deep_lemma(hg, edge[1], same_if_none)


def lemma_degrees(hg, edge: Hyperedge) -> Tuple[int, int]:
    """Compute degree sum for all atoms sharing the same lemma.

    Finds all atoms that share the same given lemma and computes
    the sum of both their degrees and deep degrees.

    If the edge is non-atomic, simply returns the degree and
    deep degree of that edge.

    Args:
        hg: Hypergraph to query.
        edge: Edge to compute degrees for.

    Returns:
        Tuple of (degree_sum, deep_degree_sum).
    """
    if edge.atom:
        roots = {edge.root()}

        # find lemma
        satom = edge.simplify()
        for edge in hg.search((const.lemma_connector, satom, '*'), strict=True):
            roots.add(edge[2].root())

        # compute degrees
        d = 0
        dd = 0
        for r in roots:
            atoms = set(hg.atoms_with_root(r))
            d += sum([hg.degree(atom) for atom in atoms])
            dd += sum([hg.deep_degree(atom) for atom in atoms])

        return d, dd
    else:
        return hg.degree(edge), hg.deep_degree(edge)
