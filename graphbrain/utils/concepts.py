from typing import Optional, Set

from graphbrain.hyperedge import Hyperedge


def strip_concept(edge: Hyperedge) -> Optional[Hyperedge]:
    """Strip away nesting edges with connectors such as triggers and
    subpredicates, to expose the outmost and leftmost concept that can be
    found. May be the edge itself.

    For example:

    (against/T (the/M (of/B treaty/C paris/C)))

    becomes

    (the/M (of/B treaty/C paris/C))

    Args:
        edge: Hyperedge to strip.

    Returns:
        The stripped concept edge, or None if no concept found.
    """
    if edge.mtype() == 'C':
        return edge
    elif edge.not_atom:
        return strip_concept(edge[1])
    else:
        return None


def has_proper_concept(edge: Hyperedge) -> bool:
    """Check if the edge is a proper concept or contains one.

    Args:
        edge: Hyperedge to check.

    Returns:
        True if edge contains a proper concept (Cp).
    """
    if edge.atom:
        return edge.type()[:2] == 'Cp'
    else:
        for subedge in edge[1:]:
            if has_proper_concept(subedge):
                return True
        return False


def has_common_or_proper_concept(edge: Hyperedge) -> bool:
    """Check if the edge is a common or proper concept or contains one.

    Args:
        edge: Hyperedge to check.

    Returns:
        True if edge contains a common (Cc) or proper (Cp) concept.
    """
    if edge.atom:
        return edge.type()[:2] == 'Cp' or edge.type()[:2] == 'Cc'
    else:
        for subedge in edge[1:]:
            if has_proper_concept(subedge):
                return True
        return False


def all_concepts(edge: Hyperedge) -> Set[Hyperedge]:
    """Recursively search for all concepts contained in the edge.

    Args:
        edge: Hyperedge to search.

    Returns:
        Set of all concept edges found, which may include the edge itself.
    """
    concepts: Set[Hyperedge] = set()
    if edge.mtype() == 'C':
        concepts.add(edge)
    if edge.not_atom:
        for item in edge:
            concepts |= all_concepts(item)
    return concepts
