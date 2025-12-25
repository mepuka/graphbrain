# Auxiliary functions for graphbrain databases based on permutations
# of string representations of edges.
#
# For example, the edge:
# (is/pd.sc (my/m name/cn.s) mary/cp.s)
#
# can be represented by permutation 0:
# is/pd.sc (my/m name/cn.s) mary/cp.s
#
# permutation 1:
# is/pd.sc mary/cp.s (my/m name/cn.s)
#
# permutation 2:
# (my/m name/cn.s) is/pd.sc mary/cp.s
#
# and so on...
import itertools
import math
from functools import lru_cache

from graphbrain.hyperedge import hedge, split_edge_str


# maximum permutations of an edge that are written to the database
MAX_PERMS = 1000


def fact(n):
    if n < 1:
        return 0
    res = 1
    for i in range(2, n + 1):
        res *= i
    return res


@lru_cache(maxsize=10000)
def nthperm(n, nper):
    """Compute the nper-th permutation of n elements.

    Uses LRU cache to limit memory usage while maintaining fast lookups.
    """
    pos = 0
    pindices = None
    for perm in itertools.permutations(range(n)):
        if pos >= nper:
            pindices = perm
            break
        pos += 1
    return tuple(pindices[i] for i in range(n))


def permutate(tokens, nper):
    """Reorder the tokens vector to perform a permutation,
       specified by nper.
    """
    n = len(tokens)
    indices = nthperm(n, nper)
    return tuple(tokens[i] for i in indices)


def unpermutate(tokens, nper):
    """Reorder the tokens vector to revert a permutation,
       specified by nper.
    """
    n = len(tokens)
    indices = nthperm(n, nper)
    res = [''] * n
    for pos, i in enumerate(indices):
        res[i] = tokens[pos]
    return res


def first_permutation(elements, positions):
    n_elements = elements
    fp = 0
    for i in range(len(positions)):
        position = positions[i]
        fp += fact(n_elements - 1) * (position - i)
        n_elements -= 1
    return fp


def do_with_edge_permutations(edge, f):
    """Applies the function f to all permutations of the given edge."""
    nperms = min(math.factorial(len(edge)), MAX_PERMS)
    for nperm in range(nperms):
        perm_str = ' '.join([e.to_str() for e in permutate(edge, nperm)])
        perm_str = ''.join((perm_str, ' ', str(nperm)))
        f(perm_str)


def perm2edge(perm_str):
    """Transforms a permutation string from a database query
       into an edge.
    """
    from graphbrain.exceptions import EdgeParseError
    try:
        tokens = split_edge_str(perm_str)
        nper = int(tokens[-1])
        tokens = tokens[:-1]
        tokens = unpermutate(tokens, nper)
        edge_str = ' '.join(tokens)
        if len(tokens) > 1:
            edge_str = ''.join(('(', edge_str, ')'))
        return hedge(edge_str)
    except (ValueError, EdgeParseError):
        return None


def str_plus_1(s):
    """Increment a string by one, regaring lexicographical ordering."""
    last_char = s[-1]
    last_char = chr(ord(last_char) + 1)
    return ''.join((s[:-1], last_char))
