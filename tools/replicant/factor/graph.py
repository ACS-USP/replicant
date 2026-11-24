"""Factor graphs with q-state variables, and samplers of the graphs the notes use.

Convention (KZ Sec. 5.2.2): P(s) = (1/Z) prod_i g_i(s_i) prod_a f_a(s_{da}), s_i in {0, ..., q - 1}.  One alphabet
size q for every variable; the factors have any arity k >= 1 and nonnegative tables of shape (q,) * k.  The
pairwise Potts / coloring weight is f(s, s') = exp(-beta delta_{s s'}) (KZ Ch. 6: beta > 0 antiferromagnetic,
beta < 0 ferromagnetic), the Ising weight exp(beta_I J s s') with s = 1 - 2 x for the state x in {0, 1}.

  FactorGraph(variables, factors)   variables: an (N, q) array of unary weights g_i (or an int N with q given),
                                    factors: (scope, table) pairs or FactorGroup blocks (a block shares one table
                                    or stacks one per factor, for vectorised BP at N ~ 1e4)
  pairwise(n, q, edges, table, unary=None)             a graph of 2-body factors
  random_tree_edges(n, rng)          a uniform labelled tree (Pruefer sequence)
  random_factor_tree(n_factors, arity, rng)            scopes of a tree factor graph with k-body factors
  random_regular_edges(n, d, rng)    configuration model, self-loops and repeated edges removed by switches
  erdos_renyi_edges(n, c, rng)       G(n, p = c/(n - 1)): M ~ Binomial, then M distinct uniform pairs
All samplers are written here (no code shared with cavity/graphs.py: the cross-checks against cavity/ and
lattice/ stay independent; tests/test_factor_bp.py audits the imports).
"""
import heapq
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from ..core import Refusal


@dataclass(frozen=True, eq=False)
class FactorGroup:
    """M factors of one arity k: scopes (M, k) variable indices; tables (M,) + (q,) * k, or (q,) * k shared."""
    scopes: np.ndarray
    tables: np.ndarray

    @property
    def arity(self):
        return self.scopes.shape[1]

    @property
    def shared(self):
        return self.tables.ndim == self.arity


def _refuse(detail):
    raise Refusal("precondition:factor-graph", detail)


class FactorGraph:
    """A factor graph; sockets (variable i, factor a) are numbered group by group, factor by factor, position
    by position: socket_var[e] is the variable of socket e, A the (N, E) incidence matrix."""

    def __init__(self, variables, factors, q=None):
        if isinstance(variables, (int, np.integer)) and not isinstance(variables, bool):
            if q is None:
                _refuse("an int number of variables needs q")
            variables = np.ones((int(variables), int(q)))
        g = np.asarray(variables, dtype=float)
        if g.ndim != 2 or g.shape[1] < 2 or g.shape[0] < 1:
            _refuse(f"variables must be an (N, q) array of unary weights with q >= 2, got shape {g.shape}")
        if not (np.all(np.isfinite(g)) and np.all(g >= 0) and np.all(g.sum(1) > 0)):
            _refuse("unary weights must be finite, nonnegative, with a positive sum per variable")
        self.unary, (self.n, self.q) = g, g.shape
        self.groups = tuple(self._groups(factors))
        self.m = sum(len(gr.scopes) for gr in self.groups)
        sv = [gr.scopes.ravel() for gr in self.groups]
        self.socket_var = np.concatenate(sv) if sv else np.zeros(0, dtype=int)
        self.n_sockets = len(self.socket_var)
        self.offsets = np.cumsum([0] + [gr.scopes.size for gr in self.groups])
        self.A = csr_matrix((np.ones(self.n_sockets), (self.socket_var, np.arange(self.n_sockets))),
                            shape=(self.n, self.n_sockets))

    def _groups(self, factors):
        loose = {}
        for f in factors:
            if isinstance(f, FactorGroup):
                yield self._check(np.asarray(f.scopes, dtype=int), np.asarray(f.tables, dtype=float))
                continue
            scope, table = f
            scope, table = np.atleast_1d(np.asarray(scope, dtype=int)), np.asarray(table, dtype=float)
            loose.setdefault(len(scope), ([], []))
            loose[len(scope)][0].append(scope), loose[len(scope)][1].append(table)
        for k, (sc, tb) in sorted(loose.items()):
            tb = [np.asarray(t) for t in tb]
            if any(t.shape != (self.q,) * k for t in tb):
                _refuse(f"a factor of arity {k} needs a table of shape {(self.q,) * k}")
            yield self._check(np.array(sc, dtype=int), np.stack(tb))

    def _check(self, scopes, tables):
        if scopes.ndim != 2 or scopes.shape[1] < 1:
            _refuse(f"scopes must be (M, k), got shape {scopes.shape}")
        k = scopes.shape[1]
        if scopes.size and (scopes.min() < 0 or scopes.max() >= self.n):
            _refuse("a factor scope names a variable out of range")
        if k > 1 and np.any(np.sort(scopes, 1)[:, 1:] == np.sort(scopes, 1)[:, :-1]):
            _refuse("a factor scope repeats a variable")
        ok = tables.shape in ((self.q,) * k, (len(scopes),) + (self.q,) * k)
        if not ok:
            _refuse(f"tables of arity {k}: shape (q,)*k shared or (M,)+(q,)*k, got {tables.shape}")
        if not (np.all(np.isfinite(tables)) and np.all(tables >= 0)):
            _refuse("factor tables must be finite and nonnegative")
        return FactorGroup(scopes, tables)

    def degrees(self):
        return np.bincount(self.socket_var, minlength=self.n)

    def is_forest(self):
        """True when the bipartite variable-factor graph has no cycle (then BP and Bethe are exact)."""
        fac = np.concatenate([np.repeat(np.arange(len(gr.scopes)), gr.arity) + o
                              for gr, o in zip(self.groups, np.cumsum([0] + [len(g.scopes) for g in self.groups]))]
                             ) if self.groups else np.zeros(0, dtype=int)
        nodes = self.n + self.m
        adj = csr_matrix((np.ones(self.n_sockets), (self.socket_var, self.n + fac)), shape=(nodes, nodes))
        ncomp = connected_components(adj, directed=False)[0]
        return self.n_sockets == nodes - ncomp


def pairwise(n, q, edges, table, unary=None):
    """FactorGraph of 2-body factors on `edges` (M, 2); `table` (q, q) shared or (M, q, q) per edge."""
    e = np.asarray(edges, dtype=int).reshape(-1, 2)
    g = np.ones((n, q)) if unary is None else unary
    return FactorGraph(g, [FactorGroup(e, np.asarray(table, dtype=float))])


def random_tree_edges(n, rng):
    """A uniform random labelled tree on n vertices, decoded from a uniform Pruefer sequence."""
    if not (isinstance(n, int) and n >= 1):
        _refuse(f"a tree needs an int n >= 1, got {n!r}")
    if n <= 2:
        return np.array([[0, 1]] if n == 2 else [], dtype=int).reshape(-1, 2)
    seq = rng.integers(n, size=n - 2)
    deg = np.ones(n, dtype=int)
    np.add.at(deg, seq, 1)
    leaves = [i for i in range(n) if deg[i] == 1]
    heapq.heapify(leaves)
    out = []
    for v in seq:
        leaf = heapq.heappop(leaves)
        out.append((leaf, int(v)))
        deg[v] -= 1
        if deg[v] == 1:
            heapq.heappush(leaves, int(v))
    out.append((heapq.heappop(leaves), heapq.heappop(leaves)))
    return np.array(out, dtype=int)


def random_factor_tree(n_factors, arity, rng):
    """(n_variables, scopes): each new factor joins one existing variable to arity - 1 new ones."""
    scopes, n = [], 1
    for _ in range(n_factors):
        old = int(rng.integers(n))
        new = list(range(n, n + arity - 1))
        n += arity - 1
        sc = [old] + new
        scopes.append([sc[i] for i in rng.permutation(arity)])
    return n, np.array(scopes, dtype=int).reshape(-1, arity)


def _bad_edges(e, n):
    key = np.minimum(e[:, 0], e[:, 1]) * n + np.maximum(e[:, 0], e[:, 1])
    _, first = np.unique(key, return_index=True)
    dup = np.ones(len(e), dtype=bool)
    dup[first] = False
    return np.flatnonzero(dup | (e[:, 0] == e[:, 1]))


def random_regular_edges(n, d, rng, rounds=10000):
    """A simple d-regular graph: the configuration model, then each self-loop or repeated edge switches an
    endpoint with a uniformly chosen edge until none is left (degrees are preserved by every switch)."""
    if not (isinstance(d, int) and isinstance(n, int) and 1 <= d < n and (n * d) % 2 == 0):
        _refuse(f"no simple {d}-regular graph on n = {n}")
    e = rng.permutation(np.repeat(np.arange(n), d)).reshape(-1, 2)
    for _ in range(rounds):
        bad = _bad_edges(e, n)
        if not len(bad):
            return np.sort(e, 1)
        for b, o in zip(bad, rng.integers(len(e), size=len(bad))):     # one switch at a time: degrees kept
            e[b, 1], e[o, 1] = e[o, 1], e[b, 1]
    raise Refusal("no-solution:graph", f"{d}-regular switches left a non-simple graph after {rounds} rounds")


def erdos_renyi_edges(n, c, rng):
    """G(n, p) with p = c/(n - 1) (mean degree c): M ~ Binomial(n(n-1)/2, p), then M distinct uniform pairs."""
    if not (isinstance(n, int) and n >= 2 and 0 < c < n - 1):
        _refuse(f"Erdos-Renyi needs an int n >= 2 and 0 < c < n - 1, got n = {n!r}, c = {c!r}")
    m = int(rng.binomial(n * (n - 1) // 2, c / (n - 1)))
    keys = np.zeros(0, dtype=np.int64)
    while len(keys) < m:
        i, j = rng.integers(n, size=(2, 2 * (m - len(keys)) + 8))
        ok = i != j
        k = np.minimum(i, j)[ok].astype(np.int64) * n + np.maximum(i, j)[ok]
        keys = np.concatenate([keys, k])
        _, first = np.unique(keys, return_index=True)
        keys = keys[np.sort(first)]
    keys = keys[:m]
    return np.stack([keys // n, keys % n], 1).astype(int)


__all__ = ["FactorGraph", "FactorGroup", "erdos_renyi_edges", "pairwise", "random_factor_tree",
           "random_regular_edges", "random_tree_edges"]
