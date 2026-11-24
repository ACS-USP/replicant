"""K3: generic factor-graph BP and the Bethe free entropy (KZ Sec. 4.2, Sec. 5.2, Eq. (5.1)): factor/.

BP marginals and the Bethe free entropy against exact enumeration (factor/enumerate.py) on random trees (exact to
1e-12) and on a loopy 3-regular graph (an approximation: the difference is asserted nonzero and labelled).
Planted errors go through factor.bp._HOOKS, restored with identity asserts.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from replicant.core import Refusal
from replicant.factor import bp as BP
from replicant.factor import enumerate as EN
from replicant.factor import graph as FG


def _planted(module, key, value, fn):
    reg = module._HOOKS
    old = reg[key]
    reg[key] = value
    try:
        return fn()
    finally:
        reg[key] = old
        assert module._HOOKS is reg and reg[key] is old


def _refusal(fn):
    try:
        fn()
    except Refusal as r:
        return r
    raise AssertionError("expected a Refusal")


def _random_pairwise(rng, n, q, edges, field=True):
    tables = np.exp(rng.normal(0, 0.8, (len(edges), q, q)))
    unary = np.exp(rng.normal(0, 0.5, (n, q))) if field else None
    return FG.pairwise(n, q, edges, tables, unary)


def _trees(rng):
    """Pairwise trees (q = 2, 3) and a tree of 3-body factors with a 1-body factor, all with random tables."""
    out = []
    for n, q in ((9, 2), (12, 2), (8, 3), (10, 3)):
        out.append(_random_pairwise(rng, n, q, FG.random_tree_edges(n, rng)))
    n, scopes = FG.random_factor_tree(4, 3, rng)                     # 4 factors of arity 3: 9 variables
    facs = [(sc, np.exp(rng.normal(0, 0.7, (2, 2, 2)))) for sc in scopes] + [((0,), np.array([0.3, 1.7]))]
    out.append(FG.FactorGraph(np.exp(rng.normal(0, 0.4, (n, 2))), facs))
    return out


def test_bp_exact_on_trees_and_approximate_on_loops():
    rng = np.random.default_rng(20261124)
    for g in _trees(rng):
        assert g.is_forest()
        st = BP.bp(g, iters=200, init="random", seed=3)
        assert st.converged, st.note
        lz, marg = EN.exact(g)
        bethe = BP.bethe_free_entropy(g, st)
        assert bethe.exact and bethe.label.startswith("exact"), bethe.label
        assert abs(bethe.log_z - lz) < 1e-12, (bethe.log_z, lz)
        assert np.max(np.abs(BP.marginals(g, st) - marg)) < 1e-12
        # planted: the edge term counted with the wrong weight breaks tree exactness
        bad = _planted(BP, "edge_term", 0.5, lambda: BP.bethe_free_entropy(g, st).log_z)
        assert abs(bad - lz) > 1e-3, (bad, lz)
    # loopy: a 3-regular graph on N = 12, ferromagnetic Ising couplings and fields
    g = FG.pairwise(12, 2, FG.random_regular_edges(12, 3, rng), np.exp(0.6 * np.array([[1, -1], [-1, 1]])),
                    np.exp(0.2 * np.array([[1, -1]] * 12)))
    assert not g.is_forest()
    st = BP.bp(g, iters=2000, damping=0.2)
    assert st.converged, st.note
    lz, marg = EN.exact(g)
    bethe = BP.bethe_free_entropy(g, st)
    assert not bethe.exact and bethe.label.startswith("approximation"), bethe.label
    gap, mgap = abs(bethe.log_z - lz), np.max(np.abs(BP.marginals(g, st) - marg))
    assert 1e-4 < gap < 1.0 and 1e-4 < mgap < 0.2, (gap, mgap)        # nonzero: Bethe is not ln Z on loops


def test_bp_messages_and_refusals():
    """Hard constraints (3-coloring of a tree, f = 1 - delta) are exact through the zero counting; a contradiction
    (K4 2-colored) stops BP unconverged, with its note; malformed graphs and inits are refused."""
    rng = np.random.default_rng(7)
    edges = FG.random_tree_edges(8, rng)
    g = FG.pairwise(8, 3, edges, 1.0 - np.eye(3))
    st = BP.bp(g, iters=100, init="random", seed=1)
    lz, marg = EN.exact(g)
    assert st.converged and abs(lz - (np.log(3) + 7 * np.log(2))) < 1e-12        # q (q - 1)^(N - 1) colorings
    assert abs(BP.bethe_free_entropy(g, st).log_z - lz) < 1e-12
    assert np.max(np.abs(BP.marginals(g, st) - marg)) < 1e-12
    k4 = FG.pairwise(4, 2, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], 1.0 - np.eye(2))
    st = BP.bp(k4, iters=50, init=("planted", [0, 1, 0, 1]))
    assert not st.converged and "contradiction" in st.note, st.note
    assert "undetermined" in BP.bethe_free_entropy(k4, st).label
    tri = FG.pairwise(3, 2, [[0, 1], [1, 2], [0, 2]], 1.0 - np.eye(2))
    assert _refusal(lambda: FG.pairwise(3, 2, [[0, 0]], np.ones((2, 2)))).reason == "precondition:factor-graph"
    assert _refusal(lambda: FG.pairwise(3, 2, [[0, 5]], np.ones((2, 2)))).reason == "precondition:factor-graph"
    assert _refusal(lambda: FG.pairwise(3, 2, [[0, 1]], np.ones((3, 3)))).reason == "precondition:factor-graph"
    assert _refusal(lambda: BP.bp(tri, init="ones")).reason == "precondition:init"
    assert _refusal(lambda: BP.bp(tri, iters=0)).reason == "precondition:bp"
    assert _refusal(lambda: EN.exact(FG.FactorGraph(30, [], q=2))).reason == "precondition:enumeration-size"


def test_graph_samplers():
    """Degrees of the samplers (d-regular exactly, simple; ER mean degree c within its sampling error); Pruefer
    trees are trees.  Positive controls: the same predicates reject planted wrong samples -- a regular sample with
    one edge doubled (not simple), the ER sampler run at c/2 (the mean-degree check, 13 sd away), and a tree with
    one extra edge (not a forest)."""
    rng = np.random.default_rng(11)
    regular_ok = lambda e, n, d: bool(np.all(np.bincount(e.ravel(), minlength=n) == d)
                                      and len(np.unique(e[:, 0] * n + e[:, 1])) == len(e) and np.all(e[:, 0] < e[:, 1]))
    simple = lambda e, n: len(np.unique(e[:, 0] * n + e[:, 1])) == len(e) and bool(np.all(e[:, 0] != e[:, 1]))
    mean_ok = lambda e, n, c: abs(2 * len(e) / n - c) < 4 * np.sqrt(2 * c / n)      # 2M/N ~ c, sd sqrt(2c/N)
    e = FG.random_regular_edges(2000, 5, rng)
    assert regular_ok(e, 2000, 5)
    assert not regular_ok(np.vstack([e[1:], e[:1], e[:1]]), 2000, 5)               # planted: one edge doubled
    e = FG.erdos_renyi_edges(4000, 3.0, rng)
    assert simple(e, 4000) and mean_ok(e, 4000, 3.0)
    assert not mean_ok(FG.erdos_renyi_edges(4000, 1.5, rng), 4000, 3.0)              # planted: c read as c/2
    assert not simple(np.vstack([e, e[:1]]), 4000)
    for n in (1, 2, 17):
        t = FG.random_tree_edges(n, rng)
        assert len(t) == n - 1 and FG.pairwise(n, 2, t, np.ones((2, 2))).is_forest() if n > 1 else len(t) == 0
    t = np.asarray(FG.random_tree_edges(17, rng))
    far = next(x for x in range(1, 17) if not any({0, x} == set(map(int, ed)) for ed in t))    # not adjacent to 0
    extra = np.vstack([t, [[0, far]]])
    assert not FG.pairwise(17, 2, extra, np.ones((2, 2))).is_forest()               # planted: a cycle
