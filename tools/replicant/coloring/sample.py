"""Graph coloring on sampled graphs: exhaustive counts at small N, the Bethe count from BP, and BP from the planted
start on planted graphs (KZ Chs. 6, 13).

Convention: q colors, proper colorings (beta = oo, factor 1 - delta_{s s'}); G(N, M) = M distinct uniform pairs.
  count_colorings(n, edges, q) -> int: every one of the q^n configurations (refused above MAX_CONFIGS =
      precondition:enumeration-size), the independent route for everything below.
  mean_count(q, n, m, *, graphs, seed) -> list of counts over `graphs` samples of G(n, m) (to compare with the exact
      first moment of coloring/first_moment.py in a 99% Student-t band).
  bethe_log_count(n, edges, q, *, seed) -> BetheCount: BP (factor/bp.py) from a random start; N Phi_Bethe and
      factor/bp's label (exact on a forest, an approximation on a loopy graph, undetermined when not converged).
  planted_bp(q, c, *, n, seeds, iters, damping=None) -> [PlantedRun]: BP on the planted graph of each seed
      (coloring/planted.py) from the planted start (chi = one-hot of s*), damping above K3's uniform-mode bound
      (coloring/potts_graph.uniform_mode_damping) by default; overlap = (max over color permutations pi of
      (1/N) sum_i b_i(pi(s*_i)) - 1/q)/(1 - 1/q) (0 at the paramagnet); Bethe Phi per node and the paramagnetic
      value on the same graph, ln q + (M/N) ln(1 - 1/q) (chi = 1/q is a fixed point on any graph).
_HOOKS["allow_mono"] (False; planted error True: the first edge of each graph may be monochromatic) and
_HOOKS["start"] ("planted"; planted error "random": a random start, which stays paramagnetic below c_l).
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..core import Refusal
from ..factor import bp as BP
from . import planted as PL
from .potts_bp import _q
from .potts_graph import potts_graph, uniform_mode_damping

_HOOKS = {"allow_mono": False, "start": "planted"}
MAX_CONFIGS = 6_000_000
_CHUNK = 1 << 19


def _edges(n, edges):
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if e.size and (e.min() < 0 or e.max() >= n or np.any(e[:, 0] == e[:, 1])):
        raise Refusal("precondition:graph", "edges must join distinct nodes in 0..n-1")
    return e


def count_colorings(n, edges, q):
    q = _q(q)
    e = _edges(n, edges)
    if q ** n > MAX_CONFIGS:
        raise Refusal("precondition:enumeration-size", f"q^N = {q}^{n} > {MAX_CONFIGS} configurations")
    if _HOOKS["allow_mono"]:
        e = e[1:]
    powers = q ** np.arange(n, dtype=np.int64)
    total = 0
    for start in range(0, q ** n, _CHUNK):
        idx = np.arange(start, min(start + _CHUNK, q ** n), dtype=np.int64)
        conf = (idx[None, :] // powers[:, None]) % q                       # (n, chunk)
        ok = np.ones(len(idx), dtype=bool)
        for i, j in e:
            ok &= conf[i] != conf[j]
        total += int(ok.sum())
    return total


def mean_count(q, n, m, *, graphs, seed):
    P = n * (n - 1) // 2
    if not (isinstance(m, int) and 0 <= m <= P):
        raise Refusal("precondition:graph", f"M = {m!r}: an int in [0, N(N-1)/2]")
    rng = np.random.default_rng([int(seed), 41])
    iu = np.array(np.triu_indices(n, 1)).T
    return [count_colorings(n, iu[rng.choice(P, size=m, replace=False)], q) for _ in range(int(graphs))]


@dataclass(frozen=True)
class BetheCount:
    phi_total: float          # N Phi_Bethe (the Bethe estimate of ln Z)
    converged: bool
    label: str


def bethe_log_count(n, edges, q, *, seed=0, iters=5000):
    g = potts_graph(q, n, _edges(n, edges), "inf")
    st = BP.bp(g, iters=iters, init="random", seed=seed, tol=1e-13)
    be = BP.bethe_free_entropy(g, st)
    return BetheCount(be.log_z, st.converged, be.label)


@dataclass(frozen=True)
class PlantedRun:
    seed: int
    overlap: float
    converged: bool
    phi: float
    phi_para: float
    iters: int
    note: str


def _overlap(b, labels, q):
    conf = np.zeros((q, q))
    np.add.at(conf, labels, b)                                             # conf[s*, s] = sum_i b_i(s)
    r, c = linear_sum_assignment(-conf)
    return (conf[r, c].sum() / len(labels) - 1.0 / q) / (1 - 1.0 / q)


def planted_bp(q, c, *, n, seeds=(0, 1, 2, 3), iters=3000, damping=None):
    q = _q(q)
    d = uniform_mode_damping(q, c, "inf")[1] if damping is None else float(damping)
    out = []
    for s in seeds:
        g = PL.planted_coloring(n, q, c, seed=s)
        fg = potts_graph(q, n, g.edges, "inf")
        init = ("planted", g.labels) if _HOOKS["start"] == "planted" else "random"
        st = BP.bp(fg, iters=iters, damping=d, init=init, seed=int(s) + 1, tol=1e-8)
        be = BP.bethe_free_entropy(fg, st)
        para = math.log(q) + len(g.edges) / n * math.log1p(-1.0 / q)
        out.append(PlantedRun(int(s), float(_overlap(BP.marginals(fg, st), g.labels, q)), st.converged, be.phi,
                              para, st.iters, st.note or be.label))
    return out


__all__ = ["BetheCount", "MAX_CONFIGS", "PlantedRun", "bethe_log_count", "count_colorings", "mean_count", "planted_bp"]
