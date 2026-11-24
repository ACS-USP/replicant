"""Potts / coloring BP on one sampled graph through the generic factor/ engine, from two starts (KZ Ch. 6).

Convention as coloring/potts_bp.py: f(s, s') = exp(-beta delta_{s s'}) (beta = inf: 1 - delta, the coloring).
  potts_graph(q, n, edges, beta) -> FactorGraph
  two_starts(graph, *, iters, damping, seed) -> TwoStarts: BP from the planted (ordered: every variable in color
      0) start and from a random start (uniform random messages from `seed`); the Bethe free entropy per variable
      of each; the phase is the larger one only when BOTH converge (otherwise "undetermined" with the reason, never
      refused: the ruling for first-order points), "same" when they reach one fixed point (within 1e-9)
  bp_convergence(q, c, beta, *, n, seed, iters, damping) -> (converged, max |b - 1/q|): BP on a sampled
      Erdos-Renyi graph from a random start (the Kesten-Stigum test of KZ Sec. 6.6)
"""
import math
from dataclasses import dataclass

import numpy as np

from ..factor import bp as BP
from ..factor import graph as FG
from .potts_bp import _q, beta_value


def potts_graph(q, n, edges, beta):
    q, b = _q(q), beta_value(beta)
    table = 1.0 - np.eye(q) if b == math.inf else np.exp(-b * np.eye(q))
    return FG.pairwise(n, q, edges, table)


@dataclass(frozen=True)
class TwoStarts:
    phi_planted: float
    phi_random: float
    b0_planted: float          # mean marginal of color 0 (the planted color)
    b0_random: float
    converged_planted: bool
    converged_random: bool
    phase: str                 # "ordered" | "paramagnetic-like" | "same" | "undetermined"
    note: str


def two_starts(graph, *, iters=3000, damping=0.0, seed=0):
    out = {}
    for name, init in (("planted", ("planted", np.zeros(graph.n, dtype=int))), ("random", "random")):
        st = BP.bp(graph, iters=iters, damping=damping, init=init, seed=seed)
        out[name] = (st, BP.bethe_free_entropy(graph, st).phi, float(BP.marginals(graph, st)[:, 0].mean()))
    (sp, fp, bp_), (sr, fr, br) = out["planted"], out["random"]
    if not (sp.converged and sr.converged):
        phase = "undetermined"
        note = "; ".join(f"{k} start: {v[0].note}" for k, v in out.items() if not v[0].converged)
    elif abs(fp - fr) < 1e-9:
        phase, note = "same", "both starts reach one fixed point"
    else:
        phase = "ordered" if fp > fr else "paramagnetic-like"
        note = f"Bethe phi planted {fp:.12g} vs random {fr:.12g}: the larger dominates (KZ Sec. 6.5.2)"
    return TwoStarts(fp, fr, bp_, br, sp.converged, sr.converged, phase, note)


def bp_convergence(q, c, beta, *, n, seed, iters=2000, damping=0.5):
    rng = np.random.default_rng(seed)
    g = potts_graph(q, n, FG.erdos_renyi_edges(n, float(c), rng), beta)
    st = BP.bp(g, iters=iters, damping=damping, init="random", seed=seed + 1)
    return st.converged, float(np.max(np.abs(BP.marginals(g, st) - 1.0 / g.q)))


__all__ = ["TwoStarts", "bp_convergence", "potts_graph", "two_starts"]
