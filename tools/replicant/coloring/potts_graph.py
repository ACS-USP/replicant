"""Potts / coloring BP on one sampled graph through the generic factor/ engine, from two starts (KZ Ch. 6).

Convention as coloring/potts_bp.py: f(s, s') = exp(-beta delta_{s s'}) (beta = inf: 1 - delta, the coloring).
  potts_graph(q, n, edges, beta) -> FactorGraph
  two_starts(graph, *, iters, damping, seed) -> TwoStarts: BP from the planted (ordered: every variable in color
      0) start and from a random start (uniform random messages from `seed`); the Bethe free entropy per variable
      of each; the phase is the larger one only when BOTH converge, each Phi with factor.bp's Bethe label (an approximation on a loopy graph) (otherwise "undetermined" with the reason, never
      refused: the ruling for first-order points), "same" when they reach one fixed point (within 1e-9)
  uniform_mode_damping(q, c, beta) -> (d_min, d_auto): parallel BP linearised at chi = 1/q has, besides the bulk
      (|mu| <= sqrt(c) theta/(q - theta), the KS modes of Eq. (6.23)), the uniform mode mu_u = -c theta/(q - theta)
      (the Perron eigenvalue c of the non-backtracking operator times lambda_max of Eq. (6.19)), unstable under
      synchronous updates for |mu_u| > 1 although the notes' variance criterion is not crossed.  Damping d maps an
      eigenvalue mu to (1 - d) mu + d, so the uniform mode is inside (-1, 1) iff d > d_min = (|mu_u| - 1)/(|mu_u| + 1)
      (= 1 - 2(q - 1)/(q - 1 + c) at beta = oo); the bulk threshold is unchanged by damping.  d_auto balances the
      two: the larger of 1/2 and (|mu_u| - r)/(|mu_u| - r + 2), r = sqrt(c) theta/(q - theta), with |mu_u| raised
      by 10% for the finite-N Perron eigenvalue.
  bp_convergence(q, c, beta, *, n, seed, iters, damping=None) -> KSRun: BP on a sampled Erdos-Renyi graph from a
      random start (the Kesten-Stigum test of KZ Sec. 6.6), damping d_auto by default; a damping at or below d_min
      makes the run's status undetermined (non-convergence would then not be the KS signal)
"""
import math
from dataclasses import dataclass

import numpy as np

from ..factor import bp as BP
from ..factor import graph as FG
from ..core import Refusal
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
    label_planted: str = ""    # factor.bp's Bethe label: exact (tree) / approximation (loopy) / undetermined
    label_random: str = ""


def two_starts(graph, *, iters=3000, damping=0.0, seed=0):
    out = {}
    for name, init in (("planted", ("planted", np.zeros(graph.n, dtype=int))), ("random", "random")):
        st = BP.bp(graph, iters=iters, damping=damping, init=init, seed=seed)
        be = BP.bethe_free_entropy(graph, st)
        out[name] = (st, be.phi, float(BP.marginals(graph, st)[:, 0].mean()), be.label)
    (sp, fp, bp_, lp), (sr, fr, br, lr) = out["planted"], out["random"]
    if not (sp.converged and sr.converged):
        phase = "undetermined"
        note = "; ".join(f"{k} start: {v[0].note}" for k, v in out.items() if not v[0].converged)
    elif abs(fp - fr) < 1e-9:
        phase, note = "same", "both starts reach one fixed point"
    else:
        phase = "ordered" if fp > fr else "paramagnetic-like"
        note = f"Bethe phi planted {fp:.12g} vs random {fr:.12g}: the larger dominates (KZ Sec. 6.5.2)"
    return TwoStarts(fp, fr, bp_, br, sp.converged, sr.converged, phase, note, lp, lr)


def uniform_mode_damping(q, c, beta):
    q, b = _q(q), beta_value(beta)
    th = 1.0 - math.exp(-b) if b != math.inf else 1.0
    lam = th / (q - th)
    mu, r = 1.1 * float(c) * lam, math.sqrt(float(c)) * lam
    d_min = max(0.0, (mu - 1) / (mu + 1))
    return d_min, max(0.5, (mu - r) / (mu - r + 2), d_min + 0.05 * (1 - d_min))


@dataclass(frozen=True)
class KSRun:
    converged: bool
    max_deviation: float       # max_i |b_i - 1/q|
    damping: float
    status: str                # "determined" | "undetermined" (damping does not tame the uniform mode)
    note: str


def bp_convergence(q, c, beta, *, n, seed, iters=2000, damping=None):
    d_min, d_auto = uniform_mode_damping(q, c, beta)
    d = d_auto if damping is None else beta_value(damping)
    if not 0 <= d < 1:
        raise Refusal("precondition:damping", f"damping = {damping!r}: a number in [0, 1)")
    rng = np.random.default_rng(seed)
    g = potts_graph(q, n, FG.erdos_renyi_edges(n, float(c), rng), beta)
    st = BP.bp(g, iters=iters, damping=d, init="random", seed=seed + 1)
    ok = d > d_min
    note = (f"damping {d:.4g} > {d_min:.4g}: the synchronous uniform mode is damped, non-convergence is the KS "
            "(variance) instability" if ok else f"damping {d:.4g} <= {d_min:.4g}: the uniform mode of the "
            "synchronous update is unstable, so (non-)convergence says nothing about Kesten-Stigum")
    return KSRun(st.converged, float(np.max(np.abs(BP.marginals(g, st) - 1.0 / g.q))), d,
                 "determined" if ok else "undetermined", note)


__all__ = ["KSRun", "TwoStarts", "bp_convergence", "uniform_mode_damping", "potts_graph", "two_starts"]
