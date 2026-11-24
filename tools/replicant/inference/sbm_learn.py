"""Learning the block-model parameters (n_a, c_ab) by expectation-maximisation with BP (KZ Sec. 10.2.2 and
Eqs. (10.27)-(10.28) / (10.45)-(10.46); DKMZ Sec. III.C, "BP-learning", Eqs. (34)-(35)).

Convention: inference/sbm_q.py (edge probability c_ab / N, sizes n_a).  One EM step: BP at the current (n, C)
(inference/sbm_q.run_bp, warm-started from the previous messages), then
    n_a  = (1/N) sum_i psi^i_a,
    c_ab = (1 / (N n_a n_b)) sum_{(ij) in E} c_ab (chi^{i->j}_a chi^{j->i}_b + chi^{i->j}_b chi^{j->i}_a) / Z^{ij},
Z^{ij} = sum_ab c_ab chi^{i->j}_a chi^{j->i}_b (KZ Eq. (10.20)): the expected numbers of nodes per group and of
edges per pair of groups under the BP marginals. start="spectral" (default): the group sizes and the edge counts
between groups of the non-backtracking spectral clustering (inference/nonbacktracking.py, k-means seeded by `seed`;
a data-driven start, the planted labels are not read); start="random": n_a = 1/q, c_ab = chat (1 + 0.5 u_ab) with u
a random symmetric matrix of U(-1, 1) entries from `seed`, rescaled to sum_ab c_ab n_a n_b = chat = 2M/N. EM is not
guaranteed to reach the planted parameters (DKMZ: "only from some finite fraction of the possible initial
parameters"); a run that ends at the factorized point (every psi^i = n) is reported as such.
Group count (DKMZ Sec. III.C: the smallest q whose free energy does not improve at q + 1): after convergence two
groups a != b are redundant when max_c |c_ac - c_bc| < REDUNDANT chat, a group is empty when n_a < EMPTY / q; either
gives status "detected:redundant-groups".  An over-specified q usually leaves EM drifting along a flat direction of
the Bethe free entropy (one group split in two): status "undetermined" (not converged).  sbm_em_group_count compares
the Bethe free entropies of EM at q - 1, q and q + 1 (DKMZ's rule, with the threshold gain(k, N)): it detects a
planted group count below or above the run's q by name.  Finite-N values are statistical.
_HOOKS["edge_update"] (1.0): the factor on the c_ab update (planted error: a wrong value biases c_in, c_out).
"""
import math
from dataclasses import dataclass

import numpy as np

from ..core import Refusal
from ..csp import _checks
from . import nonbacktracking as NB
from . import sbm_q as SQ

_HOOKS = {"edge_update": 1.0}
REDUNDANT, EMPTY, GAIN = 0.1, 0.05, 2e-3


def gain(k, N):
    """The free-entropy gain that counts as real for k - 1 -> k groups: max(GAIN, 4 k / N), a heuristic penalty of the
    order of the parameters added per node (the overfitting gain of EM at N = 2000 was ~1e-3, build)."""
    return max(GAIN, 4 * k / N)


@dataclass
class EMResult:
    q: int
    sizes: np.ndarray         # learned n_a
    affinity: np.ndarray      # learned c_ab
    phi: float                # Bethe free entropy per node at the last BP fixed point
    converged: bool
    iterations: int
    overlap: float            # against the planted labels (KZ Eq. (10.4))
    factorized: bool
    status: str               # "ok" | "detected:redundant-groups" | "undetermined"
    note: str
    label: str = "statistical"


def _start(graph, q, seed, start):
    chat = 2 * len(graph.edges) / graph.N
    n = np.full(q, 1.0 / q)
    if start == "random":
        u = np.random.default_rng([seed, 41]).uniform(-1, 1, (q, q))
        C = chat * (1 + 0.5 * (u + u.T) / 2)
        return n, C * chat / float(n @ C @ n), chat
    if start != "spectral":
        raise Refusal("precondition:start", f"start = {start!r}: 'spectral' or 'random'")
    est = NB.spectral_clustering(graph, q, seed=seed).assignment
    n = np.maximum(np.bincount(est, minlength=q) / graph.N, 1.0 / graph.N)
    cnt = np.zeros((q, q))
    np.add.at(cnt, (est[graph.edges[:, 0]], est[graph.edges[:, 1]]), 1.0)
    C = (cnt + cnt.T) / (graph.N * np.outer(n, n))
    return n / n.sum(), np.maximum(C, 1e-3 * chat), chat


def _m_step(graph, n, C, chi, psi):
    M = len(graph.edges)
    a, b = chi[:M], chi[M:]
    joint = C[None] * a[:, :, None] * b[:, None, :]
    joint /= joint.sum((1, 2), keepdims=True)
    S = joint.sum(0)
    n_new = psi.mean(0)
    C_new = _HOOKS["edge_update"] * (S + S.T) / (graph.N * np.outer(n_new, n_new))
    return n_new, C_new


def _redundant(n, C, chat):
    q = len(n)
    if np.any(n < EMPTY / q):
        return f"a group is empty (min n_a = {n.min():.3g} < {EMPTY / q:.3g})"
    for x in range(q):
        for y in range(x + 1, q):
            d = float(np.max(np.abs(C[x] - C[y])))
            if d < REDUNDANT * chat:
                return (f"groups {x} and {y} have the same affinities (max |c_{x}. - c_{y}.| = {d:.3g} < "
                        f"{REDUNDANT} chat)")
    return ""


def sbm_em(graph, q, *, iters, seed, bp_iters=300, tol=1e-4, damping=0.5, start="spectral") -> EMResult:
    if not isinstance(graph, SQ.SBMqGraph):
        raise Refusal("precondition:sbm-graph", "sbm_em needs an SBMqGraph (inference/sbm_q.py)")
    q, seed = SQ.group_count(q), _checks.seed(seed)
    if isinstance(iters, bool) or not isinstance(iters, int) or iters < 1:
        raise Refusal("precondition:positive-iters", f"iters = {iters!r} is not a positive int")
    if len(graph.edges) == 0:
        raise Refusal("precondition:sbm-graph", "the graph has no edges: nothing to learn")
    n, C, chat = _start(graph, q, seed, start)
    chi, change, conv, it, bp = None, math.inf, False, 0, None
    lab = np.where(graph.labels < q, graph.labels, 0)
    g = SQ.SBMqGraph(graph.N, lab, graph.edges, graph.seed)
    for it in range(1, iters + 1):
        bp = SQ.run_bp(g, n, C, init="random", iters=bp_iters, seed=seed, damping=damping, chi0=chi)
        chi = bp.messages
        n_new, C_new = _m_step(g, n, C, chi, bp.marginals)
        change = float(max(np.max(np.abs(C_new - C)) / chat, np.max(np.abs(n_new - n))))
        n, C = n_new, C_new
        if change < tol:
            conv = True
            break
    bp = SQ.run_bp(g, n, C, init="random", iters=bp_iters, seed=seed, damping=damping, chi0=chi)
    red = _redundant(n, C, chat)
    if not conv:
        status, note = "undetermined", f"EM not converged in {iters} steps (last change {change:.3g}, tol {tol:g})"
    elif red:
        status, note = "detected:redundant-groups", f"q = {q} exceeds the groups the graph supports: {red}"
    else:
        status, note = "ok", "EM converged"
    if not bp.converged:
        status, note = "undetermined", note + f"; BP at the learned parameters: {bp.note}"
    ov = bp.overlap if graph.labels.max() < q else float("nan")
    return EMResult(q, n, C, bp.phi, conv, it, ov, bp.factorized, status, note)


@dataclass
class GroupCount:
    q: int
    status: str               # "consistent" | "detected:too-many-groups" | "detected:too-few-groups" | "undetermined"
    phi: dict                 # q' -> Bethe free entropy per node of the EM run at q' (q' = 1: the Erdos-Renyi null)
    note: str


def _phi_null(graph):
    """q = 1: the Erdos-Renyi model with c = 2M/N; its Bethe free entropy is exact: (M/N) ln c - c/2."""
    c = 2 * len(graph.edges) / graph.N
    return len(graph.edges) / graph.N * math.log(c) - c / 2


def sbm_em_group_count(graph, q, *, iters, seed, **kw) -> GroupCount:
    """DKMZ's rule at q: EM at q - 1, q and q + 1 (q - 1 = 1 is the Erdos-Renyi null).  Too many groups when the
    q run does not raise the Bethe free entropy over q - 1 by more than gain(q, N), or is redundant; too few when the
    q + 1 run raises it by more than gain(q + 1, N) without being redundant.  A run whose BP did not converge makes
    it undetermined."""
    q = SQ.group_count(q)
    runs = {k: sbm_em(graph, k, iters=iters, seed=seed, **kw) for k in (q - 1, q, q + 1) if k >= 2}
    phi = {k: r.phi for k, r in runs.items()}
    if q == 2:
        phi[1] = _phi_null(graph)
    bad = [f"q' = {k}: {r.note}" for k, r in runs.items() if r.status == "undetermined" and "BP at" in r.note]
    if bad:
        return GroupCount(q, "undetermined", phi, "; ".join(bad))
    em = "; ".join(f"q' = {k}: {r.status} ({r.iterations} EM steps)" for k, r in runs.items())
    g0, g1 = gain(q, graph.N), gain(q + 1, graph.N)
    if runs[q].status == "detected:redundant-groups" or phi[q] - phi[q - 1] < g0:
        return GroupCount(q, "detected:too-many-groups", phi, f"Phi(q) - Phi(q - 1) = {phi[q] - phi[q - 1]:.3g} "
                          f"< {g0:.3g}; {em}")
    if runs[q + 1].status != "detected:redundant-groups" and phi[q + 1] - phi[q] > g1:
        return GroupCount(q, "detected:too-few-groups", phi, f"Phi(q + 1) - Phi(q) = {phi[q + 1] - phi[q]:.3g} > "
                          f"{g1:.3g}; {em}")
    return GroupCount(q, "consistent", phi, f"Phi(q) - Phi(q - 1) = {phi[q] - phi[q - 1]:.3g}, Phi(q + 1) - Phi(q) = "
                      f"{phi[q + 1] - phi[q]:.3g}; {em}")


__all__ = ["EMResult", "GroupCount", "gain", "sbm_em", "sbm_em_group_count"]
