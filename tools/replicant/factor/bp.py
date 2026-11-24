"""Sum-product belief propagation on a FactorGraph and the Bethe free entropy (KZ Sec. 5.2, Eq. (5.1)).

Messages per socket (i, a): chi^{i->a}(s) (variable to factor) and psi^{a->i}(s) (factor to variable), each
normalised over s (KZ Sec. 5.2.2):
    psi^{a->i}(s_i) ~ sum_{s_{da \\ i}} f_a(s_{da}) prod_{j in da \\ i} chi^{j->a}(s_j),
    chi^{i->a}(s)   ~ g_i(s) prod_{b in di \\ a} psi^{b->i}(s).
One iteration is a parallel (synchronous) update of every psi from chi, then of every chi from psi; damping
mixes chi_new = (1 - damping) chi_new + damping chi_old.  The leave-one-out products are taken in logs with the
zeros counted apart, so hard constraints (coloring at beta = oo, f = 1 - delta) are exact: chi^{i->a}(s) = 0
exactly when another factor of i forbids s.  Convergence: max |chi_new - chi_old| < tol.  A variable or
factor all of whose states are forbidden is a contradiction: the run stops, converged = False, with a note.
Marginals b_i(s) ~ g_i(s) prod_{a in di} psi^{a->i}(s).
Bethe free entropy (KZ Eq. (5.1)), at a fixed point:
    N Phi = sum_i ln Z_i + sum_a ln Z_a - sum_(ia) ln Z_ia,
    Z_i = sum_s g_i(s) prod_{a in di} psi^{a->i}(s),  Z_a = sum f_a prod_{i in da} chi^{i->a},
    Z_ia = sum_s chi^{i->a}(s) psi^{a->i}(s);
exact (= ln Z) when the factor graph is a forest, an approximation on a loopy graph: the result says which.
_HOOKS["edge_term"] (1.0, the coefficient of sum ln Z_ia): planted-error hook (tree exactness fails).

  bp(graph, *, iters=1000, damping=0.0, init="uniform", seed=0, tol=1e-12) -> BPState
      init: "uniform" | "random" (uniform random simplex points from `seed`) | ("planted", x[, eps]) (chi^{i->a}
      = one-hot of the configuration x, mixed with eps of the uniform law) | an (E, q) array of chi
  marginals(graph, state) -> (N, q);  bethe_free_entropy(graph, state) -> Bethe
"""
from dataclasses import dataclass

import numpy as np

from ..core import Refusal

_HOOKS = {"edge_term": 1.0}
_L = "bcdefghijklmnopqrstuvwxyz"


@dataclass(frozen=True)
class Messages:
    chi: np.ndarray        # (E, q) variable -> factor
    psi: np.ndarray        # (E, q) factor -> variable


@dataclass(frozen=True)
class BPState:
    messages: Messages
    converged: bool
    iters: int
    delta: float = float("nan")
    note: str = ""


@dataclass(frozen=True)
class Bethe:
    log_z: float           # N Phi_Bethe
    phi: float             # per variable
    variable: float        # sum_i ln Z_i
    factor: float          # sum_a ln Z_a
    edge: float            # sum_(ia) ln Z_ia
    exact: bool            # the factor graph is a forest and BP converged
    label: str


def _contract(gr, chi_blk, skip):
    """For each factor of group gr: sum over its scope of f prod_{j != skip} chi_j (skip None: a scalar)."""
    k = gr.arity
    t = _L[:k] if gr.shared else "a" + _L[:k]
    pos = [j for j in range(k) if j != skip]
    out = "a" + (_L[skip] if skip is not None else "")
    if not pos and gr.shared:
        return np.broadcast_to(gr.tables, (len(gr.scopes),) + gr.tables.shape).copy()
    subs = ",".join([t] + ["a" + _L[j] for j in pos]) + "->" + out
    return np.einsum(subs, gr.tables, *[chi_blk[:, j] for j in pos])


def _factor_to_var(graph, chi):
    psi = np.empty_like(chi)
    for gr, off in zip(graph.groups, graph.offsets):
        m, k = gr.scopes.shape
        blk = chi[off:off + m * k].reshape(m, k, graph.q)
        for p in range(k):
            psi[off + p:off + m * k:k] = _contract(gr, blk, p)
    s = psi.sum(1)
    bad = not np.all(s > 0)
    psi /= np.where(s > 0, s, 1.0)[:, None]
    return psi, bad


def _log_parts(graph, psi):
    """(L, z, S, Zc): logs of psi with zeros set to 0, the zero mask, and their per-variable sums."""
    z = psi <= 0
    L = np.log(np.where(z, 1.0, psi))
    return L, z, graph.A @ L, graph.A @ z.astype(float)


def _log_unary(graph):
    with np.errstate(divide="ignore"):
        return np.log(graph.unary)


def _var_to_factor(graph, psi):
    L, z, S, Zc = _log_parts(graph, psi)
    v = graph.socket_var
    lc = _log_unary(graph)[v] + S[v] - L
    lc[(Zc[v] - z) > 0.5] = -np.inf
    mx = lc.max(1)
    bad = not np.all(np.isfinite(mx))
    chi = np.exp(lc - np.where(np.isfinite(mx), mx, 0.0)[:, None])
    chi /= np.where(chi.sum(1) > 0, chi.sum(1), 1.0)[:, None]
    return chi, bad


def _init(graph, init, seed):
    E, q = graph.n_sockets, graph.q
    if isinstance(init, str) and init == "uniform":
        return np.full((E, q), 1.0 / q)
    if isinstance(init, str) and init == "random":
        x = -np.log(np.random.default_rng(seed).random((E, q)))          # Dirichlet(1, ..., 1) rows
        return x / x.sum(1)[:, None]
    if isinstance(init, tuple) and init and init[0] == "planted":
        x = np.asarray(init[1], dtype=int)
        eps = float(init[2]) if len(init) > 2 else 0.0
        if x.shape != (graph.n,) or x.min() < 0 or x.max() >= q or not 0 <= eps <= 1:
            raise Refusal("precondition:init", "planted init needs a configuration of N states in 0..q-1 and "
                          "0 <= eps <= 1")
        chi = np.zeros((E, q))
        chi[np.arange(E), x[graph.socket_var]] = 1.0
        return (1 - eps) * chi + eps / q
    a = np.asarray(init, dtype=float) if not isinstance(init, (str, tuple)) else None
    if a is None or a.shape != (E, q) or not (np.all(a >= 0) and np.all(a.sum(1) > 0)):
        raise Refusal("precondition:init", f"init {init!r}: 'uniform', 'random', ('planted', x[, eps]) or an "
                      f"(E, q) = ({E}, {q}) array of nonnegative messages")
    return a / a.sum(1)[:, None]


def bp(graph, *, iters=1000, damping=0.0, init="uniform", seed=0, tol=1e-12):
    if not (isinstance(iters, int) and iters >= 1 and 0 <= damping < 1):
        raise Refusal("precondition:bp", f"iters = {iters!r} (int >= 1), damping = {damping!r} (in [0, 1))")
    chi = _init(graph, init, seed)
    delta, note, t = float("inf"), "", 0
    for t in range(1, iters + 1):
        psi, bad = _factor_to_var(graph, chi)
        new, bad2 = _var_to_factor(graph, psi)
        if bad or bad2:
            note = f"contradiction at iteration {t}: a variable or factor has every state forbidden"
            break
        new = (1 - damping) * new + damping * chi
        delta = float(np.max(np.abs(new - chi))) if chi.size else 0.0
        chi = new
        if delta < tol:
            break
    psi, _ = _factor_to_var(graph, chi)
    ok = delta < tol and not note
    if not ok and not note:
        note = f"not converged in {iters} iterations: last max |change| = {delta:.3g} (tol {tol:g})"
    return BPState(Messages(chi, psi), ok, t, delta, note)


def marginals(graph, state):
    _, _, S, Zc = _log_parts(graph, state.messages.psi)
    lb = _log_unary(graph) + S
    lb[Zc > 0.5] = -np.inf
    mx = lb.max(1)
    b = np.exp(lb - np.where(np.isfinite(mx), mx, 0.0)[:, None])
    return b / np.where(b.sum(1) > 0, b.sum(1), 1.0)[:, None]


def _lse(x, axis):
    mx = np.max(x, axis=axis, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    with np.errstate(divide="ignore"):
        return np.squeeze(mx, axis) + np.log(np.sum(np.exp(x - mx), axis=axis))


def bethe_free_entropy(graph, state):
    chi, psi = state.messages.chi, state.messages.psi
    _, _, S, Zc = _log_parts(graph, psi)
    lv = _log_unary(graph) + S
    lv[Zc > 0.5] = -np.inf
    var = float(np.sum(_lse(lv, 1)))
    fac = 0.0
    for gr, off in zip(graph.groups, graph.offsets):
        m, k = gr.scopes.shape
        with np.errstate(divide="ignore"):
            fac += float(np.sum(np.log(_contract(gr, chi[off:off + m * k].reshape(m, k, graph.q), None))))
    with np.errstate(divide="ignore"):
        edge = float(np.sum(np.log(np.sum(chi * psi, 1))))
    log_z = var + fac - _HOOKS["edge_term"] * edge
    forest = graph.is_forest()
    if not state.converged:
        label = f"undetermined: BP not at a fixed point ({state.note})"
    elif forest:
        label = "exact: the factor graph is a tree (KZ Sec. 5.2: Bethe = ln Z on trees)"
    else:
        label = "approximation: Bethe free entropy on a loopy graph (not ln Z)"
    return Bethe(log_z, log_z / graph.n, var, fac, edge, forest and state.converged, label)


__all__ = ["BPState", "Bethe", "Messages", "bethe_free_entropy", "bp", "marginals"]
