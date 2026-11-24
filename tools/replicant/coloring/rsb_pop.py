"""Population dynamics of the 1RSB cavity for graph coloring at beta = oo (KZ Secs. 14.2-14.3; Zdeborova &
Krzakala, PRE 76, 031131 (2007), arXiv:0704.1269, "ZK", Secs. III-IV).

Convention: q colors, BP messages psi in the simplex (psi_s = the cavity probability of color s), the beta = oo BP
update (KZ Eq. (13.7)): psi ~ prod_k (1 - psi_k), Z = sum_s prod_k (1 - psi_k,s) (Z = 0: every color forbidden).
Graphs: "regular" (degree d; d - 1 incoming messages per update) or "er" (mean degree c; Poisson(c) incoming, the
excess degree of a Poisson law is Poisson).  Every update is synchronous (a whole new population from the old one).

  coloring_1rsb(q, d_or_c, m, *, graph, pop, sweeps, measure, seed, init) -> RSB
    KZ Eqs. (14.10)-(14.15) with Parisi parameter m in [0, 1]: a law P(psi) over the clusters (BP fixed points) of
    the message on an edge, the new message reweighted by Z^m (m = 0: by 1[Z > 0]).  Regular: one population of
    `pop` messages (the factorized solution, the same law on every edge).  ER: a population of populations, `pop`
    = (N1, N2): N1 laws of N2 messages each.  Every new message gets an independent random color permutation (the
    color-symmetric solution ZK use).  Replicated free entropy per node (KZ (14.12)-(14.15) averaged):
        Psi(m) = E ln <Z_i^m> - (c/2) E ln <Z_ij^m>,   Phi(m) = dPsi/dm = E <Z_i^m ln Z_i>/<Z_i^m> - (c/2) (edge)
        Sigma(m) = Psi(m) - m Phi(m)  (KZ (14.2)),
    <.> over the law of one edge (regular: the population), E over degrees and edges; Z_i from the full degree
    (d, or Poisson(c)), Z_ij = 1 - psi_a . psi_b.  At m = 1 Psi equals the annealed (paramagnetic) free entropy
    ln q + (c/2) ln(1 - 1/q) (the planted / reconstruction identity; returned as psi_annealed for the check); at
    m = 0 Sigma = Psi(0) = ln P(Z_i > 0) - (c/2) ln P(Z_ij > 0), the frozen-cluster complexity of SP.
    q1 = E <sum_s psi_s^2> (1/q at the trivial solution); phase trivial / nontrivial / undetermined from the measured
    series of q1 - 1/q and Sigma (_phase: a decay is trivial only when it reaches the rounding floor TRIVIAL or falls
    by a decade over the measured sweeps; a stationary series is nontrivial; anything else is undetermined).
    init: "frozen" (every message one-hot on a random color: the informative / planted-like start) or ("tilt", eps)
    (1/q + eps (e_s - 1/q), s random: a small perturbation of the trivial solution; q1 - 1/q then grows by the
    factor c lambda^2 = c/(q - 1)^2 per sweep at m = 1, the linear stability of KZ (13.9) at beta = oo).
  frozen_sp(q, d_or_c, *, graph, pop, sweeps, measure, seed) -> SP
    KZ Sec. 14.3, the SP of frozen messages, color-symmetric (n_p = eta for every p, as the notes assume on regular
    graphs; the update maps symmetric surveys to symmetric ones), a population of scalars eta for ER:
        eta = sum_l (-1)^(l-1) C(q-1, l-1) prod_k (1 - l eta_k) / sum_l (-1)^(l-1) C(q, l) prod_k (1 - l eta_k),
        Sigma = E ln sum_l (-1)^(l-1) C(q, l) prod_{k in full degree} (1 - l eta_k) - (c/2) E ln(1 - q eta_a eta_b).
    Start eta = (1 - 1e-3)/q (almost every survey frozen).  nontrivial: E eta > 1e-6.
Diagnostic (both): factor/population.relaxation, the shared drift / monotone-relaxation test, on the measured Sigma
(and E eta, q1 - 1/q); a run that fails it is "undetermined" with the reason (never refused).  Statistical: one seed gives one estimate; bands come from named
seeds (coloring/rsb.py).
_HOOKS["symmetrize"] (True: the color permutation of new messages; planted error False lets the population collapse
on one color, and Psi(1) leaves the annealed value).
"""
import math
from dataclasses import dataclass
from math import comb

import numpy as np

from ..core import Refusal
from ..factor.population import relaxation
from .potts_bp import _q

_HOOKS = {"symmetrize": True}
TRIVIAL = 1e-8          # q1 - 1/q below this: the trivial solution to rounding (the run starts at O(1))
DECADE = 0.1            # a decay by at least this factor (last fifth / first fifth of the window): decaying


def _int(name, v, lo=1):
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < lo:
        raise Refusal("precondition:population", f"{name} = {v!r}: an int >= {lo}")
    return int(v)


def _degree(graph, d_or_c):
    if graph == "regular":
        if isinstance(d_or_c, bool) or not isinstance(d_or_c, (int, np.integer)) or d_or_c < 2:
            raise Refusal("precondition:d", f"d = {d_or_c!r}: an int degree >= 2 (regular graph)")
        return int(d_or_c)
    if graph == "er":
        c = float(d_or_c) if not isinstance(d_or_c, bool) else math.nan
        if not (math.isfinite(c) and c > 0):
            raise Refusal("precondition:c", f"c = {d_or_c!r}: a finite mean degree > 0 (ER graph)")
        return c
    raise Refusal("unsupported:graph", f"graph = {graph!r}: 'regular' | 'er'")


def _m(m):
    try:
        v = float(m) if not isinstance(m, bool) else math.nan
    except (TypeError, ValueError):
        v = math.nan
    if not 0 <= v <= 1:
        raise Refusal("precondition:parisi-m", f"m = {m!r}: the Parisi parameter must be in [0, 1]")
    return v


def _wt(Z, m):
    return (Z > 0).astype(float) if m == 0 else np.where(Z > 0, Z, 0.0) ** m


def _lz(Z):
    return np.log(np.where(Z > 0, Z, 1.0))


def _children(rng, graph, deg, n, full):
    """(n,) numbers of incoming messages: d - 1 (d when full) or Poisson(c)."""
    if graph == "regular":
        return np.full(n, deg if full else deg - 1)
    return rng.poisson(deg, n)


def _prod(msgs, k):
    """prod over the first k[b] of msgs (B, ..., Kmax, q) of (1 - psi): (B, ..., q)."""
    mask = np.arange(msgs.shape[-2]) < k[..., None]
    return np.prod(np.where(mask[..., None], 1.0 - msgs, 1.0), axis=-2)


def _permute(x, rng):
    if not _HOOKS["symmetrize"]:
        return x
    return np.take_along_axis(x, np.argsort(rng.random(x.shape), axis=-1), -1)


def _phase(dev, sig, tol):
    """('trivial' | 'nontrivial' | 'undetermined', reason) of an m = 1 / frozen-start run from its measured series of
    q1 - 1/q and Sigma.  trivial: q1 - 1/q < TRIVIAL (the trivial point to rounding), or its mean over the last fifth
    of the measured sweeps is at most DECADE times its mean over the first fifth (below c_d the trivial solution
    attracts linearly, by c/(q - 1)^2 < 1 per sweep at m = 1, so the decay is geometric: a decade over the window);
    nontrivial: q1 - 1/q stationary by the shared diagnostic factor.population.relaxation (at 1% of its mean);
    otherwise undetermined (a population still relaxing, e.g. the power-law decay at a continuous point, is never
    counted trivial).  The phase uses the q1 series alone; the run's status (which gates every Sigma decision) also
    needs Sigma finite and stationary."""
    dev = np.asarray(dev)
    k = max(1, len(dev) // 5)
    if dev[-1] < TRIVIAL:
        return "trivial", ""
    if dev[-k:].mean() <= DECADE * dev[:k].mean():
        return "trivial", f"q1 - 1/q decays {dev[:k].mean():.3g} -> {dev[-k:].mean():.3g} over the measured sweeps"
    why = relaxation(dev, 0.01 * float(np.mean(dev)), "q1 - 1/q")
    return ("undetermined", why) if why else ("nontrivial", "")


@dataclass(frozen=True)
class RSB:
    q: int
    graph: str
    degree: float
    m: float
    psi: float
    phi: float
    sigma: float
    q1: float
    psi_annealed: float
    nontrivial: bool
    status: str               # "converged" | "undetermined"
    note: str
    sigma_series: tuple = ()
    q1_series: tuple = ()
    phase: str = "undetermined"   # "trivial" | "nontrivial" | "undetermined" (see _phase)


def _resample(w, rng):
    """Row-wise multinomial resampling of (B, N2) weights (rows of zeros keep their order)."""
    B, N = w.shape
    tot = w.sum(1, keepdims=True)
    cw = np.cumsum(w, 1) / np.where(tot > 0, tot, 1.0) + np.arange(B)[:, None]
    r = rng.random((B, N)) + np.arange(B)[:, None]
    idx = np.searchsorted(cw.ravel(), r.ravel()).reshape(B, N) - np.arange(B)[:, None] * N
    return np.where(tot > 0, np.clip(idx, 0, N - 1), np.arange(N)[None, :])


def _er_new(P, k, rng):
    """Products of (1 - psi) for B new laws: each of the N2 slots draws one message from each of k laws."""
    N1, N2, q = P.shape
    B, km = len(k), max(int(k.max()), 1)
    out = rng.integers(N1, size=(B, km))
    inn = rng.integers(N2, size=(B, N2, km))
    return _prod(P[out[:, None, :], inn], k[:, None].repeat(N2, 1))


def _init(rng, shape, q, init):
    if init == "frozen":
        P = np.zeros(shape + (q,))
        np.put_along_axis(P, rng.integers(q, size=shape)[..., None], 1.0, -1)
        return P
    if isinstance(init, tuple) and len(init) == 2 and init[0] == "tilt" and 0 < float(init[1]) < 1:
        P = np.full(shape + (q,), (1 - float(init[1])) / q)
        np.put_along_axis(P, rng.integers(q, size=shape)[..., None], (1 - float(init[1])) / q + float(init[1]), -1)
        return P
    raise Refusal("precondition:init", f"init = {init!r}: 'frozen' or ('tilt', eps) with 0 < eps < 1")


def coloring_1rsb(q, d_or_c, m, *, graph="regular", pop=None, sweeps=200, measure=100, seed=0, init="frozen",
                  tol=1e-3):
    q, m, deg = _q(q), _m(m), _degree(graph, d_or_c)
    sweeps, measure = _int("sweeps", sweeps), _int("measure", measure, 2)
    if measure > sweeps:
        raise Refusal("precondition:population", f"measure = {measure} > sweeps = {sweeps}")
    rng = np.random.default_rng([int(seed), 71])
    if graph == "regular":
        N = _int("pop", 20000 if pop is None else pop, 10)
        P = _init(rng, (N,), q, init)
    else:
        N1, N2 = (300, 300) if pop is None else ((pop, pop) if np.isscalar(pop) else pop)
        N1, N2 = _int("pop", N1, 10), _int("pop", N2, 10)
        P = _init(rng, (N1, N2), q, init)
    meas, bad = [], 0
    for t in range(sweeps):
        if graph == "regular":
            u = _prod(P[rng.integers(N, size=(N, deg - 1))], np.full(N, deg - 1))
            Z = u.sum(-1)
            w = _wt(Z, m)
            if w.sum() <= 0:
                return _undetermined(q, graph, deg, m, "every new message is a contradiction (Z = 0)")
            sel = rng.choice(N, size=N, p=w / w.sum())
            P = _permute(u[sel] / Z[sel, None], rng)
        else:
            new = np.empty_like(P)
            B = max(1, min(N1, 2_000_000 // (N2 * q * max(1, int(3 * deg)))))
            for s0 in range(0, N1, B):
                b = min(B, N1 - s0)
                u = _er_new(P, rng.poisson(deg, b), rng)
                Z = u.sum(-1)
                w = _wt(Z, m)
                dead = w.sum(1) <= 0
                bad += int(dead.sum())
                idx = _resample(np.where(dead[:, None], 1.0, w), rng)
                Zs = np.take_along_axis(Z, idx, 1)
                v = np.take_along_axis(u, idx[..., None], 1) / np.where(Zs > 0, Zs, 1.0)[..., None]
                new[s0:s0 + b] = np.where(dead[:, None, None], P[rng.integers(N1, size=b)], _permute(v, rng))
            P = new
        if t >= sweeps - measure:
            meas.append(_measure(P, q, graph, deg, m, rng))
    a = np.array(meas)
    psi, phi, sig, q1 = (float(x) for x in a.mean(0))
    psi_ann = math.log(q) + deg / 2 * math.log1p(-1.0 / q)
    phase, why = _phase(a[:, 3] - 1.0 / q, a[:, 2], tol)
    sig_why = "" if phase == "trivial" else relaxation(a[:, 2], tol, "Sigma")
    note = "; ".join(x for x in (why, sig_why, f"{bad} new laws with every message contradictory (kept an old law)"
                                 if bad else "") if x)
    return RSB(q, graph, deg, m, psi, phi, sig, q1, psi_ann, phase == "nontrivial",
               "undetermined" if phase == "undetermined" or sig_why else "converged", note, tuple(a[:, 2]),
               tuple(a[:, 3]), phase)


def _undetermined(q, graph, deg, m, why):
    nan = float("nan")
    return RSB(q, graph, deg, m, nan, nan, nan, nan, nan, False, "undetermined", why)


def _terms(Z, m):
    """(E ln <Z^m>, E <Z^m ln Z>/<Z^m>) over the last axis = the law of one edge / node."""
    w = _wt(Z, m)
    s = w.sum(-1)
    ok = s > 0
    lw = np.log(np.where(ok, s / Z.shape[-1], 1.0))
    lw = np.where(ok, lw, -np.inf)
    return float(np.mean(lw)), float(np.mean(np.where(ok, (w * _lz(Z)).sum(-1) / np.where(ok, s, 1.0), 0.0)))


def _measure(P, q, graph, deg, m, rng):
    if graph == "regular":
        N = len(P)
        Zi = _prod(P[rng.integers(N, size=(N, deg))], np.full(N, deg)).sum(-1)[None, :]
        a, b = P[rng.integers(N, size=N)], P[rng.integers(N, size=N)]
        Zij = (1.0 - (a * b).sum(-1))[None, :]
    else:
        N1, N2, _ = P.shape
        k = rng.poisson(deg, N1)
        B = max(1, min(N1, 2_000_000 // (N2 * q * max(1, int(3 * deg)))))
        Zi = np.concatenate([_er_new(P, k[s:s + B], rng).sum(-1) for s in range(0, N1, B)])
        a = P[rng.integers(N1, size=N1)][:, rng.permutation(N2)]
        Zij = 1.0 - (a * P[rng.integers(N1, size=N1)]).sum(-1)
    pi, fi = _terms(Zi, m)
    pe, fe = _terms(Zij, m)
    psi, phi = pi - deg / 2 * pe, fi - deg / 2 * fe
    return psi, phi, psi - m * phi, float(np.mean((P ** 2).sum(-1)))


@dataclass(frozen=True)
class SP:
    q: int
    graph: str
    degree: float
    eta: float                # E eta (per color)
    sigma: float
    nontrivial: bool
    status: str
    note: str
    sigma_series: tuple = ()


def _sp_sums(prods, q):
    """prods (n, q): prod_k (1 - l eta_k) for l = 1..q -> (numerator, denominator) of the SP update."""
    l = np.arange(1, q + 1)
    sg = (-1.0) ** (l - 1)
    num = prods @ (sg * np.array([comb(q - 1, j - 1) for j in l]))
    den = prods @ (sg * np.array([comb(q, j) for j in l]))
    return num, den


def _sp_prods(eta, k, rng, q):
    n, km = len(k), max(int(k.max()), 1)
    x = eta[rng.integers(len(eta), size=(n, km))]                          # (n, km)
    f = 1.0 - x[..., None] * np.arange(1, q + 1)                                             # (n, km, q)
    f = np.where((np.arange(km)[None, :] < k[:, None])[..., None], f, 1.0)
    return np.prod(f, axis=1)


def frozen_sp(q, d_or_c, *, graph="er", pop=20000, sweeps=200, measure=100, seed=0, tol=1e-4):
    q, deg = _q(q), _degree(graph, d_or_c)
    pop, sweeps, measure = _int("pop", pop, 10), _int("sweeps", sweeps), _int("measure", measure, 2)
    if measure > sweeps:
        raise Refusal("precondition:population", f"measure = {measure} > sweeps = {sweeps}")
    rng = np.random.default_rng([int(seed), 73])
    eta = np.full(pop, (1 - 1e-3) / q)
    sig, etas = [], []
    for t in range(sweeps):
        num, den = _sp_sums(_sp_prods(eta, _children(rng, graph, deg, pop, False), rng, q), q)
        eta = np.clip(np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0), 0.0, 1.0 / q)
        if t >= sweeps - measure:
            _, zi = _sp_sums(_sp_prods(eta, _children(rng, graph, deg, pop, True), rng, q), q)
            zij = 1.0 - q * eta[rng.integers(pop, size=pop)] * eta[rng.integers(pop, size=pop)]
            with np.errstate(divide="ignore"):
                sig.append(float(np.mean(np.log(zi)) - deg / 2 * np.mean(np.log(zij))))
            etas.append(float(eta.mean()))
    note = relaxation(sig, tol, "Sigma") or relaxation(etas, tol, "E eta")
    e = float(np.mean(etas))
    return SP(q, graph, deg, e, float(np.mean(sig)), e > 1e-6, "undetermined" if note else "converged", note,
              tuple(sig))


__all__ = ["DECADE", "TRIVIAL", "RSB", "SP", "coloring_1rsb", "frozen_sp"]
