"""Family (b): the negative-margin spherical perceptron at T = 0, K-step RSB and the full-RSB limit.

Continues perceptron_rsb.py (read its docstring first: the 1RSB jamming functional, the
scaling 1 - q_K = delta -> 0, m_i = mu_i delta, the variational form alpha_c = min A).  K-step
RSB has overlaps q_0 < q_1 < ... < q_{K-1} < q_K = 1 and Parisi parameters
m_0 < ... < m_{K-1} < m_K = 1; in the jamming scaling all m_i = mu_i delta with finite
0 < mu_0 < ... < mu_{K-1}.  Writing g for the gap of one constraint, the nested Gaussian
averages of FPSUZ Appendix B, Eq. (B2) (arXiv:1702.06919), become in this scaling
  F_K(g) = -g_+^2 / 2,    F_i(a) = (1/mu_i) ln E_u exp( mu_i F_{i+1}(a - sqrt(q_{i+1} - q_i) u) ),
  (a = kappa - sqrt(q_0) t at the outermost level),
and the entropy is s = (1/2)(1 + c) ln delta + Phi_K / delta + O(1) with
  Phi_K = q_0 / (2 Lam_0) + (1/2) sum_{i=0}^{K-1} ln(Lam_i / Lam_{i+1}) / mu_i + alpha E_t F_0(kappa - sqrt(q_0) t),
  Lam_K = 1,   Lam_i = Lam_{i+1} + mu_i (q_{i+1} - q_i)
(the log-determinant of FPSUZ Eq. (30) with lambda(q) = delta Lam; K = 1 is perceptron_rsb.py).
Phi_K is linear in alpha, and the K-RSB capacity is the stationary value of
A_K(mu, q) = -ent/en over the 2K parameters, a local MINIMUM found here by BFGS (the K-RSB families
are nested -- a repeated parameter reproduces the (K-1) point -- so the minima alpha_c^{K} do not
increase with K; each trial point bounds the capacity from above only if A_K is Stojnic's lifted
functional, which the agreement of K = 1, 2 with his tables supports but does not prove).  The full-RSB capacity is the K -> infinity
limit, estimated from the staircase K = 1, 2, 3, ... by geometric (Aitken) extrapolation with
the last difference as the stated band; the continuum Parisi PDE of FPSUZ Eqs. (27)-(29) is
NOT solved (a documented limit: no scaling solution of Sec. V, no kappa exponent).

Contact index (the replicon of the innermost level): z = Prob(g > 0) in the selected state is
the response of E_t F_0 to a step -eps theta(g) added to the last energy, z = -d E_t F_0 / d eps
(the posterior average of the derivative at every level: FPSUZ Eq. (B5) in its stable
backward form), so that c_K = alpha z and Lambda = 1 - c_K.  c_K -> 1 as K grows is the
marginality of full RSB (isostaticity at jamming, FPSUZ Sec. V, Eqs. (100)-(106); Franz & Parisi
2016): c_K is what labels a K-RSB value: `unstable` for c_K > 1 (every K and kappa computed here), `stable` only if
a computed c_K <= 1 (replicon sector only), and the full-RSB estimate is always `not-computed`.

Published values read: Stojnic, arXiv:2312.16531, Tables 2, 5, 6 (r-sfl RDT): 2-sfl = K = 1 here,
3-sfl = K = 2 (12.29 at kappa = -1, 36.40 at -1.5, 4.698 at -0.5, 124.8 at -2); his p2-hat/q2-hat
parameters correspond to q_0/q_1 (e.g. q_1 = 0.9691 at kappa = -1 here, his p2 = 0.9691).

Numerics: float64 (scipy log_ndtr, cubic-spline interpolation of F_i on a grid in a, trapezoid
Gaussian weights); the numerical error is the change under a refined grid (also returned).
Slow tier only; physics-level.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.special import expit, log_ndtr

from ..core import Refusal
from .common import CAVEAT
from .perceptron_rsb import _KAPPA_MIN, _KAPPA_NEAR_ZERO, _ln_y_f64, onersb_jamming

_K_MAX = 5


@dataclass(frozen=True)
class KRSBJamming:
    kappa: float
    levels: int
    alpha_c: float                   # the K-step RSB capacity (an upper bound on the next level)
    mus: tuple
    qs: tuple
    contact_index: float             # c_K = alpha P(g > 0); Lambda = 1 - c_K
    replicon: float
    grid_error: float                # |alpha_c(refined grid) - alpha_c|
    gradient_norm: float             # of A_K in the unconstrained parameters (convergence audit)
    degenerate: bool                 # two levels merged: the point is a lower level embedded
    caveat: str = CAVEAT


def _weights(n, lim):
    t = np.linspace(-lim, lim, n)
    return t, np.exp(-t**2 / 2) / math.sqrt(2 * math.pi) * (t[1] - t[0])


class _Tree:
    """F_i on a grid for given (mus, qs); evaluates ent, en and the contact index."""

    def __init__(self, mus, qs, kappa, h=None, nu=801):
        self.K, self.kappa = len(mus), kappa
        self.mus, self.q = list(mus), list(qs) + [1.0]
        b_top = math.sqrt(1 - self.q[-2])
        self.h = h or _grid_h(1 - b_top**2)
        self.L = 24.0
        n = int(2 * self.L / self.h) | 1
        self.grid = np.linspace(kappa - self.L, kappa + self.L, n)
        self.u, self.wu = _weights(nu, 9.0)
        self.F = [None] * (self.K + 1)           # F[i] on the grid for i = 0..K-1 (F[K] closed)
        self._build()

    def _fclosed(self, a):
        K = self.K
        return _ln_y_f64(a, self.mus[K - 1], 1 - self.q[K - 1]) / self.mus[K - 1]

    def _spline_eval(self, i, a):
        return CubicSpline(self.grid, self.F[i + 1])(a)

    def _build(self):
        K = self.K
        self.F[K - 1] = self._fclosed(self.grid)
        for i in range(K - 2, -1, -1):
            s = math.sqrt(self.q[i + 1] - self.q[i])
            cs = CubicSpline(self.grid, self.F[i + 1])
            a = np.clip(self.grid[:, None] - s * self.u[None, :], self.grid[0], self.grid[-1])
            vals = self.mus[i] * cs(a)
            m = vals.max(axis=1, keepdims=True)
            self.F[i] = (m[:, 0] + np.log((np.exp(vals - m) * self.wu[None, :]).sum(axis=1))) / self.mus[i]

    def ent_en(self):
        K, mus, q = self.K, self.mus, self.q
        lam = [0.0] * (K + 1)
        lam[K] = 1.0
        for i in range(K - 1, -1, -1):
            lam[i] = lam[i + 1] + mus[i] * (q[i + 1] - q[i])
        ent = q[0] / (2 * lam[0]) + 0.5 * sum(math.log(lam[i] / lam[i + 1]) / mus[i] for i in range(K))
        t, wt = _weights(2401, 12.0)
        a = self.kappa - math.sqrt(q[0]) * t
        f0 = CubicSpline(self.grid, self.F[0])(a) if K > 1 else self._fclosed(a)
        return ent, float(np.sum(wt * f0))

    def contact(self, alpha):
        """c_K = alpha z, z = -d E_t F_0 / d eps (see the module docstring)."""
        K, mus, q = self.K, self.mus, self.q
        b2 = 1 - q[K - 1]
        s2 = 1 + mus[K - 1] * b2
        a = self.grid
        l_sat = log_ndtr(-a / math.sqrt(b2))
        l_con = (-mus[K - 1] * a * a / (2 * s2) - 0.5 * math.log(s2)
                 + log_ndtr(a / (math.sqrt(b2) * math.sqrt(s2))))
        D = -np.exp(l_con - np.logaddexp(l_sat, l_con))         # dF_{K-1}/d eps
        for i in range(K - 2, -1, -1):
            s = math.sqrt(q[i + 1] - q[i])
            A = np.clip(self.grid[:, None] - s * self.u[None, :], self.grid[0], self.grid[-1])
            vals = mus[i] * CubicSpline(self.grid, self.F[i + 1])(A)
            w = np.exp(vals - vals.max(axis=1, keepdims=True)) * self.wu[None, :]
            D = (w * CubicSpline(self.grid, D)(A)).sum(axis=1) / w.sum(axis=1)
        t, wt = _weights(2401, 12.0)
        d0 = CubicSpline(self.grid, D)(self.kappa - math.sqrt(q[0]) * t)
        return float(-alpha * np.sum(wt * d0))


def _unpack(v, K):
    mus = np.cumsum(np.exp(v[:K]))
    qs, prev = [], 0.0
    for i in range(K):
        prev = prev + (1 - prev) * expit(v[K + i])
        qs.append(prev)
    return list(mus), qs


def _pack(mus, qs):
    v = [math.log(mus[0])] + [math.log(mus[i] - mus[i - 1]) for i in range(1, len(mus))]
    w, prev = [], 0.0
    for q in qs:
        r = (q - prev) / (1 - prev)
        w.append(math.log(r / (1 - r)))
        prev = q
    return np.array(v + w)


def _grid_h(q_top):
    """Grid spacing for a top overlap q_{K-1}: resolves the last level's scale sqrt(1 - q)/8 (>= 0.0025)."""
    return min(0.02, max(math.sqrt(1 - q_top) / 8, 0.0025))


def _alpha_of(v, K, kappa, h, nu=601):
    """A_K at the unconstrained parameters v on a grid of FIXED spacing h (a spacing that followed the parameters
    would make the objective jump when the grid size changes)."""
    mus, qs = _unpack(v, K)
    if qs[-1] > 0.9996 or mus[-1] > 1e6:
        return 1e30
    ent, en = _Tree(mus, qs, kappa, h=h, nu=nu).ent_en()
    return -ent / en if en < 0 else 1e30


def _starts(prev, K):
    """Candidate K-level starts from the (K-1)-level optimum: split one level, or a new top level."""
    mus, qs = prev
    out = []
    for f_mu, f_q in ((8.0, 0.1), (4.0, 0.2), (15.0, 0.05)):
        out.append((mus[:-1] + [mus[-1] * 1.4, mus[-1] * f_mu], qs[:-1] + [qs[-1], 1 - (1 - qs[-1]) * f_q]))
    # a new bottom level
    out.append(([mus[0] * 0.5] + mus, [qs[0] * 0.5] + qs))
    return out


def krsb_jamming(kappa, K, _prev=None) -> KRSBJamming:
    """K-step RSB jamming point (K <= 5).  K = 1 is perceptron_rsb.onersb_jamming (same value to
    ~1e-9); K >= 2 is minimised by BFGS from splits of the (K-1)-level optimum.  Refusals:
    precondition:kappa-range (outside [-3, -0.02]), precondition:levels (K not in 1..5),
    no-solution:krsb-degenerate (every start merges back to the (K-1)-level point)."""
    kappa = float(kappa)
    if not (_KAPPA_MIN <= kappa <= _KAPPA_NEAR_ZERO):
        raise Refusal("precondition:kappa-range", f"kappa = {kappa}: the K-RSB search is validated "
                      f"on [{_KAPPA_MIN}, {_KAPPA_NEAR_ZERO}]")
    if not (isinstance(K, int) and 1 <= K <= _K_MAX):
        raise Refusal("precondition:levels", f"levels = {K!r}: use an integer 1..{_K_MAX}")
    if K == 1:
        r = onersb_jamming(kappa)
        start = [([float(r.mu)], [float(r.q0)])]
    else:
        prev = _prev or krsb_jamming(kappa, K - 1)
        start = _starts((list(prev.mus), list(prev.qs)), K)
    best = None
    for mus0, qs0 in start:                       # stage 1: coarse grid (alpha is stationary at
        v0 = _pack(mus0, qs0)                     # the minimum, so the shift of the coarse optimum
        h0 = max(min(0.04, math.sqrt(1 - qs0[-1]) / 4), 0.005)   # costs only its square on the fine grid)
        res = minimize(_alpha_of, v0, args=(K, kappa, h0, 301), method="BFGS",
                       options=dict(gtol=1e-6, maxiter=150, eps=1e-6))
        if best is None or res.fun < best.fun:
            best = res
    h1 = _grid_h(_unpack(best.x, K)[1][-1])                                   # stage 2: fine grid, h fixed
    res = minimize(_alpha_of, best.x, args=(K, kappa, h1), method="BFGS",
                   options=dict(gtol=1e-7, maxiter=40, eps=1e-6))
    best = res if res.fun <= _alpha_of(best.x, K, kappa, h1) else best
    mus, qs = _unpack(best.x, K)
    a_ref = _alpha_of(best.x, K, kappa, 0.5 * h1, nu=1201)
    tree = _Tree(mus, qs, kappa, h=h1)
    c = tree.contact(best.fun)
    degenerate = (any(b - a < 1e-3 * max(a, 1e-30) for a, b in zip(mus, mus[1:]))
                  or any(b - a < 1e-4 for a, b in zip(qs, qs[1:])))
    if K >= 2 and _prev is None and degenerate:
        raise Refusal("no-solution:krsb-degenerate",
                      f"K = {K}, kappa = {kappa}: the minimiser merges two levels (the K-1 point)")
    return KRSBJamming(kappa, K, float(best.fun), tuple(mus), tuple(qs), c, 1 - c,
                       abs(a_ref - best.fun), float(np.linalg.norm(best.jac)), degenerate)


def aitken_tail(alphas):
    """(alpha_inf, last difference) for a decreasing staircase alpha_1 > alpha_2 > ... (>= 3 terms):
    the differences are assumed geometric, d_{k+1} = r d_k, so the remaining decrease after the last
    term is d_last r/(1 - r).  r >= 1 or r <= 0 (not geometric) gives alpha_inf = nan."""
    a = [float(x) for x in alphas]
    if len(a) < 3:
        raise Refusal("precondition:levels", "the geometric extrapolation needs a staircase of >= 3 levels")
    d1, d2 = a[-3] - a[-2], a[-2] - a[-1]
    r = d2 / d1 if d1 > 0 else float("nan")
    tail = d2 * r / (1 - r) if 0 < r < 1 else float("nan")
    return a[-1] - tail, d2


def frsb_estimate(kappa, K=3):
    """(alpha_inf, band, ladder): the staircase K = 1..K, extrapolated geometrically (aitken_tail),
    band = the last difference.  alpha_inf lies below the K-level value; a physics-level estimate of
    the full-RSB capacity (not the continuum Parisi solution).  The geometric ratio rests on the last
    two differences only (K = 3: one ratio): see GUIDE for what a drifting ratio would change."""
    ladder, prev = [], None
    for k in range(1, K + 1):
        prev = krsb_jamming(kappa, k, _prev=prev)
        if k >= 2 and prev.degenerate:
            raise Refusal("no-solution:krsb-degenerate", f"kappa = {kappa}: the {k}-level search merged two "
                          "levels (the staircase would repeat the previous rung)")
        ladder.append(prev)
    alpha_inf, band = aitken_tail([x.alpha_c for x in ladder])
    if not (math.isfinite(alpha_inf) and band > 0):
        raise Refusal("undetermined:frsb-extrapolation", f"kappa = {kappa}: the staircase "
                      f"{[round(x.alpha_c, 8) for x in ladder]} is not geometrically decreasing; the K-step "
                      "values stand, the full-RSB limit is not extrapolated")
    return alpha_inf, band, ladder


def ladder_table(kappas, K=3):
    """Rows (kappa, RS, alpha_1..alpha_K, c_1..c_K, FRSB estimate, band) for `python3 -m
    replicant.families.perceptron_krsb` (several minutes per kappa at K = 3)."""
    from .perceptron_rsb import rs_capacity
    rows = []
    for k in kappas:
        est, band, ladder = frsb_estimate(float(k), K)
        rows.append(dict(kappa=float(k), rs=float(rs_capacity(k)), alphas=[x.alpha_c for x in ladder],
                         contact=[x.contact_index for x in ladder], frsb=est, band=band))
    return rows


if __name__ == "__main__":
    import sys
    ks = [float(x) for x in sys.argv[1:]] or [-0.55, -0.85, -1.12, -1.36, -1.66]
    for r in ladder_table(ks):
        print(f"kappa {r['kappa']:+.2f}  RS {r['rs']:.6g}  " + "  ".join(
            f"K={i + 1}: {a:.6g} (c {c:.4f})" for i, (a, c) in enumerate(zip(r['alphas'], r['contact'])))
            + f"  FRSB ~ {r['frsb']:.6g} +- {r['band']:.2g}")
