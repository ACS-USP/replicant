"""Curie-Weiss at finite N (exact, certified) and the large-deviation rate functions (KZ Sec. 1.1-1.2).

Convention as meanfield/curie_weiss.py (KZ Eq. (1.1)): Sbar takes the N + 1 values m_k = (2k - N)/N and
    Z_N(beta, h) = sum_k C(N, k) exp(beta N (m_k^2/2 + h m_k))        (KZ Sec. 1.1, the binomial count)
evaluated in arb (every term positive: no cancellation), with
    phi_N = ln Z_N / N, <Sbar>, <|Sbar|>, e_N = <H_N>/N = -<Sbar^2>/2 - h <Sbar>, chi_N = N Var(Sbar)
    (Eq. (1.11): (1/beta) d^2 Phi_N / dh^2 = N beta Var(Sbar)),
    the finite-N rate  I_N(m_k) = -(1/N) ln P_N(Sbar = m_k)              (Lemma 1, Eq. (1.4)).
Lemma 2 (Eq. (1.6)) bounds phi_N - Phi to [-ln(N(N+1))/N, ln(N+1)/N]; cw_exact refuses no-solution:audit when
the certified phi_N leaves that window (not audited at beta = 1, h = 0, where the roots are not certified).
Away from a transition the Laplace sum gives (derived here, not in KZ)
    ln Z_N = N Phi - (1/2) ln(1 - beta (1 - m*^2)) + ln(number of global maxima) + o(1),
so N (phi_N - Phi) converges to finite_size_term(beta, h) (cw_finite_size_term).
Rate functions of Sbar at field h (the true rate is I*(m) = Phi - phi(m), Eq. (1.8), non-convex for beta > 1):
  "legendre" (Sec. 1.2.2): I(m) = sup_h' [beta (h' - h) m - Phi(beta, h') + Phi(beta, h)];
  "gartner_ellis" (Sec. 1.2.3, Theorem 3): I(m) = sup_k [k m - lambda(k)], lambda(k) = lim (1/N) ln
      <e^{N k Sbar}> = sup_m' [phi(m') + k m'] - sup_m' phi(m') (Varadhan on Theorem 1).
Both are the convex envelope of I*: equal to it at the exposed points, below it on (-m*, m*) when h = 0 and
beta > 1.  Floats (scipy bounded Brent on a concave objective; Phi by a grid-and-refine maximisation of phi).

  cw_exact_z(N, beta, h) -> Ball;  cw_exact(N, beta, h) -> CWExact;  cw_exact_rate(N, m, beta, h) -> Ball
  cw_finite_size_term(beta, h) -> mpf;  cw_rate_function(m, beta, h, route) -> float;  cw_ldp_rate -> float
"""
import math
from dataclasses import dataclass
from fractions import Fraction

import mpmath as mp
import numpy as np
from scipy.optimize import minimize_scalar

from ..core import Refusal
from ..lattice.ball import Ball, arb_q, ball_of, baller, rational
from .curie_weiss import cw_solution

ROUTES = ("legendre", "gartner_ellis")
_MAX_N = 20000


def _n(N):
    if isinstance(N, bool) or not isinstance(N, int) or not 1 <= N <= _MAX_N:
        raise Refusal("precondition:size", f"N = {N!r}: need an int in 1 .. {_MAX_N}")
    return N


@dataclass(frozen=True)
class CWExact:
    N: int
    z: Ball
    phi: Ball             # ln Z_N / N
    magnetization: Ball   # <Sbar>
    abs_magnetization: Ball
    energy: Ball          # <H_N>/N
    chi: Ball             # N Var(Sbar)


def _sums(N, bq, hq):
    from flint import arb
    b, h = arb_q(bq), arb_q(hq)
    z = s1 = s2 = sa = arb(0)
    for k in range(N + 1):
        m = Fraction(2 * k - N, N)
        w = arb(math.comb(N, k)) * (b * N * arb_q(m * m / 2 + hq * m)).exp()
        mm = arb_q(m)
        z, s1, s2, sa = z + w, s1 + w * mm, s2 + w * mm * mm, sa + w * abs(mm)
    return z, s1, s2, sa, h


def _prec(N):
    return 128 + 2 * N.bit_length()


def cw_exact_z(N, beta, h=0) -> Ball:
    N, bq, hq = _n(N), rational(beta, "beta"), rational(h, "h", positive=False)
    with baller().hygiene.ctx_guard(prec=_prec(N)):
        return ball_of(_sums(N, bq, hq)[0])


def cw_exact(N, beta, h=0) -> CWExact:
    N, bq, hq = _n(N), rational(beta, "beta"), rational(h, "h", positive=False)
    with baller().hygiene.ctx_guard(prec=_prec(N)):
        z, s1, s2, sa, hb = _sums(N, bq, hq)
        m1, m2 = s1 / z, s2 / z
        out = CWExact(N, *(ball_of(x) for x in (z, z.log() / N, m1, sa / z, -m2 / 2 - hb * m1,
                                                 N * (m2 - m1 * m1))))
    try:
        phi = cw_solution(beta, h).phi
    except Refusal as r:                     # beta = 1, h = 0: Phi is not certified there; no window to audit
        if r.reason != "undetermined:spinodal":
            raise
        return out
    with mp.workdps(40):
        lo = phi.mid() - phi.rad() - mp.log(N * (N + 1)) / N
        hi = phi.mid() + phi.rad() + mp.log(N + 1) / N
        if not (lo <= out.phi.mid() - out.phi.rad() and out.phi.mid() + out.phi.rad() <= hi):
            raise Refusal("no-solution:audit", f"phi_N = {out.phi} outside Lemma 2's window [{mp.nstr(lo, 12)}, "
                          f"{mp.nstr(hi, 12)}] at N = {N}")
    return out


def cw_exact_rate(N, m, beta, h=0) -> Ball:
    """I_N(m) = -(1/N) ln P_N(Sbar = m), m on the grid (2k - N)/N."""
    N, bq, hq = _n(N), rational(beta, "beta"), rational(h, "h", positive=False)
    mq = rational(m, "magnetization", positive=False)
    k = (mq * N + N) / 2
    if k.denominator != 1 or not 0 <= k <= N:
        raise Refusal("precondition:magnetization-grid", f"m = {m}: at N = {N} Sbar takes the values (2k - N)/N")
    from flint import arb
    with baller().hygiene.ctx_guard(prec=_prec(N)):
        z = _sums(N, bq, hq)[0]
        w = arb(math.comb(N, int(k))) * (arb_q(bq) * N * arb_q(mq * mq / 2 + hq * mq)).exp()
        return ball_of(-(w / z).log() / N)


def cw_finite_size_term(beta, h=0):
    """lim N (phi_N - Phi) = -(1/2) ln(1 - beta (1 - m*^2)) + ln(#global maxima) (Laplace; derived here)."""
    try:
        sol = cw_solution(beta, h)
    except Refusal as r:
        if r.reason != "undetermined:spinodal":
            raise
        raise Refusal("precondition:off-critical", f"beta = {beta}, h = {h}: {r.detail}") from None
    with mp.workdps(30):
        m, b = sol.fixed_points[-1].m.mid(), mp.mpf(rational(beta, "beta").numerator) / rational(beta).denominator
        curv = 1 - b * (1 - m * m)
        if not curv > mp.mpf("1e-6"):
            raise Refusal("precondition:off-critical", f"beta = {beta}, h = {h}: at (or too near) the critical "
                          "point the Laplace sum is not Gaussian (N^(-1/4) scaling)")
        two = sol.selected is None
        return -mp.log(curv) / 2 + (mp.log(2) if two else 0)


# ---- rate functions -------------------------------------------------------------------------------------
def _phi(m, beta, h):
    p, q = (1 + m) / 2, (1 - m) / 2
    return -(p * np.log(p) + q * np.log(q)) + beta * m * m / 2 + beta * h * m


def _sup_potential(beta, h):
    """max_m phi(m; beta, h) on (-1, 1): a grid, then a bounded refine around the best node (Theorem 1)."""
    grid = np.linspace(-1 + 1e-12, 1 - 1e-12, 4001)
    i = int(np.argmax(_phi(grid, beta, h)))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    r = minimize_scalar(lambda m: -_phi(m, beta, h), bounds=(lo, hi), method="bounded",
                        options={"xatol": 1e-13})
    return max(-r.fun, float(_phi(grid[i], beta, h)))


def _concave_sup(fn, lo, hi):
    r = minimize_scalar(lambda x: -fn(x), bounds=(lo, hi), method="bounded", options={"xatol": 1e-11})
    return -r.fun


def cw_rate_function(m, beta, h=0, route="legendre"):
    """The convex rate of Sbar at field h by the Legendre route or the Gartner-Ellis route (floats)."""
    if route not in ROUTES:
        raise Refusal("unsupported:route", f"route = {route!r}: one of {', '.join(ROUTES)}")
    m, b, hh = float(m), float(rational(beta, "beta")), float(rational(h, "h", positive=False))
    if not -1 < m < 1:
        raise Refusal("precondition:magnetization", f"m = {m}: need -1 < m < 1")
    base = _sup_potential(b, hh)
    if route == "legendre":
        return max(0.0, _concave_sup(lambda hp: b * (hp - hh) * m - _sup_potential(b, hp) + base, hh - 12, hh + 12))
    lam = lambda k: _sup_potential(b, hh + k / b) - base
    return max(0.0, _concave_sup(lambda k: k * m - lam(k), -12 * b, 12 * b))


def cw_ldp_rate(m, beta, h=0):
    """The true rate I*(m) = Phi - phi(m) (Theorem 1, Eq. (1.8)), Phi certified, phi(m) in floats."""
    phi = float(cw_solution(beta, h).phi.mid())
    return phi - float(_phi(float(m), float(rational(beta, "beta")), float(rational(h, "h", positive=False))))


__all__ = ["CWExact", "ROUTES", "cw_exact", "cw_exact_rate", "cw_exact_z", "cw_finite_size_term", "cw_ldp_rate",
           "cw_rate_function"]
