"""Family (b): Random Energy Model (Derrida, Phys. Rep. 67, 29 (1980); PRB 24, 2613 (1981)).

Normalisation.  2^N i.i.d. energies E_i ~ N(0, N_scale * N): N_scale is the
energy variance per spin.  N_scale = 1 (the default) gives
beta_c = sqrt(2 ln 2).  Derrida writes P(E) ~ exp(-E^2/(N J^2)), i.e.
N_scale = J^2/2; at J = 1 (N_scale = 1/2) beta_c = 2 sqrt(ln 2),
T_c = 1/(2 sqrt(ln 2)), and that is also the energy scale of the spherical
p-spin with couplings of variance p!/(2 N^(p-1)) (E H^2 = N/2), so the p-spin
module compares against N_scale = 1/2.

  rem_critical_beta(N_scale=1)            -> sqrt(2 ln 2 / N_scale)
  rem_entropy(e, N_scale=1)               -> s(e) = ln 2 - e^2/(2 N_scale)   (e = E/N)
  rem_free_energy(beta, N_scale=1)        -> f per spin, N -> infinity
  rem_1rsb_functional()                   -> sympy -beta f(m) (symbols m, beta, s)
  rem_1rsb(beta, N_scale=1)               -> RSBSolution(q0=0, q1=1, m=beta_c/beta, f, e)
  rem_quenched_finite_n(beta, N, N_scale) -> (1/N) E ln Z at finite N (float, quadrature)
  rem_participation_ratio(beta, N_scale=1, N=None) -> E[Y], Y = sum_i w_i^2 (w the Gibbs weights)
  rem_y2(beta, N_scale=1, N=None)         -> E[Y^2]
  rem_overlap_distribution(beta, N_scale=1) -> {1: P(q = 1), 0: P(q = 0)}

Results.  Microcanonically s(e) vanishes at e0 = -sqrt(2 N_scale ln 2) and
beta_c = s'(e0).  Above T_c the annealed and quenched free energies agree:
-beta f = ln 2 + N_scale beta^2 / 2.  Below, the system is frozen at e0:
-beta f = beta sqrt(2 N_scale ln 2) (f = e0, entropy 0).  f and e are
continuous at beta_c; the specific heat jumps.

1RSB route (n -> 0).  E Z^n with the n replicas in n/m groups of m that share
a configuration (overlap q1 = 1 inside a group, q0 = 0 between): the choice of
configurations gives (2^N)^(n/m), and E exp(-beta sum_a E_a) gives
exp(N_scale N beta^2 sum_{a,b} q_ab^p / 2), with sum_{a,b} q_ab^p = n + the
off-diagonal ansatz.sum_offdiag_power sum (any p > 0, since q in {0, 1}).
Per replica and spin, via ansatz.n_to_zero:
    -beta f(m) = ln 2 / m + N_scale beta^2 m / 2,
stationary at m* = sqrt(2 ln 2 / N_scale) / beta = beta_c / beta (Parisi
breakpoint; Derrida & Mottishaw 2015, mu = T/T_c).  For beta < beta_c,
m* > 1 leaves 0 < m <= 1 and the answer is the paramagnet (m = 1).  The n -> 0
continuation and the ansatz are physics-level, not proven; the free energy
itself is a theorem here (Derrida's direct computation), which is what makes
the REM a check on the replica route.

Condensation (Krzakala & Zdeborova, lecture notes of June 9 2024, "KZ", Secs.
15.2.4-15.2.5, Ex. 15.2; KZ's P(E) = N(0, N/2), Eq. (15.1), is N_scale = 1/2).
With m = min(beta_c/beta, 1), the replica (1RSB) predictions, physics-level:
    E[Y] = 1 - m                    (Eq. (15.39); 0 for beta <= beta_c)
    E[Y^2] = (3 - 5m + 2m^2)/3      (Eq. (15.66), stated for beta >= beta_c; 0 at m = 1)
    E P(q) = (1 - m) delta(q - 1) + m delta(q)   (Eqs. (15.41)-(15.42)),
P(q = 1) = E[Y] (Eq. (15.40)): two Gibbs draws coincide.  KZ p. 230 says the
participation ratio "diverges" at T_c; Y -> 0 there, it is 1/Y (the quantity of
Fig. 15.2.1, "inverse participation ratio") that diverges.  At finite N (N int)
the exact moments come without replicas from 1/Z^k = int_0^oo t^(k-1) e^(-tZ)
dt/(k-1)! and the independence of the 2^N terms (a_i = e^(-beta E_i), x = t a):
    E[Y]   = int du M B2(u) g^(M-1),
    E[Y^2] = int du [M B4 g^(M-1) + M (M-1) B2^2 g^(M-2)] / 6,
B_k(u) = E[x^k e^-x], g = E e^-x, x = e^(u - beta E), M = 2^N (binary64
trapezoid on a grid of step 0.2 in u and in beta E: refining to 0.02 changes
the result by < 1e-12 at N = 12; checked against direct sampling at N = 2, 4,
12 during the build and at N = 14 in tests/test_rem_condensation.py).  _HOOKS["m"] (identity) is the planted-error hook of the condensation
predictions (tests plant m -> 1 - m); rem_1rsb does not use it.
"""
import math
from functools import lru_cache

import mpmath as mp
import numpy as np
import sympy as sp

from ..ansatz import n_to_zero, sum_offdiag_power
from ..core import Refusal
from .common import RSBSolution

_DPS = 30
_HOOKS = {"m": lambda m: m}


def _prec():   # never lower the caller's precision (mp.diff raises it)
    return mp.workdps(max(_DPS, mp.mp.dps))


def _s(N_scale):
    s = mp.mpf(N_scale)
    if not s > 0:
        raise Refusal("precondition:positive-N_scale",
                      f"N_scale = {N_scale} (energy variance per spin) must be > 0")
    return s


def _beta(beta):
    b = mp.mpf(beta)
    if not b > 0:
        raise Refusal("precondition:positive-beta", f"beta = {beta} must be > 0")
    return b


def rem_critical_beta(N_scale=1):
    """beta_c = sqrt(2 ln 2 / N_scale); sqrt(2 ln 2) at the default N_scale = 1."""
    with _prec():
        return mp.sqrt(2 * mp.log(2) / _s(N_scale))


def rem_entropy(e, N_scale=1):
    """Annealed entropy per spin at energy per spin e: ln 2 - e^2 / (2 N_scale)."""
    with _prec():
        return mp.log(2) - mp.mpf(e)**2 / (2 * _s(N_scale))


def rem_free_energy(beta, N_scale=1):
    """Quenched free energy per spin, N -> infinity (Derrida)."""
    with _prec():
        b, s = _beta(beta), _s(N_scale)
        bc = rem_critical_beta(s)
        if b <= bc:
            return -(mp.log(2) + s * b**2 / 2) / b
        return -mp.sqrt(2 * s * mp.log(2))


def rem_1rsb_functional():
    """-beta f(m) per spin from the replica blocks (symbols m, beta, s = N_scale)."""
    n, m, beta, s, p = sp.symbols("n m beta s p", positive=True)
    q0, q1 = sp.Integer(0), sp.Integer(1)
    # (1/n) sum_{a,b} q_ab^p: diagonal n plus the off-diagonal 1RSB sum
    pairs = 1 + n_to_zero(sum_offdiag_power("rsb", [m], [q0, q1], p, n), n)
    entropy = n_to_zero((n / m) * sp.log(2), n)        # ln (2^N)^(n/m), per n and N
    return sp.simplify(entropy + s * beta**2 * pairs / 2)


def rem_1rsb(beta, N_scale=1) -> RSBSolution:
    with _prec():
        b, s = _beta(beta), _s(N_scale)
        bc = rem_critical_beta(s)
        m = bc / b
        if m > 1:
            raise Refusal("no-solution", f"beta = {mp.nstr(b, 8)} < beta_c = {mp.nstr(bc, 8)}: "
                          f"stationary m = beta_c/beta = {mp.nstr(m, 8)} > 1 is outside "
                          "0 < m <= 1; this is the paramagnet (m = 1, RS), use rem_free_energy")
        phi = mp.log(2) / m + s * b**2 * m / 2
        # e = d(beta f)/d beta at fixed m (stationarity) = -N_scale beta m = e0
        return RSBSolution(mp.mpf(0), mp.mpf(1), m, -phi / b, -s * b * m)


def rem_quenched_finite_n(beta, N, N_scale=1, *, de=0.05, nu=1500):
    """(1/N) E ln Z for N finite, without replicas, float64 quadrature.

    E ln Z = int_0^inf dt/t [e^-t - E e^(-t Z)], and since the 2^N energies
    are i.i.d., E e^(-t Z) = g(t)^(2^N), g(t) = E exp(-t e^(-beta E)).  Using
    1 - g = h(t) = E[1 - exp(-t e^(-beta E))] keeps the precision:
    integrand (in u = ln t) = expm1(-t) - expm1(2^N log1p(-h)).
    Energy grid: trapezoid over |E| <= (sqrt(2 N ln 2) + 8) sd, step de/beta;
    u grid: nu points down to -(ln Z_max + 20).  Grid refinement (de 0.05 ->
    0.02, nu 1500 -> 4000) at beta = 2 beta_c changes the result by < 1e-12
    for N <= 80, 1.3e-10 at N = 160, 1.8e-7 at N = 320.  float64, not certified.
    """
    beta, s = float(_beta(beta)), float(_s(N_scale))
    if not (isinstance(N, int) and 1 <= N <= 1000):
        raise Refusal("precondition:integer-N", f"N = {N!r}: need an int in [1, 1000]")
    sd = math.sqrt(s * N)
    L = (math.sqrt(2 * N * math.log(2)) + 8) * sd
    E = np.linspace(-L, L, int(2 * L * beta / de) + 2)
    w = np.exp(-E**2 / (2 * sd * sd)) / (math.sqrt(2 * math.pi) * sd) * (E[1] - E[0])
    u = np.linspace(-(N * math.log(2) + beta * L) - 20, 6, nu)
    with np.errstate(over="ignore"):
        h = (-np.expm1(-np.exp(u[:, None] - beta * E[None, :]))) @ w
    integ = np.expm1(-np.exp(u)) - np.expm1(2.0**N * np.log1p(-h))
    return float(integ.sum() * (u[1] - u[0]) / N)


def _m(beta, N_scale):
    with _prec():
        b, s = _beta(beta), _s(N_scale)
        return _HOOKS["m"](min(rem_critical_beta(s) / b, mp.mpf(1)))


def rem_participation_ratio(beta, N_scale=1, N=None):
    """E[Y]: 1 - m (KZ Eq. (15.39)); with N an int, the exact finite-N moment (float)."""
    if N is not None:
        return _y_moments(beta, N, N_scale)[0]
    with _prec():
        return 1 - _m(beta, N_scale)


def rem_y2(beta, N_scale=1, N=None):
    """E[Y^2]: (3 - 5m + 2m^2)/3 (KZ Eq. (15.66)); with N an int, the exact finite-N moment (float)."""
    if N is not None:
        return _y_moments(beta, N, N_scale)[1]
    with _prec():
        m = _m(beta, N_scale)
        return (3 - 5 * m + 2 * m * m) / 3


def rem_overlap_distribution(beta, N_scale=1):
    """{1: 1 - m, 0: m} (KZ Eq. (15.42); {1: 0, 0: 1} above T_c, Eq. (15.41))."""
    with _prec():
        m = _m(beta, N_scale)
        return {1: 1 - m, 0: m}


@lru_cache(maxsize=64)
def _y_moments(beta, N, N_scale, step=0.2):
    beta, s = float(_beta(beta)), float(_s(N_scale))
    if not (isinstance(N, int) and 2 <= N <= 400):
        raise Refusal("precondition:integer-N", f"N = {N!r}: need an int in [2, 400]")
    M, sd = 2.0**N, beta * math.sqrt(s * N)                # w = -beta E ~ N(0, sd^2)
    L = (math.sqrt(2 * N * math.log(2)) + 8) * sd
    w = np.arange(-L, L + step, step)
    p = np.exp(-w**2 / (2 * sd * sd)) / (math.sqrt(2 * math.pi) * sd) * step
    u = np.arange(-L - 40, 40, step)
    y1 = y2 = 0.0
    for chunk in np.array_split(u, max(1, len(u) // 256)):
        S = chunk[:, None] + w[None, :]
        with np.errstate(over="ignore", under="ignore"):
            x = np.exp(np.minimum(S, 700))
            h = (-np.expm1(-x)) @ p
            b2, b4 = np.exp(2 * S - x) @ p, np.exp(4 * S - x) @ p
        with np.errstate(divide="ignore"):      # h = 1: g = 0
            lg = np.log1p(-np.minimum(h, 1.0))
        g1, g2 = np.exp((M - 1) * lg), np.exp((M - 2) * lg)
        y1 += float(np.sum(M * b2 * g1))
        y2 += float(np.sum((M * b4 * g1 + M * (M - 1) * b2 * b2 * g2) / 6))
    return y1 * step, y2 * step
