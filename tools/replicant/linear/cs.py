"""The noiseless compressed-sensing phase diagram for the Gauss-Bernoulli prior (K6; KZ notes Sec. 16.6.2, Fig. 16.6.1).

Model: y = A x0 (Delta = 0), A in R^{M x N} with i.i.d. N(0, 1/N) entries, alpha = M/N, x0 i.i.d.
(1 - rho) delta_0 + rho N(0, 1).  Three boundaries in the (rho, alpha) plane, labelled separately:
  alpha_l1     l1 minimisation (basis pursuit, an LP) succeeds above it.  DMM (Donoho, Maleki & Montanari, PNAS 106,
               18914 (2009), arXiv:0907.3574, fetched) Eq. (5) for signed signals (kappa_chi = 2):
                   rho_SE(delta) = max_{z >= 0} [1 - (2/delta) S(z)] / [1 + z^2 - 2 S(z)],
                   S(z) = (1 + z^2) Phi(-z) - z phi(z),
               with delta = alpha and DMM's rho = k/n = rho/alpha: alpha_l1 solves alpha rho_SE(alpha) = rho.  DMM's
               Finding 1 (numerical) equates rho_SE with Donoho-Tanner's combinatorial-geometry curve rho_CG (not
               fetched); the LP at finite N is the independent check here.
  alpha_l1_se  the same boundary from the soft-threshold SE (linear/lasso.py) at Delta = 0 with threshold kappa tau:
               E = 0 is stable iff the slope lim_{tau -> 0} mse(tau, kappa tau)/tau^2 < alpha (E' = mse with
               tau^2 = E/alpha), so alpha_l1_se = min_kappa slope(kappa) (DMM's local-stability argument, their
               Sec. "Local stability"; no closed form used).  The slope is lasso.mse_b/tau^2 at tau = 1e-4 and 2e-4,
               extrapolated linearly to tau = 0 (the leading correction is O(tau): the Gaussian component's mass
               under the threshold; much smaller tau loses digits to Phi(t) - Phi(-t) at t ~ tau).
  alpha_bayes  Bayes-optimal AMP (linear/gamp.py, matched prior) succeeds above it: the spinodal of the Bayes SE.
               Derived here: with Delta = 0 the SE is E' = mmse(snr = alpha/E) (linear/se.py); a fixed point E > 0 is
               s mmse(s) = alpha at s = alpha/E, and s >= alpha/rho holds automatically there (mmse <= rho), so a
               non-trivial fixed point exists iff alpha <= max_s s mmse(s), and alpha_bayes = max_s s mmse(s)
               (golden section in log s; mmse by bayes/channels.py).  It is a value only when the two SE starts
               (linear/se.py, tol 1e-10 in m) confirm it: at alpha_bayes (1 - 0.02) "two-fixed-points"
               (uninformative MSE > 1e-6 rho,
               informative MSE < 1e-9 rho: exact recovery to the SE's tolerance), at alpha_bayes (1 + 0.02) "unique"
               with MSE < 1e-9 rho; otherwise
               bayes_status is "undetermined" (the ruling on first-order values).  Physics-level (KMSSZ, PRX 2,
               021005 (2012), arXiv:1109.4424, fetched: alpha_EM-BP(rho0 = 0.4) ~ 0.59, Fig. 3 caption).
  alpha_it     = rho: the information-theoretic boundary of KMSSZ's replica analysis (the global maximum of their
               Phi(D) is at D = 0 iff alpha > rho0, main text); physics-level here (no theorem fetched).
_HOOKS["snr"] (1.0) scales the snr inside s mmse(s) (planted errors only).
"""
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq, linprog, minimize_scalar
from scipy.special import ndtr

from ..bayes.channels import Prior, mmse
from ..core import Refusal
from .gamp import Channel, linear_instance
from .lasso import mse_b
from .se import gamp_se

_HOOKS = {"snr": 1.0}
_SQ2PI = math.sqrt(2 * math.pi)
_GOLD = (math.sqrt(5) - 1) / 2


def _rho(rho):
    try:
        r = float(rho)
    except (TypeError, ValueError):
        r = math.nan
    if not (0 < r < 1):
        raise Refusal("precondition:rho", f"rho={rho!r}: the non-zero fraction must lie in (0, 1)")
    return r


def _S(z):
    return (1 + z * z) * float(ndtr(-z)) - z * math.exp(-z * z / 2) / _SQ2PI


def rho_se(delta):
    """DMM Eq. (5), kappa = 2: (rho_SE, the maximising z)."""
    f = lambda z: -(1 - 2 * _S(z) / delta) / (1 + z * z - 2 * _S(z))
    r = minimize_scalar(f, bounds=(1e-9, 12.0), method="bounded", options={"xatol": 1e-13})
    return -r.fun, r.x


def l1_boundary(rho):
    """alpha with alpha rho_SE(alpha) = rho (DMM Eq. (5) with rho_DMM = rho/alpha)."""
    r = _rho(rho)
    return brentq(lambda a: a * rho_se(a)[0] - r, r * (1 + 1e-9), 1 - 1e-12, xtol=1e-14)


def _slope(r, kappa, tau=1e-4):
    pr = Prior("gauss_bernoulli", rho=repr(r))
    at = lambda t: mse_b(pr, t, kappa * t)[0] / (t * t)
    return 2 * at(tau) - at(2 * tau)


def l1_boundary_se(rho):
    """min over kappa of the soft-threshold SE's slope at E = 0 (module docstring)."""
    r = _rho(rho)
    res = minimize_scalar(lambda k: _slope(r, k), bounds=(1e-6, 8.0), method="bounded", options={"xatol": 1e-10})
    return float(res.fun)


def bayes_spinodal(rho):
    """(max_s s mmse(s), argmax s) for the Gauss-Bernoulli prior (module docstring), golden section in log s."""
    pr = Prior("gauss_bernoulli", rho=repr(_rho(rho)))
    h = lambda t: math.exp(t) * mmse(pr, _HOOKS["snr"] * math.exp(t))
    ts = np.linspace(-3, 10, 53)
    k = int(np.argmax([h(t) for t in ts]))
    a, b = ts[max(k - 1, 0)], ts[min(k + 1, len(ts) - 1)]
    c, d = b - _GOLD * (b - a), a + _GOLD * (b - a)
    hc, hd = h(c), h(d)
    while b - a > 1e-9:
        if hc > hd:
            b, d, hd = d, c, hc
            c = b - _GOLD * (b - a)
            hc = h(c)
        else:
            a, c, hc = c, d, hd
            d = a + _GOLD * (b - a)
            hd = h(d)
    t = (a + b) / 2
    return h(t), math.exp(t)


@dataclass(frozen=True)
class CSBoundary:
    rho: float
    alpha_l1: float          # DMM Eq. (5)
    alpha_l1_se: float       # the soft-threshold SE's local stability (same boundary, independent evaluation)
    alpha_bayes: float       # the Bayes SE spinodal (nan when undetermined)
    alpha_it: float          # = rho (KMSSZ replica analysis)
    bayes_status: str        # spinodal | undetermined | none | unconfirmed
    note: str = ""
    starts: tuple = ()       # (GampSE at alpha_bayes (1 - margin), at alpha_bayes (1 + margin)) when confirmed


def cs_phase_boundary(rho, *, margin=0.02, confirm=True) -> CSBoundary:
    """The three boundaries at the non-zero fraction rho (module docstring).  confirm=False skips the two-start check
    (bayes_status "unconfirmed": alpha_bayes is then the formula's maximum, not a reported value).  Memoised on
    (rho, margin, confirm, _HOOKS["snr"]): a pure function of these (the result is frozen)."""
    return _boundary(_rho(rho), float(margin), bool(confirm), _HOOKS["snr"])


@lru_cache(maxsize=32)
def _boundary(r, margin, confirm, _hook):
    a_l1, a_se = l1_boundary(r), l1_boundary_se(r)
    a_bp, s_star = bayes_spinodal(r)
    if a_bp <= r * (1 + margin):
        return CSBoundary(r, a_l1, a_se, math.nan, r, "none", "max_s s mmse(s) <= rho: no hard phase")
    if not confirm:
        return CSBoundary(r, a_l1, a_se, a_bp, r, "unconfirmed", "the two SE starts were not run")
    ch, pr = Channel("awgn", Delta=0), Prior("gauss_bernoulli", rho=repr(r))
    lo, hi = (gamp_se(a_bp * (1 + sg * margin), ch, pr, tol=1e-10) for sg in (-1, 1))
    ok = (lo.status == "two-fixed-points" and lo.mse > 1e-6 * r and lo.mse_informative < 1e-9 * r
          and hi.status == "unique" and hi.mse < 1e-9 * r)
    note = (f"alpha_bayes = max_s s mmse(s) = {a_bp:.8g} at s = {s_star:.6g}; two starts at alpha (1 -+ {margin}): "
            f"{lo.status} (MSE {lo.mse:.3g} / {lo.mse_informative:.3g}), {hi.status} (MSE {hi.mse:.3g})")
    if not ok:
        return CSBoundary(r, a_l1, a_se, a_bp, r, "undetermined", "UNDETERMINED: the SE starts do not bracket the "
                          "spinodal; " + note, (lo, hi))
    return CSBoundary(r, a_l1, a_se, a_bp, r, "spinodal", note, (lo, hi))


# ---- l1 recovery by linear programming at finite N ----------------------------------------------------------------
@dataclass(frozen=True)
class L1Run:
    success: bool            # |x_hat - x0|^2 <= 1e-12 |x0|^2 (relative error < 1e-6)
    rel_error: float
    status: str


def l1_recover(A, y):
    """argmin |x|_1 subject to A x = y, as the LP min 1.(u + v), A (u - v) = y, u, v >= 0 (scipy HiGHS)."""
    M, N = A.shape
    res = linprog(np.ones(2 * N), A_eq=np.hstack([A, -A]), b_eq=y, bounds=(0, None), method="highs")
    if res.status != 0:
        return None, res.message
    return res.x[:N] - res.x[N:], res.message


def l1_recover_instance(N, alpha, rho, seed):
    """One noiseless Gauss-Bernoulli instance (linear_instance, seed) and its LP recovery."""
    A, y, x0 = linear_instance(N, alpha, Channel("awgn", Delta=0), Prior("gauss_bernoulli", rho=repr(_rho(rho))),
                               seed)
    x, msg = l1_recover(A, y)
    if x is None:
        return L1Run(False, math.inf, f"LP failed: {msg}")
    rel = float(np.linalg.norm(x - x0) / max(np.linalg.norm(x0), 1e-300))
    return L1Run(rel < 1e-6, rel, "solved")


@dataclass(frozen=True)
class L1Fraction:
    fraction: float
    successes: int
    trials: int
    seeds: tuple
    label: str = "statistical (finite N, named seeds; never final)"


def l1_success_fraction(N, alpha, rho, seeds):
    seeds = tuple(int(s) for s in seeds)
    ok = sum(l1_recover_instance(N, alpha, rho, s).success for s in seeds)
    return L1Fraction(ok / len(seeds), ok, len(seeds), seeds)


__all__ = ["CSBoundary", "L1Fraction", "L1Run", "bayes_spinodal", "cs_phase_boundary", "l1_boundary",
           "l1_boundary_se", "l1_recover", "l1_recover_instance", "l1_success_fraction", "rho_se"]
