"""Finite-N approximate survey propagation (ASP) for SK: the iteration whose state evolution is amp/asp.py, run on a
sampled coupling matrix (KZ Sec. 12.2.5, Eqs. (12.63)-(12.65); Antenucci, Krzakala, Urbani & Zdeborova,
arXiv:1807.01296, Sec. IV B).

Model.  SK as amp/sk.py: J symmetric, zero diagonal, J_ij ~ N(0, 1/N) (`sk_matrix(N, seed)`, Gaussian law only),
field h, inverse temperature beta, Parisi parameter m in (0, 1].  ASP is the symmetric AMP
    x^{t+1} = J f_t(x^t) - b_t f_{t-1}(x^{t-1}),   f_{-1} = 0,
with the 1RSB denoiser and its x-derivative (both under the cosh^m-reweighted V-average, V ~ N(0, 1))
    f_t(x)  = <tanh Xi>,   Xi = beta (x + sqrt(q1^t - q0^t) V + h),   <g> = E_V[cosh^m Xi g] / E_V[cosh^m Xi],
    f_t'(x) = beta (<sech^2 Xi> + m (<tanh^2 Xi> - <tanh Xi>^2)),
the Onsager coefficient b_t = mean_i f_t'(x_i^t), and the empirical plug-ins
    q0^{t+1} = mean_i f_t(x_i^t)^2,   q1^{t+1} = mean_i <tanh^2 Xi>(x_i^t).
Start: x^0 = sqrt(q0_start) g, g iid N(0, 1) from default_rng([seed, 2]) (independent of J, which sk_matrix draws
from default_rng([seed, 1])), q0^0 = q0_start, q1^0 = q1_start; default (0.5, 0.95), as asp_fixed_point.

Index alignment with the SE.  amp/asp.py `asp_trajectory` returns (q0_t, q1_t), t = 0 the start, each step its
_map(q0_t, q1_t) = (E_U <tanh>^2, E_U <tanh^2>) at Xi = beta (h + sqrt(q0_t) U + sqrt(q1_t - q0_t) V).  The SE
theorem (Bayati-Montanari, for deterministic Lipschitz f_t) gives x^t ~ N(0, tau_t) with tau_0 = q0_start (the law
of x^0 exactly) and tau_{t+1} = E f_t(X_t)^2.  By induction tau_t = q0_t: true at t = 0, and if X_t ~ sqrt(q0_t) U
with f_t built from (q0_t, q1_t), then tau_{t+1} = E_U <tanh>^2 = q0_{t+1}.  So the finite-N trace entry t (q0^t,
q1^t: the mean of f_{t-1}^2 and of <tanh^2> at x^{t-1}) estimates SE entry t, with no shift; entry 0 is the start
in both.

The plug-in caveat.  The theorem is for a deterministic f_t, i.e. f_t built from the SE's q0_t, q1_t.  This run
builds f_t from the empirical q0^t, q1^t (consistent plug-ins, as amp/sk.py's Onsager term uses the empirical Q_t).
The SE values at a fixed t are statements about the theorem's algorithm; the finite-N traces are STATISTICAL (one
matrix, one start per seed, never certified; asp_amp_ensemble gives 99% Student-t bands over named seeds) and NOT
final.  Below the AT line the iterates need not converge as t grows: compare at a fixed t, not at a fixed point.
`converged` (last increment mean (f_t - f_{t-1})^2 < tol) is a label, never a refusal.

Independence.  The V-average is Gauss-Hermite of order GH_ORDER (probabilists' nodes from scipy.special.roots_hermitenorm, mirrored so that the
quadrature is exactly symmetric in V; cosh^m in log space, the row maximum subtracted), against amp/asp.py's
trapezoid on [-10, 10].  This module imports nothing from amp/asp.py or families/ (AST audit in
tests/test_asp_finite.py); only the matrix generator and the band are shared with amp/sk.py and amp/_stats.py.
At h = 0 and q0_start = 0, x^0 = 0 and the mirrored quadrature makes f = <tanh Xi> exactly 0 (each pair V, -V
cancels), so x^t = 0 and the q0 trace is exactly 0, the SE's q0 = 0 branch.

  asp_amp(J, beta, m, *, h=0, iters, seed, start=None, reaction=True) -> ASPAMPResult
  asp_amp_ensemble(N, beta, m, h=0, *, seeds, iters, start=None, reaction=True) -> ASPAMPEnsemble
reaction=False drops the Onsager term (b_t = 0): the no-Onsager planted control, not AMP.

_HOOKS (planted errors, tests only): "reweight" (True: the cosh^m weight; False sets m = 0 in both f_t and f_t',
the RS / plain AMP average: a valid AMP with the wrong m).
"""
import functools
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import roots_hermitenorm

from ..core import Refusal
from ._stats import student_band
from .sk import check_matrix, sk_matrix

_HOOKS = {"reweight": True}
GH_ORDER = 360           # even; agrees with 2 * GH_ORDER to < 1e-10 on f, f', <tanh^2> for beta <= 3, q1 - q0 <= 0.45,
                         # |x| <= 3 (tests/test_asp_finite.py; 300 is at 9e-11 there, 360 at 7e-12)


_NOTE = ("statistical (finite N, one Gaussian SK matrix and one start per seed; not certified, NOT final).  The SE "
         "(amp/asp.py asp_trajectory) is a theorem at each fixed t for the deterministic f_t built from the SE's "
         "q0_t, q1_t (Bayati-Montanari, Lipschitz f_t, start independent of J); this run uses the empirical "
         "plug-ins q0^t, q1^t, not literally the theorem's algorithm.  Below the AT line the iterates need not "
         "converge: compare at a fixed t")


@functools.lru_cache(maxsize=None)
def _nodes(order):
    """Positive half of the probabilists' Gauss-Hermite rule (order even), mirrored: (z > 0, log weight)."""
    if isinstance(order, bool) or not isinstance(order, int) or order < 2 or order % 2:
        raise Refusal("precondition:quadrature-order", f"order = {order!r} is not an even int >= 2")
    z, w = roots_hermitenorm(order)                # Golub-Welsch / asymptotics: finite at large order
    k = order // 2
    zp = (z[k:] - z[:k][::-1]) / 2                 # exact mirror: V = +zp and V = -zp
    wp = (w[k:] + w[:k][::-1]) / 2
    keep = wp > 0                                  # the outermost weights underflow to 0 at large order
    zp, wp = zp[keep], wp[keep]
    return zp, np.log(wp / (2 * wp.sum()))


def _lncosh(x):
    a = np.abs(x)
    return a + np.log1p(np.exp(-2 * a)) - math.log(2)


def _denoise(x, beta, m, h, s, order=GH_ORDER):
    """At each x: f = <tanh Xi>, f' = beta (<sech^2> + m (<tanh^2> - <tanh>^2)) and <tanh^2 Xi>, with
    Xi = beta (x + s V + h) and the cosh^m V-average (module docstring).  Pairs (V, -V) are summed first."""
    zp, lw = _nodes(order)
    x = np.asarray(x, dtype=float)[..., None]
    up, um = x + h + s * zp, x + h - s * zp
    xp, xm = beta * up, beta * um
    lcp, lcm = _lncosh(xp), _lncosh(xm)
    Lp, Lm = m * lcp + lw, m * lcm + lw
    c = np.maximum(Lp.max(axis=-1, keepdims=True), Lm.max(axis=-1, keepdims=True))
    Wp, Wm = np.exp(Lp - c), np.exp(Lm - c)
    Z = (Wp + Wm).sum(axis=-1)
    tp, tm = np.tanh(xp), np.tanh(xm)
    avg = lambda gp, gm: (Wp * gp + Wm * gm).sum(axis=-1) / Z
    t1 = avg(tp, tm)
    t2 = avg(tp * tp, tm * tm)
    sech2 = avg(np.exp(-2 * lcp), np.exp(-2 * lcm))
    return t1, beta * (sech2 + m * (t2 - t1 * t1)), t2


@dataclass
class ASPAMPResult:
    q0_trace: list           # q0^t, t = 0..iters (q0^0 = q0_start, q0^{t+1} = mean f_t(x^t)^2)
    q1_trace: list           # q1^t, t = 0..iters (q1^0 = q1_start, q1^{t+1} = mean <tanh^2 Xi>(x^t))
    converged: bool          # increments[-1] < tol (a label)
    increments: list = field(default_factory=list)   # mean (f_t - f_{t-1})^2, t = 0..iters-1 (f_{-1} = 0)
    beta: float = 0.0
    m: float = 0.0
    h: float = 0.0
    seed: int = 0
    N: int = 0
    iters: int = 0
    reaction: bool = True
    label: str = "statistical"
    note: str = _NOTE


@dataclass
class ASPAMPEnsemble:
    q0: list                 # q0^iters per seed
    q1: list                 # q1^iters per seed
    bands: dict              # {"q0": Band, "q1": Band}: 99% Student-t bands of the means
    converged: list
    seeds: tuple
    N: int
    iters: int
    note: str
    reaction: bool = True
    label: str = "statistical"


def _num(x, name, reason, ok):
    if isinstance(x, bool):
        raise Refusal(reason, f"{name} = {x!r} is not a number")
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise Refusal(reason, f"{name} = {x!r} is not a number") from None
    if not (math.isfinite(v) and ok(v)):
        raise Refusal(reason, f"{name} = {x!r} is out of range")
    return v


def _start(start):
    if start is None:
        return 0.5, 0.95
    try:
        if isinstance(start, (str, bytes)) or len(start) != 2:
            raise TypeError("not a pair")
        q0, q1 = (float(x) for x in start)
    except (TypeError, ValueError) as exc:
        raise Refusal("precondition:start", f"start = {start!r}: need a pair (q0, q1) of numbers ({exc})")
    if not 0 <= q0 <= q1 <= 1:
        raise Refusal("precondition:start", f"start = {start!r}: need 0 <= q0 <= q1 <= 1")
    return q0, q1


def asp_amp(J, beta, m, *, h=0, iters, seed, start=None, reaction=True, tol=1e-10) -> ASPAMPResult:
    """Run `iters` ASP steps on J (module docstring); traces of the empirical q0^t, q1^t for t = 0..iters."""
    J = check_matrix(J)
    b = _num(beta, "beta", "precondition:beta", lambda v: v > 0)
    mm = _num(m, "m", "precondition:parisi-m", lambda v: 0 < v <= 1)
    hh = _num(h, "h", "precondition:field", lambda v: True)
    if isinstance(iters, bool) or not isinstance(iters, int) or iters < 1:
        raise Refusal("precondition:positive-iters", f"iters = {iters!r} is not a positive int")
    q0, q1 = _start(start)
    m_eff = mm if _HOOKS["reweight"] else 0.0
    N = J.shape[0]
    x = math.sqrt(q0) * np.random.default_rng([seed, 2]).normal(size=N)
    f_old = np.zeros(N)
    q0s, q1s, incs = [q0], [q1], []
    for t in range(iters):
        f, fp, t2 = _denoise(x, b, m_eff, hh, math.sqrt(max(q1 - q0, 0.0)))
        q0, q1 = float(f @ f / N), float(np.mean(t2))
        q0s.append(q0)
        q1s.append(q1)
        incs.append(float((f - f_old) @ (f - f_old) / N))
        if t + 1 < iters:
            onsager = float(np.mean(fp)) if reaction else 0.0
            x = J @ f - onsager * f_old
        f_old = f
    note = _NOTE + ("" if reaction else "; reaction=False: NO Onsager term, the planted control, not AMP") + (
        "" if _HOOKS["reweight"] else "; _HOOKS reweight=False: m = 0 in f and f' (planted control)")
    return ASPAMPResult(q0s, q1s, bool(incs[-1] < tol), incs, b, mm, hh, seed, N, iters, reaction, note=note)


def asp_amp_ensemble(N, beta, m, h=0, *, seeds, iters, start=None, reaction=True, tol=1e-10) -> ASPAMPEnsemble:
    """One fresh matrix sk_matrix(N, seed) and start per named seed; 99% Student-t bands of q0^iters and q1^iters.
    A zero-variance sample (e.g. the exact q0 = 0 branch) gives a zero-width band, not NaN."""
    seeds = tuple(seeds)
    runs = [asp_amp(sk_matrix(N, s), beta, m, h=h, iters=iters, seed=s, start=start, reaction=reaction, tol=tol)
            for s in seeds]
    q0 = [r.q0_trace[-1] for r in runs]
    q1 = [r.q1_trace[-1] for r in runs]
    note = runs[0].note + f"; seeds {list(seeds)}, N = {N}, t = {iters}"
    return ASPAMPEnsemble(q0, q1, {"q0": student_band(q0), "q1": student_band(q1)}, [r.converged for r in runs],
                          seeds, N, iters, note, reaction)


__all__ = ["ASPAMPEnsemble", "ASPAMPResult", "GH_ORDER", "asp_amp", "asp_amp_ensemble"]
