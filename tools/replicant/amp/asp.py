"""Approximate survey propagation (ASP) for SK: the Parisi 1RSB equations as a state evolution (KZ Sec. 12.2.5,
Eqs. (12.63)-(12.65); Antenucci, Krzakala, Urbani & Zdeborova, arXiv:1807.01296, Sec. IV B, whose ASP for low-rank
estimation the notes specialise to SK; the notes' "as shown in ?" is that reference).

SK conventions as amp/sk.py: J_ij ~ N(0, 1/N), field h, inverse temperature beta.  ASP is the symmetric AMP
x^{t+1} = J f_t(x^t) - b_t f_{t-1}(x^{t-1}) with the 1RSB denoiser of (12.63),
    f_t(x) = E_V[cosh^m(Xi) tanh(Xi)] / E_V[cosh^m(Xi)],   Xi = beta x + beta sqrt(q1 - q0) V + beta h,
V ~ N(0, 1), m the Parisi parameter.  Its state evolution (Bolthausen / Bayati-Montanari for a Lipschitz f_t) has
x^t ~ sqrt(q0^t) U with q0^t = E f_{t-1}^2 ("the norm of f_t is q0"), and q1 is carried by (12.64):
    q1^{t+1} = E_U <tanh^2 Xi>,   q0^{t+1} = E_U <tanh Xi>^2,   <g> = E_V[cosh^m Xi g] / E_V[cosh^m Xi],
    Xi = beta (h + sqrt(q0^t) U + sqrt(q1^t - q0^t) V).
At fixed m its fixed points are the q0, q1 stationarity equations of the 1RSB free energy of families/sk.py (which
also solves the m equation); this module imports nothing from families/ (an independent route: a trapezoid rule on a
U x V grid of step 0.1 / grid over [-10, 10], cosh^m reweighting in log space, against sk.py's Gauss-Hermite Newton).
The notes' (12.63)-(12.65) write x where E_U sqrt(q0) U is meant; nothing else is changed.

  asp_fixed_point(beta, m, h=0, *, start=None, tol=1e-14, max_iter=5000, grid=1) -> ASPResult(q0, q1, ...)
start: (q0, q1), default (0.5, 0.95) (an informative start; at h = 0, q0 = 0 is itself a fixed point of the q0 map
because <tanh> is odd in U, so a start with q0 = 0 stays on that branch).  converged False (never a Refusal) when
|step| > tol after max_iter.

_HOOKS (planted errors, tests only): "reweight" (True: the cosh^m weight; False drops it, the RS / AMP average).
"""
import math
from dataclasses import dataclass

import numpy as np

from ..core import Refusal

_HOOKS = {"reweight": True}


def _grid(level):
    n = 200 * level + 1
    z = np.linspace(-10.0, 10.0, n)
    w = np.exp(-z * z / 2)
    return z, w / w.sum()


def _lncosh(x):
    a = np.abs(x)
    return a + np.log1p(np.exp(-2 * a)) - math.log(2)


def _map(beta, m, h, q0, q1, z, w):
    xi = beta * (h + math.sqrt(q0) * z[:, None] + math.sqrt(max(q1 - q0, 0.0)) * z[None, :])
    L = m * _lncosh(xi) if _HOOKS["reweight"] else np.zeros_like(xi)
    L -= L.max(axis=1, keepdims=True)
    W = np.exp(L) * w[None, :]
    W /= W.sum(axis=1, keepdims=True)
    t = np.tanh(xi)
    t1, t2 = (W * t).sum(axis=1), (W * t * t).sum(axis=1)
    return float(w @ (t1 * t1)), float(w @ t2)


@dataclass(frozen=True)
class ASPResult:
    beta: float
    m: float
    h: float
    q0: float
    q1: float
    converged: bool
    iterations: int
    note: str


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


def asp_fixed_point(beta, m, h=0, *, start=None, tol=1e-14, max_iter=5000, grid=1) -> ASPResult:
    """Iterate the ASP state evolution (module docstring) to a fixed point (q0, q1) at fixed Parisi m."""
    b = _num(beta, "beta", "precondition:beta", lambda v: v > 0)
    mm = _num(m, "m", "precondition:parisi-m", lambda v: 0 < v <= 1)
    hh = _num(h, "h", "precondition:field", lambda v: True)
    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 1:
        raise Refusal("precondition:grid", f"grid = {grid!r} is not a positive int")
    q0, q1 = (0.5, 0.95) if start is None else (float(start[0]), float(start[1]))
    if not 0 <= q0 <= q1 <= 1:
        raise Refusal("precondition:start", f"start = {start!r}: need 0 <= q0 <= q1 <= 1")
    z, w = _grid(grid)
    ok, it = False, 0
    for it in range(1, max_iter + 1):
        n0, n1 = _map(b, mm, hh, q0, q1, z, w)
        if start is not None and start[0] == 0 and hh == 0:
            n0 = 0.0                              # the exact q0 = 0 branch (its map value is 0 up to rounding)
        ok = abs(n0 - q0) <= tol and abs(n1 - q1) <= tol
        q0, q1 = n0, n1
        if ok:
            break
    note = (f"ASP state evolution at beta = {b}, m = {mm}, h = {hh}: (q0, q1) = ({q0:.12g}, {q1:.12g}) after {it} "
            f"steps" + ("" if ok else f"; NOT converged (|step| > {tol})"))
    return ASPResult(b, mm, hh, q0, q1, ok, it, note)


__all__ = ["ASPResult", "asp_fixed_point"]
