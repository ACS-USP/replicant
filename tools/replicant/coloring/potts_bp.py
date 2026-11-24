"""Potts / coloring BP on random graphs: the paramagnetic and ferromagnetic fixed points (KZ Ch. 6).

Convention (KZ Ch. 6): P(s) ~ prod_(ij) exp(-beta delta_{s_i s_j}), s_i in {1..q}; beta > 0 antiferromagnetic
(coloring: beta = oo counts proper colorings), beta < 0 ferromagnetic.  Against the Ising ferromagnet
exp(beta_I s s') with s = +-1: s s' = 2 delta - 1, so beta_Potts = -2 beta_Ising and, per edge,
ln Z_Ising = ln Z_Potts - beta_I (Phi_Ising = Phi_Potts - beta_I d/2 on a d-regular graph).  c = mean degree;
d = the degree of a d-regular graph; c~ = mean excess degree (d - 1 regular, c for Erdos-Renyi); theta = 1 - e^-beta.
  paramagnetic chi = 1/q (KZ Sec. 6.4):  Phi = ln q + (c/2) ln(1 - theta/q),
      e = (c/2) e^-beta / ((q - 1) + e^-beta),  s = Phi + beta e;  Phi = Phi_annealed (Sec. 6.6)
  ferromagnetic ansatz chi_1 = a, chi_s = (1 - a)/(q - 1) on d-regular graphs (Eqs. (6.9)-(6.12)):
      a = RHS(a) = A^(d-1) / (A^(d-1) + (q - 1) B^(d-1)),  A = 1 - theta a,  B = 1 - theta (1 - a)/(q - 1),
      Phi(a) = ln[(q - 1) B^d + A^d] - (d/2) ln[1 - theta ((1 - a)^2/(q - 1) + a^2)]
  stability of 1/q (Eqs. (6.13), (6.22)):  beta_stab = -ln(1 + q/(c~ - 1))
  annealed threshold (Eq. (6.17)):  c_ann(beta) = -2 ln q / ln(1 - theta/q)
  Kesten-Stigum (Eq. (6.24)):  c~_KS = (q - theta)^2 / theta^2 -> (q - 1)^2 at beta = oo
Fixed points: every root of RHS(a) - a on (0, 1) (a 4001-point grid with geometric end points, sign changes
refined by brentq; a = 1/q always), each with RHS'(a) (stable when |RHS'| < 1) and Phi(a).
First-order structure for q >= 3 (Sec. 6.5.2; beta_stab < beta_c < beta_s < 0): beta_s, where the stable
ferromagnetic root a > 1/q disappears (bisection on the root list, refined as the saddle node RHS = a,
RHS' = 1), and beta_c, where Phi(a_ferro) = Phi(1/q) (brentq).  KZ's rule, here a ruling: the points are
reported only when, at beta_c, the iteration of (6.11) from the planted start a = 1 and from the paramagnetic-side
start a = 1/q + 0.01 both converge (to the ferromagnetic and the paramagnetic roots); otherwise each is labelled
undetermined with the reason.  For q = 2 the transition is continuous: beta_s = beta_c = beta_stab (KZ).
Floats (physics-level BP formulas, not certified).

  potts_paramagnetic(q, c, beta) -> Para;  potts_ferro_fixed_points(q, d, beta) -> (FixedPoint, ...)
  potts_beta_stab(q, c_tilde);  potts_transitions(q, d) -> Transitions;  ising_beta / potts_beta (the map)
  coloring_annealed_threshold(q, beta=inf);  kesten_stigum(q, beta=inf)
"""
import math
import warnings
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.optimize import brentq, fsolve

from ..core import Refusal

# planted-error hooks (tests): ising_map (beta_Ising = ising_map * beta_Potts, KZ Sec. 6.5.1), ann_numerator (the 2 of
# 2 ln q in Eq. (6.17)), ks_power (the square in Eq. (6.24)), map_iters (the iteration budget of the two starts)
_HOOKS = {"ising_map": -0.5, "ann_numerator": 2, "ks_power": 2, "map_iters": 200000}


def ising_beta(beta_potts):
    """The Ising-convention beta (H = -J sum s s', J = 1) of a Potts q = 2 beta: beta_Ising = -beta_Potts/2."""
    return _HOOKS["ising_map"] * beta_potts


def potts_beta(beta_ising):
    """The inverse map: beta_Potts = -2 beta_Ising."""
    return beta_ising / _HOOKS["ising_map"]


def beta_value(beta):
    """A float beta from an int, float, rational string or 'inf' (beta = +oo: the hard coloring)."""
    if isinstance(beta, str) and beta.strip().lower() in ("inf", "oo", "+inf"):
        return math.inf
    try:
        b = float(Fraction(beta.strip())) if isinstance(beta, str) else float(beta)
    except (ValueError, ZeroDivisionError, TypeError):
        raise Refusal("precondition:beta", f"beta = {beta!r}: a number, a rational string or 'inf'") from None
    if isinstance(beta, bool) or math.isnan(b) or b == -math.inf:
        raise Refusal("precondition:beta", f"beta = {beta!r}: need a finite number or +inf")
    return b


def _q(q):
    if isinstance(q, bool) or not isinstance(q, (int, np.integer)) or q < 2:
        raise Refusal("precondition:q", f"q = {q!r}: need an int number of colors >= 2")
    return int(q)


def _theta(beta):
    return 1.0 - math.exp(-beta) if beta != math.inf else 1.0


@dataclass(frozen=True)
class Para:
    phi: float
    e: float
    s: float


def potts_paramagnetic(q, c, beta):
    q, b = _q(q), beta_value(beta)
    c = float(c)
    if not c >= 0:
        raise Refusal("precondition:c", f"c = {c!r}: the mean degree must be >= 0")
    x = math.exp(-b) if b != math.inf else 0.0
    phi = math.log(q) + c / 2 * math.log1p(-_theta(b) / q)
    e = c / 2 * x / ((q - 1) + x)
    return Para(phi, e, phi + (b * e if e else 0.0))


def potts_beta_stab(q, c_tilde):
    """-ln(1 + q/(c~ - 1)) (KZ Eqs. (6.13), (6.22)); -inf when c~ <= 1 (no instability: the chain, d = 2)."""
    q = _q(q)
    return -math.log1p(q / (c_tilde - 1)) if c_tilde > 1 else -math.inf


def coloring_annealed_threshold(q, beta="inf"):
    q, b = _q(q), beta_value(beta)
    if not b > 0:
        raise Refusal("precondition:beta", "c_ann is the antiferromagnetic (beta > 0) threshold (KZ Eq. (6.17))")
    return -_HOOKS["ann_numerator"] * math.log(q) / math.log1p(-_theta(b) / q)


def kesten_stigum(q, beta="inf"):
    q, b = _q(q), beta_value(beta)
    if not b > 0:
        raise Refusal("precondition:beta", "c_KS is the antiferromagnetic (beta > 0) threshold (KZ Eq. (6.24))")
    th = _theta(b)
    return (q - th) ** _HOOKS["ks_power"] / th ** 2


def _rhs(a, q, d, th):
    A, B = 1 - th * a, 1 - th * (1 - a) / (q - 1)
    u, w = A ** (d - 1), (q - 1) * B ** (d - 1)
    du, dw = -(d - 1) * th * A ** (d - 2), (d - 1) * th * B ** (d - 2)
    return u / (u + w), (du * w - u * dw) / (u + w) ** 2


def _phi(a, q, d, th):
    A, B = 1 - th * a, 1 - th * (1 - a) / (q - 1)
    return math.log((q - 1) * B ** d + A ** d) - d / 2 * math.log(1 - th * ((1 - a) ** 2 / (q - 1) + a * a))


@dataclass(frozen=True)
class FixedPoint:
    a: float
    phi: float
    slope: float
    stable: bool
    kind: str           # "paramagnetic" | "ferromagnetic" (a > 1/q) | "depleted" (a < 1/q)


_GRID = np.unique(np.concatenate([np.linspace(0, 1, 4001)[1:-1], 10.0 ** -np.arange(4, 13),
                                  1 - 10.0 ** -np.arange(4, 13)]))


def _d(d):
    if isinstance(d, bool) or not isinstance(d, (int, np.integer)) or d < 2:
        raise Refusal("precondition:d", f"d = {d!r}: need an int degree >= 2")
    return int(d)


def potts_ferro_fixed_points(q, d, beta):
    q, d, b = _q(q), _d(d), beta_value(beta)
    if b == math.inf:
        raise Refusal("precondition:beta", "the ferromagnetic ansatz needs a finite beta (beta < 0 is ferromagnetic)")
    th = _theta(b)
    g = _rhs(_GRID, q, d, th)[0] - _GRID
    roots = [1.0 / q]
    for i in np.flatnonzero(np.sign(g[:-1]) * np.sign(g[1:]) < 0):
        lo, hi = _GRID[i], _GRID[i + 1]
        if lo <= 1.0 / q <= hi:
            continue
        roots.append(brentq(lambda a: _rhs(a, q, d, th)[0] - a, lo, hi, xtol=1e-15, rtol=1e-15))
    out = []
    for a in sorted(roots):
        sl = _rhs(a, q, d, th)[1]
        kind = "paramagnetic" if a == 1.0 / q else ("ferromagnetic" if a > 1.0 / q else "depleted")
        out.append(FixedPoint(a, _phi(a, q, d, th), sl, abs(sl) < 1, kind))
    return tuple(out)


def iterate_map(q, d, beta, a0, iters=None, tol=1e-13):
    """(a, converged): the iteration a <- RHS(a) of Eq. (6.11) from a0 (iters default _HOOKS["map_iters"])."""
    th, a = _theta(beta_value(beta)), float(a0)
    iters = _HOOKS["map_iters"] if iters is None else iters
    for _ in range(iters):
        new = _rhs(a, q, d, th)[0]
        if abs(new - a) < tol:
            return new, True
        a = new
    return a, False


def _ferro(q, d, beta):
    fs = [f for f in potts_ferro_fixed_points(q, d, beta) if f.kind == "ferromagnetic" and f.stable]
    return max(fs, key=lambda f: f.a) if fs else None


class _NoFerro(Exception):
    pass


@dataclass(frozen=True)
class Transitions:
    beta_stab: float
    beta_s: float | None
    beta_c: float | None
    order: str              # "second" | "first" | "none" (d = 2)
    status: str             # "determined" | "undetermined"
    note: str


def potts_transitions(q, d):
    q, d = _q(q), _d(d)
    bst = potts_beta_stab(q, d - 1)
    if d == 2:
        return Transitions(bst, None, None, "none", "determined",
                           "d = 2, the chain: no transition at finite beta (beta_stab = -oo; KZ Sec. 6.5.1)")
    if q == 2:
        return Transitions(bst, bst, bst, "second", "determined", "continuous (KZ Sec. 6.5.1): beta_s = beta_c = beta_stab")
    lo, hi = bst + 1e-9, 0.0
    if _ferro(q, d, lo) is None:
        return Transitions(bst, None, None, "first", "undetermined",
                           "no stable ferromagnetic root just above beta_stab: the coexistence window is not resolved")
    for _ in range(60):                                       # beta_s: last beta with a stable ferro root
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _ferro(q, d, mid) is not None else (lo, mid)
    f = _ferro(q, d, lo)
    sn = lambda z: [_rhs(z[0], q, d, _theta(z[1]))[0] - z[0], _rhs(z[0], q, d, _theta(z[1]))[1] - 1]
    with warnings.catch_warnings():                          # fsolve's exit flag is replaced by the audit below
        warnings.simplefilter("ignore", RuntimeWarning)
        sol = fsolve(sn, [f.a, lo], xtol=1e-14)
    res = max(abs(x) for x in sn(sol))                       # audited directly, whatever fsolve's exit flag says
    if not (lo - 1e-6 <= sol[1] <= hi + 1e-6 and 1.0 / q < sol[0] < 1 and res < 1e-10):
        return Transitions(bst, None, None, "first", "undetermined",
                           f"beta_s not determined: the saddle-node solve (residual {res:.3g}, a = {sol[0]:.12g}, "
                           f"beta = {sol[1]:.12g}) fails its audit against the bisection bracket [{lo:.12g}, {hi:.12g}]")
    bs = float(sol[1])

    def dphi(b):
        fb = _ferro(q, d, b)
        if fb is None:
            raise _NoFerro(b)
        return fb.phi - potts_paramagnetic(q, d, b).phi
    try:
        bc = brentq(dphi, bst + 1e-9, lo, xtol=1e-14)
    except _NoFerro as e:
        return Transitions(bst, bs, None, "first", "undetermined",
                           f"beta_c not located: no stable ferromagnetic root at beta = {e.args[0]:.12g}, inside "
                           f"(beta_stab, beta_s): the window is not connected")
    af, conv_f = iterate_map(q, d, bc, 1.0)
    ap, conv_p = iterate_map(q, d, bc, 1.0 / q + 0.01)
    good = conv_f and conv_p and af > 1.0 / q + 1e-6 and abs(ap - 1.0 / q) < 1e-9
    note = (f"at beta_c the planted start a = 1 converges to a = {af:.12g} and the paramagnetic-side start "
            f"a = 1/q + 0.01 to {ap:.12g}" + ("" if good else f" (planted converged: {conv_f}, paramagnetic-side "
                                                               f"converged: {conv_p})"))
    return Transitions(bst, bs, bc, "first", "determined" if good else "undetermined", note)


__all__ = ["FixedPoint", "Para", "Transitions", "beta_value", "ising_beta", "potts_beta", "coloring_annealed_threshold", "iterate_map",
           "kesten_stigum", "potts_beta_stab", "potts_ferro_fixed_points", "potts_paramagnetic",
           "potts_transitions"]
