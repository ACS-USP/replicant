"""The fully connected random field Ising model (KZ Ch. 2).

Convention (KZ Ch. 2, p. 37, with a coupling J > 0 added): H_N(s, h) = -(J N/2) Sbar^2 - sum_i h_i s_i, Sbar =
(1/N) sum_i s_i, h_i i.i.d. from the field law: "gaussian" N(0, Delta) (Delta the VARIANCE, as in KZ) or
"bimodal" +-h0 with probability 1/2 each.  J = 1 is KZ's model.
Replica route (Sec. 2.2, Eqs. (2.1)-(2.3); Ex. 2.2 for the sup):
    Phi_RS(m) = -beta J m^2/2 + E_h ln 2 cosh beta (h + J m),   m = E_h tanh beta (h + J m),
    Phi = Phi_RS at the stationary point of largest Phi_RS, f = -Phi/beta,
    e = J m^2/2 - E_h[(h + J m) tanh beta (h + J m)]                       (Eq. (2.2)),
    T = 0: e_min = min_m [J m^2/2 - E_h |h + J m|]                         (Eq. (2.3); beta -> oo of Ex. 2.2),
E_h by a 96-node Gauss-Hermite rule (gaussian) or the two-point sum (bimodal); floats, physics-level.  Derived
here (NOT printed by KZ; KZ asks for it numerically in Ex. 2.3): for the Gaussian law the T = 0 equation is
m = erf(J m / sqrt(2 Delta)), ferromagnetic for Delta < Delta_c = 2 J^2/pi.
Exact finite-N routes (independent of the RS formulas):
  ground state by sorting: with k spins up the best choice is the k largest fields, so
      e_N = min_k [-J m_k^2/2 - (2 S_k - sum_i h_i)/N],  m_k = (2k - N)/N, S_k the sum of the k largest h_i;
  free entropy by Hubbard-Stratonovich: Z_N = sqrt(N beta J/2pi) int dx exp(-N beta J x^2/2) prod_i 2 cosh
      beta (h_i + J x), a one-dimensional trapezoid on a grid finer than the peak width (exact up to 1e-12).
Guerra's interpolation bounds on one sample (Sec. 2.3; Ex. 2.2), both rigorous at every N:
  lower: Sbar^2 >= 2 m Sbar - m^2 gives Phi_N >= max_m [-beta J m^2/2 + (1/N) sum ln 2 cosh beta (h_i + J m)]
      (the RS functional on the sample: the interpolation remainder (beta/2) <(Sbar - m)^2> is >= 0);
  upper: splitting Z_N by Sbar = m' (N + 1 values) and bounding each term by Chernoff at any mhat (the fixed-m
      interpolation of Sec. 2.3), Phi_N <= ln(N + 1)/N + max_m' min_mhat [beta J m'^2/2 - mhat m' + (1/N) sum_i
      ln 2 cosh(beta h_i + mhat)]; mhat ranges over a grid (any mhat is a valid bound).
_HOOKS (planted-error hooks): "emin_sign" (+1, the sign of J m^2/2 in e_min); "guerra_quadratic" (-1, the sign of
beta J m^2/2 in the lower bound).

  field_law(kind, Delta=None, h0=None) -> FieldLaw;  rfim_rs(beta, J, law) -> RS;  rfim_ground_energy(J, law) -> RS
  rfim_sample_fields(N, law, seed);  rfim_exact_ground_energy(h, J);  rfim_exact_phi(h, beta, J)
  rfim_guerra_bound(h, beta, J) -> (lower, upper)
"""
import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import erf

from ..core import Refusal

_HOOKS = {"emin_sign": 1, "guerra_quadratic": -1}
_GH = np.polynomial.hermite_e.hermegauss(96)            # nodes / weights for N(0, 1) (weight e^{-x^2/2})


def _pos(x, name):
    try:
        v = float(Fraction(x.strip()) if isinstance(x, str) else x)
    except (TypeError, ValueError, ZeroDivisionError):
        raise Refusal(f"precondition:{name}", f"{name} = {x!r}: need a number") from None
    if isinstance(x, bool) or not math.isfinite(v) or v <= 0:
        raise Refusal(f"precondition:{name}", f"{name} = {x!r}: need a finite number > 0")
    return v


@dataclass(frozen=True)
class FieldLaw:
    kind: str           # "gaussian" | "bimodal"
    scale: float        # Delta (variance) for gaussian, h0 for bimodal

    def nodes(self):
        if self.kind == "gaussian":
            return math.sqrt(self.scale) * _GH[0], _GH[1] / _GH[1].sum()
        return np.array([-self.scale, self.scale]), np.array([0.5, 0.5])

    def expect(self, fn):
        x, w = self.nodes()
        return float(np.dot(w, fn(x)))

    def abs_mean(self, c):
        """E |h + c| in closed form."""
        if self.kind == "gaussian":
            s = math.sqrt(self.scale)
            return s * math.sqrt(2 / math.pi) * math.exp(-c * c / (2 * s * s)) + c * math.erf(c / (s * math.sqrt(2)))
        return (abs(c + self.scale) + abs(c - self.scale)) / 2

    def sample(self, rng, n):
        if self.kind == "gaussian":
            return rng.normal(0.0, math.sqrt(self.scale), n)
        return self.scale * rng.choice((-1.0, 1.0), n)


def field_law(kind="gaussian", Delta=None, h0=None):
    if kind == "gaussian":
        if h0 is not None:
            raise Refusal("precondition:known-parameter", "h0 belongs to the bimodal law")
        return FieldLaw("gaussian", _pos(Delta if Delta is not None else 1, "Delta"))
    if kind == "bimodal":
        if Delta is not None:
            raise Refusal("precondition:known-parameter", "Delta belongs to the gaussian law")
        return FieldLaw("bimodal", _pos(h0 if h0 is not None else 1, "h0"))
    raise Refusal("unsupported:field-law", f"law = {kind!r}: gaussian | bimodal")


@dataclass(frozen=True)
class RS:
    m: float
    f: float | None      # free energy -Phi/beta (None at T = 0)
    e: float             # energy per spin (e_min at T = 0)
    phi: float | None
    stationary: tuple    # every stationary m found


def _roots(fn, lo=-1.0, hi=1.0, n=801):
    xs = np.linspace(lo, hi, n)
    ys = [fn(x) for x in xs]
    out = [x for x, y in zip(xs, ys) if y == 0]
    for a, b, ya, yb in zip(xs, xs[1:], ys, ys[1:]):
        if ya * yb < 0:
            out.append(brentq(fn, a, b, xtol=1e-14))
    return sorted(out)


def rfim_rs(beta, J=1, law=None):
    """RS solution at inverse temperature beta (KZ Eqs. (2.1)-(2.2))."""
    b, j, law = _pos(beta, "beta"), _pos(J, "J"), law or field_law()
    eq = lambda m: law.expect(lambda h: np.tanh(b * (h + j * m))) - m
    phi = lambda m: -b * j * m * m / 2 + law.expect(lambda h: np.logaddexp(b * (h + j * m), -b * (h + j * m)))
    st = tuple(_roots(eq))
    if not st:
        raise Refusal("no-solution:bracket", f"no RS root at beta = {beta}, J = {J}")
    m = max(st, key=phi)
    e = j * m * m / 2 - law.expect(lambda h: (h + j * m) * np.tanh(b * (h + j * m)))
    p = phi(m)
    return RS(m, -p / b, e, p, st)


def rfim_ground_energy(J=1, law=None):
    """T = 0: e_min = min over m of J m^2/2 - E|h + J m| (KZ Eq. (2.3)); even in m, so m >= 0 is reported."""
    j, law = _pos(J, "J"), law or field_law()
    sg = _HOOKS["emin_sign"]
    e = lambda m: sg * j * m * m / 2 - law.abs_mean(j * m)
    grid = np.linspace(0, 1, 2001)
    i = int(np.argmin([e(x) for x in grid]))
    r = minimize_scalar(e, bounds=(grid[max(i - 1, 0)], grid[min(i + 1, 2000)]), method="bounded",
                        options={"xatol": 1e-13})
    m = min((0.0, float(r.x), float(grid[i])), key=lambda x: (round(e(x), 12), x))
    if law.kind == "gaussian":
        st = tuple(_roots(lambda x: erf(j * x / math.sqrt(2 * law.scale)) - x, 0.0, 1.0))
    else:
        st = (m,)
    return RS(float(m), None, float(e(m)), None, st)


def rfim_delta_c(J=1):
    """Derived here (not in KZ): the T = 0 Gaussian RFIM orders for Delta < 2 J^2 / pi."""
    return 2 * _pos(J, "J") ** 2 / math.pi


# ---- exact finite N -----------------------------------------------------------------------------------
def rfim_sample_fields(N, law, seed):
    if isinstance(N, bool) or not isinstance(N, int) or N < 2:
        raise Refusal("precondition:size", f"N = {N!r}: need an int >= 2")
    return law.sample(np.random.default_rng(seed), N)


def rfim_exact_ground_energy(h, J=1, branch=None):
    """min of H_N / N over the 2^N configurations (branch None) or over those with Sbar >= 0 (branch "plus"),
    by sorting (exact).  For a symmetric law the unrestricted minimum picks the better of the two
    ferromagnetic states, which lowers its mean by O(N^(-1/2)) (the difference of the two branches is a sum
    of N fields); the "plus" branch converges to e_min at O(1/N) and is the one compared with RS."""
    h, j = np.asarray(h, float), _pos(J, "J")
    if branch not in (None, "plus"):
        raise Refusal("precondition:branch", f"branch = {branch!r}: None or 'plus'")
    n = len(h)
    top = np.concatenate(([0.0], np.cumsum(np.sort(h)[::-1])))
    k = np.arange(n + 1)
    m = (2 * k - n) / n
    e = -j * m * m / 2 - (2 * top - h.sum()) / n
    return float(np.min(e if branch is None else e[m >= 0]))


def _hs_grid(n, b, j):
    w = 1 / math.sqrt(n * b * j)
    lo, hi = -1 - 12 * w, 1 + 12 * w
    return np.linspace(lo, hi, int((hi - lo) / (w / 10)) + 2)


def _logcosh2(x):
    return np.logaddexp(x, -x)


def rfim_exact_phi(h, beta, J=1):
    """ln Z_N / N of one sample by the Hubbard-Stratonovich quadrature (exact up to ~1e-12)."""
    h, b, j = np.asarray(h, float), _pos(beta, "beta"), _pos(J, "J")
    n = len(h)
    xs = _hs_grid(n, b, j)
    expo = np.array([-n * b * j * x * x / 2 + _logcosh2(b * (h + j * x)).sum() for x in xs])
    top = expo.max()
    integral = np.trapezoid(np.exp(expo - top), xs)
    return float((top + math.log(integral) + 0.5 * math.log(n * b * j / (2 * math.pi))) / n)


def rfim_guerra_bound(h, beta, J=1):
    """(lower, upper) bounds on Phi_N of the sample h (rigorous; see the module docstring)."""
    h, b, j = np.asarray(h, float), _pos(beta, "beta"), _pos(J, "J")
    n = len(h)
    q = _HOOKS["guerra_quadratic"]
    rs_n = lambda m: q * b * j * m * m / 2 + _logcosh2(b * (h + j * m)).mean()
    grid = np.linspace(-1, 1, 2001)
    i = int(np.argmax([rs_n(m) for m in grid]))
    r = minimize_scalar(lambda m: -rs_n(m), bounds=(grid[max(i - 1, 0)], grid[min(i + 1, 2000)]),
                        method="bounded", options={"xatol": 1e-13})
    lower = max(-r.fun, rs_n(grid[i]))
    mh = np.linspace(-30, 30, 1201)
    L = np.array([_logcosh2(b * h + x).mean() for x in mh])
    mp_ = (2 * np.arange(n + 1) - n) / n
    best = max(float(np.max(b * j * c ** 2 / 2 + np.min(L[None, :] - mh[None, :] * c[:, None], axis=1)))
               for c in np.array_split(mp_, max(1, (n + 1) // 128)))
    return lower, best + math.log(n + 1) / n


__all__ = ["FieldLaw", "RS", "field_law", "rfim_delta_c", "rfim_exact_ground_energy", "rfim_exact_phi",
           "rfim_ground_energy", "rfim_guerra_bound", "rfim_rs", "rfim_sample_fields"]
