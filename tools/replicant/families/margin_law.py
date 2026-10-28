"""Family (b): the spherical Gardner perceptron with a margin LAW, kappa_l drawn iid per pattern.

Convention (families/perceptron.py, with kappa now per pattern).  P = alpha N patterns xi^l
with iid N(0, 1) components, J.J = N; J stores pattern l if J.xi^l/sqrt(N) >= kappa_l, where
the kappa_l are QUENCHED, iid, drawn from a stated law, independent of the patterns.

Why the average goes inside.  For one pattern the replicated weight is
  E_kappa E_xi prod_a Theta(lambda_a - kappa),   lambda_a = J^a.xi/sqrt(N),
the average over kappa sitting OUTSIDE the product over replicas (one kappa per pattern, shared
by the n replicas) and INSIDE the logarithm of the (1/N) ln E V^n: it is a second Gaussian-type
integral of the same footing as the pattern average, not a parameter of the closed form.  At RS
(lambda_a = sqrt(q) t + sqrt(1-q) z_a, x = (kappa - sqrt(q) t)/sqrt(1-q)) and n -> 0:
  S(q) = 1/2 ln(1-q) + q/(2(1-q)) + alpha E_kappa E_t ln H(x),
  q/(1-q) = alpha E_kappa E_t E(x)^2           (E = phi/H; the heat-equation identity of
                                                perceptron.py holds at every fixed kappa),
  replicon lambda = 1 - alpha E_kappa E_t [(ln H)''(x)]^2.
Replacing kappa by its mean in these integrals is the planted error the tests catch.

Laws (parameters; anything else is refused precondition:supported-margin-law):
  constant  kappa_l = kappa                              (the catalog perceptron)
  gaussian  kappa_l = kappa + kappa_s g, g ~ N(0, 1)     (kappa_s >= 0; kappa defaults to 0)
  mixture   kappa_l = kappa_values[i] w.p. kappa_weights[i]  (weights >= 0, sum 1 to 1e-12)
Each law is a finite list of components (w_i, m_i, s_i): kappa = m_i + s_i g with weight w_i.

The merged Gaussian.  For a component, sqrt(q) t - s g is a sum of independent centred
Gaussians, N(0, q + s^2): x depends on (t, g) only through u ~ N(0, 1),
  x = (m - sqrt(q + s^2) u)/sqrt(1-q)        (the code's ONE-dimensional integral),
exact at every q; _alpha_unmerged is the two-dimensional (t, g) tensor Gauss-Hermite route that
does not use the merge, the audit of the variance q + s^2.

Capacity (q -> 1).  E(x)^2 -> x^2 on the satisfied side, so alpha_c = 1/E_kappa D(kappa),
D(kappa) = (1 + kappa^2) H(-kappa) + kappa phi(kappa) (perceptron._den), and the RS replicon at
capacity is 1 - alpha_c E_kappa H(-kappa) (the violated fraction).  The Gaussian offset
collapses to the catalog perceptron:  E_g D(mu + s g) = (1 + s^2) D(mu/sqrt(1 + s^2)), i.e.
  alpha_c(gaussian mu, s) = alpha_c^catalog(kappa~)/(1 + s^2),   kappa~ = mu/sqrt(1 + s^2).
In the (a, h) form of constraints a g_l + b y_l <= h (g_l ~ N(0, 1) quenched, y_l the Gaussian
patterns, a^2 + b^2 = 1): kappa_l = (a g_l - h)/b, so mu = -h/b, s = a/b, kappa~ = -h and the
load seen by the catalog is alpha_eff = alpha/(1 - a^2) = alpha (1 + s^2) (offset_law, collapse).
This holds at q -> 1 only: the entropy and the replicon below capacity do NOT collapse, and the
replicon at capacity is 1 - alpha_c H(-kappa~), not the catalog's.  collapse_residual is the
banked internal check of the identity.

Scope.  Spherical couplings, RS (capacity: the q -> 1 endpoint).  The level-(a) derivation
(derive_margin.py) takes the same laws for spherical and Ising.  RS is exact where the version
space is convex, all kappa_l >= 0; for margins of both signs (or kappa~ < 0) the RS capacity is the
catalog's caveat (Franz-Parisi) and `stable` is the sign of the computed replicon.  Physics-level
(replica n -> 0, RS), not proven.
"""
import math
from dataclasses import dataclass

import mpmath as mp
import numpy as np
import sympy as sp

from ..core import Refusal
from ..gauss import GaussE
from . import _perceptron_float as F
from .common import CAVEAT, Capacity
from .perceptron import (RSPoint, _bracket_root, _den, _E, _gauss, _H, _phi, _polish, _prec)

LAWS = ("constant", "gaussian", "mixture")
LAW_PARAMS = {"constant": ("kappa",), "gaussian": ("kappa", "kappa_s"),
              "mixture": ("kappa_weights", "kappa_values")}
ALL_PARAMS = ("kappa_law", "kappa", "kappa_s", "kappa_weights", "kappa_values")
MAX_ATOMS = 64
WEIGHT_TOL = mp.mpf("1e-12")


def _refuse(detail):
    raise Refusal("precondition:supported-margin-law", detail)


def _decimal(name, v):
    """A finite real as a decimal string (precision-independent), or the named refusal."""
    if isinstance(v, bool) or v is None:
        _refuse(f"{name} = {v!r}: need a finite real number")
    try:
        x = mp.mpf(str(v))
    except (ValueError, TypeError):
        _refuse(f"{name} = {v!r}: need a finite real number")
    if not mp.isfinite(x):
        _refuse(f"{name} = {v!r}: need a finite real number")
    if isinstance(v, (str, int)):
        return str(v).strip()
    return repr(v) if isinstance(v, float) else mp.nstr(x, 30)


def _sequence(name, v):
    if isinstance(v, str):
        v = [x for x in v.replace(" ", "").split(",") if x]
    if not isinstance(v, (tuple, list)) or not v:
        _refuse(f"{name} = {v!r}: need a non-empty sequence of numbers (or 'a,b,c')")
    if len(v) > MAX_ATOMS:
        _refuse(f"{name}: {len(v)} atoms, at most {MAX_ATOMS}")
    return tuple(_decimal(f"{name}[{i}]", x) for i, x in enumerate(v))


@dataclass(frozen=True)
class MarginLaw:
    """kind in LAWS; mu / s: location and scale (constant: mu = kappa, s = 0); weights / values:
    the atoms of a mixture.  Numbers are decimal strings, converted at the caller's precision."""
    kind: str
    mu: str = "0"
    s: str = "0"
    weights: tuple = ()
    values: tuple = ()

    def components(self):
        """((w, m, s), ...) as mpf at the current precision: kappa = m + s g with weight w."""
        if self.kind == "mixture":
            w = [mp.mpf(x) for x in self.weights]
            tot = mp.fsum(w)
            return tuple((wi / tot, mp.mpf(v), mp.mpf(0)) for wi, v in zip(w, self.values)
                         if wi > 0)
        return ((mp.mpf(1), mp.mpf(self.mu), mp.mpf(self.s)),)

    def mean(self):
        return mp.fsum(w * m for w, m, _ in self.components())

    def describe(self):
        if self.kind == "mixture":
            atoms = ", ".join(f"{w}:{v}" for w, v in zip(self.weights, self.values))
            return f"mixture of margins ({atoms})"
        if self.kind == "gaussian":
            return f"Gaussian margin kappa_l = {self.mu} + {self.s} g"
        return f"constant margin kappa = {self.mu}"


def constant(kappa=0):
    return MarginLaw("constant", mu=_decimal("kappa", kappa))


def gaussian(mu=0, s=0):
    s = _decimal("kappa_s", s)
    if mp.mpf(s) < 0:
        _refuse(f"kappa_s = {s}: the scale of the Gaussian offset must be >= 0")
    return MarginLaw("gaussian", mu=_decimal("kappa", mu), s=s)


def mixture(weights, values):
    w, v = _sequence("kappa_weights", weights), _sequence("kappa_values", values)
    if len(w) != len(v):
        _refuse(f"kappa_weights has {len(w)} entries, kappa_values {len(v)}")
    if any(mp.mpf(x) < 0 for x in w):
        _refuse(f"kappa_weights = {w}: weights must be >= 0")
    if abs(mp.fsum(mp.mpf(x) for x in w) - 1) > WEIGHT_TOL:
        _refuse(f"kappa_weights = {w}: weights must sum to 1 (to 1e-12), "
                f"sum = {mp.nstr(mp.fsum(mp.mpf(x) for x in w), 15)}")
    return MarginLaw("mixture", weights=w, values=v)


def parse_margin_law(params):
    """The law named by `params` (kappa_law, kappa, kappa_s, kappa_weights, kappa_values), or
    None when `kappa_law` is absent (the catalog's constant margin; law parameters without
    kappa_law are refused).  A parameter the chosen law does not read is refused, never
    silently ignored."""
    law = params.get("kappa_law")
    given = [k for k in ALL_PARAMS[1:] if params.get(k) is not None]
    if law is None:
        extra = [k for k in given if k != "kappa"]
        if extra:
            _refuse(f"{', '.join(extra)} given without kappa_law (use one of {', '.join(LAWS)})")
        return None
    if law not in LAWS:
        _refuse(f"kappa_law = {law!r}: supported laws are {', '.join(LAWS)}")
    foreign = [k for k in given if k not in LAW_PARAMS[law]]
    if foreign:
        _refuse(f"{', '.join(foreign)}: not read by kappa_law = {law!r} "
                f"(it takes {', '.join(LAW_PARAMS[law])})")
    if law == "constant":
        return constant(params.get("kappa", 0))
    if law == "gaussian":
        if params.get("kappa_s") is None:
            _refuse("kappa_law = 'gaussian' needs kappa_s (the scale; kappa_s = 0 is the constant)")
        return gaussian(params.get("kappa", 0), params["kappa_s"])
    for k in LAW_PARAMS["mixture"]:
        if params.get(k) is None:
            _refuse(f"kappa_law = 'mixture' needs {k}")
    return mixture(params["kappa_weights"], params["kappa_values"])


def offset_law(a, h):
    """The Gaussian law of constraints a g_l + b y_l <= h, b = sqrt(1 - a^2), as a margin law
    kappa_l = (a g_l - h)/b on the perceptron acting on y: mu = -h/b, s = |a|/b (|a| < 1)."""
    with _prec():
        a, h = mp.mpf(str(a)), mp.mpf(str(h))
        if not mp.isfinite(a) or not mp.isfinite(h) or abs(a) >= 1:
            _refuse(f"(a, h) = ({a}, {h}): need |a| < 1 (a^2 + b^2 = 1, b > 0) and finite h")
        b = mp.sqrt(1 - a * a)
        return gaussian(mp.nstr(-h / b, 30), mp.nstr(abs(a) / b, 30))


def collapse(law):
    """(alpha_eff_factor, kappa_tilde) of a Gaussian law: alpha_c(law) = alpha_c^catalog(kappa~)
    * alpha_eff_factor, factor = 1/(1 + s^2), kappa~ = mu/sqrt(1 + s^2) (module docstring)."""
    if law.kind != "gaussian":
        _refuse(f"collapse: only the Gaussian offset collapses to the catalog, not {law.kind!r}")
    with _prec():
        mu, s = mp.mpf(law.mu), mp.mpf(law.s)
        return 1 / (1 + s * s), mu / mp.sqrt(1 + s * s)


# ---- the law-averaged t-integrals (mpmath), merged variable --------------------------------
def _merged_std(q, s):
    """Std of sqrt(q) t - s g, t and g independent N(0, 1): sqrt(q + s^2) (the one place the
    merge enters the numerics; tests/test_margin_law.py plants a wrong variance here)."""
    return mp.sqrt(q + s * s)


class _TLaw:
    """E_kappa E_t of a function of x at overlap q: sum_i w_i E_u f((m_i - c_i u)/sqrt(1-q)),
    c_i = sqrt(q + s_i^2)."""

    def __init__(self, law, q, one_m_q):
        self.r = mp.sqrt(one_m_q)
        self.parts = []
        for w, m, s in law.components():
            c = _merged_std(q, s)
            u0, wd = m / c, self.r / c
            self.parts.append((w, m, c, [u0 - 30 * wd, u0 - 3 * wd, u0, u0 + 3 * wd,
                                         u0 + 30 * wd, 0]))

    def _avg(self, f):
        return mp.fsum(w * _gauss(lambda u: f((m - c * u) / self.r), pts)
                       for w, m, c, pts in self.parts)

    def eh2(self):
        return self._avg(lambda x: _E(x)**2)

    def elnh(self):
        return self._avg(lambda x: mp.log(_H(x)))

    def d2(self):                       # E [(ln H)''(x)]^2, (ln H)'' = -E (E - x)
        return self._avg(lambda x: (_E(x) * (_E(x) - x))**2)


def _alpha_of(law, q):
    return q / ((1 - q) * _TLaw(law, q, 1 - q).eh2())


def _float_alpha(law_c, q):
    """float64 alpha(q), to bracket roots only (law_c: ((w, m, s) floats))."""
    r = math.sqrt(1 - q)
    tot = 0.0
    for w, m, s in law_c:
        c = float(_merged_std(q, s))
        u0, wd = m / c, r / c
        tot += w * F.gauss(lambda u: F.E((m - c * u) / r)**2,
                           [u0 - 30 * wd, u0 - 3 * wd, u0, u0 + 3 * wd, u0 + 30 * wd, 0.0])
    return q / ((1 - q) * tot)


def _unmerged_grid(n=160):
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / math.sqrt(2 * math.pi)


def alpha_unmerged(law, q, nodes=160):
    """alpha(q) = q/((1-q) E_kappa E_t E(x)^2) by TWO-dimensional Gauss-Hermite in (t, g), for
    x = (m + s g - sqrt(q) t)/sqrt(1-q): no merge of the Gaussians, the audit of the variance
    q + s^2 (float64; use a smooth case: the grid does not resolve sharp x for q -> 1)."""
    x, w = _unmerged_grid(nodes)
    q = float(q)
    r = math.sqrt(1 - q)
    tot = 0.0
    for wi, m, s in law.components():
        m, s, wi = float(m), float(s), float(wi)
        if s == 0:
            xx = (m - math.sqrt(q) * x) / r
            tot += wi * float(np.dot(w, F.E(xx)**2))
            continue
        xx = (m + s * x[None, :] - math.sqrt(q) * x[:, None]) / r
        tot += wi * float(np.einsum("i,j,ij->", w, w, F.E(xx)**2))
    return q / ((1 - q) * tot)


# ---- RS saddle, capacity, entropy ------------------------------------------------------------
def law_rs_point(q, law) -> RSPoint:
    """RS saddle at overlap q in [0, 1): alpha from the stationarity equation, as in
    perceptron.gardner_rs_point('spherical')."""
    with _prec():
        o = mp.mpf(q)
        k = law.mean()
        if not 0 <= o < 1:
            raise Refusal("precondition:order-in-range", f"spherical q = {q} must be in [0, 1)")
        if o == 0:
            return RSPoint(mp.mpf(0), o, None, mp.mpf(0), mp.mpf(1), "spherical", k)
        T = _TLaw(law, o, 1 - o)
        a = o / ((1 - o) * T.eh2())
        S = mp.log(1 - o) / 2 + o / (2 * (1 - o)) + a * T.elnh()
        return RSPoint(a, o, None, S, 1 - a * T.d2(), "spherical", k)


def _solve_alpha(law, alpha):
    law_c = [(float(w), float(m), float(s)) for w, m, s in law.components()]
    a = float(alpha)
    ff = lambda u: math.log(_float_alpha(law_c, 1 / (1 + math.exp(-u))) / a)
    what = f"RS saddle of the margin law at alpha = {mp.nstr(alpha, 10)}"
    u0 = _bracket_root(ff, -30.0, 28.0, what)
    order = lambda u: 1 / (1 + mp.exp(-u))
    u = _polish(lambda u: mp.log(_alpha_of(law, order(u)) / alpha), u0, what)
    pt = law_rs_point(order(u), law)
    if abs(pt.alpha / alpha - 1) > mp.mpf(10)**(-mp.mp.dps + 3):
        raise Refusal("no-solution", f"{what}: audit residual {mp.nstr(pt.alpha / alpha - 1, 3)}")
    return pt


def _e_kappa(law, f):
    return mp.fsum(w * f(m, s) for w, m, s in law.components())


def _gaussian_over_g(f, s, m):
    """E_g f(m + s g) for the offset component (tanh-sinh); the atom s = 0 needs none."""
    if s == 0:
        return f(m)
    return _gauss(lambda g: f(m + s * g), [-3, 0, 3])


def law_capacity(law) -> Capacity:
    """alpha_c = 1/E_kappa D(kappa) (the q -> 1 endpoint of the RS saddle) and the RS replicon
    there, 1 - alpha_c E_kappa H(-kappa).  E_kappa is taken over the law itself (a quadrature
    over g for the Gaussian), NOT through the collapse to the catalog."""
    with _prec():
        D = _e_kappa(law, lambda m, s: _gaussian_over_g(_den, s, m))
        ac = 1 / D
        viol = _e_kappa(law, lambda m, s: _gaussian_over_g(lambda k: _H(-k), s, m))
        lam = 1 - ac * viol
        if lam > 0:
            note = "replicon > 0 at capacity (RS self-consistent at the q -> 1 endpoint)"
        elif lam == 0:
            note = "replicon -> 0 at alpha_c (marginal)"
        else:
            note = ("replicon < 0 at capacity: the RS alpha_c is not the capacity (non-convex "
                    "version space, RSB needed; Franz & Parisi, arXiv:1501.03397)")
        note = f"{law.describe()}: alpha_c = 1/E_kappa D(kappa); {note}"
        return Capacity(ac, bool(lam >= 0), "rs", "spherical", law.mean(), lam, note)


def law_entropy_point(alpha, law):
    """(S, RSPoint | None), as perceptron.gardner_entropy_point('spherical'): same refusals."""
    with _prec():
        a = mp.mpf(alpha)
        if not a >= 0:
            raise Refusal("precondition:nonnegative-alpha", f"alpha = {alpha}")
        if a == 0:
            return mp.mpf(0), law_rs_point(0, law)
        ac = law_capacity(law).alpha_c
        if a >= ac:
            return mp.ninf, None
        pt = _solve_alpha(law, a)
        if pt.replicon < 0:
            raise Refusal("unstable:rs", f"replicon = {mp.nstr(pt.replicon, 6)} < 0 at alpha = "
                          f"{mp.nstr(a, 8)}: RS is AT-unstable, its entropy is not reported")
        return pt.S, pt


def collapse_residual(law):
    """|alpha_c(law) (1 + s^2) - alpha_c^catalog(kappa~)|: the banked internal check of the
    Gaussian-offset collapse (law_capacity integrates over the law; the catalog is the closed
    form of the collapsed perceptron)."""
    from .perceptron import gardner_capacity
    factor, kt = collapse(law)
    with _prec():
        return abs(law_capacity(law).alpha_c / factor - gardner_capacity("spherical", kt).alpha_c)


def rs_equations_symbolic():
    """The saddle equation the solver uses for the Gaussian law, in sympy form with the GaussE
    placeholder, as perceptron.rs_equations_symbolic does for the catalog:
      q/(1-q) = alpha E_u E(x)^2,  x = (kappa - sqrt(q + kappa_s^2) u)/sqrt(1-q),  E = phi/H
    (alpha(q) = q/((1-q) _TLaw.eh2()); kappa_s = 0 is the catalog's)."""
    q, alpha = sp.Symbol("q", positive=True), sp.Symbol("alpha", positive=True)
    kappa, ks = sp.Symbol("kappa", real=True), sp.Symbol("kappa_s", positive=True)
    u = sp.Symbol("t", real=True)
    x = (kappa - sp.sqrt(q + ks**2) * u) / sp.sqrt(1 - q)
    E = (sp.exp(-x**2 / 2) / sp.sqrt(2 * sp.pi)) / (sp.erfc(x / sp.sqrt(2)) / 2)
    return [sp.Eq(q / (1 - q), alpha * GaussE(E**2, u))]
