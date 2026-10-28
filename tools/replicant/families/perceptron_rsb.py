"""Family (b): the spherical perceptron with a NEGATIVE margin at T = 0, one-step RSB (1RSB).

The version space {J.J = N : J.xi^mu / sqrt(N) >= kappa for all mu} of the spherical
perceptron is convex for kappa >= 0, where the RS capacity alpha_c(kappa) = 1/D(kappa) is exact
(Gardner 1988), and non-convex for kappa < 0, where the RS replicon is negative at capacity and
the RS value is only an upper bound (Franz & Parisi, J. Phys. A 49, 145001 (2016),
arXiv:1501.03397, Fig. 1 caption and text; Stojnic, arXiv:1306.3980).  The SAT-UNSAT (jamming)
threshold is then a replica-symmetry-broken quantity (Franz, Parisi, Sevelev, Urbani &
Zamponi, SciPost Phys. 2, 019 (2017), arXiv:1702.06919; FPSUZ below).  Convention of this
module: the paper's sigma is our kappa (their gap h = J.xi/sqrt(N) - sigma >= 0), alpha =
P/N, D(kappa) as in perceptron.py, H(x) = int_x^inf Dz.

1RSB entropy of the SAT phase (FPSUZ Eq. (59); q0 between blocks, q1 inside a block of size
m, written with gap g = kappa - sqrt(q0) t - sqrt(q1 - q0) u, t, u ~ N(0, 1)):
  s(q0, q1, m) = (m-1)/(2m) ln(1-q1) + (1/(2m)) ln(1-q1 + m(q1-q0)) + q0/(2(1-q1 + m(q1-q0)))
                 + (alpha/m) E_t ln E_u H( g / sqrt(1-q1) )^m                  (`unscaled_entropy`)
(m = 1 is the RS entropy of perceptron.py).  The jamming point is where the cluster volume
shrinks to zero, q1 -> 1: put 1 - q1 = delta -> 0 and m = mu delta (the T -> 0 scaling of a
soft-sphere replica, mu = m / (1 - q1) finite).  With H(g/sqrt(delta))^m -> exp(-mu g_+^2 / 2)
(g_+ = max(g, 0): the violation) the entropy is
  s = (1/2)(1 + c) ln delta + Phi(mu, q0; alpha) / delta + O(1)   (c: the contact index below),
  Phi = ln(1 + mu(1-q0)) / (2 mu) + q0 / (2 (1 + mu(1-q0)))
        + (alpha/mu) E_t ln Y(t),   Y = E_u exp(-mu g_+^2 / 2),  g = kappa - sqrt(q0) t - sqrt(1-q0) u,
and Y has the closed form (a = kappa - sqrt(q0) t, b^2 = 1 - q0, s^2 = 1 + mu b^2)
  Y = H(a/b) + exp(-mu a^2 / (2 s^2)) / s * (1 - H(a / (b s))).
Stationarity of s in q1 forces Phi = 0 (the 1/delta term must vanish), and in m, q0 forces
dPhi/dmu = dPhi/dq0 = 0: three equations in (mu, q0, alpha).  Phi is LINEAR in alpha,
Phi = ent(mu, q0) + alpha en(mu, q0), so Phi = 0 means alpha = -ent/en =: A(mu, q0), and
dPhi = 0 at Phi = 0 is exactly grad A = 0.  The 1RSB capacity is therefore the stationary
value of A(mu, q0) (a LOCAL MINIMUM of A at every kappa < 0 checked: the Hessian is audited
positive definite).  Its minimum has the form of the rigorous upper bounds of Stojnic's lifted
random duality theory (arXiv:1306.3980, arXiv:2312.16531: every trial value of his parameters
bounds the capacity from above), and reproduces his numbers (below); that A and his functional
are the same object is verified numerically here, not proved, and every trial (mu, q0) of A is
a bound only under that identification.  mu -> 0 (m = 0) is the RS value A = 1/D: the RS capacity
is the boundary of the family, so alpha_c^{1RSB} <= alpha_c^{RS} always.

kappa >= 0: A has no stationary point below 1/D (its infimum is the mu -> 0 boundary, checked
in tests/test_perceptron_rsb.py) and the version space is convex, so 1RSB = RS.
kappa < 0: a unique nontrivial minimum (mu > 0 finite, q0 < 1), alpha_c^{1RSB}(kappa) <
1/D(kappa); as kappa -> 0^-, alpha_RS - alpha_1RSB ~ 0.50 |kappa|^3, mu ~ 1.5 / kappa^2.

Replicon (stability of the 1RSB point toward further RSB, the Gardner instability).  FPSUZ
Eqs. (62)-(63): 1 / lambda^2 = alpha int dh P(1, h) f''(1, h)^2 with lambda = 1 - q1 = delta.  At
jamming f''(1, h) = -1/delta on the violated side (h < 0) and 0 on the satisfied side, so
the leading order is the 1RSB isostaticity (contact) index
  c = alpha * Prob(g > 0 | cluster) = alpha E_t [ Y_contact(t) / Y(t) ],
  Y_contact = exp(-mu a^2 / (2 s^2)) / s * (1 - H(a / (b s))),   Y = H(a/b) + Y_contact,
and the replicon is Lambda = 1 - c: the 1RSB point is stable toward further RSB iff c <= 1.
Full RSB at jamming is isostatic (c = 1, FPSUZ Sec. V; Franz & Parisi 2016); c > 1 at 1RSB
means the 1RSB capacity is NOT the capacity (`unstable`): K-step RSB (perceptron_krsb.py)
lowers it further.  c is read off the leading order of Eq. (62) here; it is checked in
tests against `unscaled_replicon`, the same formula (62)-(63) evaluated at finite delta.

Published values read (not remembered): Stojnic, arXiv:2312.16531 (Dec 2023), Tables 2, 3, 5,
6: his second full level of lifting (2-sfl RDT) is this 1RSB capacity (12.32 at kappa = -1,
36.57 at -1.5, 125.4 at -2, 4.701 at -0.5) and his q-parameter p2-hat (0.4747 at -1.5) is
our q0; see reference/perceptron.json.  The RS rows (1-sfl) equal 1/D.

Numerics: the float64 stage minimises A over (ln mu, ln(1 - q0)) by Nelder-Mead from the best node of a
coarse grid (composite Gauss-Legendre in t); the root is polished by quasi-Newton on grad A in mpmath
(tanh-sinh, 20 digits; alpha_c is stable to 2e-20 against a 32-digit run) and audited: the three
stationarity residuals dPhi/dmu, dPhi/dq0 (<= 1e-10; Phi = 0 holds by construction, alpha = A), the Hessian, and
agreement with the float64 stage.  Physics-level (replica n -> 0, 1RSB ansatz).
"""
import math
from dataclasses import dataclass
from functools import lru_cache

import mpmath as mp
import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr

from ..core import Refusal
from .common import CAVEAT

_DPS = 20                       # working precision of the polish and the audit
_KAPPA_MIN = -3.0               # float64 stage validated on [_KAPPA_MIN, -0.02]
_KAPPA_NEAR_ZERO = -0.02
_AUDIT_TOL = 1e-10              # stationarity residuals (central differences, h = 1e-9, set the floor)
_MU_FLOOR = 1e-2                # float64 stage: ln(1+mu s)/mu and ln Y/mu cancel below this

JAMMING_CITE = ("Franz, Parisi, Sevelev, Urbani & Zamponi, SciPost Phys. 2, 019 (2017), "
                "arXiv:1702.06919, Eq. (59) and Sec. V (zero-temperature 1RSB, jamming scaling)")


@dataclass(frozen=True)
class RSBJamming:
    """The 1RSB jamming point at margin `kappa` (a stationary minimum of A(mu, q0))."""
    kappa: object
    alpha_c: object          # 1RSB capacity (SAT-UNSAT threshold)
    alpha_rs: object         # 1/D(kappa), the RS capacity
    q0: object               # overlap between clusters
    mu: object               # m / (1 - q1) in the jamming scaling
    contact_index: object    # c = alpha P(contact); 1RSB replicon Lambda = 1 - c
    replicon: object
    hessian_eigs: tuple      # eigenvalues of d^2 A / d(ln mu, ln(1-q0))^2, float64 (positive: minimum)
    residuals: tuple         # (Phi, dPhi/dmu, dPhi/dq0) at the polished point (audit)
    levels: int = 1
    caveat: str = CAVEAT


# ---- Gaussian helpers (mpmath) -----------------------------------------------------------------
def _phi(x):
    return mp.exp(-x * x / 2) / mp.sqrt(2 * mp.pi)


def _H(x):
    return mp.erfc(x / mp.sqrt(2)) / 2


def _den(kappa):
    return (1 + kappa**2) * _H(-kappa) + kappa * _phi(kappa)


def rs_capacity(kappa):
    """1/D(kappa): the RS capacity (= perceptron.gardner_capacity('spherical', kappa).alpha_c)."""
    with mp.workdps(max(_DPS, mp.mp.dps)):
        return 1 / _den(mp.mpf(kappa))


def _y_parts(a, mu, b2):
    """(Y_satisfied, Y_contact) with Y = E_u exp(-mu g_+^2/2), g = a - b u, b2 = b^2:
    Y_satisfied = P(g <= 0) = H(a/b), Y_contact = int_{g>0} = exp(-mu a^2/(2 s^2))/s * Phi(a/(b s))."""
    b = mp.sqrt(b2)
    s2 = 1 + mu * b2
    s = mp.sqrt(s2)
    return _H(a / b), mp.exp(-mu * a * a / (2 * s2)) / s * _H(-a / (b * s))


def _pts(kappa, q0, mu):
    """Quadrature breakpoints in t: a = kappa - sqrt(q0) t changes character at t0 = kappa/sqrt(q0),
    on the scales b/sqrt(q0) (the satisfied tail) and b s/sqrt(q0) (the contact part)."""
    s0 = mp.sqrt(q0)
    t0 = kappa / s0
    b = mp.sqrt(1 - q0)
    pts = {0, t0}
    for w in (b / s0, b * mp.sqrt(1 + mu * b * b) / s0):
        for c in (3, 30):
            pts |= {t0 - c * w, t0 + c * w}
    return sorted(p for p in pts if abs(p) < 12) + [mp.inf]


def _t_average(f, kappa, q0, mu):
    pts = [-mp.inf] + _pts(kappa, q0, mu)
    return mp.quad(lambda t: f(t) * _phi(t), pts)


def jamming_parts(mu, q0, kappa):
    """(ent, en) with Phi(mu, q0; alpha) = ent + alpha * en (see the module docstring)."""
    mu, q0, kappa = mp.mpf(mu), mp.mpf(q0), mp.mpf(kappa)
    s = 1 - q0
    ent = mp.log(1 + mu * s) / (2 * mu) + q0 / (2 * (1 + mu * s))
    sq0, b2 = mp.sqrt(q0), 1 - q0

    def lny(t):
        ys, yc = _y_parts(kappa - sq0 * t, mu, b2)
        return mp.log(ys + yc)

    return ent, _t_average(lny, kappa, q0, mu) / mu


def jamming_alpha(mu, q0, kappa):
    """A(mu, q0) = -ent/en: the capacity bound at the trial (mu, q0); its minimum is the 1RSB value."""
    ent, en = jamming_parts(mu, q0, kappa)
    return -ent / en


def jamming_phi(mu, q0, alpha, kappa):
    ent, en = jamming_parts(mu, q0, kappa)
    return ent + mp.mpf(alpha) * en


def contact_index(mu, q0, alpha, kappa):
    """c = alpha P(g > 0 | cluster): the leading order of FPSUZ Eq. (62), 1RSB isostaticity index."""
    mu, q0, kappa = mp.mpf(mu), mp.mpf(q0), mp.mpf(kappa)
    sq0, b2 = mp.sqrt(q0), 1 - q0

    def post(t):
        ys, yc = _y_parts(kappa - sq0 * t, mu, b2)
        return yc / (ys + yc)

    return mp.mpf(alpha) * _t_average(post, kappa, q0, mu)


# ---- unscaled 1RSB entropy (FPSUZ Eq. (59)) and its replicon (Eqs. (62)-(63)), float64 ------------
# A code path that shares nothing with the closed-form Y above: nested composite Gauss-Legendre
# quadrature over (t, u) of the entropy exactly as the paper writes it.
def _gl(pts, n=24):
    x, wt = np.polynomial.legendre.leggauss(n)
    pts = sorted(set(pts))
    ts = np.concatenate([0.5 * (hi - lo) * x + 0.5 * (hi + lo) for lo, hi in zip(pts[:-1], pts[1:])])
    ws = np.concatenate([0.5 * (hi - lo) * wt for lo, hi in zip(pts[:-1], pts[1:])])
    return ts, ws


def _u_nodes(centre, scale, lim=11.0):
    """Gauss-Legendre nodes and Gaussian weights in u, panels refined around `centre`."""
    pts = [-lim, lim] + [centre + s * c for c in (0.0, 1.0, 3.0, 10.0, 30.0, 100.0) for s in (-scale, scale)]
    u, w = _gl([p for p in pts if -lim <= p <= lim])
    return u, w * np.exp(-u**2 / 2) / math.sqrt(2 * math.pi)


def _outer(q0, q1, kappa, fun):
    """E_t fun(t, ...) over panels around t0 = kappa/sqrt(q0); fun gets the t-nodes (vector)."""
    t0, sc = kappa / math.sqrt(q0), math.sqrt(max(1 - q0, 1e-30)) / math.sqrt(q0)
    pts = [-11.0, 11.0, 0.0, t0] + [t0 + s * c * sc for c in (0.1, 1.0, 3.0, 10.0) for s in (-1, 1)]
    t, w = _gl([p for p in pts if -11.0 <= p <= 11.0], 32)
    return float(np.sum(w * np.exp(-t**2 / 2) / math.sqrt(2 * math.pi) * fun(t)))


def _log_h_neg(x):
    """ln H(x) = log_ndtr(-x)."""
    return log_ndtr(-x)


def unscaled_entropy(q0, q1, m, alpha, kappa):
    """FPSUZ Eq. (59) as written (finite 1 - q1), float64; m = 1 is the RS entropy at q0."""
    q0, q1, m, alpha, kappa = (float(x) for x in (q0, q1, m, alpha, kappa))
    lam = 1 - q1 + m * (q1 - q0)
    ent = (m - 1) / (2 * m) * math.log(1 - q1) + math.log(lam) / (2 * m) + q0 / (2 * lam)
    r, w, s0 = math.sqrt(1 - q1), math.sqrt(q1 - q0), math.sqrt(q0)

    def inner(t):
        a = kappa - s0 * t
        out = np.empty_like(t)
        for i, ai in enumerate(a):
            u, wu = _u_nodes(ai / w, r / w)
            out[i] = np.log(np.sum(wu * np.exp(m * _log_h_neg((ai - w * u) / r))))
        return out

    return ent + alpha / m * _outer(q0, q1, kappa, inner)


def unscaled_replicon(q0, q1, m, alpha, kappa):
    """Lambda = 1 - alpha lambda^2 int dh P(1, h) f''(1, h)^2, lambda = 1 - q1 (FPSUZ Eqs. (62)-(63)),
    float64: with the gap g = -h, f''(1, h) = (ln H)''(x)/(1 - q1) at x = g/sqrt(1 - q1) and P(1, h)
    the density of the gap of one constraint in the cluster, weight H(x)^m."""
    q0, q1, m, alpha, kappa = (float(x) for x in (q0, q1, m, alpha, kappa))
    r, w, s0 = math.sqrt(1 - q1), math.sqrt(q1 - q0), math.sqrt(q0)

    def post(t):
        a = kappa - s0 * t
        out = np.empty_like(t)
        for i, ai in enumerate(a):
            u, wu = _u_nodes(ai / w, r / w)
            x = (ai - w * u) / r
            e = np.exp(-x**2 / 2 - _log_h_neg(x)) / math.sqrt(2 * math.pi)       # E = phi/H
            weight = wu * np.exp(m * _log_h_neg(x))
            out[i] = np.sum(weight * (e * (e - x))**2) / np.sum(weight)
        return out

    return 1 - alpha * _outer(q0, q1, kappa, post)


# ---- float64 stage ------------------------------------------------------------------------------
def _gl_panels(kappa, q0, mu, n=32):
    """Composite Gauss-Legendre nodes and Gaussian-weighted weights for E_t over breakpoints."""
    s0, b = math.sqrt(q0), math.sqrt(1 - q0)
    t0 = kappa / s0
    pts = {-11.0, 11.0, 0.0, t0}
    for w in (b / s0, b * math.sqrt(1 + mu * b * b) / s0):
        for c in (3.0, 30.0):
            pts |= {t0 - c * w, t0 + c * w}
    pts = sorted(p for p in pts if -11.0 <= p <= 11.0)
    x, wt = np.polynomial.legendre.leggauss(n)
    ts, ws = [], []
    for lo, hi in zip(pts[:-1], pts[1:]):
        ts.append(0.5 * (hi - lo) * x + 0.5 * (hi + lo))
        ws.append(0.5 * (hi - lo) * wt)
    ts, ws = np.concatenate(ts), np.concatenate(ws)
    return ts, ws * np.exp(-ts**2 / 2) / math.sqrt(2 * math.pi)


def _ln_y_f64(a, mu, b2):
    b = math.sqrt(b2)
    s2 = 1 + mu * b2
    l_sat = log_ndtr(-a / b)
    l_con = -mu * a * a / (2 * s2) - 0.5 * math.log(s2) + log_ndtr(a / (b * math.sqrt(s2)))
    return np.logaddexp(l_sat, l_con)


def _alpha_f64(x, kappa):
    mu, q0 = math.exp(x[0]), 1 - math.exp(x[1])
    if not (_MU_FLOOR <= mu <= 1e9 and 0 < q0 < 1):
        return 1e30
    ts, ws = _gl_panels(kappa, q0, mu)
    en = float(np.sum(ws * _ln_y_f64(kappa - math.sqrt(q0) * ts, mu, 1 - q0))) / mu
    s = 1 - q0
    ent = math.log1p(mu * s) / (2 * mu) + q0 / (2 * (1 + mu * s))
    return -ent / en if en < 0 else 1e30


def _float_minimum(kappa):
    """Minimum of A over (ln mu, ln(1-q0)) in float64: a coarse grid, then Nelder-Mead from its
    best node.  Returns (x, A)."""
    grid = [(lm, lw) for lm in np.linspace(0.0, 14.0, 11) for lw in np.linspace(-0.1, -14.0, 11)]
    vals = [_alpha_f64(x, kappa) for x in grid]
    x0 = grid[int(np.argmin(vals))]
    r = minimize(_alpha_f64, list(x0), args=(kappa,), method="Nelder-Mead",
                 options=dict(xatol=1e-7, fatol=1e-12, maxiter=400))
    return r.x, float(r.fun)


def _float_hessian(x, kappa, h=1e-3):
    """Hessian of A in (ln mu, ln(1-q0)) at x, float64 central differences (relative error ~1e-5)."""
    f = lambda a, b: _alpha_f64([a, b], kappa)
    u, w = float(x[0]), float(x[1])
    f0 = f(u, w)
    fuu = (f(u + h, w) - 2 * f0 + f(u - h, w)) / h**2
    fww = (f(u, w + h) - 2 * f0 + f(u, w - h)) / h**2
    fuw = (f(u + h, w + h) - f(u + h, w - h) - f(u - h, w + h) + f(u - h, w - h)) / (4 * h * h)
    return [[fuu, fuw], [fuw, fww]]


# ---- mpmath polish and audit --------------------------------------------------------------------
def _alpha_uw(u, w, kappa):
    return jamming_alpha(mp.exp(u), 1 - mp.exp(w), kappa)


def _grad(u, w, kappa):
    """grad A in (u, w) = (ln mu, ln(1-q0)): central differences, h = 1e-7 (truncation ~1e-14)."""
    h = mp.mpf("1e-7")
    f = lambda a, b: _alpha_uw(a, b, kappa)
    return [(f(u + h, w) - f(u - h, w)) / (2 * h), (f(u, w + h) - f(u, w - h)) / (2 * h)]


def _newton(u, w, kappa, hess, what):
    """Quasi-Newton on grad A = 0 with the float64 Hessian `hess` (relative error ~1e-5: each
    step cuts the gradient by that factor); the value A is stationary, so a root accurate to
    1e-12 in (u, w) fixes A to ~1e-24."""
    det = hess[0][0] * hess[1][1] - hess[0][1]**2
    for _ in range(8):
        g = _grad(u, w, kappa)
        if abs(g[0]) + abs(g[1]) < _alpha_uw(u, w, kappa) * mp.mpf(10)**(-mp.mp.dps + 8):   # relative to A
            return u, w, g
        u -= (hess[1][1] * g[0] - hess[0][1] * g[1]) / det
        w -= (hess[0][0] * g[1] - hess[0][1] * g[0]) / det
    raise Refusal("no-solution:rsb-polish", f"{what}: quasi-Newton on grad A did not converge")


def _audit_residuals(mu, q0, alpha, kappa):
    """(Phi, dPhi/dmu, dPhi/dq0) at (mu, q0, alpha): central differences at working precision."""
    h = mp.mpf("1e-9")
    phi = lambda m, q: jamming_phi(m, q, alpha, kappa)
    return (phi(mu, q0), (phi(mu * (1 + h), q0) - phi(mu * (1 - h), q0)) / (2 * mu * h),
            (phi(mu, q0 + h) - phi(mu, q0 - h)) / (2 * h))


@lru_cache(maxsize=None)
def _jamming_cached(kappa_str):
    """The audited root at the module's own precision (_DPS), whatever the caller's mp.dps: the quasi-Newton
    tolerance and the audit are tuned to the finite-difference floor of this precision."""
    with mp.workdps(_DPS):
        k = mp.mpf(kappa_str)
        kf = float(k)
        what = f"1RSB jamming point, kappa = {kappa_str}"
        x, a_f = _float_minimum(kf)
        a_rs = float(1 / _den(k))
        if not a_f < a_rs * (1 - 1e-13) or x[0] <= math.log(_MU_FLOOR) + 1e-3:
            raise Refusal("no-solution:rsb-branch",
                          f"{what}: the float64 search finds no nontrivial minimum of A(mu, q0) "
                          f"below the RS value (A_min = {a_f:.12g}, RS = {a_rs:.12g})")
        hess = _float_hessian(x, kf)
        tr, det = hess[0][0] + hess[1][1], hess[0][0] * hess[1][1] - hess[0][1]**2
        lo = tr / 2 - math.sqrt(max(tr * tr / 4 - det, 0.0))
        hi = tr / 2 + math.sqrt(max(tr * tr / 4 - det, 0.0))
        if not (lo > 0 and 0 < 1 - math.exp(x[1]) < 1):
            raise Refusal("no-solution:rsb-branch", f"{what}: the stationary point is not a "
                          f"minimum (Hessian eigenvalues {lo:.4g}, {hi:.4g})")
        u, w, g = _newton(mp.mpf(float(x[0])), mp.mpf(float(x[1])), k, hess, what)
        mu, q0 = mp.exp(u), 1 - mp.exp(w)
        alpha = _alpha_uw(u, w, k)
        if abs(alpha / a_f - 1) > 1e-7 or abs(mu / math.exp(x[0]) - 1) > 1e-3:
            raise Refusal("no-solution:rsb-audit", f"{what}: mpmath root {mp.nstr(alpha, 12)} "
                          f"disagrees with the float64 stage {a_f:.12g}")
        res = _audit_residuals(mu, q0, alpha, k)
        if max(abs(r) for r in res) > _AUDIT_TOL:
            raise Refusal("no-solution:rsb-audit", f"{what}: stationarity residuals "
                          f"{[mp.nstr(r, 3) for r in res]}")
        c = contact_index(mu, q0, alpha, k)
        return RSBJamming(k, alpha, 1 / _den(k), q0, mu, c, 1 - c, (lo, hi), res)


def onersb_jamming(kappa) -> RSBJamming:
    """The 1RSB jamming (SAT-UNSAT) point of the spherical perceptron at margin kappa < 0.
    Refusals: precondition:finite-kappa; precondition:kappa-range (kappa >= 0: use the RS value,
    1RSB reduces to it; -0.02 < kappa < 0 and kappa < -3: outside the validated float64 stage);
    no-solution:rsb-branch / :rsb-polish / :rsb-audit (no nontrivial minimum, Newton failure, or
    a failed audit of the root)."""
    k = mp.mpf(kappa)
    if not mp.isfinite(k):
        raise Refusal("precondition:finite-kappa", f"kappa = {kappa}")
    if k >= 0:
        raise Refusal("precondition:kappa-range", f"kappa = {kappa} >= 0: the 1RSB jamming point "
                      "reduces to the RS capacity 1/D(kappa) (mu -> 0); use gardner_capacity")
    if not _KAPPA_MIN <= float(k) <= _KAPPA_NEAR_ZERO:
        raise Refusal("precondition:kappa-range", f"kappa = {kappa}: the 1RSB search is validated "
                      f"on [{_KAPPA_MIN}, {_KAPPA_NEAR_ZERO}]")
    return _jamming_cached(mp.nstr(k, 30))
