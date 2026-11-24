"""Stieltjes transforms and spectral densities of the Wigner and Wishart ensembles (KZ Ch. 3, Ex. 3.1).

Conventions (KZ Sec. 3.1-3.4):
  wigner   A = (G + G^T)/sqrt(2N), G_ij i.i.d. N(0, 1) (KZ p. 49); semicircle on [-2, 2].
  wishart  Sigma = X X^T / N with X an M x N matrix of i.i.d. N(0, 1), alpha = M/N (KZ Ex. 3.1): Sigma is M x M,
           so alpha is (matrix dimension) / (number of columns).  This alpha is the gamma = p/n of
           learning/ridge.py (X^T X / n is p x p), NOT the regression chapters' alpha = n/p = 1/gamma: the
           Ch. 3 Wishart alpha maps to gamma_ridge = alpha_notes (tests/test_random_matrix.py checks it both ways).
  S(z) = (1/N) sum_i 1/(lambda_i - z) (KZ Sec. 3.1: S_A(lambda) = -(1/N) sum 1/(lambda - lambda_i)), analytic off
  the support, Im S > 0 for Im z > 0, nu(x) = lim_{eps -> 0+} Im S(x + i eps) / pi.
Replica route (KZ Sec. 3.2, Eq. at p. 53: the RS saddle qhat = 1/q, q = -qhat - lambda, i.e. q^2 + z q + 1 = 0;
Ex. 3.1 for Wishart), certified in acb: the root that decays like -1/z at infinity, written with the branch
sqrt(z - a) sqrt(z - b) (analytic off [a, b]):
    S_wigner(z) = (-z + sqrt(z - 2) sqrt(z + 2)) / 2,
    S_wishart(z) = (1 - alpha - z + sqrt(z - l+) sqrt(z - l-)) / (2 alpha z),   l+- = (1 +- sqrt(alpha))^2;
the density is Im S(x + i0)/pi from the same formula on the real axis (the principal sqrt of a negative real
is i sqrt|.|); edges are the roots of the discriminant; for alpha > 1 the Wishart law has an atom 1 - 1/alpha at
0 (Ex. 3.1, item 4: the Stieltjes transform of delta(lambda) is -1/lambda), the continuous part is the density.
Cavity route (KZ Sec. 3.3: S_{N+1}(z) = -1/(z + S_N(z)); for Wishart the Schur-complement analogue,
derived here: S <- 1/(-z + 1/(1 + alpha S))), mpmath: both recursions are Moebius maps, so N -> oo is taken by
repeated squaring of the 2 x 2 matrix (2^k steps) from S_0 = 0 at Im z > 0; the density uses z = x + i 1e-30
at 80 digits (error O(1e-30)).  The cavity edges are where the recursion turns parabolic (trace^2 = 4 det):
the attracting real fixed point outside the support becomes a rotation (complex fixed points) inside it.

  stieltjes_wigner(z), stieltjes_mp(z, alpha) -> acb (replica);  *_cavity(z, ...) -> mpc
  density(kind, x, route="replica", **p) -> Ball (replica) | mpf (cavity);  edges(kind, route, **p) -> (Ball, Ball)
  atom_at_zero(kind, **p) -> Fraction;  sample_spectrum(kind, N, seed, alpha=None) -> eigenvalues (numpy)
_HOOKS["edge"] (2: the semicircle radius in the replica formula): planted-error hook.
"""
from fractions import Fraction

import mpmath as mp
import numpy as np

from ..core import Refusal
from ..lattice.ball import arb_q, ball_of, baller, rational

KINDS = ("wigner", "wishart")
ROUTES = ("replica", "cavity")
_HOOKS = {"edge": 2}
_PREC = 256


def _kind(kind):
    if kind not in KINDS:
        raise Refusal("unsupported:ensemble", f"kind = {kind!r}: one of {', '.join(KINDS)}")
    return kind


def _alpha(kind, alpha):
    if kind == "wigner":
        if alpha is not None:
            raise Refusal("precondition:known-parameter", "alpha belongs to the wishart ensemble")
        return None
    if alpha is None:
        raise Refusal("precondition:alpha", "the wishart ensemble needs alpha = M/N > 0")
    return rational(alpha, "alpha")


def _acb(z):
    from flint import acb
    zc = complex(z) if not isinstance(z, (int, str, Fraction)) else None
    if zc is None:
        return acb(arb_q(rational(z, "z", positive=False)))
    return acb(zc.real, zc.imag)


# ---- replica route (certified) ------------------------------------------------------------------------
def _s_wigner(z):
    r = _HOOKS["edge"]
    return (-z + (z - r).sqrt() * (z + r).sqrt()) / 2


def _s_mp(z, aq):
    a = arb_q(aq)
    lp, lm = (1 + a.sqrt()) ** 2, (1 - a.sqrt()) ** 2
    return (1 - a - z + (z - lp).sqrt() * (z - lm).sqrt()) / (2 * a * z)


def stieltjes_wigner(z):
    with baller().hygiene.ctx_guard(prec=_PREC):
        return _s_wigner(_acb(z))


def stieltjes_mp(z, alpha):
    aq = _alpha("wishart", alpha)
    with baller().hygiene.ctx_guard(prec=_PREC):
        return _s_mp(_acb(z), aq)


def _density_replica(kind, x, aq):
    from flint import arb
    xq = rational(x, "x", positive=False)
    with baller().hygiene.ctx_guard(prec=_PREC):
        if kind == "wishart" and xq <= 0:
            return ball_of(arb(0))        # continuous part only; the atom at 0 is atom_at_zero
        s = _s_wigner(_acb(xq)) if kind == "wigner" else _s_mp(_acb(xq), aq)
        return ball_of(s.imag / arb.pi())


def _edges_replica(kind, aq):
    with baller().hygiene.ctx_guard(prec=_PREC):
        if kind == "wigner":
            r = arb_q(_HOOKS["edge"])
            return ball_of(-r), ball_of(r)
        a = arb_q(aq)
        return ball_of((1 - a.sqrt()) ** 2), ball_of((1 + a.sqrt()) ** 2)    # discriminant (z-a-1)^2 - 4a


# ---- cavity route --------------------------------------------------------------------------------------
def _moebius(z, kind, alpha):
    if kind == "wigner":
        return mp.matrix([[0, -1], [1, z]])
    a = mp.mpf(alpha)
    return mp.matrix([[a, 1], [-a * z, 1 - z]])


def _iterate(Mz, squarings):
    """The Moebius map applied 2^squarings times to S_0 = 0: M^n (0) = M^n[0, 1] / M^n[1, 1]."""
    M = Mz
    for _ in range(squarings):
        M = M * M
        M = M / max(abs(M[i, j]) for i in range(2) for j in range(2))
    return M[0, 1] / M[1, 1]


def stieltjes_wigner_cavity(z, squarings=120, dps=80):
    with mp.workdps(dps):
        z = mp.mpc(z)
        if not z.imag > 0:
            raise Refusal("precondition:upper-half-plane", f"z = {z}: the cavity recursion runs at Im z > 0")
        return _iterate(_moebius(z, "wigner", None), squarings)


def stieltjes_mp_cavity(z, alpha, squarings=120, dps=80):
    aq = _alpha("wishart", alpha)
    with mp.workdps(dps):
        z = mp.mpc(z)
        if not z.imag > 0:
            raise Refusal("precondition:upper-half-plane", f"z = {z}: the cavity recursion runs at Im z > 0")
        return _iterate(_moebius(z, "wishart", mp.mpf(aq.numerator) / aq.denominator), squarings)


def _density_cavity(kind, x, aq):
    xq = rational(x, "x", positive=False)
    if kind == "wishart" and xq <= 0:
        return mp.mpf(0)
    with mp.workdps(80):
        z = mp.mpc(mp.mpf(xq.numerator) / xq.denominator, mp.mpf("1e-30"))
        s = stieltjes_wigner_cavity(z) if kind == "wigner" else stieltjes_mp_cavity(z, aq)
        return +(s.imag / mp.pi)


def _edges_cavity(kind, aq):
    """Where the recursion's Moebius matrix is parabolic: trace(z)^2 = 4 det, solved for real z."""
    with baller().hygiene.ctx_guard(prec=_PREC):
        if kind == "wigner":                               # trace z, det 1
            two = 2 * arb_q(1).sqrt()
            return ball_of(-two), ball_of(two)
        a = arb_q(aq)                                      # trace 1 + a - z, det a
        return ball_of(1 + a - 2 * (a).sqrt()), ball_of(1 + a + 2 * (a).sqrt())


# ---- front doors ---------------------------------------------------------------------------------------
def _route(route):
    if route not in ROUTES:
        raise Refusal("unsupported:route", f"route = {route!r}: one of {', '.join(ROUTES)}")
    return route


def density(kind, x, route="replica", alpha=None):
    kind, route = _kind(kind), _route(route)
    aq = _alpha(kind, alpha)
    return (_density_replica if route == "replica" else _density_cavity)(kind, x, aq)


def edges(kind, route="replica", alpha=None):
    kind, route = _kind(kind), _route(route)
    aq = _alpha(kind, alpha)
    return (_edges_replica if route == "replica" else _edges_cavity)(kind, aq)


def atom_at_zero(kind, alpha=None):
    aq = _alpha(_kind(kind), alpha)
    return Fraction(0) if aq is None or aq <= 1 else 1 - 1 / aq


def sample_spectrum(kind, N, seed, alpha=None):
    """Eigenvalues of one sampled matrix (numpy eigvalsh): wigner N x N; wishart M x M, M = round(alpha N)."""
    aq = _alpha(_kind(kind), alpha)
    if isinstance(N, bool) or not isinstance(N, int) or N < 2:
        raise Refusal("precondition:size", f"N = {N!r}: need an int >= 2")
    rng = np.random.default_rng(seed)
    if kind == "wigner":
        G = rng.standard_normal((N, N))
        return np.linalg.eigvalsh((G + G.T) / np.sqrt(2 * N))
    M = round(aq * N)
    X = rng.standard_normal((M, N))
    return np.linalg.eigvalsh(X @ X.T / N)


__all__ = ["KINDS", "ROUTES", "atom_at_zero", "density", "edges", "sample_spectrum", "stieltjes_mp",
           "stieltjes_mp_cavity", "stieltjes_wigner", "stieltjes_wigner_cavity"]
