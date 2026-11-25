"""Rank-one spiked Wigner model: Bayes-optimal state evolution, the RS potential, the threshold,
and finite-N Bayes AMP with spectral or planted initialisation.

Model (Montanari & Venkataramanan, Ann. Statist. 49, 321 (2021), arXiv:1711.01682, Eq. (2.1)):
    Y = (lambda / N) x x^T + W,   W ~ GOE(N): W_ij ~ N(0, 1/N) for i < j, W_ii ~ N(0, 2/N),
x_i i.i.d. from a zero-mean, unit-variance prior (PRIORS: gaussian N(0, 1), rademacher +-1).
The top eigenvalue of Y separates from the semicircle edge 2 iff lambda > 1 (to lambda + 1/lambda;
Feral-Peche 2007, Benaych-Georges-Nadakuditi 2011, as cited by Lelarge-Miolane, not fetched).
Lelarge & Miolane (Probab. Theory Relat. Fields 173, 859 (2019), arXiv:1611.03888) and Deshpande,
Abbe & Montanari (arXiv:1507.08685) write Y = sqrt(lambda_LM / n) X X^T + Z, Z of unit variance:
lambda_LM = lambda^2.  Barbier et al. (arXiv:1606.04142) write w = s s^T / sqrt(n) + sqrt(Delta) z:
Delta = 1 / lambda^2.

Scalar channel y = gamma X + sqrt(gamma) G (MV Eq. (2.26)), denoiser F(y; gamma) = E[X | y]
(the PRIORS hook: gaussian y / (1 + gamma), rademacher tanh(y)); mmse(gamma) = E (X - F)^2 by a
trapezoid rule over G (and Gauss-Hermite over X for the gaussian prior); the SE uses
1 - mmse = 2 E[X F] - E[F^2] (scalar_overlap: unit variance, no cancellation against E X^2 = 1, so
it keeps ~1e-15 relative accuracy as gamma -> 0).  State evolution (MV (2.24)):
    gamma_{t+1} = lambda^2 (1 - mmse(gamma_t)),
whose fixed points are found here by bracketing the roots of lambda^2 (1 - mmse(gamma)) - gamma on
[1e-9 lambda^2, lambda^2] (gamma = 0 is always one; roots below the floor are not searched, the
quadrature cannot resolve them: |lambda^2 - 1| below about 1e-9 reports gamma = 0).  At the largest fixed point gamma* (the limit from the spectral
start gamma_0 = lambda^2 - 1 and from the informative start lambda^2; MV Thm 2 states gamma_ALG,
the smallest positive root: they coincide because several nontrivial roots are refused):
    q = gamma*/lambda^2   (LM's q*: the overlap x.X of a posterior sample, Nishimori),
    overlap = sqrt(q)     (cosine |<xhat, x>| / (|xhat| |x|), MV Eq. (2.28)),
    mmse = 1 - q          (vector MSE up to the global sign, MV Eq. (2.29)),
    matrix_mmse = 1 - q^2 (LM Theorem 1; for the gaussian prior LM Eq. (10)).
`bayes` is True when gamma* is stationary for, and the lowest stationary value of, the RS potential
    Psi(gamma) = lambda^2/4 + gamma^2/(4 lambda^2) - gamma/2 + I(gamma)
(DAM Eq. (9) with lambda_DAM = lambda^2, minimised: DAM Lemma 3.3; MV Prop. 2.2 says "maximizes",
which contradicts DAM's min and the sign of Psi'' at 0: we follow DAM), with I(gamma) the scalar
mutual information from its cited closed form (gaussian 1/2 log(1 + gamma), LM Sec. 2.3; rademacher
gamma - E log cosh(gamma + sqrt(gamma) Z), DAM Eq. (6)) by QUADPACK, independent of the hook.

AMP (MV Eqs. (2.20)-(2.21), Bayes AMP): x^0 = sqrt(N lambda^2 (lambda^2 - 1)) phi_1,
    x^{t+1} = Y f_t(x^t) - b_t f_{t-1}(x^{t-1}),  f_t = lambda F(.; gamma_t),  b_t = <f_t'(x^t)>,
estimate xhat^t = F(x^t; gamma_t).  For the spectral start f_{-1} = x^0 / lambda (x^0 as used, with the (2.20)
scaling), so x^1 = Y f_0(x^0) - b_0 x^0 / lambda: the convention of the proof of MV Theorem 1 (App. A.2, after Eqs.
(A.19)-(A.21): "E{L f(X~_{-1}; -1)} = 1/lambda", L the limit of sqrt(N) phi_1).  Sec. 6's "all-zeros" f_{-1} is for
the general algorithm (Eqs. (6.5)-(6.6)) and does not apply to the spectral start, whose x^0 depends on W: with a
linear f_0(x) = s x, Y f_0(x^0) -> s (lambda + 1/lambda) x^0 while SE needs lambda s x^0, and b_0 f_{-1} = s x^0 /
lambda is the difference.  The planted start keeps f_{-1} = 0 (x^0 independent of W given the truth).  MV Theorem 2 takes gamma_t from SE; here gamma_t is
estimated from the iterate (|x^t|^2 / N -> gamma_t^2 + gamma_t), so AMP does not call this
module's SE: a consistent plug-in, not literally the theorem's algorithm.  Planted start (init="planted", any lambda > 0, no eigsh): x^0 = gamma0 x + sqrt(gamma0) g with the
truth x and g ~ N(0, 1) from default_rng([seed, 4]), the scalar channel at gamma0, so SE starts at gamma_0 = gamma0
(spiked_diluted.diluted_se_trajectory; this module does not call it).  `reaction=False` drops
every Onsager term, the first one included (non-linear power iteration, the positive control).  Finite-N values are
statistical (99% Student-t bands over named seeds), never certified.
"""
import math
from dataclasses import dataclass

import mpmath as mp
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy import integrate, optimize

from ..amp._stats import Band, student_band
from ..core import Refusal
from ..csp import _checks

# G-quadrature: trapezoid on [-10, 10], h = 0.01 (exponentially accurate for these analytic
# integrands: the poles of tanh(gamma + sqrt(gamma) z) sit at |Im z| = pi / (2 sqrt(gamma)); a
# Gauss-Hermite rule with 101 nodes was off by 3e-7 at gamma = 3.7, by 2.7e-6 at gamma = 9)
_Z = np.linspace(-10.0, 10.0, 2001)
_WZ = np.exp(-_Z**2 / 2) / math.sqrt(2 * math.pi) * (_Z[1] - _Z[0])
_WZ = _WZ / _WZ.sum()     # weights summing to 1 to rounding (the raw sum is 1 + O(1e-12))
# lower edge of the root bracketing, in units of lambda^2: at lambda = 1 the true g = -gamma^2
# (gaussian) must dominate the quadrature noise, about 1e-17 sqrt(gamma) from odd moments that do not
# cancel exactly in floating point; at 1e-9 the margin is about 3000.  A root below it (|lambda^2 - 1|
# below about 1e-9) is not searched and the SE reports gamma = 0.
_FLOOR = 1e-9
_onsager_scale = 1.0    # private: tests plant 0.5 to show the band resolves the Onsager term
_spectral_first_step = True   # private: tests plant False (f_{-1} = 0 for the spectral start) to show the t = 1 band
                              # resolves MV App. A.2's first Onsager term
_SPECTRAL = ("AMP needs lambda > 1: the spectral start of MV Theorem 2 exists only above the "
             "spectral threshold (below it the Bayes estimate is trivial, q = 0)")


@dataclass(frozen=True)
class Prior:
    """Zero-mean, unit-variance prior: X-quadrature (nodes, weights), the posterior-mean denoiser
    F(y, gamma) for y = gamma X + sqrt(gamma) G and its y-derivative dF, sample(rng, n)."""
    name: str
    nodes: np.ndarray
    weights: np.ndarray
    F: object
    dF: object
    sample: object


def _gauss_prior():
    x, w = hermegauss(21)
    return Prior("gaussian", x, w / w.sum(), lambda y, g: y / (1 + g),
                 lambda y, g: np.full_like(y, 1 / (1 + g)), lambda rng, n: rng.normal(size=n))


def _rademacher_prior():
    return Prior("rademacher", np.array([-1.0, 1.0]), np.array([0.5, 0.5]), lambda y, g: np.tanh(y),
                 lambda y, g: 1 - np.tanh(y)**2,
                 lambda rng, n: rng.choice(np.array([-1.0, 1.0]), size=n))


PRIORS = {"gaussian": _gauss_prior(), "rademacher": _rademacher_prior()}


_DILUTED = "diluted_rademacher"


def _diluted_rho(rho):
    """rho in (0, 1] as a float, else the precondition:rho refusal."""
    bad = Refusal("precondition:rho", f"rho={rho!r}: the diluted prior needs a number in (0, 1]")
    if isinstance(rho, bool) or rho is None:
        raise bad
    try:
        r = float(rho)
    except (TypeError, ValueError):
        raise bad from None
    if not (math.isfinite(r) and 0 < r <= 1):
        raise bad
    return r


_DILUTED_CACHE = {}


def diluted_prior(rho):
    """Unit-variance diluted Rademacher prior x in {-a, 0, a}, a = 1/sqrt(rho), P(0) = 1 - rho (cached per rho).

    Denoiser on y = gamma x + sqrt(gamma) G, in closed form with the exponentials divided through by e^{u},
    u = |a y|:  F = a rho sinh(a y) / (rho cosh(a y) + (1 - rho) e^{gamma a^2 / 2}),  dF = Var[x | y] =
    a^2 rho cosh(a y) / (same) - F^2.  Written in numpy only: nothing from bayes/ or spiked_diluted."""
    r = _diluted_rho(rho)
    if r in _DILUTED_CACHE:
        return _DILUTED_CACHE[r]
    a = 1 / math.sqrt(r)

    log_half = math.log(r / 2)
    log_zero = math.log1p(-r) if r < 1 else -math.inf     # the zero atom is dropped exactly at rho = 1

    def _exp(x):
        return np.exp(np.maximum(x, -700.0))              # e^-700 is normal: no underflow trap

    def parts(y, g):
        # log-sum-exp over the atoms {+a, -a, 0}: log-weights (common factors dropped) log(rho/2) +- a y - g a^2/2
        # and log(1 - rho); every exponent after the shift by their maximum is <= 0
        u = np.abs(a * np.asarray(y, dtype=float))
        s = g * a * a / 2
        lp, lm = log_half + u - s, log_half - u - s
        m = np.maximum(lp, log_zero)
        ep, em, e0 = _exp(lp - m), _exp(lm - m), (_exp(log_zero - m) if r < 1 else 0.0)
        den = ep + em + e0
        return a * (ep - em) / den, a * a * (ep + em) / den

    def F(y, g):
        return np.sign(y) * parts(y, g)[0]

    def dF(y, g):
        f, second = parts(y, g)
        f = np.where(np.abs(f) < 1e-150, 0.0, f)          # f^2 would underflow (harmless, but trapped)
        return np.maximum(second - f * f, 0.0)

    def sample(rng, n):
        mask = rng.random(n) < r
        return a * mask * rng.choice(np.array([-1.0, 1.0]), size=n)

    P = Prior(_DILUTED, np.array([-a, 0.0, a]), np.array([r / 2, 1 - r, r / 2]), F, dF, sample)
    _DILUTED_CACHE[r] = P
    return P


def _prior(name, rho=None):
    if name == _DILUTED:
        return diluted_prior(rho)
    if not isinstance(name, str) or name not in PRIORS:
        raise Refusal("precondition:known-prior",
                      f"prior={name!r}; known: {', '.join(PRIORS)}, {_DILUTED} (zero mean, unit variance)")
    return PRIORS[name]


def _snr(x):
    if isinstance(x, bool) or x is None:
        raise Refusal("precondition:snr", f"snr={x!r}: lambda must be a number >= 0")
    try:
        v = mp.mpf(x)
    except (TypeError, ValueError):
        raise Refusal("precondition:snr", f"snr={x!r}: lambda must be a number >= 0") from None
    if not mp.isfinite(v) or v < 0:
        raise Refusal("precondition:snr", f"snr={x!r}: lambda must be a finite number >= 0")
    return v


def scalar_mmse(prior, gamma):
    """E (X - F(gamma X + sqrt(gamma) G; gamma))^2 by quadrature over X and G (binary64)."""
    P = _prior(prior)
    g = float(gamma)
    y = g * P.nodes[:, None] + math.sqrt(g) * _Z[None, :]
    err = (P.nodes[:, None] - P.F(y, g))**2
    return float(P.weights @ err @ _WZ)


def scalar_overlap(prior, gamma):
    """1 - mmse(gamma) for a unit-variance prior, as 2 E[X F] - E[F^2] (no cancellation)."""
    P = _prior(prior)
    g = float(gamma)
    y = g * P.nodes[:, None] + math.sqrt(g) * _Z[None, :]
    F = P.F(y, g)
    return float(P.weights @ (2 * P.nodes[:, None] * F - F * F) @ _WZ)


def _mutual_info(prior, g):
    if prior == "gaussian":
        return 0.5 * math.log1p(g)
    if g == 0:
        return 0.0
    s = math.sqrt(g)

    def f(z):
        a = abs(g + s * z)
        return math.exp(-z * z / 2) * (a + math.log1p(math.exp(-2 * a)) - math.log(2))
    v, _ = integrate.quad(f, -math.inf, math.inf, epsabs=1e-15, epsrel=1e-13, limit=200)
    return g - v / math.sqrt(2 * math.pi)


def spiked_potential(snr, gamma, prior="gaussian"):
    """RS potential Psi(gamma; lambda) (DAM Eq. (9), lambda_DAM = lambda^2), minimised over gamma."""
    lam = float(_snr(snr))
    _prior(prior)
    g = float(gamma)
    if lam == 0:
        raise Refusal("precondition:snr", "the potential needs lambda > 0")
    if not math.isfinite(g) or g < 0:
        raise Refusal("precondition:gamma", f"gamma = {gamma!r}: the scalar SNR must be >= 0")
    return mp.mpf(lam**2 / 4 + g**2 / (4 * lam**2) - g / 2 + _mutual_info(prior, g))


@dataclass(frozen=True)
class SEResult:
    snr: object
    prior: str
    gamma: float
    q: float
    overlap: float
    mmse: float
    matrix_mmse: float
    fixed_points: tuple
    bayes: bool
    potential: float
    note: str = ""
    below_floor: bool = False     # lambda > 1 but no root above the search floor: gamma = 0 is a floor


def _roots(lam2, prior):
    g = lambda x: lam2 * scalar_overlap(prior, x) - x
    grid = np.concatenate([lam2 * np.geomspace(_FLOOR, 1e-2, 25), lam2 * np.linspace(1e-2, 1, 40)[1:]])
    vals = [g(x) for x in grid]
    out = [0.0]
    for (x0, v0), (x1, v1) in zip(zip(grid, vals), zip(grid[1:], vals[1:])):
        if v0 == 0:
            out.append(float(x0))
        elif v0 * v1 < 0:
            out.append(optimize.brentq(g, x0, x1, xtol=1e-15, rtol=1e-15, maxiter=200))
    return tuple(out)


def spiked_state_evolution(snr, prior="gaussian", rho=None):
    """Fixed points of MV Eq. (2.24); reports the largest (see the module docstring).  prior="diluted_rademacher"
    (with rho) goes to inference/spiked_diluted.py: every fixed point, both starts, the hard phase (a DilutedSE)."""
    if prior == "diluted_rademacher":
        from .spiked_diluted import diluted_state_evolution
        return diluted_state_evolution(_snr(snr), rho)
    if rho is not None:
        raise Refusal("precondition:known-parameter", f"rho = {rho!r} applies to prior diluted_rademacher only, "
                      f"not {prior!r}")
    lam = _snr(snr)
    _prior(prior)
    L2 = float(lam)**2
    if L2 == 0:
        return SEResult(lam, prior, 0.0, 0.0, 0.0, 1.0, 1.0, (0,), True, 0.0, "lambda = 0: no signal")
    roots = _roots(L2, prior)
    if len(roots) > 2:
        raise Refusal("ambiguous:several-fixed-points",
                      f"lambda = {mp.nstr(lam, 8)}, prior {prior}: nontrivial SE fixed points "
                      f"{[f'{r:.10g}' for r in roots[1:]]}; this module reports only the unique one")
    gs = roots[-1]
    psi = {r: float(spiked_potential(lam, r, prior)) for r in roots}
    h = min(1e-5 * max(gs, 1e-3), gs / 2)       # central difference inside gamma >= 0
    stationary = gs == 0 or abs(float(spiked_potential(lam, gs + h, prior)) -
                                float(spiked_potential(lam, gs - h, prior))) / (2 * h) < 1e-6
    bayes = bool(stationary and psi[gs] <= min(psi.values()) + 1e-13)
    q = gs / L2
    below = bool(lam > 1 and len(roots) == 1)
    note = (f"largest SE fixed point gamma* = {gs:.12g} of {len(roots)} (gamma = 0 included); "
            f"{'' if bayes else 'NOT '}the global minimum of the RS potential (DAM Eq. (9))")
    if below:
        note = (f"lambda > 1 and below the root-search floor (gamma < 1e-9 lambda^2): reported gamma = 0 is a floor, not the fixed point (LM: q* > 0 for every lambda > 1; the nontrivial root "
                "gamma* ~ lambda^2 - 1 is not resolved for 1 < lambda < about 1 + 5e-10)")
    roots = tuple(0 if r == 0 else r for r in roots)
    return SEResult(lam, prior, gs, q, math.sqrt(q), 1 - q, 1 - q * q, roots, bayes, psi[gs], note,
                    below)


def spiked_threshold(prior="gaussian"):
    """lambda_c = (-mmse'(0))^(-1/2): where gamma = 0 stops being a stable SE fixed point
    (slope lambda^2 (-mmse'(0)) = 1), by Richardson extrapolation of (1 - mmse(h))/h = scalar_overlap(h)/h (for a prior of variance v the
    same expression is 2 E[XF] - E[F^2] = v - mmse, so a planted variance shows up)."""
    _prior(prior)
    D = lambda h: scalar_overlap(prior, h) / h
    hs = (1e-4, 5e-5, 2.5e-5)
    r1 = [2 * D(hs[i + 1]) - D(hs[i]) for i in range(2)]
    slope = (4 * r1[1] - r1[0]) / 3
    return mp.mpf(1 / math.sqrt(slope))


# ---- finite N --------------------------------------------------------------------------------
def _amp_prior(prior, rho):
    """The finite-N prior: rho is the diluted prior's parameter and is refused with any other prior."""
    if rho is not None and prior != _DILUTED:
        raise Refusal("precondition:known-parameter", f"rho = {rho!r} applies to prior {_DILUTED} only, not {prior!r}")
    return _prior(prior, rho)


def spiked_matrix(N, snr, prior, seed, rho=None):
    """(Y, x): W ~ GOE(N) from default_rng([seed, 1]), x ~ prior (with rho for diluted_rademacher) from
    default_rng([seed, 2])."""
    if isinstance(N, bool) or not isinstance(N, int) or N < 10:
        raise Refusal("precondition:size", f"N={N!r} must be an int >= 10")
    seed = _checks.seed(seed)
    lam = float(_snr(snr))
    P = _amp_prior(prior, rho)
    A = np.random.default_rng([seed, 1]).normal(size=(N, N)) / math.sqrt(N)
    W = (A + A.T) / math.sqrt(2)
    x = P.sample(np.random.default_rng([seed, 2]), N)
    return lam / N * np.outer(x, x) + W, x


@dataclass
class AMPResult:
    xhat: np.ndarray
    x: np.ndarray
    gamma_trace: list            # empirical gamma_t, t = 0..iters
    increments: list             # <xhat^t - xhat^{t-1}, same>, t = 1..iters
    converged: bool
    snr: float = 0.0
    prior: str = ""
    seed: int = 0
    reaction: bool = True
    label: str = "statistical"
    init: str = "spectral"
    gamma0: object = None


def _check_Y(Y):
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2 or Y.shape[0] != Y.shape[1] or Y.shape[0] < 10:
        raise Refusal("precondition:square-matrix", f"Y has shape {Y.shape}; need N x N, N >= 10")
    if not np.all(np.isfinite(Y)):
        raise Refusal("precondition:finite-matrix", "Y has non-finite entries")
    if float(np.max(np.abs(Y - Y.T))) > 1e-12 * (float(np.max(np.abs(Y))) or 1.0):
        raise Refusal("precondition:symmetric-matrix", "the spiked Wigner Y must be symmetric")
    return Y


def _gamma_hat(x):
    return (-1 + math.sqrt(1 + 4 * float(x @ x) / x.size)) / 2


def _gamma0(gamma0):
    """gamma0 as a finite float > 0, else precondition:gamma0."""
    bad = Refusal("precondition:gamma0", f"gamma0 = {gamma0!r}: the planted start needs a finite number > 0 "
                  "(x^0 = gamma0 x + sqrt(gamma0) g)")
    if isinstance(gamma0, bool) or gamma0 is None:
        raise bad
    try:
        g = float(gamma0)
    except (TypeError, ValueError):
        raise bad from None
    if not (math.isfinite(g) and g > 0):
        raise bad
    return g


def _check_init(init, lam, gamma0, x0=None):
    """The start's preconditions; returns gamma0 as a float for the planted start, None for the spectral one."""
    if init == "spectral":
        if gamma0 is not None or x0 is not None:
            raise Refusal("precondition:known-parameter", "gamma0 and x0 apply to init='planted' only; the spectral "
                          "start is x^0 = sqrt(N lambda^2 (lambda^2 - 1)) phi_1")
        if lam <= 1:
            raise Refusal("precondition:snr-above-one", f"lambda = {lam}: {_SPECTRAL}")
        return None
    if init == "planted":
        g = _gamma0(gamma0)
        if lam <= 0:
            raise Refusal("precondition:snr", "the planted start needs lambda > 0")
        return g
    raise Refusal("precondition:known-init", f"init={init!r}; supported: 'spectral' (MV Eq. (2.20)) and 'planted' "
                  "(x^0 = gamma0 x + sqrt(gamma0) g, the scalar channel at gamma0, given the truth x); a start "
                  "independent of Y and x has zero overlap for these zero-mean priors, so SE would stay at gamma = 0")


def _planted_start(x0, gamma0, seed):
    """x^0 = gamma0 x0 + sqrt(gamma0) g, g ~ N(0, 1)^N from default_rng([seed, 4]): the scalar channel
    y = gamma X + sqrt(gamma) G at gamma = gamma0 (SE's gamma_0 = gamma0, independent of W given x0)."""
    g = np.random.default_rng([seed, 4]).normal(size=x0.size)
    return gamma0 * x0 + math.sqrt(gamma0) * g


def spiked_amp(Y, prior, *, snr, iters, seed, init="spectral", reaction=True, tol=1e-12, rho=None, gamma0=None,
               x0=None) -> AMPResult:
    """Bayes AMP on Y (module docstring).  init="spectral": `seed` seeds the eigensolver's start vector (lambda > 1).
    init="planted": x^0 = gamma0 x0 + sqrt(gamma0) g with the truth x0 and g from default_rng([seed, 4]); no eigsh,
    any lambda > 0.  gammas[0] is the empirical gamma of x^0 in both cases."""
    Y = _check_Y(Y)
    P = _amp_prior(prior, rho)
    lam = float(_snr(snr))
    seed = _checks.seed(seed)
    if isinstance(iters, bool) or not isinstance(iters, int) or iters < 1:
        raise Refusal("precondition:positive-iters", f"iters={iters!r} is not a positive int")
    g0 = _check_init(init, lam, gamma0, x0)
    N = Y.shape[0]
    if init == "planted":
        if x0 is None:
            raise Refusal("precondition:x0", "init='planted' needs x0, the planted truth of Y")
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (N,) or not np.all(np.isfinite(x0)):
            raise Refusal("precondition:x0", f"x0 has shape {x0.shape}; need a finite vector of length N = {N}")
        x = _planted_start(x0, g0, seed)
    else:
        from scipy.sparse.linalg import eigsh
        v0 = np.random.default_rng([seed, 3]).normal(size=N)
        _, vec = eigsh(Y, k=1, which="LA", v0=v0, tol=1e-12)
        x = math.sqrt(N * lam**2 * (lam**2 - 1)) * vec[:, 0]
    # f_{-1}: x^0 / lambda for the spectral start (x^0 depends on W; MV App. A.2's convention E{L f(X~_{-1}; -1)} =
    # 1/lambda), 0 for the planted start (x^0 independent of W given x0)
    f_old = x / lam if init == "spectral" and _spectral_first_step else np.zeros(N)
    gam = _gamma_hat(x)
    gammas, incs, xhat = [gam], [], P.F(x, gam)
    for _ in range(iters):
        f = lam * P.F(x, gam)
        b = lam * float(np.mean(P.dF(x, gam))) if reaction else 0.0
        x = Y @ f - _onsager_scale * b * f_old
        f_old = f
        gam = _gamma_hat(x)
        new = P.F(x, gam)
        incs.append(float((new - xhat) @ (new - xhat) / N))
        xhat = new
        gammas.append(gam)
    return AMPResult(xhat, x, gammas, incs, bool(incs[-1] < tol), lam, prior, seed, reaction, init=init, gamma0=g0)


def amp_metrics(xhat, x0):
    """Sign-free accuracy of xhat against the truth: q = |<xhat, x0>|/N, cosine, the vector MSE
    min_s |s xhat - x0|^2 / N and the matrix MSE |x0 x0^T - xhat xhat^T|_F^2 / N^2."""
    N = x0.size
    c, a, b = abs(float(xhat @ x0)) / N, float(xhat @ xhat) / N, float(x0 @ x0) / N
    return {"q": c, "overlap": c / math.sqrt(a * b) if a > 0 else 0.0, "mse": a + b - 2 * c,
            "matrix_mse": b * b - 2 * c * c + a * a}


@dataclass
class AMPEnsemble:
    q: list
    overlap: list
    mse: list
    matrix_mse: list
    converged: list
    bands: dict                  # key -> amp._stats.Band (99% Student-t); gamma_t included
    seeds: tuple
    N: int
    snr: float
    prior: str
    iters: int
    reaction: bool
    label: str = "statistical"
    note: str = ""
    gamma_t: list = None         # each run's empirical gamma at t = iters (gamma_trace[-1])
    init: str = "spectral"
    gamma0: object = None
    rho: object = None


def spiked_amp_ensemble(N, snr, prior, *, seeds, iters, reaction=True, rho=None, init="spectral",
                        gamma0=None) -> AMPEnsemble:
    """One fresh (W, x) per named seed; 99% Student-t bands of the sign-free metrics and of the empirical gamma_t.
    init="planted" starts each run from its own truth x (spiked_matrix's) at gamma0."""
    seeds = _checks.seeds(seeds)
    lam = float(_snr(snr))
    _amp_prior(prior, rho)
    g0 = _check_init(init, lam, gamma0)
    runs = []
    for s in seeds:
        Y, x0 = spiked_matrix(N, snr, prior, s, rho=rho)
        r = spiked_amp(Y, prior, snr=snr, iters=iters, seed=s, reaction=reaction, rho=rho, init=init,
                       gamma0=g0, x0=x0 if init == "planted" else None)
        runs.append((amp_metrics(r.xhat, x0), r.converged, r.gamma_trace[-1]))
    cols = {k: [m[k] for m, _, _ in runs] for k in ("q", "overlap", "mse", "matrix_mse")}
    gam = [g for _, _, g in runs]
    bands = {k: student_band(v) for k, v in cols.items()}
    bands["gamma_t"] = student_band(gam)
    start = ("spectral start x^0 = sqrt(N lambda^2 (lambda^2 - 1)) phi_1" if init == "spectral" else
             f"planted start x^0 = gamma0 x + sqrt(gamma0) g at gamma0 = {g0:.6g} (the truth x of each matrix)")
    note = (f"statistical: Bayes AMP ({start}) on {len(seeds)} matrices, N = {N}, t = {iters}, seeds {list(seeds)}"
            + (f", prior {prior} rho = {rho}" if rho is not None else "") +
            "; 99% Student-t bands; gamma_t is the empirical plug-in (|x^t|^2 / N = gamma_t^2 + gamma_t), not SE's: "
            "MV Theorem 2 is the N -> oo statement at fixed t (with SE gamma_t)" +
            ("" if reaction else "; reaction=False: NO Onsager term (power iteration), the control"))
    return AMPEnsemble(cols["q"], cols["overlap"], cols["mse"], cols["matrix_mse"],
                       [c for _, c, _ in runs], bands, seeds, N, lam, prior, iters, reaction, note=note,
                       gamma_t=gam, init=init, gamma0=g0, rho=rho)


__all__ = ["AMPEnsemble", "AMPResult", "Band", "PRIORS", "Prior", "SEResult", "amp_metrics",
           "diluted_prior", "scalar_mmse", "scalar_overlap", "spiked_amp", "spiked_amp_ensemble", "spiked_matrix", "spiked_potential",
           "spiked_state_evolution", "spiked_threshold"]
