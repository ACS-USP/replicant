"""The diluted-Rademacher (Rademacher-Bernoulli) spiked Wigner model: every state-evolution fixed point, the spinodals,
the information-theoretic threshold, the hard phase and the tricritical point (KZ Ex. 8.2, model 2; Lesieur, Krzakala
& Zdeborova, J. Stat. Mech. (2017) 073403, arXiv:1701.00858 (LKZ), Sec. IV C and Eqs. (33), (213)-(214); Antenucci,
Franz, Urbani & Zdeborova, PRX 9, 011020 (2019), arXiv:1805.05857 (AFUZ), Fig. 1 and its text).

Convention.  replicant's spiked Wigner (inference/spiked.py): Y = (lambda/N) x x^T + W, W ~ GOE(N), unit-variance
prior.  Here x_i = 0 with probability 1 - rho and +-a with probability rho/2 each, a = rho^(-1/2) (variance 1).  LKZ
and the notes take x in {0, +-1} (variance rho) with Y = x x^T / sqrt(N) + sqrt(Delta) xi (LKZ (1), (33)); the
notes write Y = sqrt(lambda_notes / N) x x^T + xi, so lambda_notes = 1/Delta.  Rescaling x by sqrt(rho) gives
    lambda^2 = rho^2 / Delta = rho^2 lambda_notes,   Delta / rho^2 = 1 / lambda^2,
so LKZ's and AFUZ's horizontal axis Delta / rho^2 is replicant's 1 / lambda^2, and LKZ's Delta_c = rho^2 (Eq. (193))
is lambda = 1 (snr_from_lkz, lkz_from_snr, snr_from_notes, notes_from_snr).

Scalar channel y = gamma X + sqrt(gamma) G (spiked.py's): posterior weights over {0, +a, -a} proportional to
w_x exp(x y - gamma x^2 / 2); F = E[X|y] and V = Var[X|y] in closed form (log-sum-exp).  Then
    m(gamma)  = E[X F]                         (diluted_overlap; = 1 - mmse for this unit-variance prior),
    m'(gamma) = E[V^2]                         (diluted_overlap_slope: Guo, Shamai & Verdu, mmse' = -E Var^2),
    I(gamma)  = -E log sum_x' w_x' exp((x' - X) y - gamma (x'^2 - X^2) / 2)   (diluted_mutual_info; the log-partition
                formula, a different integrand from F),
by the trapezoid rule over G on [-10, 10], h = 0.01, exact over X (three atoms).  SE: gamma_{t+1} = lambda^2 m(gamma_t)
(MV Eq. (2.24), as spiked.py; LKZ (213) with x = m/Delta).  Every nontrivial fixed point solves
    D(gamma) = m(gamma) / gamma = 1 / lambda^2,
and is stable iff D is decreasing there (the map's slope lambda^2 m' = gamma m'/m < 1).  D(0+) = 1.  The critical
points of D (gamma m' = m) are found once per rho; on each monotone piece there is at most one root (bracketed), so no
pair of close roots is missed.  The local minimum of D is the algorithmic spinodal Delta_alg / rho^2 (the low branch
ends), the local maximum the dynamic spinodal Delta_dyn / rho^2 (the high branch is born); none means no first-order
transition (rho above the tricritical rho_tri, where the two merge: diluted_tricritical).
RS potential Psi(gamma) = lambda^2/4 + gamma^2/(4 lambda^2) - gamma/2 + I(gamma) (DAM Eq. (9), minimised; spiked.py),
stationary exactly at the fixed points.  The IT threshold is where Psi of the two stable fixed points is equal.

Runs (diluted_state_evolution).  Following LKZ Sec. IV C 1, SE is iterated from the uninformative start m_0 = 1e-6 and
from the informative start m_0 = E x^2 = 1.  The Bayes fixed point is the one of lower Psi among the two limits, and is
reported only when both iterations converged (|step| <= 1e-13 lambda^2 + 1e-12 gamma within max_iter) to enumerated
fixed points; otherwise the Bayes values are None and the phase is `undetermined` (never a Refusal).  Phases (LKZ
Sec. IV C 1): `hard` - the limits differ and the informative one has the lower Psi (AMP suboptimal); `easy` - AMP's
(uninformative) limit is the Bayes one, q > 0; `impossible` - the Bayes q is 0.  `metastable` marks two distinct limits.

_HOOKS (planted errors, tests only): "amplitude_power" (-1/2: a = rho^power; -1 is the unnormalised slip).
"""
import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

import numpy as np
from scipy import optimize

from ..core import Refusal

_HOOKS = {"amplitude_power": -0.5}
_Z = np.linspace(-10.0, 10.0, 2001)
_WZ = np.exp(-_Z**2 / 2)
_WZ = _WZ / _WZ.sum()
_GRID = np.geomspace(1e-6, 1e3, 361)      # where the critical points of D are searched
_EPS_START = 1e-6
PRIOR = "diluted_rademacher"


def _rho(rho):
    if rho is None:
        raise Refusal("precondition:rho", f"prior {PRIOR} needs rho, the non-zero fraction in (0, 1]")
    if isinstance(rho, bool):
        raise Refusal("precondition:rho", f"rho = {rho!r}: need a number in (0, 1]")
    try:
        r = float(Fraction(rho)) if isinstance(rho, str) else float(rho)
    except (TypeError, ValueError, ZeroDivisionError):
        raise Refusal("precondition:rho", f"rho = {rho!r}: need a number in (0, 1]") from None
    if not (math.isfinite(r) and 0 < r <= 1):
        raise Refusal("precondition:rho", f"rho = {rho!r}: the non-zero fraction must lie in (0, 1]")
    return r


def _atoms(rho):
    a = rho ** _HOOKS["amplitude_power"]
    x = np.array([0.0, a, -a])
    w = np.array([1 - rho, rho / 2, rho / 2])
    return x, w


def _post(rho, g, y):
    """(F, V) for y (array) at scalar SNR g: posterior over the three atoms by log-sum-exp."""
    x, w = _atoms(rho)
    with np.errstate(divide="ignore"):
        lw = np.log(w)
    L = lw + y[..., None] * x - np.asarray(g)[..., None] * x**2 / 2
    L -= L.max(axis=-1, keepdims=True)
    p = np.exp(L)
    p /= p.sum(axis=-1, keepdims=True)
    F = p @ x
    return F, p @ (x * x) - F * F


def _avg(rho, g, fn):
    """E over X (atoms) and G of fn(X, F, V) at y = g X + sqrt(g) G."""
    x, w = _atoms(rho)
    tot = 0.0
    for xi, wi in zip(x, w):
        if wi == 0:
            continue
        F, V = _post(rho, g, g * xi + math.sqrt(g) * _Z)
        tot += wi * float(_WZ @ fn(xi, F, V))
    return tot


def _g(gamma):
    g = float(gamma)
    if not (math.isfinite(g) and g >= 0):
        raise Refusal("precondition:gamma", f"gamma = {gamma!r}: the scalar SNR must be >= 0")
    return g


def _m(r, g):
    """m(gamma) = E[X F]: the atom 0 contributes nothing and +a, -a contribute equally (F is odd), so
    m = rho a E_G F(gamma a + sqrt(gamma) G)."""
    a = r ** _HOOKS["amplitude_power"]
    u = a * (g * a + math.sqrt(g) * _Z)
    if r == 1:
        return a * float(_WZ @ np.tanh(u))
    c = math.log(r / 2) - g * a * a / 2 - math.log1p(-r)        # log(w_+- / w_0) - gamma a^2 / 2
    top = np.maximum(0.0, c + np.abs(u))
    ep, em = np.exp(c + u - top), np.exp(c - u - top)
    return r * a * a * float(_WZ @ ((ep - em) / (np.exp(-top) + ep + em)))


def diluted_overlap(rho, gamma):
    """m(gamma) = E[X E[X|y]] for y = gamma X + sqrt(gamma) G (binary64 trapezoid over G, exact over X)."""
    return _m(_rho(rho), _g(gamma))


def diluted_overlap_slope(rho, gamma):
    """m'(gamma) = E[Var(X|y)^2] (Guo-Shamai-Verdu: d mmse / d snr = -E Var^2; m = E X^2 - mmse)."""
    return _avg(_rho(rho), _g(gamma), lambda x, F, V: V * V)


def diluted_mutual_info(rho, gamma):
    """I(X; y) in nats from the log-partition: -E log sum_x' w_x' exp((x' - X) y - gamma (x'^2 - X^2)/2)."""
    r, g = _rho(rho), _g(gamma)
    if g == 0:
        return 0.0
    x, w = _atoms(r)
    tot = 0.0
    for xi, wi in zip(x, w):
        if wi == 0:
            continue
        y = g * xi + math.sqrt(g) * _Z
        with np.errstate(divide="ignore"):
            lw = np.log(w)
        L = lw[None, :] + (x[None, :] - xi) * y[:, None] - g * (x[None, :]**2 - xi**2) / 2
        top = L.max(axis=1)
        tot -= wi * float(_WZ @ (top + np.log(np.exp(L - top[:, None]).sum(axis=1))))
    return tot


def diluted_potential(snr, gamma, rho):
    """RS potential Psi(gamma; lambda) of DAM Eq. (9) (lambda_DAM = lambda^2), minimised over gamma."""
    lam2, g = float(snr)**2, _g(gamma)
    return lam2 / 4 + g * g / (4 * lam2) - g / 2 + diluted_mutual_info(rho, g)


# ---- conventions ---------------------------------------------------------------------------------------------------
def snr_from_lkz(delta, rho):
    """LKZ / AFUZ (x in {0, +-1}, noise variance Delta) -> replicant lambda = rho / sqrt(Delta)."""
    return _rho(rho) / math.sqrt(float(delta))


def lkz_from_snr(snr, rho):
    return _rho(rho)**2 / float(snr)**2


def snr_from_notes(lam_notes, rho):
    """The notes' Y = sqrt(lambda_notes / N) x x^T + xi, x in {0, +-1} -> lambda = rho sqrt(lambda_notes)."""
    return _rho(rho) * math.sqrt(float(lam_notes))


def notes_from_snr(snr, rho):
    return float(snr)**2 / _rho(rho)**2


# ---- the critical structure of D(gamma) = m / gamma ---------------------------------------------------------------
def _s(rho, g):
    """gamma m'(gamma) - m(gamma): the sign of D'(gamma); g a scalar or a 1-d array (vectorised over gamma)."""
    gs = np.atleast_1d(np.asarray(g, dtype=float))
    x, w = _atoms(rho)
    out = np.zeros(gs.size)
    for xi, wi in zip(x, w):
        if wi == 0:
            continue
        F, V = _post(rho, gs[:, None], gs[:, None] * xi + np.sqrt(gs)[:, None] * _Z[None, :])
        out += wi * ((gs[:, None] * V * V - xi * F) @ _WZ)
    return out if np.ndim(g) else float(out[0])


@lru_cache(maxsize=256)
def _critical(rho):
    vals = _s(rho, _GRID)
    out = []
    for (g0, v0), (g1, v1) in zip(zip(_GRID, vals), zip(_GRID[1:], vals[1:])):
        if v0 * v1 < 0:
            out.append((optimize.brentq(lambda g: _s(rho, g), g0, g1, xtol=1e-15, rtol=1e-14), v0 < 0))
    return tuple(out)


@dataclass(frozen=True)
class Spinodals:
    rho: float
    gamma_alg: float       # local minimum of D: the uninformative (low) branch ends at lambda = snr_alg
    gamma_dyn: float       # local maximum of D: the informative (high) branch is born at lambda = snr_dyn
    snr_alg: float
    snr_dyn: float


def diluted_spinodals(rho):
    """Spinodals(...) from the two critical points of D, or None (D monotone: no first-order transition)."""
    r = _rho(rho)
    crit = _critical(r)
    if not crit:
        return None
    if len(crit) != 2 or not crit[0][1] or crit[1][1]:
        raise Refusal("unsupported:critical-structure",
                      f"rho = {r}: D(gamma) = m/gamma has critical points {crit} (gamma, rising after); the "
                      "spinodal reading assumes one local minimum followed by one local maximum")
    (ga, _), (gd, _) = crit
    return Spinodals(r, ga, gd, 1 / math.sqrt(diluted_overlap(r, ga) / ga), 1 / math.sqrt(diluted_overlap(r, gd) / gd))


def clear_caches():
    _critical.cache_clear()


# ---- fixed points at one lambda --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FixedPoint:
    gamma: float
    q: float
    stable: bool
    slope: float           # of the SE map lambda^2 m(gamma) at the fixed point
    potential: float


@dataclass(frozen=True)
class Start:
    gamma0: float
    gamma: object          # the enumerated fixed point the iteration reached (None if not converged / not matched)
    raw: float
    iters: int
    converged: bool


def _roots(lam2, r):
    G = lambda x: lam2 * _m(r, x) - x
    edges = [lam2 * 1e-12] + [c for c, _ in _critical(r) if c < lam2] + [lam2]
    out = [0.0]
    for a, b in zip(edges, edges[1:]):
        va, vb = G(a), G(b)
        if va == 0:
            out.append(a)
        elif va * vb < 0:
            out.append(optimize.brentq(G, a, b, xtol=1e-15, rtol=1e-15, maxiter=300))
    return out


def _iterate(lam2, r, g0, max_iter):
    g = g0
    for t in range(1, max_iter + 1):
        new = lam2 * _m(r, g)
        if abs(new - g) <= 1e-13 * lam2 + 1e-12 * g:
            return new, t, True
        g = new
    return g, max_iter, False


def _start(lam2, r, g0, roots, max_iter):
    raw, its, ok = _iterate(lam2, r, g0, max_iter)
    best = min(roots, key=lambda x: abs(x - raw))
    hit = ok and abs(best - raw) <= 1e-6 * lam2
    return Start(g0, best if hit else None, raw, its, hit)


@dataclass(frozen=True)
class DilutedSE:
    snr: float
    prior: str
    rho: float
    gamma: object          # Bayes values: None when undetermined
    q: object
    overlap: object
    mmse: object
    matrix_mmse: object
    fixed_points: tuple
    bayes: bool
    potential: object
    note: str
    phase: str
    metastable: bool
    uninformative: Start
    informative: Start
    below_floor: bool = False


def diluted_state_evolution(snr, rho, *, max_iter=20000) -> DilutedSE:
    """Every SE fixed point at lambda = snr, the two iterated starts, the Bayes values and the phase (module doc)."""
    lam, r = float(snr), _rho(rho)
    if not (math.isfinite(lam) and lam > 0):
        raise Refusal("precondition:snr", f"snr = {snr!r}: the diluted prior needs a finite lambda > 0")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise Refusal("precondition:positive-iters", f"max_iter = {max_iter!r} is not a positive int")
    lam2 = lam * lam
    roots = _roots(lam2, r)
    fps = []
    for g in roots:
        slope = lam2 * diluted_overlap_slope(r, g)
        fps.append(FixedPoint(g, g / lam2, bool(slope < 1), slope, diluted_potential(lam, g, r)))
    uni = _start(lam2, r, lam2 * _EPS_START, roots, max_iter)
    inf = _start(lam2, r, lam2, roots, max_iter)
    pot = {f.gamma: f.potential for f in fps}
    nfp = f"{len(fps)} fixed points gamma = {[f'{f.gamma:.10g}' for f in fps]} (stable: {[f.stable for f in fps]})"
    tie = uni.converged and inf.converged and uni.gamma != inf.gamma and abs(pot[uni.gamma] - pot[inf.gamma]) < 1e-12
    if not (uni.converged and inf.converged) or tie:
        why = ("an SE iteration did not converge to an enumerated fixed point within max_iter"
               if not (uni.converged and inf.converged) else "the two limits have equal potential (the IT point)")
        return DilutedSE(lam, PRIOR, r, None, None, None, None, None, tuple(fps), False, None,
                         f"undetermined: {why}; {nfp}", "undetermined", False, uni, inf)
    meta = uni.gamma != inf.gamma
    gs = inf.gamma if meta and pot[inf.gamma] < pot[uni.gamma] else uni.gamma
    q = gs / lam2
    phase = "hard" if gs != uni.gamma else ("easy" if gs > 0 else "impossible")
    note = (f"{nfp}; uninformative start -> {uni.gamma:.10g}, informative start -> {inf.gamma:.10g}; Bayes = the lower "
            f"RS potential: gamma* = {gs:.10g}; phase {phase}" + (" (a metastable second branch)" if meta else ""))
    return DilutedSE(lam, PRIOR, r, gs, q, math.sqrt(q), 1 - q, 1 - q * q, tuple(fps), True, pot[gs], note, phase,
                     meta, uni, inf)


# ---- thresholds ---------------------------------------------------------------------------------------------------
def diluted_it_threshold(rho):
    """lambda_IT: Psi of the lowest and the highest stable fixed point coincide (brentq between the spinodals)."""
    sp = diluted_spinodals(rho)
    if sp is None:
        raise Refusal("no-transition:first-order", f"rho = {rho}: no spinodals (rho above the tricritical point), so "
                      "no first-order IT threshold")
    r = sp.rho

    def gap(lam):
        lam2 = lam * lam
        st = [g for g in _roots(lam2, r) if lam2 * diluted_overlap_slope(r, g) < 1]
        return diluted_potential(lam, st[-1], r) - diluted_potential(lam, st[0], r)
    return optimize.brentq(gap, sp.snr_dyn * (1 + 1e-9), sp.snr_alg * (1 - 1e-9), xtol=1e-14, rtol=1e-14)


@dataclass(frozen=True)
class Tricritical:
    rho: float
    gamma: float
    snr: float             # Delta_tri / rho_tri^2 = 1 / snr^2


def _local_max_s(rho):
    """The interior local maximum of s(gamma) = gamma m' - m (s' = gamma m'': it sits where m'' changes sign from + to
    -), or (-1, None) when s has none on [1e-3, 30] (m concave: no first-order structure at all)."""
    lg = np.linspace(math.log(1e-3), math.log(30.0), 48)
    f = lambda u: _s(rho, math.exp(u))
    vals = _s(rho, np.exp(lg))
    ks = [k for k in range(1, len(lg) - 1) if vals[k - 1] < vals[k] > vals[k + 1]]
    if not ks:
        return -1.0, None
    k = ks[0]
    res = optimize.minimize_scalar(lambda u: -f(u), bounds=(lg[k - 1], lg[k + 1]), method="bounded",
                                   options={"xatol": 1e-10})
    return -res.fun, math.exp(res.x)


def diluted_tricritical(bracket=(0.085, 0.11)):
    """rho_tri: the interior local maximum of s = gamma m' - m (the sign of D') crosses zero; below rho_tri it is
    positive, D is non-monotone and has the two spinodals; at rho_tri D' = D'' = 0 (s = s' = 0)."""
    lo, hi = (float(b) for b in bracket)
    if not _local_max_s(lo)[0] > 0 > _local_max_s(hi)[0]:
        raise Refusal("no-transition:tricritical", f"max s does not change sign on rho in [{lo}, {hi}]")
    rt = optimize.brentq(lambda r: _local_max_s(r)[0], lo, hi, xtol=1e-12, rtol=1e-12)
    g = _local_max_s(rt)[1]
    return Tricritical(rt, g, 1 / math.sqrt(diluted_overlap(rt, g) / g))


__all__ = ["DilutedSE", "FixedPoint", "PRIOR", "Spinodals", "Start", "Tricritical", "clear_caches",
           "diluted_it_threshold", "diluted_mutual_info", "diluted_overlap", "diluted_overlap_slope",
           "diluted_potential",
           "diluted_spinodals", "diluted_state_evolution", "diluted_tricritical", "lkz_from_snr", "notes_from_snr",
           "snr_from_lkz", "snr_from_notes"]
