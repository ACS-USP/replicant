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

Scalar channel y = gamma X + sqrt(gamma) G (spiked.py's) is sqrt(gamma) times bayes/channels.py's y' = sqrt(snr) x + z
at snr = gamma, so the prior and every scalar quantity come from channels (the one place priors live; the unit-variance
`diluted` family keeps the amplitude exact as Sqrt(1/rho)):
    m(gamma)  = E[X E[X|y]]       channels.scalar_moments (diluted_overlap; = 1 - mmse for this unit-variance prior),
    m'(gamma) = E[Var(X|y)^2]     channels.scalar_moments(..., slope=True) (Guo, Shamai & Verdu: mmse' = -E Var^2),
    I(gamma)                      channels.mutual_information (h(Y) - h(Z): a different integrand from m).
SE: gamma_{t+1} = lambda^2 m(gamma_t) (MV Eq. (2.24), as spiked.py; LKZ (213) with x = m/Delta).  Every nontrivial
fixed point solves
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

_HOOKS (planted errors, tests only): "unit_variance" (True; False uses channels' sparse(rho, a = 1), the notes' and
LKZ's unnormalised x in {0, +-1}: the convention slip).
"""
import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

import numpy as np
from scipy import optimize

from ..bayes import channels as CH
from ..core import Refusal
from .spiked import _snr as _spiked_snr

_HOOKS = {"unit_variance": True}
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


@lru_cache(maxsize=64)
def _channel_prior(rho, unit):
    r = Fraction(repr(rho))                 # the decimal the float prints (0.08 -> 2/25)
    return CH.Prior("diluted", rho=r) if unit else CH.Prior("sparse", rho=r, a=1)


def _prior(r):
    return _channel_prior(r, bool(_HOOKS["unit_variance"]))


def _g(gamma):
    g = float(gamma)
    if not (math.isfinite(g) and g >= 0):
        raise Refusal("precondition:gamma", f"gamma = {gamma!r}: the scalar SNR must be >= 0")
    return g


def _m(r, g):
    return CH.scalar_moments(_prior(r), g)


def diluted_overlap(rho, gamma):
    """m(gamma) = E[X E[X|y]] for y = gamma X + sqrt(gamma) G (channels.scalar_moments at snr = gamma)."""
    return _m(_rho(rho), _g(gamma))


def diluted_overlap_slope(rho, gamma):
    """m'(gamma) = E[Var(X|y)^2] (channels.scalar_moments with slope)."""
    return CH.scalar_moments(_prior(_rho(rho)), _g(gamma), slope=True)[1]


def diluted_mutual_info(rho, gamma):
    """I(X; y) in nats (channels.mutual_information at snr = gamma; scaling y does not change I)."""
    return CH.mutual_information(_prior(_rho(rho)), _g(gamma))


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
    m, sl = CH.scalar_moments(_prior(rho), g, slope=True)
    return np.asarray(g) * sl - m if np.ndim(g) else g * sl - m


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


_S_RESOLUTION = 1e-10     # |max s| below this: the sign of D' is not resolved (at the tricritical point)


def _refined_critical(r):
    """_critical, plus the pair the 361-point grid misses within ~1e-7 of rho_tri: when the interior maximum of s is
    positive but no grid sign change was found, the two zeros are bracketed on either side of that maximum."""
    crit = _critical(r)
    if crit:
        return crit
    smax, g = _local_max_s(r)
    if not smax > 0:
        return ()
    k = int(np.searchsorted(_GRID, g))
    lo, hi = _GRID[max(k - 1, 0)], _GRID[min(k, len(_GRID) - 1)]
    f = lambda x: _s(r, x)
    if not (f(lo) < 0 < f(g) and f(hi) < 0):
        raise Refusal("unsupported:critical-structure", f"rho = {r}: max s = {smax:.3g} at gamma = {g:.6g} not "
                      "bracketed by the grid neighbours")
    return ((optimize.brentq(f, lo, g, xtol=1e-15, rtol=1e-14), True),
            (optimize.brentq(f, g, hi, xtol=1e-15, rtol=1e-14), False))


def diluted_first_order(rho):
    """'first-order' | 'continuous' | 'undetermined': undetermined when |max s| < _S_RESOLUTION (within numerical
    resolution of the tricritical point, where the sign of D' is not resolved), never a silent False."""
    r = _rho(rho)
    smax = _local_max_s(r)[0]
    if abs(smax) < _S_RESOLUTION:
        return "undetermined"
    return "first-order" if _refined_critical(r) else "continuous"


def diluted_spinodals(rho):
    """Spinodals(...) from the two critical points of D (the grid's, or refined on either side of the maximum of s
    near rho_tri), or None (D monotone: no first-order transition).  diluted_first_order says when the sign is not
    resolved at all (|max s| < 1e-10)."""
    r = _rho(rho)
    crit = _refined_critical(r)
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
    _channel_prior.cache_clear()
    _it.cache_clear()


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
    lam, r = float(_spiked_snr(snr)), _rho(rho)
    if not lam > 0:
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


def diluted_se_trajectory(snr, rho, gamma0, iters):
    """SE from a given start: gamma_0 = gamma0, gamma_{t+1} = lambda^2 m(gamma_t) (MV Eq. (2.24)), t = 0..iters, as a list
    of length iters + 1.  The fixed-t comparison for planted AMP (spiked_amp init="planted" starts at x^0 = gamma0 x +
    sqrt(gamma0) g, the scalar channel at gamma0); the overlap at t is diluted_overlap(rho, gamma_t)."""
    lam, r = float(_spiked_snr(snr)), _rho(rho)
    if not lam > 0:
        raise Refusal("precondition:snr", f"snr = {snr!r}: the diluted prior needs a finite lambda > 0")
    bad = Refusal("precondition:gamma0", f"gamma0 = {gamma0!r}: the start needs a finite number > 0")
    if isinstance(gamma0, bool) or gamma0 is None:
        raise bad
    try:
        g = float(gamma0)
    except (TypeError, ValueError):
        raise bad from None
    if not (math.isfinite(g) and g > 0):
        raise bad
    if isinstance(iters, bool) or not isinstance(iters, int) or iters < 1:
        raise Refusal("precondition:positive-iters", f"iters = {iters!r} is not a positive int")
    lam2, out = lam * lam, [g]
    for _ in range(iters):
        # m = E[E[X|y]^2] >= 0; near gamma = 0 the quadrature returns about -1e-27, clipped (else the next step refuses)
        g = max(lam2 * _m(r, g), 0.0)
        out.append(g)
    return out


# ---- thresholds ---------------------------------------------------------------------------------------------------
def diluted_it_threshold(rho):
    """lambda_IT: Psi of the lowest and the highest stable fixed point coincide (brentq between the spinodals)."""
    return _it(_rho(rho))


@lru_cache(maxsize=64)
def _it(rho):
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
           "diluted_potential", "diluted_se_trajectory",
           "diluted_first_order", "diluted_spinodals", "diluted_state_evolution", "diluted_tricritical", "lkz_from_snr", "notes_from_snr",
           "snr_from_lkz", "snr_from_notes"]
