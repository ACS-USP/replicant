"""Clusters in graph coloring at beta = oo: colorability (frozen SP), clustering and condensation (1RSB), rigidity
(KZ Secs. 13.2.3, 14.1-14.3; Zdeborova & Krzakala, PRE 76, 031131 (2007), arXiv:0704.1269, "ZK").

Convention: q >= 3 colors; graph "regular" (degree d, read as a real number in the closed forms, as the notes treat
it, and as an integer where a population is run) or "er" (Erdos-Renyi, mean degree c).  Every threshold names its
graph.  The populations are in coloring/rsb_pop.py.

  sp_regular(q, d) -> SPRegular: the notes' symmetric SP fixed point and complexity on d-regular graphs (KZ
      Sec. 14.3, p. 220): eta = sum_l (-1)^(l-1) C(q-1, l-1) (1 - l eta)^(d-1) / sum_l (-1)^(l-1) C(q, l)
      (1 - l eta)^(d-1), Sigma = ln sum_l (-1)^(l-1) C(q, l) (1 - l eta)^d - (d/2) ln(1 - q eta^2); iterated from
      eta = (1 - 1e-3)/q (the frozen start); nontrivial when eta > 1e-9.
  colorability_threshold(q, graph="regular") -> Colorability: the real root of Sigma(d) on the nontrivial branch
      (physics-level), with d_sp (the smallest integer d with a nontrivial eta, ZK Table I c_SP) and d_uncol (the
      smallest integer d with Sigma < 0, ZK Table I c_s: the first uncolorable degree).
  colorability_threshold(q, graph="er", seeds, ...) -> Colorability: the root of the frozen-SP complexity
      (rsb_pop.frozen_sp) in c: per seed, `steps` bisections of `bracket` (default (c_ann - 1.5, c_ann), c_ann the
      annealed bound of Eq. (6.17)) on "colorable" = trivial SP or Sigma > 0, then the linear interpolation between
      the last nontrivial points on either side; a 99% Student-t band over the seeds (statistical, never final).
  rigidity_threshold(q, graph) -> Rigidity: KZ Sec. 13.2.3 (p. 211), eta_l = 1 + sum_{r=1}^{q-1} (-1)^r C(q-1, r)
      (1 - r eta_{l+1}/(q - 1))^(d - 1) on the d-regular tree; on ER (Poisson(c) children, E x^K = e^{-c(1 - x)},
      derived here) the sum closes: eta = (1 - e^{-c eta/(q - 1)})^(q - 1).  c_r = the smallest degree with a
      nonzero fixed point = min over eta in (0, 1) of the degree solving F(eta) = eta (the fold of the map).  This
      is ZK's c_r(m = 1) (the planted / m = 1 freezing).  Floats, physics-level.
  rigidity_corrections(q) -> {"leading": q ln q (the notes' stated form), "zk_eq44": q (ln q + ln ln q + 1) (ZK
      Eq. (44), the next terms, labelled)}.
  clustering_threshold(q, graph, seeds, ...) / condensation_threshold(q, graph, seeds, ...) -> Threshold:
      every decision uses converged runs only (rsb_pop phases: trivial / nontrivial / undetermined; an undetermined
      or disagreeing decisive run makes the threshold undetermined, never refused).  regular: scanning down from the
      first uncolorable degree, c_d = the smallest d whose m = 1 frozen start is nontrivial on every seed with d - 1
      trivial on every seed, c_c = the smallest nontrivial d with Sigma(m = 1) < 0 in the 99% band over seeds, every
      smaller nontrivial d resolved > 0 -- ZK's integer definitions (Table I).  ER: the growth factor per sweep of the
      trivial m = 1 solution from a small tilt (degrees 0.8 and 1.2 (q - 1)^2, interpolated to 1 per seed) gives
      c_inst; if the frozen start is trivial at 0.95 c_inst on every seed the transition is continuous, c_d = c_inst;
      c_c is not located (Sigma(1) -> 0 at c_inst): only the bound c_c <= 1.15 c_inst when Sigma(1) < 0 is resolved
      there (kind "upper-bound", status undetermined).  ER q >= 4 is refused up front, before any population runs,
      unsupported:er-first-order (ZK 2007: the ER clustering transition is discontinuous for q >= 4, continuous for
      q = 3; the first-order ER path is not implemented, the sbm entry at c_in = 0 covers q = 5).  At q = 3 a frozen
      start that stays nontrivial at 0.95 c_inst labels c_d, c_c undetermined (kind "first-order"), never refused.
_HOOKS["children"] (1: d - 1 incoming messages; planted error 0 uses d) and _HOOKS["er_rate"] (1.0: the factor on
the Poisson mean; planted error 0.8 = (q - 1)/q at q = 5 reads the mean degree c as c_out = q c/(q - 1)).
"""
import math
from dataclasses import dataclass
from math import comb

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from ..amp._stats import student_band
from ..core import Refusal
from . import rsb_pop as RP
from .potts_bp import coloring_annealed_threshold

_HOOKS = {"children": 1, "er_rate": 1.0}
_GRAPHS = ("regular", "er")


def _q3(q):
    if isinstance(q, bool) or not isinstance(q, (int, np.integer)) or q < 3:
        raise Refusal("precondition:q", f"q = {q!r}: clusters of colorings need an int q >= 3")
    return int(q)


def _graph(graph):
    if graph not in _GRAPHS:
        raise Refusal("unsupported:graph", f"graph = {graph!r}: 'regular' | 'er'")
    return graph


# ---- the notes' symmetric SP on d-regular graphs -------------------------------------------------------
@dataclass(frozen=True)
class SPRegular:
    q: int
    d: float
    eta: float
    sigma: float
    nontrivial: bool


def _sums(q, eta, p):
    num = math.fsum((-1) ** (l - 1) * comb(q - 1, l - 1) * (1 - l * eta) ** p for l in range(1, q + 1))
    den = math.fsum((-1) ** (l - 1) * comb(q, l) * (1 - l * eta) ** p for l in range(1, q + 1))
    return num, den


def sp_regular(q, d):
    q, d = _q3(q), float(d)
    if not d >= 2:
        raise Refusal("precondition:d", f"d = {d!r}: a degree >= 2")
    p = d - _HOOKS["children"]
    g = lambda e: (lambda nd: nd[0] / nd[1] - e)(_sums(q, e, p))
    eta, e0 = 0.0, (1 - 1e-3) / q                     # the iteration from e0 decreases to the largest root below e0
    grid = e0 * np.concatenate([np.linspace(1, 0.02, 99), np.logspace(np.log10(0.02), -9, 60)[1:]])
    if g(e0) >= 0:
        eta = e0
    else:
        for a, b in zip(grid[1:], grid[:-1]):
            if g(a) >= 0:
                eta = brentq(g, a, b, xtol=1e-16, rtol=1e-15)
                break
    eta = max(eta, 0.0)
    _, z = _sums(q, eta, d)
    return SPRegular(q, d, eta, math.log(z) - d / 2 * math.log1p(-q * eta * eta), eta > 1e-9)


@dataclass(frozen=True)
class Colorability:
    q: int
    graph: str
    value: float
    status: str                    # "determined" | "undetermined"
    note: str
    d_sp: int | None = None        # regular only
    d_uncol: int | None = None
    stderr: float = 0.0
    half_width: float = 0.0
    estimates: tuple = ()


def colorability_threshold(q, graph="regular", *, seeds=(0, 1, 2, 3), pop=10000, sweeps=150, measure=100,
                           bracket=None, steps=6):
    q, graph = _q3(q), _graph(graph)
    if graph == "regular":
        d, d_sp = q, None
        while True:
            s = sp_regular(q, d)
            if s.nontrivial and d_sp is None:
                d_sp = d
            if s.nontrivial and s.sigma < 0:
                break
            d += 1
            if d > 50 * q * q:
                return Colorability(q, graph, math.nan, "undetermined", "no uncolorable degree below 50 q^2")
        root = brentq(lambda x: sp_regular(q, x).sigma, d_sp, d, xtol=1e-14)
        return Colorability(q, graph, root, "determined", "real root of the notes' Sigma(d) on the nontrivial "
                            "branch (d read as real; physics-level)", d_sp, d)
    lo0, hi0 = bracket or (coloring_annealed_threshold(q) - 1.5, coloring_annealed_threshold(q))
    est, bad = [], []
    for s in seeds:
        run = lambda c: RP.frozen_sp(q, c, graph="er", pop=pop, sweeps=sweeps, measure=measure, seed=s)
        lo, hi, flo, fhi = lo0, hi0, None, None
        for _ in range(steps):
            mid = (lo + hi) / 2
            r = run(mid)
            if r.status != "converged":
                bad.append(f"seed {s}, c = {mid:.4g}: {r.note}")
            if not r.nontrivial or r.sigma > 0:
                lo, flo = mid, (r.sigma if r.nontrivial else None)
            else:
                hi, fhi = mid, r.sigma
        if fhi is None:
            bad.append(f"seed {s}: no nontrivial point with Sigma < 0 in the bracket ({lo0:.4g}, {hi0:.4g})")
        est.append(lo + flo * (hi - lo) / (flo - fhi) if flo is not None and fhi is not None else (lo + hi) / 2)
    b = student_band(est)
    return Colorability(q, graph, b.mean, "undetermined" if bad else "determined", "; ".join(bad) or
                        f"frozen-SP root over seeds {tuple(seeds)} (statistical, NOT final)", None, None,
                        b.stderr, b.half_width, tuple(est))


# ---- rigidity ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Rigidity:
    q: int
    graph: str
    c_r: float
    eta_r: float


def _deg_regular(q, eta):
    f = lambda d: 1 + math.fsum((-1) ** r * comb(q - 1, r) * (1 - r * eta / (q - 1)) ** (d - _HOOKS["children"])
                                for r in range(1, q)) - eta
    return brentq(f, _HOOKS["children"] + 1e-9, 100.0 * q * q, xtol=1e-13)


def _deg_er(q, eta):
    """The c with eta = (1 - e^{-rate c eta/(q - 1)})^(q - 1): c = -(q - 1) ln(1 - eta^{1/(q-1)}) / (rate eta)."""
    return -(q - 1) * math.log(-math.expm1(math.log(eta) / (q - 1))) / (_HOOKS["er_rate"] * eta)


def rigidity_threshold(q, graph="regular"):
    q, graph = _q3(q), _graph(graph)
    f = (lambda e: _deg_regular(q, e)) if graph == "regular" else (lambda e: _deg_er(q, e))
    grid = np.concatenate([np.linspace(0.02, 0.98, 49), 1 - np.logspace(-2, -12, 41)])
    vals = [f(e) for e in grid]
    i = int(np.argmin(vals))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    r = minimize_scalar(f, bounds=(a, b), method="bounded", options={"xatol": 1e-13})
    return Rigidity(q, graph, float(r.fun), float(r.x))


def rigidity_corrections(q):
    q = _q3(q)
    lq = math.log(q)
    return {"leading": q * lq, "zk_eq44": q * (lq + math.log(lq) + 1)}


# ---- clustering / condensation ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Threshold:
    q: int
    graph: str
    value: float
    kind: str                 # "integer" (regular) | "continuous" | "first-order"
    status: str
    note: str
    estimates: tuple = ()


def _m1(q, x, graph, seed, pop, sweeps, measure, init="frozen"):
    return RP.coloring_1rsb(q, x, 1, graph=graph, pop=pop, sweeps=sweeps, measure=measure, seed=seed, init=init)


def _verdict(runs):
    """'trivial' / 'nontrivial' when every run is converged and agrees, else 'undetermined' with the phases."""
    ph = {r.phase for r in runs}
    return (ph.pop(), "") if len(ph) == 1 and "undetermined" not in ph else \
        ("undetermined", "phases " + ", ".join(r.phase + (f" ({r.note})" if r.note else "") for r in runs))


def _regular_scan(q, seeds, pop, sweeps, measure):
    """{d: (verdict, why, runs)} at m = 1 from the frozen start, downward from the first uncolorable degree until a
    degree that is not nontrivial."""
    d = colorability_threshold(q, "regular").d_uncol
    out = {}
    while d >= 3:
        runs = [_m1(q, d, "regular", s, pop, sweeps, measure) for s in seeds]
        out[d] = _verdict(runs) + (runs,)
        if out[d][0] != "nontrivial":
            break
        d -= 1
    return out


def _growth(q, c, seed):
    """Per-sweep growth factor of q1 - 1/q from a tilt of the trivial m = 1 solution (2000 laws of 100 messages;
    sweeps 3..10: q1 - 1/q stays below 1e-4, the linear regime)."""
    r = RP.coloring_1rsb(q, c, 1, graph="er", pop=(2000, 100), sweeps=10, measure=8, seed=seed, init=("tilt", 1e-3))
    dev = np.array(r.q1_series) - 1.0 / q
    return float(np.exp(np.mean(np.diff(np.log(dev))))) if np.all(dev > 0) else math.nan


def _er_instability(q, seeds):
    ks = (q - 1) ** 2
    est = []
    for s in seeds:
        l1, l2 = _growth(q, 0.8 * ks, s), _growth(q, 1.2 * ks, s)
        est.append(0.8 * ks + (1 - l1) * 0.4 * ks / (l2 - l1))
    return est


def _sigma_band(runs):
    """The 99% band of Sigma(1) over the runs; (nan, nan) -- never resolved -- unless every run is converged."""
    if any(r.status != "converged" for r in runs):
        return math.nan, math.nan
    b = student_band([r.sigma for r in runs])
    return b.mean - b.half_width, b.mean + b.half_width


def _regular(q, seeds, pop, sweeps, measure):
    scan = _regular_scan(q, seeds, pop, sweeps, measure)
    und = lambda why: Threshold(q, "regular", math.nan, "integer", "undetermined", why)
    low = min(scan)
    if scan[low][0] != "trivial":
        why = f"d = {low}: {scan[low][1] or scan[low][0]} (no trivial degree below the nontrivial ones)"
        return und(why), und(f"c_d undetermined: {why}")
    dd = low + 1
    if dd not in scan:
        return und("the first uncolorable degree is already trivial"), und("c_d undetermined")
    cd = Threshold(q, "regular", dd, "integer", "determined", f"m = 1 frozen start trivial at d = {low}, nontrivial "
                   f"(stationary) at d = {dd} on seeds {tuple(seeds)}")
    for d in sorted(x for x in scan if x >= dd):
        lo, hi = _sigma_band(scan[d][2])
        if not lo <= hi or lo <= 0 <= hi:
            return cd, und(f"Sigma(m = 1) at d = {d}: 99% band ({lo:.3g}, {hi:.3g}) not resolved")
        if hi < 0:
            return cd, Threshold(q, "regular", d, "integer", "determined", f"Sigma(m = 1) < 0 first at d = {d}: "
                                 f"99% band ({lo:.3g}, {hi:.3g}) over seeds {tuple(seeds)}")
    return cd, und("Sigma(m = 1) >= 0 up to d_uncol")


def _er(q, seeds, pop, sweeps, measure):
    """ER: the continuous case only.  c_d = c_inst when the frozen start is trivial at 0.95 c_inst on every seed; c_c
    is NOT located (Sigma(m = 1) vanishes at c_inst, so its sign just above cannot be resolved): only the upper
    bound c_c <= 1.15 c_inst when Sigma(1) < 0 is resolved there.  Called for q = 3 only (q >= 4 is refused up
    front by cluster_thresholds); a nontrivial probe (unexpectedly first-order) labels both undetermined."""
    est = _er_instability(q, seeds)
    und = lambda why, kind="continuous": Threshold(q, "er", math.nan, kind, "undetermined", why)
    if not all(math.isfinite(e) for e in est):
        why = "the tilt did not grow or decay cleanly"
        return und(why), und(why)
    ci = float(np.mean(est))
    # below c_d the trivial solution attracts by 0.95 per sweep at 0.95 c_inst: a decade between the first and last
    # fifths of the window (0.8 measure sweeps apart) needs measure >= ln 10 / (0.8 ln(1/0.95)) = 56; take 2 x measure
    probe_pop = pop if pop is not None else (200, 200)
    verdict, why = _verdict([_m1(q, 0.95 * ci, "er", s, probe_pop, 2 * sweeps, 2 * measure) for s in seeds])
    if verdict == "nontrivial":
        why = (f"the frozen start stays nontrivial at 0.95 c_inst = {0.95 * ci:.4g} (a first-order signature; the ER "
               "first-order 1RSB path is not implemented)")
        return und(why, "first-order"), und(why, "first-order")
    if verdict == "undetermined":
        return und(f"probe at 0.95 c_inst: {why}"), und("c_d undetermined")
    b = student_band(est)
    cd = Threshold(q, "er", b.mean, "continuous", "determined", f"the frozen start decays to the trivial solution at "
                   f"0.95 c_inst = {0.95 * ci:.4g} on every seed; c_d = c_inst (growth factor 1), 99% half-width "
                   f"{b.half_width:.3g} over seeds {tuple(seeds)} (statistical)", tuple(est))
    runs = [_m1(q, 1.15 * ci, "er", s, pop, sweeps, measure) for s in seeds]
    lo, hi = _sigma_band(runs)
    verdict, why = _verdict(runs)
    ok = verdict == "nontrivial" and hi < 0
    return cd, Threshold(q, "er", 1.15 * ci if ok else math.nan, "upper-bound", "undetermined",
                         f"c_c NOT located: Sigma(m = 1) at 1.15 c_inst = {1.15 * ci:.4g}: 99% band ({lo:.3g}, "
                         f"{hi:.3g}) {'< 0, so c_d <= c_c <= ' + format(1.15 * ci, '.4g') if ok else why or 'unresolved'}",
                         tuple(r.sigma for r in runs))


def refuse_er_first_order(q):
    """The input check for ER clustering, shared by cluster_thresholds and the catalog's first step."""
    if q >= 4:
        raise Refusal("unsupported:er-first-order", f"q = {q} on ER: the clustering transition is first-order for "
                      "q >= 4 (ZK 2007, arXiv:0704.1269); the ER 1RSB path here does the continuous case q = 3 only "
                      "(the sbm entry at c_in = 0 locates the q = 5 c_d, c_c by K4's planted population)")


def cluster_thresholds(q, graph="regular", *, seeds=(0, 1, 2, 3), pop=None, sweeps=150, measure=60):
    """(c_d, c_c) as Thresholds, from one set of population runs."""
    q, graph = _q3(q), _graph(graph)
    if graph == "er":
        refuse_er_first_order(q)
    if graph == "regular":
        return _regular(q, tuple(seeds), pop, sweeps, measure)
    return _er(q, tuple(seeds), pop, sweeps, measure)


def clustering_threshold(q, graph="regular", **kw):
    return cluster_thresholds(q, graph, **kw)[0]


def condensation_threshold(q, graph="regular", **kw):
    return cluster_thresholds(q, graph, **kw)[1]


__all__ = ["Colorability", "Rigidity", "SPRegular", "Threshold", "cluster_thresholds", "clustering_threshold",
           "colorability_threshold",
           "condensation_threshold", "rigidity_corrections", "rigidity_threshold", "sp_regular"]
