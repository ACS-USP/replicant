"""Population dynamics of the symmetric q-group block model at the planted (Nishimori) measure, and the first-order
thresholds c_d, c_c along a line of fixed c_in (KZ Sec. 10.4.3; DKMZ Sec. IV.B, Fig. 3).

Model (inference/sbm_q.py): n_a = 1/q, c_aa = c_in, c_ab = c_out, mean degree c = (c_in + (q - 1) c_out)/q,
Poisson degrees (the N -> oo law of the sampled graphs).  On the line c_in fixed, c_out = (q c - c_in)/(q - 1).
By the colour symmetry one population represents the BP messages sent by a node of planted colour 0; a message from
a neighbour of planted colour t is a population member with the components 0 and t swapped (then the components
1..q-1 of every new member are permuted at random, so the population stays symmetric under the permutations that fix
0).  For Poisson degrees the excess degree and the degree have the same law, so the population is also a sample of
the full marginals psi of a colour-0 node.  One sweep replaces the whole population:
    chi ~ n e^{-h} prod_{k=1}^{d} C chi_k,  d ~ Poisson(c),  colour of k ~ c_{0 t} n_t / c,
with h = C n (the field of the factorized block model).  Observables (averaged over the last `measure` sweeps):
    m   = (E chi_0 - 1/q) / (1 - 1/q)                (the planted-colour magnetisation; 0 at the paramagnet)
    Phi = E ln Z^i - (c/2) E ln Z^{ij} + c/2          (KZ Eq. (10.23) in the N -> oo average)
with ln Z^i = ln sum_t n_t e^{-h_t} prod_k (C chi_k)_t and Z^{ij} = chi_i . C chi_j over a planted edge.  At the
factorized point Phi = -(c/2)(1 - ln c) (KZ Eq. (10.31)), the reference of Delta Phi.  The estimator averages
ln Z^i - (d ln c - c) and ln (Z^{ij} / c), whose factorized values are 0 (a control variate: the degree noise
d ln c, standard deviation sqrt(c) ln c per member, cancels), and adds Phi_para back.
Starts: "planted" (every member one-hot on the planted colour, KZ Eq. (10.38)) and "random" (independent uniform
points of the simplex, uncorrelated with the planted colour).
Thresholds (planted_informative, sbm_q_transitions): c_d is where the planted start stops decaying to the paramagnet
(bisection on m after `sweeps`; finite sweeps bias c_d low near the spinodal, the note says by how much the last
bracket moved); c_c is the zero of Delta Phi = Phi_planted - Phi_para on the informative branch (bisection); c_l is the
KS point (inference/sbm_q.sbm_q_ks_degree).  The random start must stay at the paramagnet on [c_d, c_l) for the pair
to show a first-order transition; otherwise the thresholds are undetermined, with the reason.  Statistical: one
seed gives one estimate; bands come from named seeds.
_HOOKS["swap"] (True): the colour relabelling of incoming messages (planted error: False feeds colour-0 messages to
every neighbour, which breaks the planted measure); _HOOKS["degree"] (1.0): the factor on the Poisson mean (planted
error 0.8 = (q - 1)/q at q = 5: reading the mean degree c as c_out, the convention slip of review focus 2).
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from ..core import Refusal
from ..csp import _checks
from . import sbm_q as SQ

_HOOKS = {"swap": True, "degree": 1.0}
INFORMATIVE = 0.05          # m above this after the sweeps: the planted start did not decay to the paramagnet
PARA = 1e-9                 # every member within this of 1/q: the run is at the factorized point and stops


@dataclass(frozen=True)
class PDResult:
    c: float
    m: float
    phi: float
    phi_para: float
    phi_stderr: float
    contradictions: int
    sweeps: int


def _swap(chi, t):
    if not _HOOKS["swap"]:
        return chi
    out = chi.copy()
    r = np.arange(len(chi))
    out[r, 0], out[r, t] = chi[r, t], chi[r, 0]
    return out


def _logs(W):
    z = W <= 0
    with np.errstate(divide="ignore"):
        return np.where(z, 0.0, np.log(np.where(z, 1.0, W))), z


def population(q, c_in, c, *, init, pop, sweeps, measure, seed):
    """Planted-measure population dynamics at mean degree c on the line c_in fixed -> PDResult."""
    q, seed = SQ.group_count(q), _checks.seed(seed)
    c = float(c)
    if not (c > 0 and float(c_in) <= q * c):
        raise Refusal("precondition:affinity", f"c = {c}: need c > 0 and c_in <= q c (c_out >= 0)")
    for name, v in (("pop", pop), ("sweeps", sweeps), ("measure", measure)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise Refusal("precondition:population", f"{name} = {v!r}: a positive int")
    if measure > sweeps:
        raise Refusal("precondition:population", f"measure = {measure} > sweeps = {sweeps}")
    C = SQ.affinity(q, c_in, (q * c - float(c_in)) / (q - 1))
    n = np.full(q, 1.0 / q)
    base = np.log(n) - n @ C
    p_nb = C[0] * n / c
    cum = np.cumsum(p_nb)
    rng = np.random.default_rng([seed, 21])
    if init == "planted":
        P = np.zeros((pop, q))
        P[:, 0] = 1.0
    elif init == "random":
        x = -np.log(rng.random((pop, q)))
        P = x / x.sum(1, keepdims=True)
    else:
        raise Refusal("precondition:init", f"init = {init!r}: 'planted' or 'random'")
    ms, phis, bad = [], [], 0
    for s in range(sweeps):
        d = rng.poisson(_HOOKS["degree"] * c, pop)
        K = int(d.sum())
        own = csr_matrix((np.ones(K), np.arange(K), np.r_[0, np.cumsum(d)]), shape=(pop, K))
        t = np.minimum(np.searchsorted(cum, rng.random(K), side="right"), q - 1)
        L, z = _logs(_swap(P[rng.integers(pop, size=K)], t) @ C)
        S, Z = own @ L, own @ z.astype(float)
        lw = base + S
        lw[Z > 0.5] = -np.inf
        mx = lw.max(1, keepdims=True)
        ok = np.isfinite(mx[:, 0])
        bad += int((~ok).sum())
        w = np.exp(lw[ok] - mx[ok])
        new = P.copy()
        new[ok] = w / w.sum(1, keepdims=True)
        perm = np.argsort(rng.random((pop, q - 1)), 1) + 1
        new[:, 1:] = np.take_along_axis(new, perm, 1)
        if s >= sweeps - measure:
            lzi = mx[ok, 0] + np.log(w.sum(1)) - d[ok] * math.log(c) + c        # ln Z^i - ln Z^i_para
            te = np.minimum(np.searchsorted(cum, rng.random(pop), side="right"), q - 1)
            zij = np.sum(P[rng.integers(pop, size=pop)] * (_swap(P[rng.integers(pop, size=pop)], te) @ C), 1)
            phis.append(float(np.mean(lzi) - c / 2 * np.mean(np.log(zij / c))))
            ms.append((float(np.mean(new[:, 0])) - 1 / q) / (1 - 1 / q))
        P = new
        if s < sweeps - measure and np.max(np.abs(P.mean(0) - 1 / q)) < PARA and np.max(np.abs(P - 1 / q)) < PARA:
            ms, phis = [0.0], [0.0]              # at the paramagnet (stable below c_l): nothing moves any more
            break
    se = float(np.std(phis, ddof=1) / math.sqrt(len(phis))) if len(phis) > 1 else float("nan")
    para = -(c / 2) * (1 - math.log(c))
    return PDResult(c, float(np.mean(ms)), para + float(np.mean(phis)), para, se, bad, s + 1)


@dataclass(frozen=True)
class Transitions:
    q: int
    c_in: float
    c_d: object               # float, or None when undetermined
    c_c: object
    c_l: float                # KS point (closed-form condition, bisection at 30 digits)
    status: str               # "determined" | "undetermined"
    note: str
    probes: tuple             # (c, init, m, Delta Phi, stderr of Delta Phi) of every population run


def sbm_q_transitions(q, c_in, *, seed, pop=3000, sweeps=400, equil=100, measure=40, steps=8, grid=5):
    """(c_d, c_c, c_l) on the line c_in fixed, from one population seed.
    c_d: bisection (`steps` halvings of [c_l / 2, c_l]) of "the planted start is informative after `sweeps`
    sweeps"; the random start is then run at the midpoint of [c_d, c_l) and must decay to the paramagnet.
    c_c: Delta Phi of the planted start on a `grid` of points of [c_d+, c_l], then on `grid` points across the first
    sign change, and the root of the least-squares line through the fine points (each run: `equil` sweeps, then
    `measure` measured ones)."""
    q, seed = SQ.group_count(q), _checks.seed(seed)
    c_l = float(SQ.sbm_q_ks_degree(q, c_in))
    probes, notes = [], []

    def probe(c, init, meas):
        sw = sweeps if meas == 5 else equil + meas
        r = population(q, c_in, c, init=init, pop=pop, sweeps=sw, measure=meas, seed=seed)
        probes.append((c, init, r.m, r.phi - r.phi_para, r.phi_stderr))
        if r.contradictions:
            notes.append(f"{r.contradictions} contradictory members at c = {c:.4g} ({init})")
        return r

    def out(c_d, c_c, status, why=""):
        return Transitions(q, float(c_in), c_d, c_c, c_l, status, "; ".join(([why] if why else []) + notes),
                           tuple(probes))

    lo, hi = c_l / 2, c_l
    if probe(lo, "planted", 5).m > INFORMATIVE or probe(hi, "planted", 5).m <= INFORMATIVE:
        return out(None, None, "undetermined", f"the planted start does not change behaviour on [{lo:.4g}, "
                   f"{hi:.4g}]: no c_d bracket")
    for _ in range(steps):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if probe(mid, "planted", 5).m > INFORMATIVE else (mid, hi)
    c_d = (lo + hi) / 2
    notes.append(f"c_d bracket [{lo:.4g}, {hi:.4g}] ({steps} bisections, {sweeps} sweeps of {pop} members per probe)")
    if c_l - hi < 4 * (hi - lo):
        return out(c_d, None, "undetermined", "c_d is within resolution of c_l: no first-order window for c_c")
    c_r = (c_d + c_l) / 2
    rnd = probe(c_r, "random", 5)
    if rnd.m > INFORMATIVE:
        return out(c_d, None, "undetermined", f"the random start is informative (m = {rnd.m:.3g}) at c = {c_r:.4g} "
                   "< c_l: no planted / random pair")
    notes.append(f"random start at c = {c_r:.4g}: m = {rnd.m:.2g} (paramagnet)")
    xs = np.linspace(hi, c_l, 2 * grid - 1)
    for x in xs:
        probe(x, "planted", measure)
    ds = [p[3] for p in probes[-len(xs):]]
    sign = [k for k in range(1, len(xs)) if ds[k - 1] < 0 <= ds[k]]
    if ds[0] >= 0 or not sign or any(p[2] <= INFORMATIVE for p in probes[-len(xs):]):
        return out(c_d, None, "undetermined", f"Delta Phi on the planted branch does not cross 0 from below on "
                   f"[{hi:.4g}, {c_l:.4g}] (values {', '.join(f'{d:.3g}' for d in ds)})")
    a, b = xs[sign[0] - 1], xs[sign[0]]
    fx = np.linspace(a, b, grid)
    for x in fx:
        probe(x, "planted", measure)
    fd = np.array([p[3] for p in probes[-grid:]])
    slope, icpt = np.polyfit(fx, fd, 1)
    if not slope > 0:
        return out(c_d, None, "undetermined", f"the least-squares line of Delta Phi on [{a:.4g}, {b:.4g}] has slope "
                   f"{slope:.3g} <= 0")
    notes.append(f"c_c: line through Delta Phi at {grid} points of [{a:.4g}, {b:.4g}], slope {slope:.3g}, {measure} "
                 "measured sweeps per point")
    return out(c_d, float(-icpt / slope), "determined")


__all__ = ["INFORMATIVE", "PDResult", "Transitions", "population", "sbm_q_transitions"]
