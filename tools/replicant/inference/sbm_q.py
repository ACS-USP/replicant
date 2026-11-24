"""The stochastic block model with q groups (KZ Ch. 10; Decelle, Krzakala, Moore & Zdeborova, Phys. Rev. E 84,
066106 (2011), arXiv:1109.3041, "DKMZ"): a sampler, belief propagation with the non-edge field, the Bethe free
entropy, and the Kesten-Stigum point.

Convention (DKMZ Sec. II; KZ Eq. (10.5)): labels s_i in {0, ..., q - 1} i.i.d. with probabilities n_a; each pair
{i, j} is an edge independently with probability c_{s_i s_j} / N.  The symmetric case: n_a = 1/q, c_aa = c_in,
c_ab = c_out (a != b); mean degree c = sum_ab c_ab n_a n_b = (c_in + (q - 1) c_out) / q (KZ Eq. (10.2)); the
notes' epsilon = c_out / c_in.  Map to inference/sbm.py (q = 2, edge probability a/N inside, b/N across):
c_in = a, c_out = b, c = (a + b) / 2.  This module shares no code with inference/sbm.py (the q = 2 cross-check
in tests/test_sbm_q.py goes through two implementations; an AST audit checks the imports).

BP (KZ Eqs. (10.15)-(10.17), DKMZ (26)-(28)), messages chi^{i->j} on the 2M directed edges:
    chi^{i->j}_t ~ n_t e^{-h_t} prod_{k in di \\ j} sum_s c_{t s} chi^{k->i}_s,
    h_t = (1/N) sum_k sum_s c_{s t} psi^k_s,   psi^i_t ~ n_t e^{-h_t} prod_{k in di} sum_s c_{t s} chi^{k->i}_s.
Update (synchronous): from (chi(t), h(t)) compute the marginals psi(t) and the new messages chi(t+1) with the same
field h(t), then h(t+1) from psi(t); damping d mixes chi and h alike, (1 - d) new + d old (the map then has the
eigenvalue d, d on the uniform mode below; damping chi alone makes it (d (1 - kappa) +- ...)/2, unstable at
kappa = -4, d = 1/2).  For the uniform (global) mode delta chi = x of the factorized fixed point this
map is nilpotent (x' = kappa x - y, y' = kappa x' with y = delta h / q, kappa = (c_in - c_out)/q), so neither the
assortative period-2 mode nor the disassortative mode of the coloring BP (|kappa| > 1) appears; a field solved
self-consistently with the marginals would give x' = kappa x / (1 + kappa), unstable for kappa < -1/2.
Logs are taken with the zeros counted apart (c_in = 0 and planted one-hot messages give exact zeros).
Default damping 1/2 (sbm_q_bp): from a random start above the KS point at q = 5, c_in = 0 the undamped
synchronous update oscillates (N = 1e4, c = 17: max |change| = 1 after 1000 iterations; converged at d = 1/2).
init: "random" (each message an independent uniform point of the simplex, DKMZ's random initial condition) or
"planted" (chi^{i->j} = delta_{t, s*_i}, KZ Eq. (10.38)).  Convergence: max |chi(t+1) - chi(t)| and
max |h(t+1) - h(t)| / max(1, max |h|) < tol.
Overlap (KZ Eq. (10.4)): max over permutations (an assignment problem) of
((1/N) sum_i delta(s*_i, pi(shat_i)) - max n) / (1 - max n), shat_i = argmax psi^i (groups within TIE = 1e-6 of
the largest marginal are ties, broken at random by the seed; `factorized`: every |psi^i - n| < TIE);
`overlap_nishimori` is Eq. (10.19), computed from the marginals alone.
Bethe free entropy (KZ Eq. (10.23) / (10.44)): Phi = (1/N) sum_i ln Z^i - (1/N) sum_{(ij) in E} ln Z^{ij} + c/2;
on a sampled (loopy) graph it is an approximation, and undetermined when BP did not converge.  Factorized value on
the same graph: Phi_para = (M/N) ln c - c + c/2, which is KZ Eq. (10.31) -(c/2)(1 - ln c) when 2M/N = c.
Kesten-Stigum (KZ Eqs. (10.36)-(10.37)): c lambda_2^2 > 1, lambda_2 = (c_in - c_out)/(q c), i.e.
|c_in - c_out| > q sqrt(c); margin tau = (c_in - c_out)^2 / (q^2 c).  Finite-N values are statistical.
_HOOKS["field"] (1.0): the coefficient of h in the updates (planted error: 0 drops the non-edge field).
"""
import math
from dataclasses import dataclass

import mpmath as mp
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix

from ..core import Refusal
from ..csp import _checks

_HOOKS = {"field": 1.0}
TIE = 1e-6                   # marginals within this of the largest are a tie (the factorized test uses it too)


# ---- parameters ---------------------------------------------------------------------------------
def _num(name, x, *, positive=False):
    if isinstance(x, bool) or x is None:
        raise Refusal("precondition:affinity", f"{name} = {x!r}: a finite number >= 0")
    try:
        v = mp.mpf(x)
    except (TypeError, ValueError):
        raise Refusal("precondition:affinity", f"{name} = {x!r}: a finite number >= 0") from None
    if not mp.isfinite(v) or v < 0 or (positive and v == 0):
        raise Refusal("precondition:affinity", f"{name} = {x!r}: a finite number {'> 0' if positive else '>= 0'}")
    return v


def group_count(q):
    if isinstance(q, bool) or not isinstance(q, (int, np.integer)) or q < 2:
        raise Refusal("precondition:group-count", f"q = {q!r}: an int >= 2")
    return int(q)


def affinity(q, c_in, c_out):
    """The symmetric (q, q) affinity matrix (floats) after the parameter checks."""
    q, ci, co = group_count(q), _num("c_in", c_in), _num("c_out", c_out)
    if ci + co == 0:
        raise Refusal("precondition:affinity", "c_in + c_out = 0: the graph has no edges")
    C = np.full((q, q), float(co))
    np.fill_diagonal(C, float(ci))
    return C


def mean_degree(q, c_in, c_out):
    q = group_count(q)
    with mp.workdps(30):
        return (_num("c_in", c_in) + (q - 1) * _num("c_out", c_out)) / q


@dataclass(frozen=True)
class KSq:
    q: int
    c_in: object
    c_out: object
    mean_degree: object
    lambda2: object           # (c_in - c_out) / (q c)
    margin: object            # c lambda_2^2 = (c_in - c_out)^2 / (q^2 c); unstable factorized point iff > 1
    detectable: bool


def sbm_q_kesten_stigum(q, c_in, c_out) -> KSq:
    q = group_count(q)
    affinity(q, c_in, c_out)
    ci, co = _num("c_in", c_in), _num("c_out", c_out)
    with mp.workdps(30):
        c = (ci + (q - 1) * co) / q
        lam = (ci - co) / (q * c)
        tau = c * lam**2
    return KSq(q, ci, co, c, lam, tau, bool(tau > 1))


@dataclass(frozen=True)
class KSqLine:
    q: int
    c: object
    c_in: object
    c_out: object
    ratio: object             # c_in / c_out on the line = 1 / eps_c (KZ caption of Fig. 10.4.1)


def sbm_q_ks_line(q, c):
    """The assortative KS point at mean degree c: bisection of tau - 1 in c_in on [c, q c] with
    c_out = (q c - c_in)/(q - 1) (tau rises from 0 at c_in = c to tau = c at c_in = q c, c_out = 0; so c <= 1 has
    no KS point), 120 halvings at 30 digits."""
    q, c = group_count(q), _num("c", c, positive=True)
    with mp.workdps(30):
        top = sbm_q_kesten_stigum(q, q * c, 0).margin
        if top <= 1:
            raise Refusal("no-transition:ks-line", f"q = {q}, c = {c}: tau <= {mp.nstr(top, 6)} <= 1 on the line")
        lo, hi = c, q * c
        for _ in range(120):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if sbm_q_kesten_stigum(q, mid, (q * c - mid) / (q - 1)).margin > 1 else (mid, hi)
        ci = (lo + hi) / 2
        co = (q * c - ci) / (q - 1)
        return KSqLine(q, c, ci, co, ci / co)


def sbm_q_ks_degree(q, c_in):
    """c_l: the mean degree at which the factorized point turns unstable along c_in fixed, c_out = (q c - c_in)/(q-1),
    c > c_in (the disassortative side when c_in < c): bisection of tau(c) - 1 on [c_in, hi], tau increasing there."""
    q, ci = group_count(q), _num("c_in", c_in)
    with mp.workdps(30):
        tau = lambda c: sbm_q_kesten_stigum(q, ci, (q * c - ci) / (q - 1)).margin
        lo, hi = ci, max(ci, mp.mpf(1)) * 4 + (q - 1)**2 * 4
        if tau(hi) <= 1:
            raise Refusal("no-transition:ks-line", f"tau <= 1 up to c = {mp.nstr(hi, 6)}")
        for _ in range(120):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if tau(mid) > 1 else (mid, hi)
        return (lo + hi) / 2


# ---- sampler ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SBMqGraph:
    N: int
    labels: np.ndarray        # int64 in 0..q-1
    edges: np.ndarray         # (M, 2) int64, i < j, distinct
    seed: object = None


def from_edges(N, edges, labels):
    """An SBMqGraph from given edges (M, 2) and labels in 0..q-1 (e.g. a graph sampled elsewhere)."""
    e = np.sort(np.asarray(edges, dtype=np.int64).reshape(-1, 2), axis=1)
    lab = np.asarray(labels, dtype=np.int64)
    if lab.shape != (N,) or (e.size and (e.min() < 0 or e.max() >= N or np.any(e[:, 0] == e[:, 1]))):
        raise Refusal("precondition:sbm-graph", "edges must join distinct nodes in 0..N-1, labels one per node")
    return SBMqGraph(int(N), lab, e)


def _distinct_pairs(rng, m, k):
    """k distinct unordered pairs of range(m), uniformly: draw, drop i == j, deduplicate, repeat."""
    keys = np.zeros(0, dtype=np.int64)
    while len(keys) < k:
        i, j = rng.integers(m, size=(2, 2 * (k - len(keys)) + 8))
        ok = i != j
        keys = np.concatenate([keys, np.minimum(i, j)[ok].astype(np.int64) * m + np.maximum(i, j)[ok]])
        _, first = np.unique(keys, return_index=True)
        keys = keys[np.sort(first)]
    keys = keys[:k]
    return keys // m, keys % m


def sbm_q_sample(N, q, c_in, c_out, seed, sizes=None) -> SBMqGraph:
    """Labels from default_rng([seed, 11]) with probabilities `sizes` (default 1/q each); per block pair an exact
    Binomial edge count and a uniform set of that many distinct pairs (each pair an independent Bernoulli)."""
    if isinstance(N, bool) or not isinstance(N, int) or N < 10:
        raise Refusal("precondition:size", f"N = {N!r} must be an int >= 10")
    seed, C = _checks.seed(seed), affinity(q, c_in, c_out)
    q = C.shape[0]
    n = np.full(q, 1.0 / q) if sizes is None else np.asarray(sizes, dtype=float)
    if n.shape != (q,) or np.any(n < 0) or abs(n.sum() - 1) > 1e-12:
        raise Refusal("precondition:group-sizes", f"sizes must be q = {q} probabilities summing to 1")
    if C.max() > N:
        raise Refusal("precondition:edge-probability", f"max c_ab / N = {C.max() / N:.3g} must be <= 1")
    rng = np.random.default_rng([seed, 11])
    labels = rng.choice(q, size=N, p=n).astype(np.int64)
    groups = [np.flatnonzero(labels == a) for a in range(q)]
    parts = []
    for a in range(q):
        ga = groups[a]
        k = int(rng.binomial(len(ga) * (len(ga) - 1) // 2, C[a, a] / N)) if len(ga) > 1 else 0
        i, j = _distinct_pairs(rng, len(ga), k)
        parts.append(np.stack([ga[i], ga[j]], 1))
        for b in range(a + 1, q):
            gb = groups[b]
            P = len(ga) * len(gb)
            k = int(rng.binomial(P, C[a, b] / N)) if P else 0
            t = rng.choice(P, size=k, replace=False) if k else np.zeros(0, dtype=np.int64)
            parts.append(np.stack([ga[t // max(len(gb), 1)], gb[t % max(len(gb), 1)]], 1))
    E = np.sort(np.concatenate(parts).astype(np.int64), axis=1)
    return SBMqGraph(N, labels, E[np.lexsort((E[:, 1], E[:, 0]))], seed)


# ---- belief propagation ---------------------------------------------------------------------------
@dataclass
class BPqResult:
    overlap: float
    overlap_nishimori: float
    converged: bool
    iterations: int
    max_change: float
    factorized: bool
    marginals: np.ndarray     # (N, q)
    messages: np.ndarray      # (2M, q): e < M edges[e, 0] -> edges[e, 1], e >= M the reverse
    field: np.ndarray         # h (q,)
    phi: float                # Bethe free entropy per node (KZ Eq. (10.23))
    phi_para: float           # its factorized value on the same graph
    phi_label: str
    note: str = ""
    label: str = "statistical"


def _logs(W):
    z = W <= 0
    with np.errstate(divide="ignore"):
        return np.where(z, 0.0, np.log(np.where(z, 1.0, W))), z.astype(float)


def _softmax(lw):
    mx = lw.max(1, keepdims=True)
    bad = ~np.isfinite(mx[:, 0])
    w = np.exp(lw - np.where(np.isfinite(mx), mx, 0.0))
    s = w.sum(1, keepdims=True)
    return w / np.where(s > 0, s, 1.0), bad


def _sums(inc, A):
    return np.asarray(inc @ A)


class _Wiring:
    def __init__(self, graph):
        E = graph.edges
        M = len(E)
        self.N, self.M = graph.N, M
        self.src = np.concatenate([E[:, 0], E[:, 1]])
        self.dst = np.concatenate([E[:, 1], E[:, 0]])
        self.rev = np.concatenate([np.arange(M, 2 * M), np.arange(M)])
        self.inc = csr_matrix((np.ones(2 * M), (self.dst, np.arange(2 * M))), shape=(graph.N, 2 * M))


def _sweep(w, n, C, chi, h):
    """One synchronous update: (marginals psi(t), new messages chi(t+1), new field h(t+1), contradiction flag)."""
    L, z = _logs(chi @ C)                       # (C chi^{k->i})_t: C symmetric
    S, Z = _sums(w.inc, L), _sums(w.inc, z)
    base = np.log(n)[None, :] - _HOOKS["field"] * h[None, :]
    lm = base + S
    lm[Z > 0.5] = -np.inf
    psi, bad1 = _softmax(lm)
    lc = base + S[w.src] - L[w.rev]
    lc[(Z[w.src] - z[w.rev]) > 0.5] = -np.inf
    new, bad2 = _softmax(lc)
    return psi, new, psi.mean(0) @ C, bool(bad1.any() or bad2.any())


def _overlap(graph, psi, n, seed):
    q = psi.shape[1]
    tie = psi >= psi.max(1, keepdims=True) - TIE              # groups within TIE of the largest marginal
    est = np.argmax(np.where(tie, 1.0 + np.random.default_rng([seed, 13]).random(psi.shape), 0.0), axis=1)
    conf = np.zeros((q, q))
    np.add.at(conf, (graph.labels, est), 1.0)
    r, c = linear_sum_assignment(-conf)
    agree = conf[r, c].sum() / graph.N
    nmax = float(np.max(n))
    nish = float(np.mean(psi[np.arange(graph.N), est]))
    return (agree - nmax) / (1 - nmax), (nish - nmax) / (1 - nmax)


def bethe(graph, n, C, chi, h, w=None):
    """KZ Eq. (10.23) at messages chi and field h (per node)."""
    w = w or _Wiring(graph)
    L, z = _logs(chi @ C)
    S, Z = _sums(w.inc, L), _sums(w.inc, z)
    lm = np.log(n)[None, :] - h[None, :] + S
    lm[Z > 0.5] = -np.inf
    mx = lm.max(1, keepdims=True)
    with np.errstate(divide="ignore"):
        lzi = (mx[:, 0] + np.log(np.exp(lm - np.where(np.isfinite(mx), mx, 0)).sum(1)))
        zij = np.sum(chi[:w.M] * (chi[w.M:] @ C), 1)
        lzij = np.log(zij)
    c = float(n @ C @ n)
    return float((lzi.sum() - lzij.sum()) / w.N + c / 2)


def phi_para(graph, n, C):
    """Factorized Bethe free entropy on this graph, closed form: (M/N) ln c - c + c/2 (degree-balanced C)."""
    c = float(n @ C @ n)
    return len(graph.edges) / graph.N * math.log(c) - c + c / 2


def _bp(graph, n, C, *, init, iters, seed, tol, damping, chi0=None):
    w = _Wiring(graph)
    q = len(n)
    if chi0 is not None:
        chi = chi0.copy()
    elif init == "random":
        x = -np.log(np.random.default_rng([seed, 12]).random((2 * w.M, q)))
        chi = x / x.sum(1, keepdims=True)
    elif init == "planted":
        chi = np.zeros((2 * w.M, q))
        chi[np.arange(2 * w.M), graph.labels[w.src]] = 1.0
    else:
        raise Refusal("precondition:init", f"init = {init!r}: 'planted' or 'random'")
    h = n @ C
    change, it, conv, note = math.inf, 0, False, ""
    for it in range(1, iters + 1):
        psi, new, h_new, bad = _sweep(w, n, C, chi, h)
        if bad:
            note = f"contradiction at iteration {it}: a node or message has every group forbidden"
            break
        if damping:
            new = (1 - damping) * new + damping * chi
            h_new = (1 - damping) * h_new + damping * h
        change = max(float(np.max(np.abs(new - chi))) if w.M else 0.0,
                     float(np.max(np.abs(h_new - h))) / max(1.0, float(np.max(np.abs(h)))))
        chi, h = new, h_new
        if change < tol:
            conv = True
            break
    psi, _, _, _ = _sweep(w, n, C, chi, h)
    if not conv and not note:
        note = f"not converged in {iters} iterations: last max |change| = {change:.3g} (tol {tol:g})"
    return chi, h, psi, conv, it, change, note, w


def _iters(iters, damping, tol):
    if isinstance(iters, bool) or not isinstance(iters, int) or iters < 1:
        raise Refusal("precondition:positive-iters", f"iters = {iters!r} is not a positive int")
    if not 0 <= float(damping) < 1:
        raise Refusal("precondition:damping", f"damping = {damping!r} outside [0, 1)")
    if not float(tol) > 0:
        raise Refusal("precondition:tolerance", f"tol = {tol!r} must be > 0")


def run_bp(graph, n, C, *, init="random", iters=500, seed=0, tol=1e-7, damping=0.0, chi0=None) -> BPqResult:
    """BP at given group sizes n (q,) and affinity C (q, q) (sbm_learn uses this with learned parameters)."""
    if not isinstance(graph, SBMqGraph):
        raise Refusal("precondition:sbm-graph", "sbm_q_bp needs an SBMqGraph (sbm_q_sample or from_edges)")
    _iters(iters, damping, tol)
    seed = _checks.seed(seed)
    if graph.labels.size and graph.labels.max() >= len(n):
        raise Refusal("precondition:sbm-graph", f"labels exceed q = {len(n)}")
    chi, h, psi, conv, it, change, note, w = _bp(graph, n, C, init=init, iters=iters, seed=seed, tol=tol,
                                                damping=float(damping), chi0=chi0)
    Q, Qn = _overlap(graph, psi, n, seed)
    phi = bethe(graph, n, C, chi, h, w)
    lab = (f"undetermined: BP not at a fixed point ({note})" if not conv else
           "approximation: Bethe free entropy on a sampled (loopy) graph, not ln Z / N")
    return BPqResult(Q, Qn, conv, it, change, bool(np.max(np.abs(psi - n)) < TIE), psi, chi, h, phi,
                     phi_para(graph, n, C), lab, note)


def sbm_q_bp(graph, q, c_in, c_out, *, init, iters, seed, tol=1e-7, damping=0.5) -> BPqResult:
    C = affinity(q, c_in, c_out)
    return run_bp(graph, np.full(C.shape[0], 1.0 / C.shape[0]), C, init=init, iters=iters, seed=seed, tol=tol,
                  damping=damping)


__all__ = ["BPqResult", "KSq", "KSqLine", "SBMqGraph", "affinity", "bethe", "from_edges", "group_count",
           "mean_degree", "phi_para", "run_bp", "sbm_q_bp", "sbm_q_kesten_stigum", "sbm_q_ks_degree",
           "sbm_q_ks_line", "sbm_q_sample"]
