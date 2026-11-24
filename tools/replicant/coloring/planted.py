"""The planted coloring and its link to the stochastic block model (KZ Ch. 13; Krzakala & Zdeborova, PRL 102,
238701 (2009), arXiv:0901.2130, quiet planting).

Convention (KZ Sec. 13.1): q colors, planted colors s*_i i.i.d. uniform on {0..q-1}; M = round(c N / 2) distinct
edges, a fraction f_out = c_out (q - 1)/(c q) = (q - 1)/(q - 1 + e^-beta) of them (rounded to an integer count)
between different planted colors, uniform among such pairs, the rest among same-color pairs.  beta = oo (c_in = 0):
every edge is bichromatic and s* is a proper coloring.  The SBM with n_a = 1/q, c_aa = c_in, c_ab = c_out is the
planted coloring at inverse temperature beta, e^-beta = c_in / c_out (Eq. (13.3)), mean degree
c = (c_in + (q - 1) c_out)/q (KZ Eq. (10.2)); beta > 0 is the disassortative side.

  planted_coloring(n, q, c, *, beta="inf", seed) -> PlantedGraph (labels (n,), edges (M, 2) with i < j)
  sbm_affinities(q, c, beta="inf") -> (c_in, c_out);  beta_from_affinities(c_in, c_out) -> beta (inf at c_in = 0)
  bp_threshold(q, beta="inf") -> c_l = ((q - 1 + e^-beta)/(1 - e^-beta))^2 (Eq. (13.9), read on PDF p. 207: the
      square is on the whole fraction; (q - 1)^2 at beta = oo): randomly initialised BP leaves the paramagnet
  quiet_planting(q, c, *, n, seeds, ...) -> Quiet: BP from the planted start on planted graphs (coloring/sample.py);
      Delta = Phi_planted - Phi_para (Bethe, per node, on each sampled graph: an approximation on a loopy graph);
      verdict "quiet" when the planted start decays to the paramagnet on every seed (Delta = 0) or the 99% Student-t
      band of Delta lies below 0 (c_d < c < c_c: the planted cluster is one of e^{N Sigma}, Sigma = -Delta, KZ
      Eq. (13.11)), "not quiet" when it lies above 0 (c > c_c: the planted state dominates), otherwise undetermined;
      any unconverged run makes the verdict undetermined (never refused).
_HOOKS["cl_power"] (2: the square of Eq. (13.9); planted error 1).
"""
import math
from dataclasses import dataclass

import numpy as np

from ..amp._stats import student_band
from ..core import Refusal
from .potts_bp import _q, beta_value

_HOOKS = {"cl_power": 2}


@dataclass(frozen=True)
class PlantedGraph:
    n: int
    q: int
    c: float
    beta: float
    labels: np.ndarray
    edges: np.ndarray
    seed: int


def _x(beta):
    b = beta_value(beta)
    if not b > 0:
        raise Refusal("precondition:beta", f"beta = {beta!r}: the planted coloring is the disassortative side, beta > 0 "
                      "(or inf)")
    return b, (math.exp(-b) if b < math.inf else 0.0)


def _c(c):
    v = float(c) if not isinstance(c, bool) else math.nan
    if not (math.isfinite(v) and v > 0):
        raise Refusal("precondition:c", f"c = {c!r}: a finite mean degree > 0")
    return v


def sbm_affinities(q, c, beta="inf"):
    q, c = _q(q), _c(c)
    _, x = _x(beta)
    c_out = q * c / (q - 1 + x)
    return x * c_out, c_out


def beta_from_affinities(c_in, c_out):
    if not (float(c_out) > 0 and 0 <= float(c_in) < float(c_out)):
        raise Refusal("precondition:affinity", f"c_in = {c_in!r}, c_out = {c_out!r}: need 0 <= c_in < c_out")
    return math.inf if float(c_in) == 0 else -math.log(float(c_in) / float(c_out))


def bp_threshold(q, beta="inf"):
    q = _q(q)
    _, x = _x(beta)
    return ((q - 1 + x) / (1 - x)) ** _HOOKS["cl_power"]


def _pairs(rng, n, k, keep):
    """k distinct pairs (i < j) of range(n) with keep(i, j), uniform: draw, reject, deduplicate, repeat."""
    keys = np.zeros(0, dtype=np.int64)
    while len(keys) < k:
        a, b = rng.integers(n, size=(2, 2 * (k - len(keys)) + 16))
        ok = (a != b) & keep(a, b)
        keys = np.concatenate([keys, np.minimum(a, b)[ok].astype(np.int64) * n + np.maximum(a, b)[ok]])
        _, first = np.unique(keys, return_index=True)
        keys = keys[np.sort(first)]
    keys = keys[:k]
    return np.stack([keys // n, keys % n], 1)


def planted_coloring(n, q, c, *, beta="inf", seed):
    q, c = _q(q), _c(c)
    b, x = _x(beta)
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2 * q:
        raise Refusal("precondition:size", f"n = {n!r}: an int >= 2 q")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise Refusal("precondition:named-seeds", f"seed {seed!r}: a non-negative int")
    n = int(n)
    rng = np.random.default_rng([int(seed), 31])
    labels = rng.integers(q, size=n).astype(np.int64)
    m = int(round(c * n / 2))
    m_out = int(round((q - 1) / (q - 1 + x) * m))
    sizes = np.bincount(labels, minlength=q)
    n_mono = int(np.sum(sizes * (sizes - 1) // 2))
    n_bi = n * (n - 1) // 2 - n_mono
    if m_out > n_bi // 2 or m - m_out > n_mono // 2:
        raise Refusal("precondition:size", f"n = {n} is too small for M = {m} edges at c = {c}")
    out = _pairs(rng, n, m_out, lambda a, b: labels[a] != labels[b])
    inn = _pairs(rng, n, m - m_out, lambda a, b: labels[a] == labels[b])
    e = np.concatenate([out, inn]).astype(np.int64)
    return PlantedGraph(n, q, c, b, labels, e[np.lexsort((e[:, 1], e[:, 0]))], int(seed))


@dataclass(frozen=True)
class Quiet:
    q: int
    c: float
    verdict: str              # "quiet" | "not quiet" | "undetermined"
    status: str               # "determined" | "undetermined"
    delta: float              # mean of Phi_planted - Phi_para over seeds
    half_width: float
    overlaps: tuple
    note: str


def quiet_planting(q, c, *, n, seeds=(0, 1, 2, 3), iters=3000):
    from . import sample as S                       # sample imports this module (the generator)
    runs = S.planted_bp(q, c, n=n, seeds=seeds, iters=iters)
    ov = tuple(r.overlap for r in runs)
    bad = [f"seed {s}: {r.note}" for s, r in zip(seeds, runs) if not r.converged]
    deltas = [r.phi - r.phi_para for r in runs]
    b = student_band(deltas)
    if bad:
        return Quiet(q, float(c), "undetermined", "undetermined", b.mean, b.half_width, ov, "; ".join(bad))
    if all(abs(d) < 1e-9 for d in deltas):
        return Quiet(q, float(c), "quiet", "determined", b.mean, b.half_width, ov,
                     "the planted start decays to the paramagnet on every seed (c < c_d)")
    if b.mean + b.half_width < 0:
        return Quiet(q, float(c), "quiet", "determined", b.mean, b.half_width, ov, "Phi_planted < Phi_para in the 99% "
                     "band: the planted cluster is subdominant (c_d < c < c_c; Bethe on sampled graphs: approximation)")
    if b.mean - b.half_width > 0:
        return Quiet(q, float(c), "not quiet", "determined", b.mean, b.half_width, ov, "Phi_planted > Phi_para in the "
                     "99% band: the planted state dominates (c > c_c; Bethe on sampled graphs: approximation)")
    return Quiet(q, float(c), "undetermined", "undetermined", b.mean, b.half_width, ov,
                 f"the 99% band of Phi_planted - Phi_para ({b.mean:.3g} +- {b.half_width:.3g}) contains 0")


__all__ = ["PlantedGraph", "Quiet", "beta_from_affinities", "bp_threshold", "planted_coloring", "quiet_planting",
           "sbm_affinities"]
