"""K4: the non-backtracking matrix of block-model graphs (KZ Sec. 10.4.4; Krzakala, Moore, Mossel, Neeman, Sly,
Zdeborova & Zhang, PNAS 110, 20935 (2013), arXiv:1306.5550, "KMMNSZZ").

Convention: inference/sbm_q.py (edge probability c_ab / N, c_in inside, c_out across, mean degree
c = (c_in + (q - 1) c_out)/q).  KMMNSZZ (N -> oo, argued): leading eigenvalue c, q - 1 real eigenvalues near
mu_c = (c_in - c_out)/q outside the bulk disk |mu| <= sqrt(c) iff the KS condition |c_in - c_out| > q sqrt(c).
Statistic per graph (`second`): the modulus of the first real eigenvalue after the leading one among the k
leading eigenvalues, or, when none is real, the modulus of the first one (then it is a bulk eigenvalue).
"""
import math
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.amp._stats import student_band
from replicant.core import Refusal
from replicant.inference import nonbacktracking as NB
from replicant.inference import sbm_q as SQ
from replicant.refdata import load
from replicant.testing import slow

BANK = load("sbm")


def _second(vals):
    real = [v for v in vals[1:] if abs(v.imag) < 1e-8 * max(1.0, abs(v))]
    return abs(real[0]) if real else abs(vals[1])


def _stats(N, q, c_in, c_out, seeds, k=4):
    """(leading eigenvalue, `second`, sqrt(2M/N), spectral overlap) per seed."""
    out = []
    for s in seeds:
        g = SQ.sbm_q_sample(N, q, c_in, c_out, s)
        sp = NB.nonbacktracking_spectrum(g, k)
        out.append((sp.eigenvalues[0].real, _second(sp.eigenvalues), math.sqrt(sp.mean_degree),
                    NB.cluster(g, q, sp, seed=s).overlap))
    return [np.array(x) for x in zip(*out)]


def test_ihara_bass_matches_the_full_matrix():
    """On a sampled graph with N = 60 (q = 2, c_in = 5, c_out = 1, seed 0) the eigenvalues mu != +-1 of the full 2M x 2M
    matrix B equal those of the 2N x 2N matrix B' (KMMNSZZ Eqs. (8)-(9)) as multisets, to 1e-6 (best one-to-one
    matching; the zero eigenvalues of dangling trees are defective and spread numerically, so |mu| < 0.05 is left
    out). Planted error: B' with D in place of D - 1 (backtracking allowed, _HOOKS["backtrack"] = 0) does not."""
    g = SQ.sbm_q_sample(60, 2, 5, 1, 0)
    key = lambda v: v[(np.abs(np.abs(v) - 1) > 1e-6) & (np.abs(v) > 0.05)]   # zeros: defective (dangling trees)

    def gap(a, b):                                  # the largest distance of the best one-to-one matching
        if len(a) != len(b):
            return math.inf
        d = np.abs(a[:, None] - b[None, :])
        r, c = linear_sum_assignment(d)
        return float(d[r, c].max())

    full = key(np.linalg.eigvals(NB.nonbacktracking_matrix(g).toarray()))
    small = key(np.linalg.eigvals(NB.ihara_bass(g).toarray()))
    assert len(full) > 50 and gap(full, small) < 1e-6
    saved = NB._HOOKS["backtrack"]
    try:
        NB._HOOKS["backtrack"] = 0.0
        bad = key(np.linalg.eigvals(NB.ihara_bass(g).toarray()))
    finally:
        NB._HOOKS["backtrack"] = saved
    assert NB._HOOKS["backtrack"] is saved
    assert gap(bad, full) > 1e-2


def test_nonbacktracking_detects_above_ks():
    """N = 1000, seeds 0-3, mean degree about 3.  Above KS (c_in = 6, c_out = 0.5: mu_c = 2.75 > sqrt(3.25) = 1.80):
    the leading eigenvalue's 99% band contains the graphs' mean degree within 0.15, the second real eigenvalue is
    outside the disk sqrt(2M/N) on every graph and its band contains mu_c within 0.25, the spectral clustering
    overlap band is above 0.5.  Below KS (3.5, 2.5: mu_c = 0.5 < sqrt(3) = 1.73): `second` is inside the disk
    within 5% on every graph and the overlap band is below 0.1.  Power: the two `second` bands are about 1 apart
    with half-widths ~0.1 (build).  Planted error: backtracking allowed moves the leading eigenvalue off c."""
    seeds = (0, 1, 2, 3)
    top, sec, edge, ov_hi = _stats(1000, 2, 6, 0.5, seeds)
    b1, b2 = student_band(top), student_band(sec)
    assert abs(b1.mean - 3.25) < b1.half_width + 0.15, b1
    assert np.all(sec > edge * 1.05) and abs(b2.mean - 2.75) < b2.half_width + 0.25, (sec, edge, b2)
    _, lo_sec, lo_edge, ov_lo = _stats(1000, 2, 3.5, 2.5, seeds)
    assert np.all(lo_sec < lo_edge * 1.05), (lo_sec, lo_edge)
    bh, bl = student_band(ov_hi), student_band(ov_lo)
    assert bh.mean - bh.half_width > 0.5 and bl.mean + bl.half_width < 0.1, (bh, bl)
    saved = NB._HOOKS["backtrack"]
    try:
        NB._HOOKS["backtrack"] = 0.0
        bad = NB.nonbacktracking_spectrum(SQ.sbm_q_sample(1000, 2, 6, 0.5, 0), 4).eigenvalues[0]
    finally:
        NB._HOOKS["backtrack"] = saved
    assert NB._HOOKS["backtrack"] is saved
    assert abs(bad - 3.25) > b1.half_width + 0.15, (bad, b1)


# Finite-N margins of the slow test, measured on seeds 100-109 at N = 20000 (c_in = 5, c_out = 1), which the test does
# not use: mu1 - E[d(d-1)]/E[d] of the same graph = -0.0011 +- 0.0069 (99%); mu2 - 2 = -0.0151 +- 0.0242; |mu3| -
# sqrt(E[d(d-1)]/E[d]) = +0.0127 +- 0.0038.  Each margin is |mean| + half-width of that calibration.
M1, M2, M3 = 0.008, 0.040, 0.017


def _per_graph(N, c_in, c_out, seeds):
    """(mu1, `second`, |mu3|, E[d(d-1)]/E[d], spectral overlap) per seed, q = 2."""
    out = []
    for s in seeds:
        g = SQ.sbm_q_sample(N, 2, c_in, c_out, s)
        d = np.bincount(g.edges.ravel(), minlength=g.N)
        sp = NB.nonbacktracking_spectrum(g, 6)
        v, sec = sp.eigenvalues, _second(sp.eigenvalues)
        third = next(abs(x) for x in v[1:] if abs(abs(x) - sec) > 1e-9)
        out.append((v[0].real, sec, third, float((d * (d - 1)).sum() / d.sum()), NB.cluster(g, 2, sp, seed=s).overlap))
    return [np.array(x) for x in zip(*out)]


@slow
def test_nonbacktracking_bank_values():
    """KMMNSZZ Fig. 2 parameters (c_in = 5, c_out = 1, c = 3) at N = 20000, seeds 0-5, against per-graph predictors
    and margins calibrated on seeds 100-109 (M1, M2, M3 above).  Leading eigenvalue: the band of mu1 - E[d(d-1)]/E[d]
    (the graph's own non-backtracking branching number) contains 0 within M1, and the band of E[d(d-1)]/E[d] (no
    margin: a sample mean of a Poisson(3) law) contains the bank 3.  Second: the band of the second real eigenvalue
    contains the bank 2 within M2.  Bulk: the band of |mu3| - sqrt(E[d(d-1)]/E[d]) contains 0 within M3.  Spectral
    overlap band above 0.3.  Positive controls: the adjacency-matrix prediction mu + (c - 1)/mu = 3 (KMMNSZZ Eq. (2)
    form) for the second eigenvalue lies outside its band + M2, and B' with backtracking (_HOOKS["backtrack"] = 0)
    moves mu1 off the per-graph predictor by more than the band + M1.  q = 3 (c_in = 9, c_out = 1.5, mu_c = 2.5 >
    sqrt(4) = 2): two real eigenvalues near mu_c."""
    seeds = tuple(range(6))
    top, sec, third, ratio, ov = _per_graph(20000, 5, 1, seeds)
    b1, br = student_band(top - ratio), student_band(ratio)
    b2, b3 = student_band(sec), student_band(third - np.sqrt(ratio))
    assert abs(b1.mean) <= b1.half_width + M1, b1
    assert abs(br.mean - float(BANK["nb_leading_cin5_cout1"].value)) <= br.half_width, br
    assert abs(b2.mean - float(BANK["nb_second_cin5_cout1"].value)) <= b2.half_width + M2, b2
    assert abs(b3.mean) <= b3.half_width + M3, b3
    assert student_band(ov).mean - student_band(ov).half_width > 0.3, ov
    assert abs(b2.mean - (2 + (3 - 1) / 2)) > b2.half_width + M2
    g0 = SQ.sbm_q_sample(20000, 2, 5, 1, 0)
    saved = NB._HOOKS["backtrack"]
    try:
        NB._HOOKS["backtrack"] = 0.0
        bad = NB.nonbacktracking_spectrum(g0, 6).eigenvalues[0]
    finally:
        NB._HOOKS["backtrack"] = saved
    assert NB._HOOKS["backtrack"] is saved
    assert abs(abs(bad) - ratio[0]) > b1.half_width + M1, (bad, ratio[0])
    g = SQ.sbm_q_sample(20000, 3, 9, 1.5, 0)
    cl = NB.spectral_clustering(g, 3, seed=0)
    assert cl.separated and np.all(np.abs(cl.informative.real - 2.5) < 0.2), cl.informative
    assert cl.overlap > 0.3, cl.overlap


def test_nonbacktracking_refusals():
    g = SQ.sbm_q_sample(50, 2, 5, 1, 0)
    for bad in (0, True, 2.0):
        try:
            NB.nonbacktracking_spectrum(g, bad)
        except Refusal as r:
            assert r.reason == "precondition:eigen-count"
        else:
            raise AssertionError(bad)
    for bad in ("graph", SQ.SBMqGraph(5, np.zeros(5, dtype=int), np.zeros((0, 2), dtype=int))):
        try:
            NB.nonbacktracking_spectrum(bad, 2)
        except Refusal as r:
            assert r.reason == "precondition:graph"
        else:
            raise AssertionError(bad)
