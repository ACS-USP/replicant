"""K4: learning the block-model parameters by EM with BP (KZ Eqs. (10.27)-(10.28); DKMZ Sec. III.C).

Convention: inference/sbm_q.py (edge probability c_ab / N, sizes n_a; c_in = c_aa, c_out = c_ab, a != b).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.amp._stats import student_band
from replicant.core import Refusal
from replicant.inference import sbm_learn as EM
from replicant.inference import sbm_q as SQ
from replicant.testing import slow


def test_m_step_counts_edges_and_nodes():
    """With one-hot messages on the planted labels the M-step (10.27)-(10.28) returns the empirical group sizes and
    the empirical edge densities 2 E_aa / (N n_a^2), E_ab / (N n_a n_b) of the sampled graph, whatever the current
    C; these are within 4 binomial standard deviations of the planted c_in, c_out (q = 3, N = 2000, c_in = 9,
    c_out = 2, seed 0).  Planted error: a factor 1/2 on the edge update (_HOOKS["edge_update"]) halves them."""
    g = SQ.sbm_q_sample(2000, 3, 9, 2, 0)
    M = len(g.edges)
    chi = np.zeros((2 * M, 3))
    src = np.r_[g.edges[:, 0], g.edges[:, 1]]
    chi[np.arange(2 * M), g.labels[src]] = 1.0
    psi = np.eye(3)[g.labels]
    n, C = EM._m_step(g, None, np.full((3, 3), 4.0) + np.eye(3), chi, psi)
    sz = np.bincount(g.labels, minlength=3)
    assert np.allclose(n, sz / 2000)
    lab = g.labels[g.edges]
    for a in range(3):
        for b in range(3):
            e = np.sum(((lab[:, 0] == a) & (lab[:, 1] == b)) | ((lab[:, 0] == b) & (lab[:, 1] == a)))
            want = (2 * e / (2000 * n[a] ** 2)) if a == b else e / (2000 * n[a] * n[b])
            assert abs(C[a, b] - want) < 1e-9, (a, b, C[a, b], want)
            truth, pairs = (9, sz[a] ** 2 / 2) if a == b else (2, sz[a] * sz[b])
            sd = np.sqrt(pairs * truth / 2000) / (pairs / 2000)
            assert abs(C[a, b] - truth) < 4 * sd, (a, b, C[a, b], sd)
    saved = EM._HOOKS["edge_update"]
    try:
        EM._HOOKS["edge_update"] = 0.5
        _, bad = EM._m_step(g, None, np.full((3, 3), 4.0), chi, psi)
    finally:
        EM._HOOKS["edge_update"] = saved
    assert EM._HOOKS["edge_update"] is saved and np.allclose(bad, C / 2)


@slow
def test_em_recovers_parameters():
    """q = 2 (c_in = 8, c_out = 2) and q = 3 (c_in = 12, c_out = 2), N = 3000, seeds 0-3, EM from the spectral
    start: every run converges (status ok) and the 99% Student-t bands of the learned mean diagonal and
    off-diagonal affinities contain the planted c_in, c_out within 0.3 (the per-run standard deviation from the
    finite graph is ~ 0.1-0.2, build).  Planted error: the edge update scaled by 0.9 biases c_in by ~0.8, outside
    the band."""
    seeds = (0, 1, 2, 3)
    for q, ci, co in ((2, 8, 2), (3, 12, 2)):
        runs = [EM.sbm_em(SQ.sbm_q_sample(3000, q, ci, co, s), q, iters=80, seed=s) for s in seeds]
        assert all(r.status == "ok" for r in runs), [(r.status, r.note) for r in runs]
        off = ~np.eye(q, dtype=bool)
        bi = student_band([np.mean(np.diag(r.affinity)) for r in runs])
        bo = student_band([np.mean(r.affinity[off]) for r in runs])
        assert abs(bi.mean - ci) < bi.half_width + 0.3 and abs(bo.mean - co) < bo.half_width + 0.3, (q, bi, bo)
        assert all(abs(r.sizes - 1 / q).max() < 0.06 for r in runs) and all(r.overlap > 0.5 for r in runs)
    saved = EM._HOOKS["edge_update"]
    try:
        EM._HOOKS["edge_update"] = 0.9
        bad = [EM.sbm_em(SQ.sbm_q_sample(3000, 2, 8, 2, s), 2, iters=80, seed=s) for s in seeds]
    finally:
        EM._HOOKS["edge_update"] = saved
    assert EM._HOOKS["edge_update"] is saved
    bb = student_band([np.mean(np.diag(r.affinity)) for r in bad])
    assert abs(bb.mean - 8) > bb.half_width + 0.3, bb


@slow
def test_em_detects_a_wrong_group_count():
    """N = 1500, named seeds 10-13 (not the seeds of the other EM tests), 200 EM steps, the BIC threshold gain(k, N)
    = (k + 1) ln N / (2N) (derived, not fitted: 0.0098 for k = 4).  A graph planted with q* = 2 (8, 2) and learned at
    q = 3 is detected:too-many-groups, and one planted with q* = 3 (12, 2) learned at q = 2 is
    detected:too-few-groups, on 4/4 seeds each (required rate 4/4).  Margins (build): the true gain 2 -> 3 groups
    is 0.037-0.059, the overfitting gain 2 -> 3 is below 5e-4, against the threshold 0.0098.  A decision resting on
    a run that is not stationary would be undetermined and fail the test."""
    for s in (10, 11, 12, 13):
        two, three = SQ.sbm_q_sample(1500, 2, 8, 2, s), SQ.sbm_q_sample(1500, 3, 12, 2, s)
        a = EM.sbm_em_group_count(two, 3, iters=200, seed=s)
        b = EM.sbm_em_group_count(three, 2, iters=200, seed=s)
        assert a.status == "detected:too-many-groups", (s, a.status, a.note)
        assert b.status == "detected:too-few-groups", (s, b.status, b.note)


def test_group_count_threshold_is_bic():
    """gain(k, N) = (k + 1) ln N / (2N), the Schwarz penalty of the k + 1 parameters added by the k-th group."""
    for k, N in ((2, 1000), (3, 1500), (5, 10 ** 5)):
        assert abs(EM.gain(k, N) - (k + 1) * np.log(N) / (2 * N)) < 1e-15


def test_em_refusals():
    g = SQ.sbm_q_sample(100, 2, 8, 2, 0)
    for fn, reason in ((lambda: EM.sbm_em("g", 2, iters=3, seed=0), "precondition:sbm-graph"),
                       (lambda: EM.sbm_em(g, 1, iters=3, seed=0), "precondition:group-count"),
                       (lambda: EM.sbm_em(g, 2, iters=0, seed=0), "precondition:positive-iters"),
                       (lambda: EM.sbm_em(g, 2, iters=3, seed=-1), "precondition:named-seeds"),
                       (lambda: EM.sbm_em(g, 2, iters=3, seed=0, start="truth"), "precondition:start")):
        try:
            fn()
        except Refusal as r:
            assert r.reason == reason, (r.reason, reason)
        else:
            raise AssertionError(reason)
