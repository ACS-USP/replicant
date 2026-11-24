"""K7 (KZ Ch. 13): the planted coloring, its link to the SBM (c_in = 0, e^-beta = c_in / c_out, Eq. (13.3)), the BP
algorithmic threshold on planted graphs, quiet planting, and exhaustive counts of q-colorings at small N.

Conventions: q colors, c = mean degree (planted graphs: M = round(c N / 2) edges), beta = oo is the proper
coloring.  Sources: KZ = Krzakala & Zdeborova lecture notes (June 9 2024), Ch. 13; Zdeborova & Krzakala, PRE 76,
031131 (2007) (ZK, Table II); Krzakala & Zdeborova, PRL 102, 238701 (2009) (quiet planting).  The SBM side of every
cross-check is K4's inference/sbm_q.py (no code shared: coloring/ imports nothing from inference/).
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.amp._stats import student_band
from replicant.coloring import first_moment as FM
from replicant.coloring import planted as P
from replicant.coloring import potts_bp as PB
from replicant.coloring import sample as S
from replicant.core import Refusal
from replicant.factor import graph as FG
from replicant.inference import sbm_q as SQ
from replicant.refdata import load
from replicant.testing import slow

BANK = load("factor")


def _planted(module, key, value, fn):
    reg = module._HOOKS
    old = reg[key]
    reg[key] = value
    try:
        return fn()
    finally:
        reg[key] = old
        assert module._HOOKS is reg and reg[key] is old


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


def test_planted_coloring_and_the_sbm_link():
    """The generator (KZ Sec. 13.1): M = round(cN/2) distinct edges, a fraction (q - 1)/(q - 1 + e^-beta) of them
    between different planted colors (= c_out (q - 1)/(c q) of the notes); at beta = oo every edge is bichromatic
    (the planted configuration is a proper coloring).  The SBM link both ways: (c_in, c_out) from (q, c, beta) has
    mean degree (c_in + (q - 1) c_out)/q = c (K4's sbm_q.mean_degree) and e^-beta = c_in/c_out (Eq. (13.3)); and back.
    The BP threshold (13.9) c_l = ((q - 1 + e^-beta)/(1 - e^-beta))^2 equals K3's Kesten-Stigum closed form and is
    exactly K4's |c_in - c_out| = q sqrt(c) boundary at the mapped affinities; (q - 1)^2 at beta = oo (4 and 16, the
    notes' table).  A planted error (the square of (13.9) dropped) breaks both."""
    g = P.planted_coloring(3000, 5, 13.0, seed=4)
    lab, E = g.labels, g.edges
    assert len(E) == round(13.0 * 3000 / 2) and len({tuple(e) for e in E}) == len(E) and np.all(E[:, 0] < E[:, 1])
    assert np.all(lab[E[:, 0]] != lab[E[:, 1]])
    gb = P.planted_coloring(3000, 4, 6.0, beta="1", seed=4)
    frac = float(np.mean(gb.labels[gb.edges[:, 0]] != gb.labels[gb.edges[:, 1]]))
    want = 3 / (3 + math.exp(-1))
    assert abs(frac - want) < 1e-3, (frac, want)                    # exact count: round(f M) of M edges
    for q, c, beta in ((3, 4.0, "inf"), (5, 13.0, "inf"), (4, 6.0, "1"), (2, 3.0, "0.5")):
        cin, cout = P.sbm_affinities(q, c, beta)
        assert abs(float(SQ.mean_degree(q, cin, cout)) - c) < 1e-12, (q, c, beta)
        b = PB.beta_value(beta)
        assert abs(cin / cout - (math.exp(-b) if b < math.inf else 0.0)) < 1e-12
        assert P.beta_from_affinities(cin, cout) == b or abs(P.beta_from_affinities(cin, cout) - b) < 1e-12
        cl = P.bp_threshold(q, beta)
        assert abs(cl - PB.kesten_stigum(q, beta)) < 1e-9 * cl, (q, beta)
        ci, co = P.sbm_affinities(q, cl, beta)
        assert abs(abs(ci - co) - q * math.sqrt(cl)) < 1e-9 * cl, (q, beta)
    assert P.bp_threshold(3) == 4 and P.bp_threshold(5) == 16
    assert _planted(P, "cl_power", 1, lambda: P.bp_threshold(5)) != 16
    _refuses(lambda: P.planted_coloring(100, 3, -1.0, seed=0), "precondition:c")
    _refuses(lambda: P.sbm_affinities(3, 4.0, "-1"), "precondition:beta")


def test_small_n_counts():
    """Exhaustive counts of proper q-colorings (sample.count_colorings, every configuration) at small N:
    (1) on cycles C_n the count is (q - 1)^n + (-1)^n (q - 1) exactly, while the paramagnetic BP (Bethe) value is
    n ln(q - 1): Bethe on a loopy graph differs from the enumeration, by ln(1 + (-1)^n (q - 1)^(1 - n)), and the test
    asserts that difference; on a random tree BP is exact (the generic factor/ BP from a random start).
    (2) Against the annealed first moment of K3: the mean count over 200 graphs of G(N = 9, M = 9), q = 3 agrees with
    the exact E Z = 422.9 (first_moment.first_moment_log, the composition sum) in a 99% Student-t band (half-width
    about 22); a planted error in the counter (one edge per graph may be monochromatic) moves the mean by about 250,
    some 10 half-widths (seeds 50-52, not used here: 663-687 against 417-439), so the power is essentially 1.  (3) A sampled loopy ER graph at N = 12, c = 3, q = 3:
    BP converges and its Bethe value differs from the exact ln Z (labelled an approximation)."""
    for n in (5, 6, 7):
        edges = np.array([(i, (i + 1) % n) for i in range(n)])
        for q in (3, 4):
            z = S.count_colorings(n, edges, q)
            assert z == (q - 1) ** n + (-1) ** n * (q - 1), (n, q, z)
            b = S.bethe_log_count(n, edges, q, seed=1)
            assert b.converged and abs(b.phi_total - n * math.log(q - 1)) < 1e-9, (n, q, b)
            gap = math.log(z) - b.phi_total
            assert abs(gap - math.log1p((-1) ** n * (q - 1) ** (1 - n))) < 1e-9 and abs(gap) > 1e-3, (n, q, gap)
            assert "approximation" in b.label
    rng = np.random.default_rng(11)
    tree = FG.random_tree_edges(10, rng)
    b = S.bethe_log_count(10, tree, 3, seed=2)
    assert b.converged and abs(b.phi_total - math.log(S.count_colorings(10, tree, 3))) < 1e-9 and "exact" in b.label
    stat = S.mean_count(3, 9, 9, graphs=200, seed=5)
    ez = math.exp(FM.first_moment_log(3, 9, 9, "inf"))
    band = student_band(stat)
    assert abs(band.mean - ez) <= band.half_width, (band, ez)
    bad = student_band(_planted(S, "allow_mono", True, lambda: S.mean_count(3, 9, 9, graphs=200, seed=5)))
    assert abs(bad.mean - ez) > bad.half_width, (bad, ez)
    e = FG.erdos_renyi_edges(12, 3.0, np.random.default_rng(3))
    b = S.bethe_log_count(12, e, 3, seed=4)
    z = S.count_colorings(12, e, 3)
    assert b.converged and z > 0 and abs(b.phi_total - math.log(z)) > 1e-6 and "approximation" in b.label
    _refuses(lambda: S.count_colorings(30, e, 3), "precondition:enumeration-size")


@slow
def test_small_n_counts_reference():
    """N = 14 (q = 3, 4.78e6 colorings per graph): 40 graphs of G(14, 14) against the exact first moment (99% band)."""
    stat = S.mean_count(3, 14, 14, graphs=40, seed=6)
    ez = math.exp(FM.first_moment_log(3, 14, 14, "inf"))
    band = student_band(stat)
    assert abs(band.mean - ez) <= band.half_width, (band, ez)


@slow
def test_bp_algorithmic_threshold_on_planted():
    """KZ Secs. 13.2.2-13.2.3 / ZK Table II, q = 5, ER, c_d = 12.837, c_c = 13.23, c_l = 16.  BP from the planted
    start on planted graphs (N = 5000, seeds 0-3, damping clearing K3's uniform-mode bound) converges to the planted
    state (overlap > 0.5) at c = 13.8 > c_d and to the paramagnet (overlap < 0.01) at c = 11.8 < c_d, on every seed.
    Power: the two outcomes are separated by > 0.4 in overlap on every seed, so a plant that removes the effect
    (the random start in place of the planted one, below c_l: KZ's 4-case table) fails with probability 1 on 4/4
    seeds.  Cross-check (KZ: the n e^-h term "does not asymptotically influence" BP): K4's sbm_q BP with the non-edge
    field, c_in = 0, on the same graphs gives the same verdicts.  Quiet planting (KZ Sec. 13.2.2, Krzakala-Zdeborova
    2009): at c = 11.8 the planted start decays (Phi_planted = Phi_para: quiet); at c = 13.03, the midpoint of ZK's
    (c_d, c_c) (a margin argument: farthest from both ends; checked beforehand on unused seeds 10-13, quiet with 10000
    iterations, Delta = -0.013 +- 0.0075), the planted state's Bethe free entropy is below the paramagnetic one in the
    99% band (quiet); above c_c it is higher (c = 14.5, not quiet); bands over seeds 0-3."""
    q, n, seeds = 5, 5000, (0, 1, 2, 3)
    lo = S.planted_bp(q, 11.8, n=n, seeds=seeds)
    hi = S.planted_bp(q, 13.8, n=n, seeds=seeds)
    assert all(r.converged for r in lo + hi), (lo, hi)
    assert all(r.overlap < 0.01 for r in lo) and all(r.overlap > 0.5 for r in hi), (lo, hi)
    rnd = _planted(S, "start", "random", lambda: S.planted_bp(q, 13.8, n=n, seeds=seeds))
    assert all(r.overlap < 0.1 for r in rnd), rnd
    for c, runs in ((11.8, lo), (13.8, hi)):
        for s, r in zip(seeds, runs):
            g = P.planted_coloring(n, q, c, seed=s)
            sg = SQ.from_edges(n, g.edges, g.labels)
            k = SQ.sbm_q_bp(sg, q, 0.0, q * c / (q - 1), init="planted", iters=3000, seed=s, damping=0.6)
            assert k.converged and (k.overlap > 0.5) == (r.overlap > 0.5), (c, s, k.overlap, r.overlap)
    for c, verdict in ((13.03, "quiet"), (14.5, "not quiet")):
        qp = P.quiet_planting(q, c, n=n, seeds=seeds, iters=10000)
        assert qp.verdict == verdict and qp.status == "determined", (c, qp)
