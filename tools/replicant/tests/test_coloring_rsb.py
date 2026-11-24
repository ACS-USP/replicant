"""K7 (KZ Ch. 13-14): clusters in graph coloring at beta = oo -- the frozen survey propagation (colorability),
the 1RSB cavity (clustering, condensation) and the rigidity (freezing) recursion.

Conventions: q colors; c = mean degree of an Erdos-Renyi graph ("er"), d = degree of a d-regular graph
("regular"); every threshold states its graph.  Sources: KZ = Krzakala & Zdeborova lecture notes (June 9 2024),
ZK = Zdeborova & Krzakala, PRE 76, 031131 (2007), arXiv:0704.1269 (fetched; Table I regular, Table II ER).
Bank: reference/factor.json.  The notes' table (p. 208, Sec. 13.2.1) prints "c_col ~ 5.4" (q = 3) and "c_col ~ 14.4"
(q = 5): these are the ANNEALED thresholds 2 ln q / ln(q/(q - 1)) (banked as coloring_c_ann_q3 / _q5), not the
1RSB colorability thresholds of ZK Table II (4.687, 13.669).
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.amp._stats import student_band
from replicant.catalog import catalog, check_ref, run_cached
from replicant.coloring import potts_bp as PB
from replicant.coloring import rsb as R
from replicant.coloring import rsb_pop as RP
from replicant.core import Refusal
from replicant.refdata import load
from replicant.testing import slow

BANK = load("factor")


def _v(key):
    return float(BANK[key].value)


def _t(key):
    return float(BANK[key].tol)


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


def test_colorability_table():
    """(a) The notes' table (p. 208) prints c_col ~ 5.4 (q = 3) and 14.4 (q = 5): both are c_ann of Eq. (6.17) (the
    first-moment upper bound), 0.7 or more above ZK's 1RSB c_col (Table II), so they are banked as c_ann only.
    (b) Regular graphs: the notes' symmetric SP fixed point and complexity (Sec. 14.3, p. 220) give, at integer d,
    the smallest d with a frozen (nontrivial) solution and the first uncolorable d; both equal ZK Table I (c_SP, c_s)
    for q = 3, 4, 5, and the real root lies between d_uncol - 1 and d_uncol.  A planted error in the SP exponent
    (d instead of d - 1 incoming surveys) moves them.  (c) Independent route: the 1RSB population of BP messages at
    m = 0 (weight 1[Z > 0]: rsb_pop.coloring_1rsb, no SP formula) reproduces Sigma of the closed form at q = 3, d = 5
    within 0.012 (calibrated: seeds 100-109 at this size gave max |dev| 0.0062 at d = 5, 0.0035 at d = 6; about 2x).
    (d) m = 1 on a regular graph (q = 4, d = 10 > c_d = 9): the reweighted population is nontrivial, Psi(1) equals
    the annealed free entropy within 0.05 (seeds 100-109 at this size: max |dev| 0.024) and Sigma(1) < 0 (d = 10 is
    ZK's c_c); without the color symmetrisation the population collapses and Psi(1) is off by more than 1.
    (e) ER: the colour-symmetric frozen-SP population (pop 4000) has Sigma > 0 at c = 4.5 and < 0 at c = 4.9 for
    q = 3: the root lies in ZK's bracket around c_col = 4.687 (the reference root is the slow test)."""
    for q, key in ((3, "coloring_c_ann_q3"), (5, "coloring_c_ann_q5")):
        ca = PB.coloring_annealed_threshold(q)
        assert abs(ca - _v(key)) <= _t(key), (q, ca)
        assert ca - _v(f"coloring_c_col_q{q}") > 0.7, (q, ca)                   # the print is not ZK's c_col
    for q in (3, 4, 5):
        col = R.colorability_threshold(q, "regular")
        assert col.d_sp == _v(f"coloring_regular_d_sp_q{q}") and col.d_uncol == _v(f"coloring_regular_d_uncol_q{q}"), col
        assert col.d_uncol - 1 < col.value < col.d_uncol and col.status == "determined", col
        assert R.sp_regular(q, col.d_sp - 1).nontrivial is False and R.sp_regular(q, col.d_sp).sigma > 0
        assert abs(R.sp_regular(q, col.value).sigma) < 1e-10
    wrong = _planted(R, "children", 0, lambda: R.colorability_threshold(4, "regular"))
    assert (wrong.d_sp, wrong.d_uncol) != (9, 10), wrong
    sp = R.sp_regular(3, 5)
    r = RP.coloring_1rsb(3, 5, 0, graph="regular", pop=10000, sweeps=60, measure=40, seed=7)
    assert r.status == "converged" and abs(r.sigma - sp.sigma) < 0.012 and r.sigma > 0, (r, sp)
    r = RP.coloring_1rsb(4, 10, 1, graph="regular", pop=2000, sweeps=30, measure=15, seed=7)
    assert r.nontrivial and abs(r.psi - r.psi_annealed) < 0.05 and r.sigma < 0, r
    bad = _planted(RP, "symmetrize", False, lambda: RP.coloring_1rsb(4, 10, 1, graph="regular", pop=2000, sweeps=30,
                                                                      measure=15, seed=7))
    assert abs(bad.psi - bad.psi_annealed) > 1, bad
    hi = RP.frozen_sp(3, 4.5, graph="er", pop=4000, sweeps=60, measure=30, seed=3)
    lo = RP.frozen_sp(3, 4.9, graph="er", pop=4000, sweeps=60, measure=30, seed=3)
    assert hi.nontrivial and hi.sigma > 0.005 and lo.sigma < -0.01, (hi, lo)
    _refuses(lambda: R.colorability_threshold(3, "lattice"), "unsupported:graph")
    _refuses(lambda: RP.coloring_1rsb(3, 5, 1.5, graph="regular"), "precondition:parisi-m")


def _refs(prefixes):
    e = catalog()["coloring"]
    for ref in e.refs:
        if ref.key.startswith(prefixes):
            out = check_ref(e, ref)
            assert out.ok, (ref.key, out)
    return e


@slow
def test_colorability_table_reference():
    """Reference sizes (about 90 s), through the coloring entry's RefChecks (ansatz=1rsb, thresholds=colorability,
    seeds 0-3): the ER frozen-SP root c_col for q = 3, 4, 5 against ZK Table II in the 99% Student-t band (build:
    4.6880, 8.9010, 13.6694)."""
    _refs(("coloring_c_col_",))


@slow
def test_colorability_table_q3_by_1rsb():
    """The q = 3 values by the 1RSB cavity (population of populations, m = 1, seeds 0-3; about 80 s): the trivial
    solution's growth factor per sweep crosses 1 at c = 4 (continuous), the frozen start decays at 0.95 of it and is
    nontrivial at 1.15 of it with Sigma(m = 1) < 0 in the 99% band: c_d = c_c = 4 (ZK: "c_RS = c_d = c_c = 4")."""
    e = _refs(("coloring_c_d_q3", "coloring_c_c_q3"))
    v, _ = run_cached(e, "cavity", {"q": 3, "ansatz": "1rsb", "thresholds": "clustering"})
    assert v["c_d_1rsb_kind"] == v["c_c_1rsb_kind"] == "continuous" and v["c_c_1rsb"] == v["c_d_1rsb"], v
    assert e.finality_of(v)["c_d_1rsb"].state == "not-computed"


@slow
def test_colorability_table_regular_by_1rsb():
    """Regular graphs, the single factorized population at m = 1 (seeds 0-3; about 55 s): the smallest d with a
    nontrivial solution (c_d) and with Sigma(1) < 0 in the 99% band (c_c) equal ZK Table I: q = 3 c_c = 6, q = 4
    (9, 10), q = 5 (14, 14)."""
    _refs(("coloring_regular_d_d", "coloring_regular_d_c"))


def test_rigidity_scaling():
    """The notes' freezing recursion (Sec. 13.2.3, p. 211) on d-regular trees, eta = 1 + sum_r (-1)^r C(q-1, r)
    (1 - r eta/(q - 1))^(d - 1), and its ER form (Poisson(c) children: eta = (1 - e^{-c eta/(q-1)})^(q-1), derived
    here) give c_r where a nonzero fixed point appears.  ER c_r reproduces ZK Table II's c_r(m = 1) for q = 3..10 to
    the printed digit, and is above ZK's c_c for q = 3, 4, 5 (the notes: c_r > c_c for q < 9).  Scaling (the notes:
    c_r = q log q + o(q log q)): at q = 5..10, regular and ER, the leading term alone leaves c_r/(q ln q) - 1 in
    (0.6, 0.9) -- a correction, labelled: ZK Eq. (44) gives q[ln q + ln ln q + 1 + o(1)], whose residual
    |c_r/q - ln q - ln ln q - 1| < 0.35 at every q here; for ER at q = 1e2..1e6 the ratio c_r/(q ln q) decreases
    toward 1 and the residual shrinks.  A planted convention slip (c read as c_out = q c/(q - 1)) breaks the ZK match."""
    for q in range(3, 11):
        r = R.rigidity_threshold(q, "er")
        key = f"coloring_c_r_m1_q{q}"
        assert abs(r.c_r - _v(key)) <= _t(key), (q, r)
    for q in (3, 4, 5):
        assert R.rigidity_threshold(q, "er").c_r > _v(f"coloring_c_c_q{q}")
    for q in range(5, 11):
        for g in ("regular", "er"):
            c = R.rigidity_threshold(q, g).c_r
            lead = R.rigidity_corrections(q)
            assert 0.6 < c / lead["leading"] - 1 < 0.9, (q, g, c)
            assert abs(c / q - lead["zk_eq44"] / q) < 0.35, (q, g, c)
    ratios = [R.rigidity_threshold(q, "er").c_r / (q * math.log(q)) for q in (100, 10 ** 3, 10 ** 4, 10 ** 6)]
    assert all(a > b > 1 for a, b in zip(ratios, ratios[1:])), ratios
    res = [abs(R.rigidity_threshold(q, "er").c_r / q - R.rigidity_corrections(q)["zk_eq44"] / q) for q in (100, 10 ** 6)]
    assert res[1] < res[0], res
    slip = _planted(R, "er_rate", 0.8, lambda: R.rigidity_threshold(5, "er").c_r)
    assert abs(slip - _v("coloring_c_r_m1_q5")) > 1, slip
    _refuses(lambda: R.rigidity_threshold(2, "er"), "precondition:q")
