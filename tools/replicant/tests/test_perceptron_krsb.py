"""Negative-margin spherical perceptron at T = 0: K-step RSB and the full-RSB staircase
(families/perceptron_krsb.py).  Published numbers: Stojnic, arXiv:2312.16531, Tables 5-6 (his third
full level, 3-sfl RDT, is two-step RSB here).  Default tier: the grid code against the audited mpmath
1RSB point, the geometric extrapolation, refusals.  Slow tier: the 2RSB bank, the ladder."""
import math
import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.catalog import catalog
from replicant.core import Refusal
from replicant.families import perceptron_krsb as K
from replicant.families import perceptron_rsb as R
from replicant.refdata import check, load
from replicant.testing import slow

REF = load("perceptron")


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, r.reason
        return r
    raise AssertionError(f"expected Refusal {reason}")


def test_grid_code_reproduces_the_one_rsb_point():
    """K = 1 of the nested-grid code against the float64 stage of perceptron_rsb (two quadratures): same
    capacity to 1e-9, and the contact index from the backward derivative recursion is > 1."""
    tree = K._Tree([12.39531565], [0.6649244864], -1.0)           # the audited mpmath point (perceptron_rsb)
    ent, en = tree.ent_en()
    assert abs(-ent / en - 12.3196343466) < 1e-8 and abs(tree.contact(-ent / en) - 1.267219) < 1e-5


@slow
def test_grid_contact_index_matches_the_audited_mpmath_value():
    j = R.onersb_jamming(-1)
    tree = K._Tree([float(j.mu)], [float(j.q0)], -1.0)
    ent, en = tree.ent_en()
    assert abs(-ent / en - float(j.alpha_c)) < 1e-9
    assert abs(tree.contact(-ent / en) - float(j.contact_index)) < 1e-6


def test_two_level_functional_embeds_the_one_level_point():
    """The K-RSB families are nested: two equal levels (q_1 = q_0 within a hair, mu_1 = mu_0) give the 1RSB value."""
    mu, q0 = 12.39531565, 0.6649244864
    tree = K._Tree([mu * (1 - 1e-6), mu], [q0 - 1e-5, q0], -1.0)
    ent, en = tree.ent_en()
    assert abs(-ent / en - 12.3196343466) < 1e-6


def test_two_rsb_trial_point_lowers_the_one_rsb_value():
    """Upper-bound structure at kappa = -1 with the 2RSB parameters found by the slow tier (rounded): this
    trial point already sits below the 1RSB minimum, and A_2 at a perturbed point is larger (a minimum)."""
    tree = K._Tree([18.7, 85.6], [0.599, 0.969], -1.0)
    ent, en = tree.ent_en()
    a2 = -ent / en
    assert a2 < 12.3196 - 0.02 and a2 < 12.2873 + 1e-3        # below the 1RSB minimum 12.31963
    pert = K._Tree([18.7, 85.6], [0.58, 0.969], -1.0)
    e2, n2 = pert.ent_en()
    assert -e2 / n2 > a2


def test_geometric_extrapolation_of_the_staircase():
    # exact geometric: alpha_K = L + c r^K -> L
    seq = [12.0 + 3.0 * 0.4**k for k in (1, 2, 3)]
    est, band = K.aitken_tail(seq)
    assert abs(est - 12.0) < 1e-12 and abs(band - (seq[1] - seq[2])) < 1e-15
    # planted control: a staircase that is not geometric-decreasing (linear / increasing differences) is not extrapolated
    assert math.isnan(K.aitken_tail([10.0, 9.0, 8.0])[0])
    assert math.isnan(K.aitken_tail([10.0, 9.5, 8.0])[0])
    _refuses(lambda: K.aitken_tail([10.0, 9.0]), "precondition:levels")
    # a wrong tail (dropping the 1/(1-r) resummation) would stay above the true limit by d_last r^2/(1-r)
    r = (seq[1] - seq[2]) / (seq[0] - seq[1])
    assert seq[2] - (seq[1] - seq[2]) * r > 12.0 + 1e-6


def test_catalog_frsb_route_is_never_final():
    """The `frsb` ansatz of the rsb route reports the extrapolated value with its band and is `not-computed`
    (checked with a stub for the ladder: the real ladder is the slow test); K-step values are `unstable`."""
    from types import SimpleNamespace as NS
    ladder = [NS(alpha_c=12.0 + d, contact_index=c) for d, c in ((0.4, 1.3), (0.1, 1.15), (0.03, 1.1))]
    orig, K.frsb_estimate = K.frsb_estimate, lambda kappa, levels: (11.99, 0.07, ladder)
    try:
        e = catalog()["perceptron"]
        v = e.run("rsb", kappa=-1.5, ansatz="frsb", levels=3)
        fin = e.finality_of(v)
    finally:
        K.frsb_estimate = orig
    assert v["alpha_c"] == 11.99 and v["frsb_band"] == 0.07 and v["rsb_ladder"] == (12.4, 12.1, 12.03)
    assert fin["alpha_c"].state == "not-computed" and not fin["alpha_c"].final
    assert fin["frsb_band"].state == "not-computed" and fin["rsb_contact_index"].state == "not-computed"


def test_frsb_nonmonotone_staircase_is_refused_not_nan():
    """Review finding: a ladder that is not geometrically decreasing (here 10, 9, 8) used to give alpha_c = nan as a
    value; it is now the named refusal undetermined:frsb-extrapolation (and a merged rung no-solution:krsb-degenerate)."""
    from types import SimpleNamespace as NS
    for seq, reason in (((10.0, 9.0, 8.0), "undetermined:frsb-extrapolation"),
                        ((10.0, 9.0, 9.0), "no-solution:krsb-degenerate")):
        it = iter(seq)
        stub = lambda kappa, k, _prev=None, it=it, seq=seq: NS(
            alpha_c=next(it), degenerate=(k == 3 and seq[2] == 9.0), mus=(1.0,), qs=(0.5,), contact_index=1.2)
        orig, K.krsb_jamming = K.krsb_jamming, stub
        try:
            _refuses(lambda: K.frsb_estimate(-1.0, 3), reason)
        finally:
            K.krsb_jamming = orig
    e = catalog()["perceptron"]
    for bad in ("2.0x", "x", "nan", "inf", "1e999"):
        _refuses(lambda: e.run("rsb", kappa=-1, ansatz="krsb", levels=bad), "precondition:levels")


def test_krsb_refusals():
    _refuses(lambda: K.krsb_jamming(-1.0, 0), "precondition:levels")
    _refuses(lambda: K.krsb_jamming(-1.0, 6), "precondition:levels")
    _refuses(lambda: K.krsb_jamming(-5.0, 2), "precondition:kappa-range")
    _refuses(lambda: K.krsb_jamming(0.5, 1), "precondition:kappa-range")


# ---- slow tier ------------------------------------------------------------------------------------
@slow
def test_two_rsb_reproduces_stojnics_third_level_at_four_kappa():
    for tag, k in (("m2p0", -2.0), ("m1p5", -1.5), ("m1p0", -1.0), ("m0p5", -0.5)):
        one = K.krsb_jamming(k, 1)
        two = K.krsb_jamming(k, 2, _prev=one)
        key = f"spherical_2rsb_capacity_kappa_{tag}"
        assert check(REF[key], two.alpha_c, label=key).ok, (key, two.alpha_c)
        assert two.alpha_c < one.alpha_c - 1e-4 and not two.degenerate
        assert two.gradient_norm < 1e-5 and two.grid_error < 1e-5
        assert 1 < two.contact_index < one.contact_index        # closer to the marginal value 1, still unstable
        # the bank separates the levels at kappa = -1.5, -1, -2 (the 1RSB number is outside the 2RSB tolerance)
        if tag != "m0p5":
            assert not check(REF[key], one.alpha_c, label="1RSB in place of 2RSB").ok
    # his p2-hat at kappa = -1 is 0.9691 and equals our q_1 there (4 digits)
    two = K.krsb_jamming(-1.0, 2)
    assert abs(two.qs[1] - 0.9691) < 2e-4, two.qs


@slow
def test_three_rsb_continues_the_ladder_and_the_full_rsb_estimate_is_below():
    """K = 3 at kappa = -1.5: alpha_1 > alpha_2 > alpha_3 > FRSB estimate, the differences shrink geometrically
    (ratio < 0.3) and the contact index falls towards, but stays above, 1 (not yet marginal at K = 3)."""
    est, band, ladder = K.frsb_estimate(-1.5, 3)
    a = [x.alpha_c for x in ladder]
    assert a[0] > a[1] > a[2] > est > 0
    r = (a[1] - a[2]) / (a[0] - a[1])
    assert 0.05 < r < 0.3, r
    c = [x.contact_index for x in ladder]
    assert c[0] > c[1] > c[2] > 1, c
    assert a[0] - a[2] < 0.02 * a[0] and est > 0.97 * a[2]
    # the catalog route reports the K = 2 number and labels it
    vk = catalog()["perceptron"].run("rsb", kappa=-1.5, ansatz="krsb", levels=2)
    assert abs(vk["alpha_c"] - a[1]) < 1e-5
    assert catalog()["perceptron"].finality_of(vk)["alpha_c"].state == "unstable"
