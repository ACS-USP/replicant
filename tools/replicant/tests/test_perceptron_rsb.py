"""Negative-margin spherical perceptron at T = 0: the 1RSB jamming capacity (perceptron_rsb.py) and
its catalog route.  The K-step RSB ladder and the full-RSB estimate are in the slow tier below.

Published numbers (read from the sources, reference/perceptron.json): Stojnic, arXiv:2312.16531,
Tables 2, 5, 6 (his r-sfl RDT levels; the second full level is the 1RSB capacity here) and
arXiv:1306.3980 Table 2 (partially lifted upper bound); Franz & Parisi, arXiv:1501.03397.
Independent routes: the scaling limit of the unscaled entropy (a float64 nested quadrature of
FPSUZ Eq. (59) sharing no code with the closed form), the integer-n replica functional against a
Gaussian orthant probability, and the planted controls (wrong block structure, dropped m, wrong
scaling), each of which must leave the banked number.
"""
import math
import os
import sys

import mpmath as mp
import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.catalog import catalog
from replicant.core import Refusal
from replicant.families import perceptron_rsb as R
from replicant.families.perceptron import gardner_capacity, gardner_rs_point
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


def _banked(key):
    return float(REF[key].value)


# ---- the 1RSB capacity against the published numbers -----------------------------------------
def test_one_rsb_capacity_kappa_minus_one_float_stage_matches_the_bank():
    """Default tier (float64 stage, 0.3 s): the minimum of A(mu, q0) at kappa = -1 against Stojnic's second level
    of lifting and his p2-hat, below RS and below his 2013 bound, with the contact index > 1; and the capacity at
    the audited mpmath root is the slow test below."""
    from replicant.families.perceptron_krsb import _Tree
    x, a = R._float_minimum(-1.0)
    assert check(REF["spherical_1rsb_capacity_kappa_m1p0"], a, label="1RSB alpha_c(-1)").ok
    assert check(REF["spherical_1rsb_q0_kappa_m1p0"], 1 - math.exp(x[1]), label="1RSB q0(-1)").ok
    assert abs(a - 12.3196343466) < 1e-8
    assert a < _banked("stojnic13_upper_bound_kappa_m1") < float(R.rs_capacity(-1))
    assert _Tree([math.exp(x[0])], [1 - math.exp(x[1])], -1.0).contact(a) > 1      # unstable toward further RSB


@slow
def test_one_rsb_capacity_kappa_minus_one_reproduces_the_banked_numbers():
    j = R.onersb_jamming(-1)
    assert check(REF["spherical_1rsb_capacity_kappa_m1p0"], j.alpha_c, label="1RSB alpha_c(-1)").ok
    assert check(REF["spherical_1rsb_q0_kappa_m1p0"], j.q0, label="1RSB q0(-1)").ok
    assert abs(j.alpha_c - mp.mpf("12.3196343465869")) < mp.mpf("1e-12")   # our audited digits
    # below the RS value, below Stojnic's partially lifted rigorous upper bound (arXiv:1306.3980, Table 2)
    assert j.alpha_c < _banked("stojnic13_upper_bound_kappa_m1") < float(j.alpha_rs)
    assert abs(j.alpha_rs - 1 / (2 * mp.ncdf(-1) - mp.npdf(1))) < 1e-15            # 1/D(-1), D = (1+k^2) H(-k) + k phi(k)
    # the audit: stationarity at working precision, a local minimum, a nontrivial branch
    assert max(abs(r) for r in j.residuals) < 1e-9, j.residuals
    assert 0 < j.hessian_eigs[0] < j.hessian_eigs[1] and 0 < j.q0 < 1 and j.mu > 1
    # cross-check of the two float / mpmath quadratures: the float64 stage lands on the same root
    x, a = R._float_minimum(-1.0)
    assert abs(a - float(j.alpha_c)) < 1e-9 and abs(1 - math.exp(x[1]) - float(j.q0)) < 1e-6


def test_the_bank_comparison_discriminates_the_levels():
    """Planted controls at the bank level: the RS value (13.27), Stojnic's 2013 bound (12.784) and the
    2RSB value (12.2872) each fail the 1RSB number 12.32, so the kappa = -1 check is not vacuous."""
    rs = R.rs_capacity(-1)
    assert not check(REF["spherical_1rsb_capacity_kappa_m1p0"], rs, label="RS in place of 1RSB").ok
    assert not check(REF["spherical_1rsb_capacity_kappa_m1p0"], _banked("stojnic13_upper_bound_kappa_m1"),
                     label="2013 bound in place of 1RSB").ok
    assert not check(REF["spherical_1rsb_capacity_kappa_m1p0"], "12.2872", label="2RSB in place of 1RSB").ok
    # and the 2RSB bank entry rejects the 1RSB value
    assert not check(REF["spherical_2rsb_capacity_kappa_m1p0"], "12.3196343466",
                     label="1RSB in place of 2RSB").ok


@slow
def test_one_rsb_capacity_at_four_kappa_from_stojnics_table_two():
    """Float stage of the same minimisation (no mpmath) at four more of his ten kappa; the other
    six are in the slow tier.  Agreement is to his four printed digits, his values at or above ours."""
    for tag in ("m2p0", "m1p5", "m0p5", "m0p3"):
        key = f"spherical_1rsb_capacity_kappa_{tag}"
        k = {"m2p0": -2.0, "m1p5": -1.5, "m0p5": -0.5, "m0p3": -0.3}[tag]
        x, a = R._float_minimum(k)
        assert check(REF[key], a, label=key).ok, (key, a)
        assert a < float(R.rs_capacity(k))
        assert a <= _banked(key) * (1 + 1e-5)           # his numerically optimised bound is at or above ours


@slow
def test_the_stationary_point_is_the_unique_nontrivial_minimum_at_kappa_minus_one():
    """Several starts of the float64 minimisation reach the same point or the mu -> 0 boundary (the RS
    value); no second nontrivial minimum (the 450-start fsolve scan over kappa is in the slow tier)."""
    ends = []
    for lm, lw in ((0.5, -0.5), (5.0, -1.0), (8.0, -6.0), (11.0, -9.0)):
        r = minimize(R._alpha_f64, [lm, lw], args=(-1.0,), method="Nelder-Mead",
                     options=dict(xatol=1e-7, fatol=1e-12, maxiter=500))
        ends.append((round(r.fun, 6), round(math.exp(r.x[0]), 3)))
    nontrivial = {a for a, mu in ends if mu > 1.0 and a < float(R.rs_capacity(-1)) - 1e-6}
    assert nontrivial == {12.319634}, ends


# ---- reduces to RS ----------------------------------------------------------------------------
def test_kappa_nonnegative_reduces_to_rs_and_two():
    c0 = gardner_capacity("spherical", 0, ansatz="1rsb")
    assert c0.alpha_c == 2 and c0.stable is True and c0.ansatz == "1rsb" and "reduces to RS" in c0.note
    c = gardner_capacity("spherical", "0.5", ansatz="1rsb")
    assert c.alpha_c == gardner_capacity("spherical", "0.5").alpha_c and c.stable is True
    _refuses(lambda: R.onersb_jamming(0), "precondition:kappa-range")
    _refuses(lambda: R.onersb_jamming(0.5), "precondition:kappa-range")


@slow
def test_kappa_nonnegative_has_no_stationary_point_below_rs():
    """Why: the infimum of the 1RSB upper-bound functional A(mu, q0) is the mu -> 0 boundary, the RS value."""
    for k in (0.0, 0.5):
        best = min(minimize(R._alpha_f64, [lm, lw], args=(k,), method="Nelder-Mead",
                            options=dict(xatol=1e-7, fatol=1e-13, maxiter=400)).fun
                   for lm, lw in ((1.0, -1.0), (7.0, -4.0)))
        assert best >= float(R.rs_capacity(k)) * (1 - 1e-9), (k, best)


@slow
def test_kappa_to_zero_from_below_closes_on_rs_like_abs_kappa_cubed():
    """alpha_RS - alpha_1RSB ~ 0.50 |kappa|^3 (mu ~ 1.5 / kappa^2): the branch joins the RS value at 0."""
    for k in (-0.1, -0.2):
        x, a = R._float_minimum(k)
        gap = float(R.rs_capacity(k)) - a
        assert 0.45 < gap / abs(k)**3 < 0.52, (k, gap / abs(k)**3)
        assert 1.2 < math.exp(x[0]) * k**2 < 2.4, (k, math.exp(x[0]) * k**2)


# ---- the replicon / contact index, and the labels ---------------------------------------------
@slow
def test_one_rsb_point_is_unstable_toward_further_rsb():
    j = R.onersb_jamming(-1)
    assert j.contact_index > 1 and j.replicon < 0 and abs(j.contact_index + j.replicon - 1) < 1e-15
    c = gardner_capacity("spherical", -1, ansatz="1rsb")
    assert c.stable is False and c.ansatz == "1rsb" and c.point is j and "unstable toward further RSB" in c.note
    # c -> 1 (marginal) as kappa -> 0^-: the RS point is marginal at kappa = 0
    from replicant.families.perceptron_krsb import _Tree
    cs = []
    for k in (-0.5, -0.05):
        x, a = R._float_minimum(k)
        cs.append(_Tree([math.exp(x[0])], [1 - math.exp(x[1])], k).contact(a))
    assert cs[1] < cs[0] and 1 < cs[1] < 1.02, cs


@slow
def test_scaling_limit_of_the_unscaled_entropy_gives_phi_over_delta():
    """FPSUZ Eq. (59) as written (float64 nested Gauss-Legendre, no closed form) with 1 - q1 = delta, m = mu delta:
    s = (1+c) ln(delta)/2 + Phi/delta + O(1).  Off the stationary point (alpha = 0.97 alpha_c) Phi != 0 and
    delta (s - (1+c) ln(delta)/2) -> Phi linearly in delta; the closed-form Phi is the limit."""
    j = R.onersb_jamming(-1)
    mu, q0, al = float(j.mu), float(j.q0), 0.97 * float(j.alpha_c)
    phi = float(R.jamming_phi(mu, q0, al, -1))
    c = float(R.contact_index(mu, q0, al, -1))
    errs = []
    for d in (1e-5, 1e-6):
        s = R.unscaled_entropy(q0, 1 - d, mu * d, al, -1.0)
        errs.append(abs(d * (s - 0.5 * (1 + c) * math.log(d)) - phi) / phi)
    assert errs[1] < 1e-3 and errs[0] > 5 * errs[1], errs
    # m = 1 is the RS entropy of perceptron.py at q0 whatever q1 is
    pt = gardner_rs_point("spherical", "0.6", -1)
    assert abs(R.unscaled_entropy(0.6, 0.9, 1.0, float(pt.alpha), -1.0) - float(pt.S)) < 1e-9


@slow
def test_one_rsb_replicon_formula_limits_to_the_contact_index():
    """FPSUZ Eqs. (62)-(63) at finite 1 - q1 (float64) approach 1 - c as delta -> 0, with the boundary
    layer correction O(sqrt(delta)); at q0 = q1 it is the RS replicon of perceptron.py."""
    j = R.onersb_jamming(-1)
    mu, q0, al = float(j.mu), float(j.q0), float(j.alpha_c)
    errs = [abs(R.unscaled_replicon(q0, 1 - d, mu * d, al, -1.0) - float(j.replicon)) for d in (1e-4, 1e-5)]
    assert errs[1] < 0.02 and errs[0] > 2 * errs[1], errs
    pt = gardner_rs_point("spherical", "0.5", "-0.5")
    near = R.unscaled_replicon(0.5, 0.5 + 1e-9, 0.5, float(pt.alpha), -0.5)
    assert abs(near - float(pt.replicon)) < 1e-4, (near, pt.replicon)


# ---- the integer-n replica functional ----------------------------------------------------------
def _nested_orthant(q0, q1, n, m, kappa, swap=False, drop_power=False):
    """P(h_a >= kappa, a = 1..n) for 1RSB covariance (diag 1, q1 in a block of size m, q0 between blocks)
    as E_t (E_u H(x)^m)^(n/m), x = (kappa - sqrt(q0) t - sqrt(q1 - q0) u)/sqrt(1 - q1), by Gauss-Hermite."""
    t, w = np.polynomial.hermite_e.hermegauss(120)
    w = w / np.sqrt(2 * np.pi)
    from scipy.special import ndtr
    a, b = (q0, q1 - q0) if not swap else (q1 - q0, q0)     # swap: the two Gaussian layers' variances exchanged
    x = (kappa - np.sqrt(a) * t[:, None] - np.sqrt(b) * t[None, :]) / np.sqrt(1 - q1)
    inner = (w[None, :] * ndtr(-x)**m).sum(axis=1)
    return float((w * inner**(1 if drop_power else n // m)).sum())


def _mvn_orthant(q0, q1, n, m, kappa):
    Q = np.full((n, n), q0)
    for b in range(n // m):
        Q[b * m:(b + 1) * m, b * m:(b + 1) * m] = q1
    np.fill_diagonal(Q, 1.0)
    return multivariate_normal(mean=np.zeros(n), cov=Q, abseps=1e-9, releps=1e-9,
                               maxpts=400_000).cdf(np.full(n, -kappa))


def test_integer_n_replica_functional_matches_the_gaussian_orthant_probability():
    """The nested form (E_u H^m inside E_t, the integer-m content of FPSUZ Eq. (59)) equals the n-dimensional
    Gaussian orthant probability of the 1RSB overlap matrix (scipy's Genz integrator), for (n, m) = (4, 2), (6, 3).  Planted controls: exchanged layer variances (wrong block structure) and a dropped n/m power (dropped m)."""
    cases = ((0.25, 0.65, 4, 2, -0.8), (0.3, 0.7, 6, 3, -0.5))
    for q0, q1, n, m, k in cases:
        want = _mvn_orthant(q0, q1, n, m, k)
        got = _nested_orthant(q0, q1, n, m, k)
        assert abs(got - want) < 1e-6, (n, m, got, want)
        if True:
            assert abs(_nested_orthant(q0, q1, n, m, k, swap=True) - want) > 20 * abs(got - want) + 1e-5, (n, m)
            assert abs(_nested_orthant(q0, q1, n, m, k, drop_power=True) - want) > 1e-3 * want, (n, m)


# ---- planted controls on the jamming functional -------------------------------------------------
def _alpha_control(x, kappa, control):
    """A(mu, q0) of perceptron_rsb with one planted error; the true one for control = None."""
    mu, q0 = math.exp(x[0]), 1 - math.exp(x[1])
    if not (0 < q0 < 1 and 1e-2 <= mu <= 1e9):
        return 1e30
    ts, ws = R._gl_panels(kappa, q0, mu)
    s = 1 - q0
    a = kappa - math.sqrt(q0) * ts
    lny = R._ln_y_f64(a, mu, q0 if control == "block" else s)     # block: the u-variance q0, not 1 - q0
    en = float(np.sum(ws * lny))
    en = en if control == "dropped_m" else en / mu                # dropped m: no 1/m in front of the average
    ent = math.log1p(mu * s) / (2 * mu) + q0 / (2 * (1 + mu * s))
    if control == "scaling":                                      # the log term without its 1/2 (wrong scale of ln det)
        ent = math.log1p(mu * s) / mu + q0 / (2 * (1 + mu * s))
    return -ent / en if en < 0 else 1e30


@slow
def test_planted_controls_leave_the_banked_number():
    j = R.onersb_jamming(-1)
    x0 = [math.log(float(j.mu)), math.log(1 - float(j.q0))]
    ok = _alpha_control(x0, -1.0, None)
    assert abs(ok - float(j.alpha_c)) < 1e-9                       # the replica of the code is faithful
    for control in ("block", "dropped_m", "scaling"):
        best = min(minimize(_alpha_control, x0 + np.array(d), args=(-1.0, control), method="Nelder-Mead",
                            options=dict(xatol=1e-7, fatol=1e-12, maxiter=500)).fun
                   for d in ([0.0, 0.0], [1.0, -1.0]))
        assert not check(REF["spherical_1rsb_capacity_kappa_m1p0"], best, label=control).ok, (control, best)


# ---- ordering against the published bounds ----------------------------------------------------
@slow
def test_ordering_against_the_published_upper_bounds():
    """1RSB < Stojnic 2013 partially lifted bound < RS at kappa = -0.8 (float stage; -1 above); and the
    capacity at Franz & Parisi's off-equilibrium jammed margin kappa* = -0.431 exceeds alpha = 3 (their
    algorithmic jamming point lies below the thermodynamic one, arXiv:1501.03397)."""
    x, a = R._float_minimum(-0.8)
    assert a < _banked("stojnic13_upper_bound_kappa_m0p8") < float(R.rs_capacity(-0.8))
    kstar = _banked("fp16_algorithmic_jamming_kappa_alpha3")
    x, a = R._float_minimum(kstar)
    assert float(R.rs_capacity(kstar)) > a > 3


# ---- refusals ----------------------------------------------------------------------------------
def test_refusals_and_ranges():
    _refuses(lambda: R.onersb_jamming(-0.01), "precondition:kappa-range")
    _refuses(lambda: R.onersb_jamming(-3.5), "precondition:kappa-range")
    _refuses(lambda: R.onersb_jamming(mp.inf), "precondition:finite-kappa")
    _refuses(lambda: gardner_capacity("spherical", -1, ansatz="2rsb"), "precondition:known-ansatz")


# ---- catalog ------------------------------------------------------------------------------------
def test_catalog_rsb_route_reduces_to_rs_for_kappa_nonnegative():
    e = catalog()["perceptron"]
    v0 = e.run("rsb", kappa=0)
    assert v0["alpha_c"] == 2 and e.finality_of(v0)["alpha_c"].state == "stable"     # the RS replicon at capacity
    assert "S" not in v0 and "rsb_q0" not in v0


@slow
def test_catalog_route_labels_the_one_rsb_capacity_unstable_and_not_final():
    e = catalog()["perceptron"]
    v = e.run("rsb", kappa=-1)
    assert check(REF["spherical_1rsb_capacity_kappa_m1p0"], v["alpha_c"], label="catalog").ok
    fin = e.finality_of(v)
    assert fin["alpha_c"].state == "unstable" and not fin["alpha_c"].final and fin["alpha_c"].ansatz == "1rsb"
    assert v["capacity_stable"] is False and v["rsb_contact_index"] > 1
    # RS path unchanged: no 1RSB keys
    rs = e.run(kappa=-1)
    assert "rsb_q0" not in rs and e.finality_of(rs)["alpha_c"].ansatz == "rs"
    # kappa >= 0: reduces to RS, labelled by its computed RS replicon (stable)
    v0 = e.run("rsb", kappa=0)
    assert v0["alpha_c"] == 2 and e.finality_of(v0)["alpha_c"].state == "stable"
    # the route computes the capacity only: no RS entropy values
    assert "S" not in e.run("rsb", kappa=-1)


def test_catalog_refusals():
    e = catalog()["perceptron"]
    _refuses(lambda: e.run("rsb", kappa=-1, ansatz="2rsb"), "precondition:known-ansatz")
    _refuses(lambda: e.run("rsb", kappa=-1, ansatz="rs"), "precondition:known-ansatz")
    _refuses(lambda: e.run("rsb", kappa=-1, ansatz="1rsb", levels=2), "precondition:levels")
    _refuses(lambda: e.run("rsb", kappa=-1, ansatz="krsb", levels=1), "precondition:levels")
    _refuses(lambda: e.run("rsb", kappa=-1, ansatz="krsb", levels=9), "precondition:levels")
    _refuses(lambda: e.run("rsb", kappa=-1, ansatz="frsb", levels=2), "precondition:levels")
    # the RS route stays RS only: ansatz and levels are not its parameters (composer)
    from replicant.compose import compose
    _refuses(lambda: compose(e.spec).run(alpha="1.5", ansatz="1rsb"), "precondition:known-parameter")
    _refuses(lambda: compose(e.spec, "rsb").run(kapa="0.3"), "precondition:known-parameter")


# ---- slow tier ----------------------------------------------------------------------------------
@slow
def test_all_ten_kappa_of_stojnics_table_two_with_the_audited_mpmath_root():
    for tag, k in (("m2p7", -2.7), ("m2p3", -2.3), ("m2p0", -2.0), ("m1p7", -1.7), ("m1p5", -1.5),
                   ("m1p3", -1.3), ("m0p7", -0.7), ("m0p5", -0.5), ("m0p3", -0.3)):
        j = R.onersb_jamming(k)
        key = f"spherical_1rsb_capacity_kappa_{tag}"
        assert check(REF[key], j.alpha_c, label=key).ok, (key, j.alpha_c)
        assert max(abs(r) for r in j.residuals) < 1e-9 and j.replicon < 0 and j.hessian_eigs[0] > 0
    j = R.onersb_jamming(-1.5)
    assert check(REF["spherical_1rsb_q0_kappa_m1p5"], j.q0, label="q0(-1.5)").ok


@slow
def test_branch_uniqueness_scan_over_kappa():
    """A(mu, q0) on a 60 x 60 grid in (ln mu, ln(1 - q0)): exactly one interior local minimum, equal to the
    audited 1RSB point to the grid resolution, and every other grid value above it (so the nontrivial minimum is
    unique and global in the family; other stationary points -- saddles, maxima -- are not excluded)."""
    lm = np.linspace(math.log(0.05), 12.0, 60)
    lw = np.linspace(-12.0, -0.05, 60)
    for k in (-0.5, -1.0, -1.5):
        want = R.onersb_jamming(k)
        G = np.array([[R._alpha_f64([a, b], k) for b in lw] for a in lm])
        mins = [(i, j) for i in range(1, 59) for j in range(1, 59)
                if G[i, j] < 1e29 and G[i, j] < min(G[i + di, j + dj] for di in (-1, 0, 1) for dj in (-1, 0, 1)
                                                    if (di, dj) != (0, 0))]
        assert len(mins) == 1, (k, mins)
        i, j = mins[0]
        assert abs(math.exp(lm[i]) / float(want.mu) - 1) < 0.35 and abs(G[i, j] / float(want.alpha_c) - 1) < 5e-3
        assert G.min() > float(want.alpha_c) * (1 - 1e-6), (k, G.min())
