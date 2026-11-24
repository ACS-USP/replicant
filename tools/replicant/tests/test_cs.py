"""K6: the noiseless compressed-sensing phase diagram for the Gauss-Bernoulli prior (linear/cs.py; KZ notes Sec. 16.6.2,
Fig. 16.6.1).

Conventions: the notes' (and KMSSZ's) rho = fraction of non-zeros = k/N and alpha = M/N; Donoho, Maleki & Montanari
(PNAS 106, 18914 (2009), arXiv:0907.3574, "DMM") use delta = n/N (= alpha) and rho_DMM = k/n, so rho = rho_DMM delta
and DMM's curve rho_SE(delta) is the l1 boundary rho/alpha = rho_SE(alpha).  Bank: reference/linear.json.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.bayes.channels import Prior
from replicant.core import Refusal
from replicant.linear import cs as CS
from replicant.linear import gamp as G
from replicant.linear import se as SE
from replicant.refdata import check, load
from replicant.testing import slow

BANK = load("linear")


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


def _planted(key, value, fn):
    old = CS._HOOKS[key]
    CS._HOOKS[key] = value
    try:
        return fn()
    finally:
        CS._HOOKS[key] = old
        assert CS._HOOKS[key] is old


def test_cs_phase_diagram():
    """At rho = 0.4.  The l1 boundary two ways: DMM Eq. (5) solved for alpha (rho/alpha = rho_SE(alpha)) and the
    local stability of the soft-threshold state evolution at E -> 0 minimised over the threshold multiplier
    (lasso.mse_b, no DMM formula) agree to 1e-5 (no l1 value is banked: none is printed; the LP is the slow
    finite-N route).  The Bayes boundary: the SE spinodal alpha_BP = max_s s mmse(s) (derived in linear/cs.py) is
    KMSSZ's printed alpha_EM-BP ~ 0.59 (bank, tol 0.01), and the two SE starts confirm it: at alpha_BP (1 - 0.02) they
    converge to different fixed points (uninformative MSE > 1e-3, informative < 1e-9), at alpha_BP (1 + 0.02) both
    reach exact recovery (MSE < 1e-9); rho < alpha_BP < alpha_l1 (the hard phase lies inside the l1-impossible
    region).  Control: the snr inside s mmse(s) scaled by 0.9 (_HOOKS["snr"]) moves the formula's maximum to
    alpha_BP/0.9 = 0.655, off the bank (the full planted two-start confirmation, labelled undetermined, and the
    rho = 0.1, 0.2 points are slow tests: default-tier time)."""
    b = CS.cs_phase_boundary(0.4)
    assert abs(b.alpha_l1 - b.alpha_l1_se) < 1e-5 and 0.4 < b.alpha_bayes < b.alpha_l1 < 1, b
    assert check(BANK["cs_alpha_bp_rho0p4"], b.alpha_bayes, label="bp").ok
    assert b.bayes_status == "spinodal" and b.alpha_it == 0.4, b
    lo, hi = b.starts
    assert abs(lo.alpha - b.alpha_bayes * 0.98) < 1e-12 and abs(hi.alpha - b.alpha_bayes * 1.02) < 1e-12
    assert lo.status == "two-fixed-points" and lo.mse > 1e-3 and lo.mse_informative < 1e-9, lo
    assert hi.status == "unique" and hi.mse < 1e-9, hi
    bad = _planted("snr", 0.9, lambda: CS.bayes_spinodal(0.4))[0]
    assert abs(bad - b.alpha_bayes / 0.9) < 1e-6 and not check(BANK["cs_alpha_bp_rho0p4"], bad, label="bp").ok
    _refuses(lambda: CS.cs_phase_boundary(0), "precondition:rho")
    _refuses(lambda: CS.cs_phase_boundary(1), "precondition:rho")


def test_dmm_rho_map_both_ways():
    """DMM's rho is k/n = rho_notes/alpha (delta = alpha).  notes -> DMM: at alpha_l1(rho) for rho = 0.2, 0.4, DMM's
    rho_SE(alpha_l1) equals rho/alpha_l1 (1e-9).  DMM -> notes: from DMM's curve at delta = 0.3, 0.5, 0.7,
    rho_notes = delta rho_SE(delta) handed to the notes-side SE route (l1_boundary_se, which never sees DMM's
    variables) returns alpha = delta (1e-5).  Control: the unmapped rho (rho_SE(delta) taken as the notes' rho)
    misses delta by > 1e-2."""
    for rho in (0.2, 0.4):
        a = CS.l1_boundary(rho)
        assert abs(CS.rho_se(a)[0] - rho / a) < 1e-9, (rho, a)
    for delta in (0.3, 0.5, 0.7):
        r_dmm = CS.rho_se(delta)[0]
        assert abs(CS.l1_boundary_se(delta * r_dmm) - delta) < 1e-5, delta
        assert abs(CS.l1_boundary_se(r_dmm) - delta) > 1e-2, delta


@slow
def test_cs_boundaries_at_other_densities():
    """rho = 0.1, 0.2: the two l1 evaluations agree to 1e-5 and rho < alpha_BP < alpha_l1 < 1 (the two starts
    confirm alpha_BP)."""
    for rho in (0.1, 0.2):
        b = CS.cs_phase_boundary(rho)
        assert abs(b.alpha_l1 - b.alpha_l1_se) < 1e-5 and b.bayes_status == "spinodal", (rho, b)
        assert rho < b.alpha_bayes < b.alpha_l1 < 1, b


def test_l1_lp_recovers_a_sparse_vector():
    """Default-tier smoke check of the LP (scipy linprog, HiGHS): N = 100, rho = 0.1, alpha = 0.6 (well inside the l1
    success region, alpha_l1(0.1) ~ 0.33), seed 0: basis pursuit recovers x0 to relative error < 1e-6; at alpha = 0.12
    (below alpha_l1) it does not."""
    assert CS.l1_recover_instance(100, 0.6, 0.1, 0).success
    assert not CS.l1_recover_instance(100, 0.12, 0.1, 0).success


@slow
def test_cs_planted_spinodal_is_undetermined():
    """The full planted confirmation: with the snr inside s mmse(s) scaled by 0.9 the SE at the lower check point
    0.98 x 0.655 has a single (exact-recovery) fixed point, so the boundary's two-start check fails and bayes_status
    is undetermined (alpha_bayes reported only as the unconfirmed formula value)."""
    a = _planted("snr", 0.9, lambda: CS.bayes_spinodal(0.4))[0]
    assert SE.gamp_se(a * 0.98, G.Channel("awgn", Delta=0), Prior("gauss_bernoulli", rho="0.4"),
                      tol=1e-10).status == "unique"
    bad = _planted("snr", 0.9, lambda: CS.cs_phase_boundary(0.4))
    assert bad.bayes_status == "undetermined" and not check(BANK["cs_alpha_bp_rho0p4"], bad.alpha_bayes, label="bp").ok


@slow
def test_cs_l1_boundary_vs_lp_at_finite_n():
    """N = 300, rho = 0.2 (alpha_l1 = 0.5111): over named seeds 0-39, the LP success fraction is below 1/2 at
    alpha_l1 - 0.08 and above 1/2 at alpha_l1 + 0.08 (the crossing of 1/2 lies in between).  Power: on the unused
    seeds 100-139 the fractions were 0.025 and 1.0; with 40 trials a true success rate <= 0.2 (>= 0.8) exceeds (falls
    below) 1/2 with probability < 1e-4.  Control (a wrong boundary: the Bayes spinodal 0.355 taken for the l1 one):
    the LP fails there (fraction 0.0 on the unused seeds; asserted < 1/2)."""
    a = CS.cs_phase_boundary(0.2).alpha_l1
    below = CS.l1_success_fraction(300, a - 0.08, 0.2, range(40))
    above = CS.l1_success_fraction(300, a + 0.08, 0.2, range(40))
    assert below.fraction < 0.5 < above.fraction, (below, above)
    bayes = CS.cs_phase_boundary(0.2).alpha_bayes
    assert CS.l1_success_fraction(300, bayes, 0.2, range(40)).fraction < 0.5


@slow
def test_bayes_gamp_recovers_above_the_spinodal():
    """KZ Ex. 16.1 at rho = 0.4: Bayes-optimal GAMP (noiseless awgn, Gauss-Bernoulli prior) at N = 2000, seeds 0-3,
    150 steps, reaches MSE < 1e-6 at alpha = 0.65 (above alpha_BP = 0.59) in every run and stays above 1e-3 at
    alpha = 0.5 (hard phase: alpha_IT = 0.4 < 0.5 < alpha_BP) in every run."""
    ch, pr = G.Channel("awgn", Delta=0), Prior("gauss_bernoulli", rho="0.4")
    for alpha, good in ((0.65, True), (0.5, False)):
        for s in range(4):
            A, y, x0 = G.linear_instance(2000, alpha, ch, pr, s)
            r = G.gamp(A, y, ch, pr, iters=150, seed=s, teacher=x0)
            assert (r.mse_trace[-1] < 1e-6) == good and (good or r.mse_trace[-1] > 1e-3), (alpha, s, r.mse_trace[-1])


def test_lp_success_fraction_error_is_clopper_pearson():
    """Final review (ledger K6 l.130): at fraction 1 (alpha = 0.8 far above the l1 line at rho = 0.2, N = 60, seeds
    0-4) the binomial stderr would be 0; the entry reports the exact 99% Clopper-Pearson interval and its larger
    half-width as the error (5/5: [0.3466, 1], the closed form 0.005^(1/5) for the lower end)."""
    from replicant.catalog import catalog
    from replicant.compose import compose
    r = compose(catalog()["compressed_sensing"].spec, "amp").run(rho="0.2", alpha="0.8", N=60, seeds=[0, 1, 2, 3, 4])
    v = r.values
    assert v["l1_success_fraction"] == 1.0 and v["l1_success_ci99_hi"] == 1.0, v
    assert abs(v["l1_success_ci99_lo"] - 0.005 ** (1 / 5)) < 1e-9, v
    assert abs(r.errors["l1_success_fraction"] - (1 - 0.005 ** (1 / 5))) < 1e-9, r.errors
