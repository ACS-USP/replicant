"""K5: the spiked Wishart (bipartite) model, its rectangular AMP, the two-overlap state evolution and the BBP edge
(KZ Sec. 8.1 and Sec. 12.3.3, Eqs. (12.87)-(12.100); Lesieur, Krzakala & Zdeborova (LKZ), arXiv:1701.00858, Eqs.
(195)-(198), (248); Baik, Ben Arous & Peche (BBP), Ann. Probab. 33, 1643 (2005), arXiv:math/0403022, Thm 1.1).

Convention (inference/wishart.py): Y = (lambda/n) u v^T + W, n x m, W_ij ~ N(0, 1/n), alpha = m/n, u in R^n, v in R^m,
unit-variance priors.  The notes' (12.87) W = sqrt(lambda_notes / n) u v^T + xi (alpha = m/n) is sqrt(n) Y with
lambda_notes = lambda^2; LKZ's Y = U V^T / sqrt(N) + sqrt(Delta) xi (alpha = M/N, U in R^N) is sqrt(N Delta) Y with
lambda = 1/sqrt(Delta).  With v Gaussian the m columns of sqrt(n) Y are i.i.d. N(0, I + lambda^2 uhat uhat^T) samples
of dimension n: BBP's N = n, M = m, gamma^2 = M/N = alpha, l_1 = 1 + lambda^2.  Bank: reference/wishart.json.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.amp._stats import student_band
from replicant.bayes import channels as CH
from replicant.core import Refusal
from replicant.inference import wishart as WS
from replicant.refdata import check, load
from replicant.testing import slow

BANK = load("wishart")


def _ok(key, got):
    return check(BANK[key], got, label=key).ok


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


def _planted(key, value, fn):
    old = WS._HOOKS[key]
    WS._HOOKS[key] = value
    try:
        return fn()
    finally:
        WS._HOOKS[key] = old
        assert WS._HOOKS[key] is old


def test_denoiser_dy_is_snr_scaled_variance():
    """The Onsager term needs d E[x|y] / dy.  For y = sqrt(snr) x + z, differentiating the posterior mean under the
    integral gives d E[x|y]/dy = sqrt(snr) Var[x|y] (channels.denoiser_dy); checked by a central difference for four
    priors.  Control: the bare variance (the unscaled slip) misses it by a factor sqrt(snr) = 1.5."""
    y = np.linspace(-4, 4, 41)
    for pr in (CH.Prior("gaussian"), CH.Prior("rademacher"), CH.Prior("sparse", rho="1/4", a=2),
               CH.Prior("gauss_bernoulli")):
        mean, var = CH.denoiser(pr, "9/4")
        fd = (mean(y + 1e-6) - mean(y - 1e-6)) / 2e-6
        assert np.max(np.abs(CH.denoiser_dy(pr, "9/4")(y) - fd)) < 1e-7, pr
        assert np.max(np.abs(var(y) - fd)) > 0.1, pr


def test_wishart_threshold_conventions_both_ways():
    """lambda_c = alpha^(-1/4) in replicant's snr <-> LKZ Delta_c = sqrt(alpha) (Eq. (198), unit variances) <-> the
    notes' lambda_notes = lambda^2 = alpha^(-1/2) <-> BBP l_c - 1 = gamma^(-1) with gamma^2 = alpha.  The banked
    lambda_c(alpha = 4) is reproduced by the SE's linear instability (numerical slopes of the two scalar channels),
    and the closed form agrees.  Control: the planted power alpha^(-1/2) (lambda_notes read as lambda) misses."""
    se = WS.wishart_se_threshold(4, "gaussian", "rademacher")
    assert _ok("lambda_c_alpha4", se) and _ok("lambda_c_alpha4", WS.bbp_threshold(4))
    for a in (0.25, 1.0, 4.0):
        lam = WS.bbp_threshold(a)
        assert abs(WS.lkz_delta_from_snr(lam) - math.sqrt(a)) < 1e-15                  # LKZ Delta_c
        assert abs(WS.snr_from_lkz_delta(math.sqrt(a)) - lam) < 1e-15
        assert abs(WS.notes_from_snr(lam) - a**-0.5) < 1e-15 and abs(WS.snr_from_notes(a**-0.5) - lam) < 1e-15
        assert abs((1 + lam**2) - (1 + 1 / math.sqrt(a))) < 1e-15                       # BBP (36), gamma^2 = alpha
        edge = WS.bbp_top_singular(lam, a)
        assert abs(edge - (1 + math.sqrt(a))) < 1e-12                                     # BBP (4): ((1+g)/g)^2 alpha
        assert abs(WS.wishart_se_threshold(a, "gaussian", "gaussian") - lam) < 1e-9
    assert not _planted("threshold_power", -0.5, lambda: _ok("lambda_c_alpha4", WS.bbp_threshold(4)))


def test_wishart_se_gaussian_closed_form():
    """Gaussian u and v: the SE maps are LKZ (248), m_u = alpha m_v / (Delta + alpha m_v) and likewise for v
    (m = snr / (1 + snr) for the unit Gaussian channel), here iterated in closed form and compared with wishart_se's
    quadrature route; m = 0 below lambda_c, > 0 above.  Control: dropping alpha from the u-channel moves the
    fixed point."""
    def closed(lam, a):
        mu = mv = 1.0
        for _ in range(4000):
            mu = a * lam**2 * mv / (1 + a * lam**2 * mv)
            mv = lam**2 * mu / (1 + lam**2 * mu)
        return mu, mv
    for lam, a in ((1.5, 0.5), (1.2, 2.0), (2.0, 1.0)):
        r = WS.wishart_se(lam, a, "gaussian", "gaussian")
        mu, mv = closed(lam, a)
        assert r.converged and abs(r.m_u - mu) < 1e-9 and abs(r.m_v - mv) < 1e-9, (lam, a, r, mu, mv)
    below = WS.wishart_se(0.95 * WS.bbp_threshold(0.5), 0.5, "gaussian", "rademacher")
    assert below.m_u < 1e-6 and below.m_v < 1e-6, below
    bad = _planted("alpha_in_u_channel", False, lambda: WS.wishart_se(1.5, 0.5, "gaussian", "gaussian"))
    assert abs(bad.m_u - closed(1.5, 0.5)[0]) > 1e-2, bad


def _svd_tops(n, a, lam, seeds):
    return [float(np.linalg.svd(WS.wishart_matrix(n, a, lam, "gaussian", "gaussian", s)[0], compute_uv=False)[0])
            for s in seeds]


def test_wishart_bbp_vs_svd():
    """alpha = 1/4, n = 1200 (m = 300), Gaussian u, v, seeds 0-11.  Above the edge (lambda = 1.9 > lambda_c = 1.414):
    the top singular value sits at BBP's outlier sqrt((1 + lambda^2)(alpha lambda^2 + 1))/lambda = 1.5587 inside the
    99% Student-t band.  The planted wrong power alpha^(-1/2) puts lambda_c at 2 and predicts the edge 1 + sqrt(alpha)
    = 1.5 there, 0.059 away.  Calibration seeds 100-115 (not used here): per-seed SD 0.033, mean 1.5529 (finite-n
    bias -0.006, inside the band); 12 seeds give a half-width about 3.1 x 0.033 / sqrt(12) = 0.030, so the plant is
    excluded when the mean falls 0.029 (3 stderr) short of it: power about 0.998.  Below the edge (lambda = 1): the
    band excludes the outlier formula (1.5811) and its mean lies within 3 BBP edge scales (0.0064 here) of 1.5 (BBP
    Eq. (5): fluctuations of order (1 + gamma)^(4/3) / (gamma M^(2/3)) in lambda_1(S); calibration mean 1.4933, 1.05
    scales below).  Statistical: never final."""
    n, a, seeds = 1200, 0.25, range(12)
    hi = student_band(_svd_tops(n, a, 1.9, seeds))
    pred = WS.bbp_top_singular(1.9, a)
    assert abs(pred - 1.5586) < 1e-3 and abs(hi.mean - pred) <= hi.half_width, (hi, pred)
    planted = _planted("threshold_power", -0.5, lambda: WS.bbp_top_singular(1.9, a))
    assert planted == 1 + math.sqrt(a) and abs(hi.mean - planted) > hi.half_width, (hi, planted)
    lo = student_band(_svd_tops(n, a, 1.0, seeds))
    out = math.sqrt((1 + 1.0) * (a + 1)) / 1.0
    assert lo.mean + lo.half_width < out, (lo, out)
    assert abs(lo.mean - WS.bbp_top_singular(1.0, a)) <= 3 * WS.bbp_edge_scale(n, a), (lo, WS.bbp_edge_scale(n, a))


def _amp_runs(n, a, lam, seeds, iters, s0):
    out = []
    for s in seeds:
        Y, u, v = WS.wishart_matrix(n, a, lam, "gaussian", "rademacher", s)
        v_init = WS.informative_start(v, "rademacher", s0, s)
        r = WS.wishart_amp(Y, "gaussian", "rademacher", snr=lam, iters=iters, v_init=v_init)
        out.append(([float(x @ u) / u.size for x in r.u_hats], [float(x @ v) / v.size for x in r.v_hats]))
    return out


def _se_vs_amp(n, seeds, iters, plant=None):
    a, lam, s0 = 0.5, 1.6, 0.3
    se = WS.wishart_se(lam, a, "gaussian", "rademacher", iters=iters, init_snr=s0)
    runs = (_amp_runs(n, a, lam, seeds, iters, s0) if plant is None else
            _planted("onsager_scale", plant, lambda: _amp_runs(n, a, lam, seeds, iters, s0)))
    bad = 0
    for t in range(iters):
        bu = student_band([r[0][t] for r in runs])
        bv = student_band([r[1][t + 1] for r in runs])
        bad += abs(bu.mean - se.trajectory[t + 1][0]) > bu.half_width
        bad += abs(bv.mean - se.trajectory[t + 1][1]) > bv.half_width
    return bad


def test_wishart_se_vs_amp_small():
    """Default tier: n = 600, alpha = 1/2, lambda = 1.6, u Gaussian, v Rademacher, an informative start independent of
    W (vhat^0 = E[v | sqrt(0.3) v + z]); m_u^t and m_v^{t+1} of AMP over seeds 0-7 against the SE trajectory for
    t = 0..4 (10 comparisons, 99% Student-t bands): at most one excursion.  Dropping the Onsager term must put at
    least 5 outside.  Calibrated on 20 disjoint seed sets 100-107, ..., 2000-2007 (not used here): the true AMP had 0
    excursions in all 20, the no-Onsager plant 9 in all 20 (power ~1 for the rule >= 5)."""
    assert _se_vs_amp(600, range(8), 5) <= 1
    assert _se_vs_amp(600, range(8), 5, plant=0.0) >= 5


@slow
def test_wishart_se_vs_amp():
    """The reference size n = 2000 (m = 1000), seeds 0-9, t = 0..7 (16 comparisons): at most one outside its 99%
    band; the no-Onsager control out in at least 8 (seed set 100-109, not used here: 0 and 15 of 16)."""
    assert _se_vs_amp(2000, range(10), 8) <= 1
    assert _se_vs_amp(2000, range(10), 8, plant=0.0) >= 8


def test_wishart_amp_divergence_is_undetermined():
    """A divergent run is not banded: with the Onsager term planted 1e300 times too large and the (unbounded) Gaussian
    denoisers the iterates overflow to non-finite values, wishart_amp reports finite False (never a Refusal), and the catalog's m_u_amp,
    m_v_amp are absent with amp_status undetermined while the SVD band stays.  With Rademacher v the same plant keeps
    the iterates bounded but oscillating: finite, not converged, amp_status not-converged and the finality says so.
    The true run at the same point is converged."""
    from replicant import catalog_wishart as CW
    Y, u, v = WS.wishart_matrix(200, 0.5, 1.6, "gaussian", "gaussian", 0)
    r = _planted("onsager_scale", 1e300, lambda: WS.wishart_amp(Y, "gaussian", "gaussian", snr=1.6, iters=20,
                                                                 v_init=WS.informative_start(v, "gaussian", 0.3, 0)))
    assert not r.finite and not r.converged, r.increments
    run = lambda pv: CW._finite({"alpha": "0.5", "snr": "1.6", "prior_v": pv, "N": 200, "seeds": [0, 1]}).values
    bad = _planted("onsager_scale", 1e300, lambda: run("gaussian"))
    assert bad["amp_status"] == "undetermined" and "m_u_amp" not in bad, bad
    assert "sigma_top_svd" in bad and bad["amp_finite_fraction"] == 0
    osc = _planted("onsager_scale", 1e6, lambda: run("rademacher"))
    assert osc["amp_status"] == "not-converged" and "NOT converged" in CW._amp_label(osc).note, osc
    good = run("rademacher")
    assert good["amp_status"] == "converged" and good["amp_converged_fraction"] == 1, good
    assert "m_u_amp" in good and "NOT converged" not in CW._amp_label(good).note
    fin = CW.spiked_wishart().finality["m_u_amp"]
    assert fin is CW._amp_label


def test_wishart_refusals():
    _refuses(lambda: WS.wishart_se(1.5, 0.5, "cauchy", "gaussian"), "precondition:known-prior")
    _refuses(lambda: WS.wishart_se(1.5, 0.5, CH.Prior("sparse"), "gaussian"), "precondition:unit-variance")
    _refuses(lambda: WS.bbp_threshold(0), "precondition:alpha")
    _refuses(lambda: WS.wishart_matrix(100, 0.333, 1.5, "gaussian", "gaussian", 0), "precondition:size")
