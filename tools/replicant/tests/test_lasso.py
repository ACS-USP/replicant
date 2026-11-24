"""K6: the LASSO by AMP, its state evolution and calibration, and an exact solver (linear/lasso.py; KZ notes Sec. 12.7,
16.6.2 Alg. 4, 18.1).

Convention (the notes'): A is M x N with i.i.d. N(0, 1/N) entries, alpha = M/N, y = A x0 + sqrt(Delta) xi (`sigma2` =
Delta), LASSO (1/2)|y - A x|^2 + lam |x|_1.  Bayati & Montanari (arXiv:1008.2581, "BM"): A_BM with entries of variance
1/n (n = M), delta = n/N, w with E W^2 = sigma_BM^2, the same cost (1.2).  Map: A = sqrt(delta) A_BM, so dividing the
notes' cost by delta gives BM's with y/sqrt(delta): delta = alpha, sigma_BM^2 = Delta/alpha, lam_BM = lam/alpha.
"""
import math
import os
import sys

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.amp._stats import student_band
from replicant.bayes.channels import Prior
from replicant.core import Refusal
from replicant.learning import ridge as R
from replicant.linear import gamp as G
from replicant.linear import lasso as L
from replicant.testing import slow


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


def _planted(key, value, fn):
    old = L._HOOKS[key]
    L._HOOKS[key] = value
    try:
        return fn()
    finally:
        L._HOOKS[key] = old
        assert L._HOOKS[key] is old


# ---- Bayati-Montanari, coded here from the fetched paper in its own convention ------------------------------------
def _bm_eta(x, th):
    return math.copysign(max(abs(x) - th, 0.0), x)


def _q(f, lo, hi, pts):
    return quad(f, lo, hi, points=pts, limit=200, epsabs=1e-12, epsrel=1e-11)[0]


def _bm_mse_on(t, th, atoms, gauss):
    """(E[(eta(X0 + t Z; th) - X0)^2], P(|X0 + t Z| > th)) for X0 ~ sum_k p_k delta_{x_k} + gauss N(0, 1), by scipy
    quad (kinks as break points).  Atoms: over Z.  The Gaussian part: over Y = X0 + t Z ~ N(0, 1 + t^2), with
    E[(eta(Y) - X0)^2 | Y] = (eta(Y) - c Y)^2 + c t^2, c = 1/(1 + t^2) (the Gaussian posterior of X0 given Y)."""
    mse = on = 0.0
    for p, x in atoms:
        pts = sorted(((th - x) / t, (-th - x) / t))
        mse += p * _q(lambda z: (_bm_eta(x + t * z, th) - x) ** 2 * norm.pdf(z), -40, 40, pts)
        on += p * (norm.cdf(pts[0]) + norm.sf(pts[1]))
    if gauss:
        s = math.sqrt(1 + t * t)
        c = 1 / (s * s)
        mse += gauss * _q(lambda y: ((_bm_eta(y, th) - c * y) ** 2 + c * t * t) * norm.pdf(y / s) / s,
                          -40 * s, 40 * s, (-th, th))
        on += gauss * 2 * norm.sf(th / s)
    return mse, on


def _bm_point(delta, sigma2, a, atoms, gauss):
    """BM Eqs. (1.7), (1.9), (1.15), Cor. 1.6 at threshold multiplier a (BM's alpha): tau_*^2 the root of
    tau^2 = F(tau^2, a tau) (BM Prop. 1.3: unique for a > a_min, (1.14)), lam(a) = a tau_* (1 - E eta'/delta);
    returns (lam_BM, risk, delta (tau^2 - sigma^2), tau_*^2, theta_* = a tau_*)."""
    amin = brentq(lambda k: (1 + k * k) * norm.cdf(-k) - k * norm.pdf(k) - delta / 2, 0, 10)
    assert a > amin
    F = lambda t2: sigma2 + _bm_mse_on(math.sqrt(t2), a * math.sqrt(t2), atoms, gauss)[0] / delta
    hi = 2 * sigma2 + 1
    while F(hi) > hi:
        hi *= 2
    t2 = brentq(lambda t2: F(t2) - t2, sigma2, hi, xtol=1e-12, rtol=1e-11)
    t = math.sqrt(t2)
    risk, on = _bm_mse_on(t, a * t, atoms, gauss)
    return a * t * (1 - on / delta), risk, delta * (t2 - sigma2), t2, a * t


_FIG1 = (0.64, 0.2, Prior("sparse", rho="0.128", a=1), [(0.064, 1.0), (0.064, -1.0), (0.872, 0.0)], 0.0)
_GB = (0.6, 0.1, Prior("gauss_bernoulli", rho="0.3"), [(0.7, 0.0)], 0.3)


def _bm_both_ways(delta, s2, prior, atoms, gauss, a, lam_notes):
    """BM -> notes: BM's lam_BM(a) handed to lasso_se as lam = delta lam_BM, Delta = delta sigma_BM^2; notes -> BM:
    lasso_se at the notes' lam_notes, its kappa handed to BM's formula, which must return lam_BM = lam_notes/alpha
    (alpha = delta) at sigma_BM^2 = Delta/alpha.  Risk, tau^2 and threshold agree to 1e-8 both ways; BM's identity
    risk = delta (tau^2 - sigma^2) to 1e-9.  Control: the slope scaled by 0.8 moves the risk by > 1e-4."""
    lam_bm, risk, ident, t2, th = _bm_point(delta, s2, a, atoms, gauss)
    assert abs(risk - ident) < 1e-9 and lam_bm > 0
    got = L.lasso_se(delta, None, delta * lam_bm, delta * s2, prior=prior)
    assert abs(got.risk - risk) < 1e-8 and abs(got.tau2 - t2) < 1e-8 and abs(got.threshold - th) < 1e-8, \
        (delta, a, got, risk, t2, th)
    bad = _planted("slope", 0.8, lambda: L.lasso_se(delta, None, delta * lam_bm, delta * s2, prior=prior))
    assert abs(bad.risk - risk) > 1e-4, (delta, a, bad.risk, risk)
    alpha, Delta = delta, delta * s2                   # the notes' parameters of this BM problem
    se = L.lasso_se(alpha, None, lam_notes, Delta, prior=prior)
    lam_back, risk_back, _, t2_back, th_back = _bm_point(alpha, Delta / alpha, se.kappa, atoms, gauss)
    assert abs(lam_back - lam_notes / alpha) < 1e-8, (lam_back, lam_notes / alpha)
    assert abs(risk_back - se.risk) < 1e-8 and abs(t2_back - se.tau2) < 1e-8 and abs(th_back - se.threshold) < 1e-8


def test_lasso_risk_vs_bayati_montanari():
    """lasso_se (the notes' convention, exact Gaussian partial moments, lam -> kappa by brentq) against BM's formula
    coded above from the paper (scipy quad, its own convention, threshold multiplier -> lam) at BM's Fig. 1 law
    p(+-1) = 0.064, p(0) = 0.872, delta = 0.64, sigma_BM^2 = 0.2: BM -> notes at their multiplier a = 2, and notes ->
    BM at the notes' lam = 0.1 (_bm_both_ways: the convention map checked in both directions).  The Gauss-Bernoulli
    case is the slow test below (default-tier time)."""
    delta, s2, prior, atoms, gauss = _FIG1
    _bm_both_ways(delta, s2, prior, atoms, gauss, 2.0, 0.1)


@slow
def test_lasso_risk_vs_bayati_montanari_gauss_bernoulli():
    """As above for a Gauss-Bernoulli law (rho = 0.3) at delta = 0.6, sigma_BM^2 = 0.1: multiplier a = 1.5, and the
    notes' lam = 0.2."""
    delta, s2, prior, atoms, gauss = _GB
    _bm_both_ways(delta, s2, prior, atoms, gauss, 1.5, 0.2)


def test_lasso_calibration_and_least_squares_limit():
    """calibrate(kappa) -> lam inverts lasso_se (lam -> kappa) to 1e-10 at three points; the SE fixed point satisfies
    risk = alpha tau^2 - Delta (the notes' tau^2 = (E + Delta)/alpha) and lam = threshold (alpha - b) (the derived
    calibration); at lam = 0 and alpha = 2 the LASSO is least squares and its risk is the ridgeless HMRT value
    Delta/(alpha - 1) = ridge.ridgeless_test_error(1/alpha, Delta) for two priors (the prior drops out).  Bad iters
    (0, negative, non-int, bool) are refused by name in AMP, proximal gradient, the SE iteration, coordinate descent
    and the lasso entry."""
    for alpha, rho, lam, D in ((0.5, 0.1, 0.05, 0.01), (0.8, 0.3, 0.4, 0.2), (1.5, 0.5, 1.0, 0.5)):
        se = L.lasso_se(alpha, rho, lam, D)
        assert abs(L.calibrate(se.kappa, alpha, rho, D) - lam) < 1e-10
        assert abs(se.risk - (alpha * se.tau2 - D)) < 1e-12
        assert abs(se.threshold * (alpha - se.b) - lam) < 1e-12
    for prior in (Prior("gauss_bernoulli", rho="0.2"), Prior("rademacher")):
        ls = L.lasso_se(2, None, 0, 1, prior=prior)
        assert abs(ls.risk - float(R.ridgeless_test_error("0.5", 1))) < 1e-12, ls
    _refuses(lambda: L.lasso_se(0.5, 0.1, 0, 0.1), "precondition:lambda")
    _refuses(lambda: L.lasso_se(0.5, 0.1, 0.1, 0), "precondition:sigma2")
    _refuses(lambda: L.lasso_se(0.5, 1.5, 0.1, 0.1), "precondition:rho")
    _refuses(lambda: L.lasso_se(0.5, 0.1, 0.1, 0.1, prior=Prior("rademacher")), "precondition:known-parameter")
    A, y = np.eye(3)[:2], np.ones(2)
    for bad in (0, -1, 2.5, True):
        _refuses(lambda: L.lasso_amp(A, y, 0.1, iters=bad), "precondition:positive-iters")
        _refuses(lambda: L.proximal_gradient(A, y, 0.1, iters=bad), "precondition:positive-iters")
        _refuses(lambda: L.lasso_se_iterate(0.5, 0.1, 0.1, 0.1, iters=bad), "precondition:positive-iters")
    _refuses(lambda: L.lasso_exact(A, y, 0.1, max_sweeps=0), "precondition:positive-iters")
    from replicant.catalog import catalog
    _refuses(lambda: catalog()["lasso"].run("amp", alpha="0.5", rho="0.1", lam="0.1", sigma2="0.01", N=50, iters=0),
             "precondition:positive-iters")


def test_lasso_amp_is_the_lasso_at_small_n():
    """N = 250, alpha = 0.5, Gauss-Bernoulli rho = 0.1, Delta = 0.01, lam = 0.1, named seeds 0-3 (no seed picked
    from a survey: every seed whose undamped Alg. 4 converges is checked -- it can diverge at small N, e.g. N = 200,
    alpha = 0.6, and a non-converged seed is reported converged False and skipped, at least one must converge; the
    checks are deterministic fixed-point equalities): KZ Alg. 4 converges, its
    fixed point satisfies the calibration lam = threshold (alpha - b) with the empirical b = <1{|R| > threshold}>
    (derived: Sigma = (1 + V)/alpha and V = Sigma b give Sigma (alpha - b) = 1), is a fixed point of the proximal-
    gradient map with eta = 1/(alpha - b) (KZ (12.154)), and equals the coordinate-descent LASSO (KKT residual
    < 1e-9) to 1e-7; plain proximal gradient (eta = 1/|A|^2) reaches it too.  Control: the slope scaled by 0.8 in
    the Onsager term (_HOOKS["slope"]) leaves the fixed point, where Alg. 4 still converges, a LASSO solution (any consistent Onsager coefficient c
    gives a = ST(a + A^T (y - A a)/(alpha - c b); lam/(alpha - c b)): measured 2e-11 from the LASSO) but breaks the
    calibration lam = threshold (alpha - b) by > 1e-3: the calibration, not the fixed point, carries the slope."""
    ch, pr = G.Channel("awgn", Delta="0.01"), Prior("gauss_bernoulli", rho="0.1")
    checked, planted = [], []
    for seed in (0, 1, 2, 3):
        A, y, x0 = G.linear_instance(250, 0.5, ch, pr, seed)
        amp = L.lasso_amp(A, y, 0.1)
        if not amp.converged:
            continue
        checked.append(seed)
        ex = L.lasso_exact(A, y, 0.1)
        assert ex.converged and ex.kkt_residual < 1e-9, seed
        assert abs(amp.threshold * (0.5 - amp.b) - 0.1) < 1e-10, seed
        eta = 1 / (0.5 - amp.b)
        pg = L.soft_threshold(amp.x + eta * A.T @ (y - A @ amp.x), eta * 0.1)
        assert np.max(np.abs(pg - amp.x)) < 1e-8, seed
        assert np.max(np.abs(amp.x - ex.x)) < 1e-7, seed
        ista = L.proximal_gradient(A, y, 0.1, iters=20000, tol=1e-13)
        assert ista.converged and np.max(np.abs(ista.x - ex.x)) < 1e-7, seed
        bad = _planted("slope", 0.8, lambda: L.lasso_amp(A, y, 0.1))
        if bad.converged:              # the planted slope may also stop Alg. 4 converging (seed 3): caught there
            assert np.max(np.abs(bad.x - ex.x)) < 1e-7, seed
            assert abs(bad.threshold * (0.5 - bad.b) - 0.1) > 1e-3, seed
            planted.append(seed)
    assert checked and planted, ("no seed converged", checked, planted)


@slow
def test_lasso_amp_vs_exact_solver():
    """Reference size N = 1500, alpha = 0.5, Gauss-Bernoulli rho = 0.1, Delta = 0.01, lam = 0.05, named seeds 0-7:
    per seed, AMP's fixed point equals the coordinate-descent LASSO (KKT < 1e-9) to 1e-6 (deterministic: both solve
    the same convex problem) and lam = threshold (alpha - b) holds; across seeds the 99% Student-t bands of the
    LASSO MSE and of the empirical threshold contain the SE predictions (lasso_se: risk 0.01207, threshold 0.2466;
    the calibration derived, never fitted).  Power (planted error: slope 0.8 in the SE calibration, threshold 0.2255):
    on the unused seeds 100-107 the threshold band's half-width was 0.0118 (stderr 0.0034), so the planted shift
    0.021 is 6 stderr (power > 0.99); the risk shift (1.1e-3) is below the risk band's half-width (2.3e-3) and is not
    asserted."""
    ch, pr = G.Channel("awgn", Delta="0.01"), Prior("gauss_bernoulli", rho="0.1")
    risks, ths = [], []
    for s in range(8):
        A, y, x0 = G.linear_instance(1500, 0.5, ch, pr, s)
        amp = L.lasso_amp(A, y, 0.05)
        ex = L.lasso_exact(A, y, 0.05)
        assert amp.converged and ex.kkt_residual < 1e-9, s
        assert np.max(np.abs(amp.x - ex.x)) < 1e-6, s
        assert abs(amp.threshold * (0.5 - amp.b) - 0.05) < 1e-10
        risks.append(float((ex.x - x0) @ (ex.x - x0)) / 1500)
        ths.append(amp.threshold)
    se = L.lasso_se(0.5, 0.1, 0.05, 0.01)
    rb, tb = student_band(risks), student_band(ths)
    assert abs(rb.mean - se.risk) <= rb.half_width, (rb, se.risk)
    assert abs(tb.mean - se.threshold) <= tb.half_width, (tb, se.threshold)
    bad = _planted("slope", 0.8, lambda: L.lasso_se(0.5, 0.1, 0.05, 0.01))
    assert abs(tb.mean - bad.threshold) > tb.half_width, (tb, bad.threshold)


def test_lasso_alg4_state_evolution_reaches_the_calibrated_fixed_point():
    """The amp route's SE (KZ Alg. 4 iterated from a = v = 0, linear/lasso.lasso_se_iterate: theta_t = lam Sigma_t,
    Sigma_t = (1 + V_t)/alpha) converges to the replica route's fixed point (lasso_se: BM's calibration solved by
    root finding) to 1e-12 in risk and threshold at four points, including lam = 0 at alpha = 2 (least squares).
    Two computations of one fixed point: the iteration never uses (*), the root solve never iterates.  Control: the
    iteration stopped after 3 steps misses the fixed point by > 1e-3."""
    for args in ((0.5, 0.1, 0.05, 0.01), (0.6, 0.15, 0.05, 0.01), (2, 0.2, 0, 1), (0.3, 0.05, 0.01, 0.001)):
        it, fp = L.lasso_se_iterate(*args), L.lasso_se(*args)
        assert it.converged and abs(it.risk - fp.risk) < 1e-12 and abs(it.threshold - fp.threshold) < 1e-12, args
        assert abs(L.lasso_se_iterate(*args, iters=3).risk - fp.risk) > 1e-3, args
