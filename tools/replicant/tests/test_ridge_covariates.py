"""K6: ridge with a covariance spectrum (KZ notes Sec. 18.1.2) and the notes' ridge errata (Eqs. (12.163)-(12.164),
(18.17)-(18.18), and the q trace closing Sec. 18.1.2).

Conventions (learning/ridge.py): HMRT's gamma = p/n and penalty lam; the notes' regression alpha = n/p = 1/gamma
(re-derived here: the notes' A is m x n with E A_ij^2 = 1/n and alpha = m/n, i.e. samples over dimension), their
noise Delta = sigma^2, their signal rho = |beta|^2 = r^2 and their ridge penalty lam_notes = lam/gamma (the notes'
loss (1/2)|y - A x|^2 + (lam_notes/2)|x|^2 with b = x/sqrt(p) is (1/2)|y - X b|^2 + (lam_notes p/2)|b|^2, HMRT's
(n lam/2)|b|^2).  Bank: reference/ridge.json (HMRT values).
"""
import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dataclasses import replace

from replicant.catalog import catalog, check_ref
from replicant.core import Refusal
from replicant.learning import ridge as R
from replicant.learning import ridge_sim as SIM
from replicant.refdata import check, load
from replicant.testing import slow

BANK = load("ridge")
_RIDGELESS = (("ridgeless_g0p5_s1_r1", "0.5", "1", "1"), ("ridgeless_g0p9_s1_r1", "0.9", "1", "1"),
              ("ridgeless_g1p1_s1_r1", "1.1", "1", "1"), ("ridgeless_g2_s1_r1", "2", "1", "1"),
              ("ridgeless_g4_s0p25_r1", "4", "0.25", "1"))
_RIDGE = (("ridge_g1_lam0p5_s1_r1", "1", "0.5", "1", "1"), ("ridge_g2_lam0p1_s0p5_r2", "2", "0.1", "0.5", "2"),
          ("ridge_g0p5_lam1_s1_r1", "0.5", "1", "1", "1"))
TWO = "1/2:1/2,2:1/2"          # the two-point spectrum of the checks: eigenvalues 1/2 and 2, equal weights


def _ok(key, got):
    return check(BANK[key], got, label=key).ok


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


def _planted(term, value, fn):
    old = R._SCALE[term]
    R._SCALE[term] = value
    try:
        return fn()
    finally:
        R._SCALE[term] = old
        assert R._SCALE[term] is old


def test_ridge_covariates_reduce_to_isotropic():
    """An identity covariance, written three ways (one atom, two equal atoms, a string), runs the general kappa path
    and reproduces the 8 banked ridge / ridgeless risks (tol 1e-12); through the catalog entry with covariance="1:1",
    every one of the 14 keys of reference/ridge.json (ridgeless, ridge, the 4 optimal test MSEs and the 2 optimal
    penalties) is reproduced by the RefChecks' own extractors (the RS-saddle extractor excepted: that step is
    isotropic-only; its keys are covered by `risk`); the default covariance gives results identical (==) to the old
    call without the argument.  Controls: a missing bias in the general path fails the
    bank; the two-point spectrum (mean eigenvalue 5/4) does not reproduce the isotropic values."""
    for cov in ([(1, 1)], [(1, "1/2"), (1, "1/2")], "1:1"):
        for key, g, s2, r2 in _RIDGELESS:
            assert _ok(key, R.ridgeless_test_error(g, s2, r2, covariance=cov)), (cov, key)
        for key, g, lam, s2, r2 in _RIDGE:
            assert _ok(key, R.ridge_test_error(g, lam, s2, r2, covariance=cov)), (cov, key)
    e = catalog()["ridge"]
    checked = set()
    for ref in e.refs:
        if ref.key.startswith("ridge_") and ref is not next(r for r in e.refs if r.key == ref.key):
            continue                                   # the second RefCheck of a ridge key reads risk_rs
        assert check_ref(e, replace(ref, params={**ref.params, "covariance": "1:1"})).ok, ref.key
        checked.add(ref.key)
    assert checked == set(BANK) and len(checked) == 14, sorted(set(BANK) - checked)
    for key, g, s2, r2 in _RIDGELESS:
        assert R.ridgeless_test_error(g, s2, r2) == R.ridgeless_test_error(g, s2, r2, covariance="isotropic")
    for key, g, lam, s2, r2 in _RIDGE:
        assert R.ridge_bias_variance(g, lam, s2, r2) == R.ridge_bias_variance(g, lam, s2, r2, "isotropic")
    key, g, lam, s2, r2 = _RIDGE[1]
    assert not _planted("bias", 0, lambda: _ok(key, R.ridge_test_error(g, lam, s2, r2, covariance=[(1, 1)])))
    assert not any(_ok(k, R.ridge_test_error(g, lam, s2, r2, covariance=TWO)) for k, g, lam, s2, r2 in _RIDGE)
    _refuses(lambda: R.ridge_test_error(2, 1, covariance="1:1/2"), "precondition:covariance")
    _refuses(lambda: R.ridge_test_error(2, 1, covariance="0:1"), "precondition:covariance")
    _refuses(lambda: R.ridge_test_error(2, 1, covariance="a:b"), "precondition:covariance")


def _H(spec):
    return [(mp.mpf(e.numerator) / e.denominator, mp.mpf(w.numerator) / w.denominator) for e, w in spec]


def _hmrt_ridge(spec, g, lam, s2, r2):
    """HMRT Definition 2 (Eqs. (36)-(39)), read from the fetched paper, coded independently of ridge.py: m(z) from
    (36) at z = -lam by bisection on (0, 1/lam] (the right side decreases in m, the left increases: one root),
    m'(z) by implicit differentiation of (36), m_1 from (37); G = H
    (beta uniform on the sphere, HMRT Sec. 4.2)."""
    H = _H(spec)
    with mp.workdps(30):
        g, lam, z = mp.mpf(g), mp.mpf(lam), -mp.mpf(lam)
        lo, hi = mp.mpf(0), 1 / lam
        for _ in range(200):
            m = (lo + hi) / 2
            lo, hi = (m, hi) if sum(w / (s * (1 - g - g * z * m) - z) for s, w in H) > m else (lo, m)
        m = (lo + hi) / 2
        # d/dz of m = int 1/(s(1 - g - g z m) - z): m' = int (s g (m + z m') + 1)/D^2  ->  solve for m'
        D2 = [(s, w, (s * (1 - g - g * z * m) - z) ** 2) for s, w in H]
        a = sum(w * (s * g * m + 1) / d for s, w, d in D2)
        b = sum(w * s * g * z / d for s, w, d in D2)
        dm = a / (1 - b)
        c = 1 - g - g * z * m                                  # = 1 - gamma + gamma lam m(-lam)
        num = sum(w * s * s * c / (s * c - z) ** 2 for s, w in H)
        den = 1 - g * sum(w * z * s / (s * c - z) ** 2 for s, w in H)
        m1 = num / den
        bias = lam ** 2 * mp.mpf(r2) * (1 + g * m1) * sum(w * s / (lam + c * s) ** 2 for s, w in H)
        var = mp.mpf(s2) * g * sum(w * s * s * (1 - g + g * lam ** 2 * dm) / (lam + s * c) ** 2 for s, w in H)
        return bias, var


def _hmrt_ridgeless(spec, g, s2, r2):
    """HMRT Definition 1 (Eqs. (10)-(11)) and Cor. 2 (gamma > 1): c0 from (10) by bisection; bias (11) with G = H.
    The variance is (12) with the factor c0 that its printed form lacks: as printed, (12) at Sigma = I gives
    sigma^2 gamma, against Cor. 2's 1/(gamma - 1) (sigma = 1) and Thm 1; with c0 it gives sigma^2 gamma c0 =
    sigma^2/(gamma - 1) (checked below)."""
    H = _H(spec)
    with mp.workdps(30):
        g = mp.mpf(g)
        lo, hi = mp.mpf(0), mp.mpf(10) ** 6
        f = lambda c: sum(w / (1 + c * g * s) for s, w in H) - (1 - 1 / g)
        for _ in range(300):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
        c0 = (lo + hi) / 2
        i2 = sum(w * s * s / (1 + c0 * g * s) ** 2 for s, w in H)
        i1 = sum(w * s / (1 + c0 * g * s) ** 2 for s, w in H)
        bias = mp.mpf(r2) * (1 + g * c0 * i2 / i1) * i1
        printed_var = mp.mpf(s2) * g * i2 / i1
        return bias, printed_var * c0, printed_var, mp.mpf(r2) / (c0 * g * g)


def test_ridge_covariance_spectrum_matches_hmrt_definitions():
    """The kappa form (notes Sec. 18.1.2, derived in learning/ridge.py) against HMRT's own Definitions 1-2 coded here:
    bias and variance agree to 1e-20 on the two-point spectrum and on a three-point one, ridge at lam = 0.1, 0.5, 2
    and gamma = 0.5, 2, ridgeless at gamma = 2, 4; Cor. 2's bias r^2/(c0 gamma^2) agrees with (11).  Control: the
    isotropic closed form at the same points differs by > 1e-3."""
    three = R.covariance_spectrum("1/4:1/4,1:1/2,3:1/4")
    two = R.covariance_spectrum(TWO)
    for spec in (two, three):
        for g in ("0.5", "2"):
            for lam in ("0.1", "0.5", "2"):
                b, v = R.ridge_bias_variance(g, lam, "0.7", "1.3", covariance=spec)
                hb, hv = _hmrt_ridge(spec, g, lam, "0.7", "1.3")
                assert abs(b - hb) < 1e-20 and abs(v - hv) < 1e-20, (spec, g, lam, b, hb, v, hv)
                assert abs(b + v - R.ridge_test_error(g, lam, "0.7", "1.3")) > 1e-3
        for g in ("2", "4"):
            b, v = R.ridgeless_bias_variance(g, "0.7", "1.3", covariance=spec)
            hb, hv, printed, cor2 = _hmrt_ridgeless(spec, g, "0.7", "1.3")
            assert abs(b - hb) < 1e-20 and abs(v - hv) < 1e-20 and abs(hb - cor2) < 1e-20, (spec, g)
    _, hv, printed, _ = _hmrt_ridgeless(R.covariance_spectrum("1:1"), "2", "1", "1")
    assert abs(hv - 1) < 1e-25 and abs(printed - 2) < 1e-25      # HMRT (12) as printed lacks c0


@slow
def test_ridge_two_point_spectrum_matches_finite_n_fits():
    """SVD fits with Sigma = diag(1/2 on half the coordinates, 2 on the other half) over named seeds 0-19 at
    n = 200, 400 (disjoint seed sets per size), extrapolated in 1/n (learning/ridge_sim.limit_ensemble): the 99%
    band of R_inf contains the spectrum's limit at (gamma, lam, sigma2) = (2, 0, 1), (4, 0.2, 1), (2, 0.5, 0.5).
    Power (planted error: the isotropic limit, i.e. the spectrum ignored): it differs from the spectrum's limit by
    0.25, 0.20, 0.065, against band half-widths 0.042 measured on the unused seeds 100-119 (the points were chosen
    there; gamma < 1 points were dropped, where the spectrum moves the risk by < 0.02)."""
    for g, lam, s2 in (("2", 0, 1), ("4", "0.2", 1), ("2", "0.5", "0.5")):
        _, band = SIM.limit_ensemble((200, 400), g, lam, s2, range(20), covariance=TWO)
        want = float(R.ridge_test_error(g, lam, s2, 1, covariance=TWO))
        iso = float(R.ridge_test_error(g, lam, s2, 1))
        assert abs(band.mean - want) <= band.half_width, (g, lam, band, want)
        assert abs(band.mean - iso) > band.half_width, (g, lam, band, iso)


@slow
def test_optimal_penalty_is_the_argmin_with_a_covariance():
    """lam* = sigma^2 gamma / r^2 (the Gaussian-prior posterior mean, derived for any Sigma) is the numerical argmin
    of the kappa-form risk (golden section, no formula for lam*) for the two-point spectrum at (gamma, sigma2, r2) =
    (2, 1, 1), (0.5, 1, 2), to 1e-8; and with the identity covariance at the banked lambda_opt points (2, 1, 1),
    (2, 1, 5) (against the bank at the second).  Control: 2 lam* misses the argmin by > 1e-3 (the comparison
    resolves a wrong penalty)."""
    for g, s2, r2, cov in (("2", 1, 1, TWO), ("0.5", 1, 2, TWO), ("2", 1, 1, "1:1"), ("2", 1, 5, "1:1")):
        best = R.argmin_lambda(g, s2, r2, tol="1e-10", covariance=cov)
        lam = R.optimal_lambda(g, s2, r2)
        assert abs(best - lam) < 1e-8 * (1 + lam) and abs(best - 2 * lam) > 1e-3, (g, cov, best, lam)
    assert _ok("lambda_opt_g2_s1_r5", R.argmin_lambda("2", 1, 5, tol="1e-14", covariance="1:1"))


def test_ridge_covariance_finite_n_small():
    """Default-tier smoke check of the spectrum in ridge_sim: one fit at n = 60, p = 120 with the two-point spectrum
    has Sigma-weighted risk equal to (beta_hat - beta)^T Sigma (beta_hat - beta) and R_X = B_X + V_X; a weight that
    gives a non-integer coordinate count is refused."""
    import numpy as np
    f = SIM.ridge_fit(60, 120, "0.3", 1, 7, covariance=TWO)
    sig = np.array([0.5] * 60 + [2.0] * 60)
    d = f.beta_hat - f.beta
    assert abs(f.risk - float(d @ (sig * d))) < 1e-12 and abs(f.risk_x - f.bias_x - f.variance_x) < 1e-12
    iso = SIM.ridge_fit(60, 120, "0.3", 1, 7)
    assert iso.risk_x == SIM.ridge_fit(60, 120, "0.3", 1, 7, covariance="isotropic").risk_x
    _refuses(lambda: SIM.ridge_fit(60, 121, "0.3", 1, 7, covariance=TWO), "precondition:covariance")


def test_catalog_ridge_covariance_parameter():
    """The ridge entry reads `covariance`: identity reproduces the default run's numbers; a spectrum changes the
    closed-form risk, keeps the optimal penalty (kappa-form risk at lam*) and skips the isotropic-only RS saddle with
    a stated reason."""
    e = catalog()["ridge"]
    base = e.run(gamma="2", lam="0.1", sigma2="0.5", signal="2")
    ident = e.run(gamma="2", lam="0.1", sigma2="0.5", signal="2", covariance="1:1")
    assert abs(base["risk"] - ident["risk"]) < 1e-25 and abs(base["risk_ridgeless"] - ident["risk_ridgeless"]) < 1e-25
    two = e.run(gamma="2", lam="0.1", sigma2="0.5", signal="2", covariance=TWO)
    assert abs(two["risk"] - R.ridge_test_error("2", "0.1", "0.5", "2", covariance=TWO)) < 1e-25
    assert "risk_rs" not in two and two["covariance"] == TWO
    assert abs(two["risk_opt"] - R.ridge_test_error("2", two["lambda_opt"], "0.5", "2", covariance=TWO)) < 1e-25


# ---- the notes' ridge errata -------------------------------------------------------------------------------------
def _notes_eps_gen_printed(alpha, Delta):
    """Eqs. (12.163)-(12.164) = (18.17)-(18.18) as printed (rho = 1)."""
    return (1 + alpha ** 2 + Delta) / (1 - alpha) if alpha < 1 else Delta * alpha / (1 - alpha)


def _notes_eps_gen_corrected(alpha, Delta):
    """Redoing the notes' own steps: alpha < 1: m = alpha, q = alpha(1 - alpha + Delta)/(1 - alpha), E = 1 + q - 2m =
    ((1 - alpha)^2 + alpha Delta)/(1 - alpha) (the -2 alpha is lost in the printed simplification); alpha > 1:
    E = Delta/(alpha - 1) (printed just above (12.163)), so E + Delta = Delta alpha/(alpha - 1) (sign)."""
    E = ((1 - alpha) ** 2 + alpha * Delta) / (1 - alpha) if alpha < 1 else Delta / (alpha - 1)
    return E + Delta


def _notes_q_trace(alpha, lam, Delta, rho, factor):
    """The closing q, m traces of Sec. 18.1.2 at Omega = Phi = I (gamma = 1): V from V = 1/(lam + V_hat),
    V_hat = alpha/(1 + V) (18.30)-(18.31); m = V_hat rho/(lam + V_hat);
    q = V_hat^2 (rho + c (E + Delta))/(lam + V_hat)^2
    with E = rho + q - 2m, solved for q (linear).  factor: c as printed (alpha) or as derived ((E + Delta)/alpha,
    which is (18.15)'s 1 + alpha^-1 (1 + q - 2m + Delta) at rho = 1).  Returns E."""
    with mp.workdps(30):
        a, lm, D, r = (mp.mpf(x) for x in (alpha, lam, Delta, rho))
        V = mp.findroot(lambda V: V * (lm + a / (1 + V)) - 1, (mp.mpf(0), 1 / lm), solver="anderson")
        Vh = a / (1 + V)
        k = Vh ** 2 / (lm + Vh) ** 2
        m = Vh * r / (lm + Vh)
        c = a if factor == "printed" else 1 / a
        q = k * (r + c * (r - 2 * m + D)) / (1 - k * c)
        return r + q - 2 * m


def test_notes_ridge_errata_recorded():
    """(12.163)-(12.164) = (18.17)-(18.18) as printed disagree with the banked HMRT ridgeless values, and the corrected
    forms agree: ridgeless_g4_s0p25_r1 (gamma = 4: alpha = 1/4 < 1, Delta = 1/4) and ridgeless_g0p5_s1_r1 (gamma =
    1/2: alpha = 2 > 1, Delta = 1); banked values are excess risks, the notes' eps_gen = E + Delta.  The q trace
    closing Sec. 18.1.2 as printed (alpha (E + Delta)) misses ridge_g2_lam0p1_s0p5_r2 (gamma = 2: alpha = 1/2,
    lam_notes = lam/gamma = 0.05, Delta = 0.5, rho = 2) and the derived (E + Delta)/alpha reproduces it.  The map
    alpha_notes = 1/gamma and lam_notes = lam/gamma is checked against the bank in both directions: the corrected
    forms evaluated at alpha = gamma (K1's Ch. 3 map) miss the banked values, and the notes' saddle point with the
    penalty unmapped or alpha = gamma misses ridge_g2_lam0p1_s0p5_r2."""
    for key, g, s2, alpha in (("ridgeless_g4_s0p25_r1", "4", "0.25", mp.mpf(1) / 4),
                              ("ridgeless_g0p5_s1_r1", "0.5", "1", mp.mpf(2))):
        D = mp.mpf(s2)
        assert not _ok(key, _notes_eps_gen_printed(alpha, D) - D), key
        assert _ok(key, _notes_eps_gen_corrected(alpha, D) - D), key
        # the convention alpha_notes = 1/gamma, checked against the bank: the K1 (Ch. 3) map alpha = gamma misses
        assert not _ok(key, _notes_eps_gen_corrected(mp.mpf(g), D) - D), key
    key = "ridge_g2_lam0p1_s0p5_r2"
    assert _ok(key, _notes_q_trace("0.5", "0.05", "0.5", "2", "derived"))
    assert not _ok(key, _notes_q_trace("0.5", "0.05", "0.5", "2", "printed"))
    # both convention maps, through the notes' own saddle point: alpha = 1/gamma with lam_notes = lam/gamma
    # reproduces the bank; alpha = gamma (with lam_notes = lam gamma, its consistent penalty map) or lam_notes = lam
    # (the penalty left unmapped) misses it
    assert not _ok(key, _notes_q_trace("2", "0.2", "0.5", "2", "derived"))
    assert not _ok(key, _notes_q_trace("0.5", "0.1", "0.5", "2", "derived"))
