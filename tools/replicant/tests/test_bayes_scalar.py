"""K2: the Gaussian scalar channel, its priors and denoisers (KZ Ch. 7, Sec. 7.2) and sparse-vector denoising
(KZ Sec. 7.3, App. 7.B): bayes/channels.py, bayes/sparse_vector.py and the entries `scalar_denoising`,
`sparse_denoising`.

Convention (bayes/channels.py): y = sqrt(snr) x + z, z ~ N(0, 1); KZ's y_i = x* + sqrt(Delta) z_i (Eq. (7.4)) is
snr = N/Delta for N observations.  Banked: reference/bayes.json (the three printed scalar posteriors, read with the
inferred Delta = 1; Delta_c and f(1/2) of Theorem 15 evaluated).  Independent routes: mpmath quadrature of
P(y|x) P(x), the closed forms of inference/spiked.py, the REM (families/rem.py) for the sparse vector, Monte Carlo
over named seeds.  Planted errors go through the modules' _HOOKS, restored with identity asserts.
"""
import math
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import mpmath as mp
import numpy as np

from replicant.amp._stats import student_band
from replicant.bayes import channels as CH
from replicant.bayes import sparse_vector as SV
from replicant.catalog import catalog, check_ref
from replicant.core import Refusal
from replicant.families import rem as REM
from replicant.inference import spiked as SP
from replicant.refdata import check, load
from replicant.testing import slow

BANK = load("bayes")


def _planted(module, key, value, fn):
    reg = module._HOOKS
    old = reg[key]
    reg[key] = value
    try:
        return fn()
    finally:
        reg[key] = old
        assert module._HOOKS is reg and reg[key] is old


def _refusal(fn):
    try:
        fn()
    except Refusal as r:
        return r
    raise AssertionError("expected a Refusal")


def _posterior(prior, data, delta):
    return CH.posterior_ball(CH.Prior(prior), CH.DATASETS[data], delta)


# ---- the printed posteriors (KZ Eqs. (7.6)-(7.11)) -----------------------------------------------------------
def test_scalar_posteriors_vs_printed():
    """KZ prints 0.630, 0.091 and 0.8233787142909471 without Delta: Delta = 1 (inferred, stated in each cite)
    reproduces all three; Delta = 1/2 and 2 reproduce none (positive control)."""
    for key in ("gauss_posterior_mean_eq7_8", "gauss_posterior_var_eq7_8", "gb_p_nonzero_eq7_11"):
        assert "inferred" in BANK[key].cite.lower() and "Delta = 1" in BANK[key].cite, key
    got = lambda d: {"gauss_posterior_mean_eq7_8": _posterior("gaussian", "7.8", d)["mean"].mid(),
                     "gauss_posterior_var_eq7_8": _posterior("gaussian", "7.8", d)["var"].mid(),
                     "gb_p_nonzero_eq7_11": _posterior("gauss_bernoulli", "7.11", d)["p_nonzero"].mid()}
    for key, v in got(1).items():
        assert check(BANK[key], v, label=key).ok, (key, v)
    for d in ("1/2", 2):
        assert not any(check(BANK[k], v, label=k).ok for k, v in got(d).items()), d
    # Eq. (7.7): "larger than 0.999" (a bound, not banked); at Delta = 1 it is 0.99999942
    p = _posterior("rademacher", "7.6", 1)["p_plus"]
    assert p.mid() > mp.mpf("0.999") and abs(p.mid() - mp.mpf("0.9999994223654517")) < 1e-15
    # the catalog entry reproduces the bank through its own run
    e = catalog()["scalar_denoising"]
    for ref in e.refs:
        assert check_ref(e, ref).ok, ref.key


def test_posterior_closed_forms_vs_quadrature():
    """The arb posterior (sufficient statistic, mixture closed form) against an mpmath quadrature of
    P(y|x) P(x) over the N observations (KZ Eq. (7.5)), and against the float denoiser at snr = N/Delta."""
    for prior, data in (("gaussian", "7.8"), ("gauss_bernoulli", "7.11"), ("rademacher", "7.6")):
        P = CH.Prior(prior)
        post = CH.posterior_ball(P, CH.DATASETS[data], 1)
        with mp.workdps(30):
            ys = [mp.mpf(y) for y in CH.DATASETS[data]]
            lik = lambda x: mp.exp(-mp.fsum((y - x)**2 for y in ys) / 2)
            q = lambda f: mp.mpf(f.numerator) / f.denominator
            cont = [(q(w), q(v)) for w, mu, v in P.components if v > 0]      # every Gaussian has mean 0 here
            atoms = [(q(w), q(mu)) for w, mu, v in P.components if v == 0]
            dens = lambda x, k: sum(w * mp.npdf(x, 0, mp.sqrt(v)) for w, v in cont) * x**k * lik(x)
            z = [mp.quad(lambda x: dens(x, k), [-mp.inf, 0, 2, mp.inf]) if cont else mp.mpf(0) for k in (0, 1, 2)]
            for k in (0, 1, 2):
                z[k] += mp.fsum(w * mu**k * lik(mu) for w, mu in atoms)
            mean, var = z[1] / z[0], z[2] / z[0] - (z[1] / z[0])**2
            assert abs(post["mean"].mid() - mean) < 1e-20 and abs(post["var"].mid() - var) < 1e-20, prior
        s = len(ys)
        mf, vf = CH.denoiser(P, s)
        y = np.array([math.sqrt(s) * float(sum(Fraction(v) for v in CH.DATASETS[data]) / s)])
        assert abs(mf(y)[0] - float(mean)) < 1e-12 and abs(vf(y)[0] - float(var)) < 1e-12, prior


# ---- denoisers, MMSE, information ------------------------------------------------------------------------------
def test_denoisers_vectorise_and_match_the_closed_forms():
    y = np.linspace(-4, 4, 41)
    for s in (0.3, 1.0, 7.5):
        m, v = CH.denoiser(CH.Prior("gaussian", var=2), s)
        assert m(y).shape == y.shape and np.allclose(m(y), math.sqrt(s) * y * 2 / (1 + 2 * s), atol=1e-14)
        assert np.allclose(v(y), 2 / (1 + 2 * s), atol=1e-14)
        m, v = CH.denoiser(CH.Prior("rademacher"), s)
        assert np.allclose(m(y), np.tanh(math.sqrt(s) * y), atol=1e-14)
        assert np.allclose(v(y), 1 - np.tanh(math.sqrt(s) * y)**2, atol=1e-14)
        # inference/spiked.py's channel y' = gamma X + sqrt(gamma) G is sqrt(gamma) times this one at snr = gamma
        for name in ("gaussian", "rademacher"):
            m, _ = CH.denoiser(CH.Prior(name), s)
            assert np.allclose(m(y), SP.PRIORS[name].F(math.sqrt(s) * y, s), atol=1e-14), name
    # a sparse prior with atoms at 0, +-2: weights are a softmax over the atoms
    m, v = CH.denoiser(CH.Prior("sparse", rho="1/4", a=2), 2.0)
    w = np.array([0.75, 0.125, 0.125])[None, :] * np.exp(-(y[:, None] - math.sqrt(2) * np.array([0, 2, -2]))**2 / 2)
    w /= w.sum(1, keepdims=True)
    assert np.allclose(m(y), w @ np.array([0, 2, -2]), atol=1e-13)
    assert np.allclose(v(y), w @ np.array([0, 4, 4]) - m(y)**2, atol=1e-13)


def test_mmse_and_information_balls_against_quadrature_and_spiked():
    """Certified MMSE and I (GSV Eqs. (11), (13), (17), (18)) contain the float quadrature of bayes/channels.py and
    agree with inference/spiked.py's independent code."""
    for s in ("1/2", 1, "37/10"):
        sf = float(Fraction(str(s)))
        for name in ("gaussian", "rademacher"):
            P = CH.Prior(name)
            mb, ib = CH.mmse_ball(P, s), CH.information_ball(P, s)
            assert mb.rad() < 1e-30 and ib.rad() < 1e-30, (name, s)
            assert abs(mb.mid() - CH.mmse(P, sf)) < 1e-12 and abs(ib.mid() - CH.mutual_information(P, sf)) < 1e-12
            assert abs(mb.mid() - SP.scalar_mmse(name, sf)) < 1e-11, (name, s)
            assert abs(ib.mid() - SP._mutual_info(name, sf)) < 1e-11, (name, s)
    r = _refusal(lambda: CH.mmse_ball(CH.Prior("gauss_bernoulli"), 1))
    assert r.reason == "unsupported:certified-prior", r
    # Gauss-Bernoulli: q0 - mmse = E[eta^2]; at snr -> 0 mmse -> rho var, at large snr -> 0
    gb = CH.Prior("gauss_bernoulli")
    assert abs(CH.mmse(gb, 0) - 0.5) < 1e-14 and CH.mmse(gb, 400) < 3e-3
    assert CH.mutual_information(gb, 0) == 0 or abs(CH.mutual_information(gb, 0)) < 1e-14


def test_map_estimate():
    y = np.array([-2.0, -0.1, 0.3, 1.7])
    assert np.array_equal(CH.map_estimate(CH.Prior("rademacher"), y, 1.0), np.sign(y))
    g = CH.Prior("gaussian")
    assert np.allclose(CH.map_estimate(g, y, 2.0), CH.denoiser(g, 2.0)[0](y))
    sp_ = CH.Prior("sparse", rho="1/10")         # atoms 0, +-1: the zero atom wins near y = 0
    assert np.array_equal(CH.map_estimate(sp_, np.array([0.2, 3.0, -3.0]), 4.0), np.array([0.0, 1.0, -1.0]))
    r = _refusal(lambda: CH.map_estimate(CH.Prior("gauss_bernoulli"), y, 1.0))
    assert r.reason == "unsupported:map-mixed-measure", r


def test_prior_registry_refuses_unknowns():
    assert _refusal(lambda: CH.Prior("laplace")).reason == "precondition:known-prior"
    assert _refusal(lambda: CH.Prior("gaussian", rho="1/2")).reason == "precondition:known-parameter"
    assert _refusal(lambda: CH.Prior("gauss_bernoulli", rho="3/2")).reason == "precondition:rho"
    assert _refusal(lambda: CH.denoiser(CH.Prior("gaussian"), -1)).reason == "precondition:snr"
    p = CH.Prior("gauss_bernoulli", rho="1/3", var=2)
    assert p.second_moment == mp.mpf(2) / 3 and sum(w for w, _, _ in p.components) == 1


# ---- sparse-vector denoising (KZ Sec. 7.3) ----------------------------------------------------------------------
def test_sparse_denoising_matches_rem():
    """f(Delta) = 1/(2 Delta) - log 2 below Delta_c = 1/(2 log 2), 0 above (Theorem 15), against families/rem.py at
    the mapped parameters: the d - 1 wrong components are a REM with beta = sqrt(2/Delta), N_scale = 1/2, whose
    transition beta_c = 2 sqrt(ln 2) maps to Delta_c = 2/beta_c^2."""
    with mp.workdps(30):
        dc = SV.delta_c()
        assert abs(dc - 1 / (2 * mp.log(2))) < mp.mpf(10)**-28
        assert abs(dc - 2 / REM.rem_critical_beta(N_scale="0.5")**2) < mp.mpf(10)**-28
        for d in ("0.3", "0.5", "0.7", "0.75", "1", "2.5"):
            f, fr = SV.sparse_vector_free_entropy(d), SV.free_entropy_via_rem(d)
            want = max(1 / (2 * mp.mpf(d)) - mp.log(2), 0)
            assert abs(f - want) < mp.mpf(10)**-25 and abs(fr - want) < mp.mpf(10)**-25, d
            assert SV.sparse_vector_mmse(d) == (0 if mp.mpf(d) < dc else 1), d
        assert abs(SV.sigma_c2(2**20) - 1 / (2 * 20 * mp.log(2))) < mp.mpf(10)**-28
    # sigma_c^2 against the REM: noise variance Delta/N with d = 2^N, so sigma_c^2 = Delta_c_rem / N, Delta_c_rem =
    # 2/beta_c^2 from families/rem.py; a planted beta = sqrt(1/Delta) halves Delta_c_rem and fails
    agree = lambda: all(abs(SV.sigma_c2(2**n) - SV.delta_c_via_rem() / n) < mp.mpf(10)**-25 for n in (4, 10, 20))
    with mp.workdps(30):
        assert agree() and abs(SV.delta_c_via_rem() - 2 / REM.rem_critical_beta(N_scale="0.5")**2) < mp.mpf(10)**-28
        assert not _planted(SV, "beta_factor", 1, agree)
    # plant: the REM map at beta = sqrt(1/Delta) (a dropped factor 2) moves the transition: the REM route fails
    bad = _planted(SV, "beta_factor", 1, lambda: [SV.free_entropy_via_rem(d) for d in ("0.6", "1")])
    assert abs(bad[1]) > 0.05 or abs(bad[0] - (1 / mp.mpf("1.2") - mp.log(2))) > 0.05, bad
    e = catalog()["sparse_denoising"]
    v = e.run(Delta="1/2")
    assert abs(mp.mpf(v["f"]) - mp.mpf(v["f_rem"])) < 1e-25 and v["mmse"] == 0
    for ref in e.refs:
        assert check_ref(e, ref).ok, ref.key


def test_sparse_finite_d_quadrature_vs_sampling():
    """The exact finite-d (1/N) E log Z (quadrature over the planted and the d - 1 null terms of Eq. (7.54)) lies in
    the 99% Student-t band of direct samples over named seeds, and above the N -> infinity value (Lemma 8)."""
    N, delta = 10, "0.5"
    q = SV.sparse_vector_free_entropy(delta, d=2**N)
    band = student_band([SV.sample_log_z(delta, N, seed) for seed in range(200)])
    assert abs(band.mean - q) <= band.half_width, (q, band)
    f = float(SV.sparse_vector_free_entropy(delta))
    assert 0 < q - f < 0.1, (q, f)          # KZ Lemma 8 (labelled "Upper bound", it proves Phi_N >= f)
    # control: the planted term dropped (Z of pure noise, a REM) leaves the band: a shift of about 0.33 against a
    # half-width of about 0.08 (power ~ 1)
    bad = _planted(SV, "planted_copies", 0, lambda: SV.sparse_vector_free_entropy(delta, d=2**N))
    assert abs(band.mean - bad) > 3 * band.half_width, (bad, band)


def test_scalar_denoising_entry_values_and_finality():
    e = catalog()["scalar_denoising"]
    for prior in ("gaussian", "rademacher"):
        st, _ = e.execute(prior=prior, Delta="1/2")
        v, err = st["values"], st["errors"]
        assert mp.mpf(err["mmse"]) < mp.mpf("1e-35") and mp.mpf(err["mutual_information"]) < mp.mpf("1e-35")
        fin = e.finality_of(v)
        assert fin["mmse"].state == "exact-by-literature" and "Guo" in fin["mmse"].cite
    st, _ = e.execute(prior="gauss_bernoulli", rho="1/2", Delta="1/2")
    fin = e.finality_of(st["values"])
    assert "mmse" not in st["errors"] and not fin["mmse"].final and "quadrature" in fin["mmse"].note
    # the Result's rigor follows the prior: certified only when every value is a ball
    from replicant.catalog_bayes import QUADRATURE_RIGOR
    from replicant.compose import compose
    run = lambda **p: compose(e.spec, "exact").run(**p)
    assert run(prior="rademacher", data="7.6", Delta=1).rigor == "certified"
    assert run(prior="gaussian", Delta="1/2").rigor == "certified"
    for p in ({"prior": "gauss_bernoulli", "data": "7.11", "Delta": 1}, {"prior": "sparse", "rho": "1/4", "a": 2}):
        res = run(**p)
        assert res.rigor == QUADRATURE_RIGOR and "binary64 quadrature" in res.rigor and res.rigor != "certified", p


@slow
def test_sparse_finite_d_approaches_theorem_15():
    """(1/N) E log Z - f(Delta) > 0 shrinks with N on both sides of Delta_c (Theorem 15; App. 7.A: above Delta_c the
    total E log Z itself decays)."""
    for delta in ("0.5", "1"):
        f = float(SV.sparse_vector_free_entropy(delta))
        gaps = [SV.sparse_vector_free_entropy(delta, d=2**n) - f for n in (8, 12, 16, 20)]
        assert all(g > 0 for g in gaps) and all(a > b for a, b in zip(gaps, gaps[1:])), (delta, gaps)
