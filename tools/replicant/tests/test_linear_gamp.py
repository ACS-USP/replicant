"""K6: generalized linear estimation by GAMP with a Channel x Prior interface (linear/gamp.py) and its Bayes state
evolution (linear/se.py); KZ notes Sec. 12.6, 12.8 and Ch. 16 (Alg. 3).

Convention (the notes' Ch. 16): A is M x N with i.i.d. N(0, 1/N) entries, alpha = M/N, z = A x0, y ~ P_out(y | z);
priors are bayes/channels.py priors (K2), channels `awgn` (y = z + sqrt(Delta) xi) and `sign` (y = sign z).
amp/gamp.py (E-series) runs the sign channel on Phi with N(0, 1) entries and z = Phi w/sqrt(N): the same A = Phi/sqrt N.
"""
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.amp import gamp as OLD
from replicant.amp import gamp_se as OLD_SE
from replicant.bayes.channels import Prior
from replicant.core import Refusal
from replicant.linear import gamp as G
from replicant.linear import se as SE
from replicant.testing import imports, slow

PKG = Path(__file__).resolve().parent.parent


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


_imports = imports      # testing.imports: modules and the imported names


def _uses_amp(mods):
    """Imports reaching the amp package (`..amp`, `..amp.gamp`, `from .. import amp`, `replicant.amp...`)."""
    return {m for m in mods if any(p in ("..amp", "replicant.amp") or p.startswith(("..amp.", "replicant.amp."))
                                   or p in (":amp",) for p in m.replace(":", " :").split())}


def _planted(mod, key, value, fn):
    old = mod._HOOKS[key]
    mod._HOOKS[key] = value
    try:
        return fn()
    finally:
        mod._HOOKS[key] = old
        assert mod._HOOKS[key] is old


def test_linear_gamp_reproduces_amp_gamp():
    """On the sign channel with the Gaussian prior, linear.gamp (variance="nishimori": Sigma^-1 = (1/N) sum g^2, the
    form amp/gamp.py uses) reproduces amp/gamp.gamp_perceptron on the same instances (N = 300, alpha = 1.5 and 3,
    seeds 0-1, 12 steps): overlap traces to 1e-10, x_hat to 1e-8.  The Rangan form (variance="derivative",
    Sigma^-1 = -(1/N) sum dg/domega) agrees with them to 0.02 in the final overlap (the two coincide in expectation
    on the Nishimori line).  linear/ imports nothing from amp/ (AST audit; planted imports are caught), and
    linear.se.gamp_se reproduces amp/gamp_se's overlap to 1e-8 at alpha = 0.5, 1, 2, 4 with a different quadrature
    of E[g_out^2].  Controls: the Onsager term scaled by 0.8 (_HOOKS["onsager"]) moves the overlaps by > 1e-3; the
    SE's q_hat scaled by 0.95 (_HOOKS["qhat"]) misses amp/gamp_se by > 1e-3."""
    for f in sorted((PKG / "linear").glob("*.py")):
        assert not _uses_amp(_imports(f.read_text())), f
    assert _uses_amp(_imports("from ..amp import gamp\n")) and _uses_amp(_imports("from ..amp.gamp_se import x\n"))
    assert _uses_amp(_imports("from .. import amp\n")) and not _uses_amp(_imports("from ..bayes import channels\n"))
    sign, gauss = G.Channel("sign"), Prior("gaussian")
    for alpha in (1.5, 3):
        for seed in (0, 1):
            X, y, w = OLD.perceptron_instance(300, alpha, "gaussian", seed)
            old = OLD.gamp_perceptron(X, y, "gaussian", iters=12, seed=seed, teacher=w)
            A = X / math.sqrt(300)
            new = G.gamp(A, y, sign, gauss, iters=12, seed=seed, teacher=w, variance="nishimori")
            assert np.max(np.abs(np.array(old.overlap_trace) - np.array(new.overlap_trace))) < 1e-10
            assert np.max(np.abs(old.x_hat - new.x_hat)) < 1e-8
            rangan = G.gamp(A, y, sign, gauss, iters=12, seed=seed, teacher=w)
            assert abs(rangan.overlap_trace[-1] - old.overlap_trace[-1]) < 0.02, (alpha, seed)
            bad = _planted(G, "onsager", 0.8,
                           lambda: G.gamp(A, y, sign, gauss, iters=12, seed=seed, teacher=w, variance="nishimori"))
            assert abs(bad.overlap_trace[-1] - old.overlap_trace[-1]) > 1e-3
    for alpha in (0.5, 1, 2, 4):
        want = OLD_SE.perceptron_state_evolution(alpha).overlap
        got = SE.gamp_se(alpha, sign, gauss)
        assert got.status == "unique" and abs(got.overlap - want) < 1e-8, (alpha, got.overlap, want)
        assert abs(_planted(SE, "qhat", 0.95, lambda: SE.gamp_se(alpha, sign, gauss)).overlap - want) > 1e-3


def test_gamp_awgn_fixed_point_is_the_gaussian_posterior_mean():
    """AWGN channel, Gaussian prior: the posterior mean is (A^T A/Delta + I)^-1 A^T y/Delta exactly, and GAMP's fixed
    point is it for any scalar Sigma (g = (y - A x)/Delta at the fixed point: derived in linear/gamp.py).  N = 200,
    alpha = 0.7 and 2, Delta = 0.3, seed 4: agreement to 1e-8 for both variance forms; the SE fixed point is the
    closed form q = q_hat/(1 + q_hat), q_hat = alpha/(Delta + 1 - q) (a quadratic) to 1e-10.  Control: the Onsager
    term scaled by 0.8 moves the fixed point by > 1e-4."""
    ch, pr = G.Channel("awgn", Delta="0.3"), Prior("gaussian")
    for alpha in (0.7, 2):
        A, y, x0 = G.linear_instance(200, alpha, ch, pr, 4)
        exact = np.linalg.solve(A.T @ A / 0.3 + np.eye(200), A.T @ y / 0.3)
        for form in ("derivative", "nishimori"):
            r = G.gamp(A, y, ch, pr, iters=300, seed=4, teacher=x0, variance=form)
            assert r.converged and np.max(np.abs(r.x_hat - exact)) < 1e-8, (alpha, form)
        bad = _planted(G, "onsager", 0.8, lambda: G.gamp(A, y, ch, pr, iters=300, seed=4))
        assert np.max(np.abs(bad.x_hat - exact)) > 1e-4
        # q (1 - q) ... closed form: q = a/(a + D + 1 - q) <=> q^2 - (a + D + 1) q + a = 0, the root in [0, 1)
        b = alpha + 0.3 + 1
        q = (b - math.sqrt(b * b - 4 * alpha)) / 2
        assert abs(SE.gamp_se(alpha, ch, pr).overlap - q) < 1e-10


def test_gamp_divergence_is_never_converged():
    """A run whose iterates go non-finite stops and reports finite=False, converged=False, frozen=False, with its last
    finite iterate and a NON-FINITE note (planted divergence: noiseless awgn with y scaled by 1e200, N = 200).
    Control: the same instance with the true y collapses to exact recovery, reported frozen (a float64 fixed point),
    finite and converged."""
    ch, pr = G.Channel("awgn", Delta=0), Prior("gauss_bernoulli", rho="0.4")
    A, y, x0 = G.linear_instance(200, 0.7, ch, pr, 0)
    bad = G.gamp(A, y * 1e200, ch, pr, iters=20, seed=0)
    assert not bad.finite and not bad.converged and not bad.frozen and "NON-FINITE" in bad.note
    assert np.all(np.isfinite(bad.x_hat))
    good = G.gamp(A, y, ch, pr, iters=150, seed=0, teacher=x0)
    assert good.finite and good.frozen and good.converged and good.mse_trace[-1] < 1e-6


def test_gamp_refusals_are_named():
    """Unknown channel, unknown or bad channel parameter, a non-+-1 label on the sign channel, shape mismatches,
    an unknown variance form, non-positive iters: named refusals."""
    _refuses(lambda: G.Channel("probit"), "precondition:known-channel")
    _refuses(lambda: G.Channel("awgn", sigma=1), "precondition:known-parameter")
    _refuses(lambda: G.Channel("awgn", Delta=-1), "precondition:Delta")
    _refuses(lambda: G.Channel("sign", Delta=1), "precondition:known-parameter")
    A = np.ones((3, 4)) / 2
    _refuses(lambda: G.gamp(A, np.array([1.0, 0.5, -1]), G.Channel("sign"), Prior("gaussian"), iters=3, seed=0),
             "precondition:sign-labels")
    _refuses(lambda: G.gamp(A, np.ones(2), G.Channel("awgn"), Prior("gaussian"), iters=3, seed=0),
             "precondition:label-count")
    _refuses(lambda: G.gamp(A, np.ones(3), G.Channel("awgn"), Prior("gaussian"), iters=3, seed=0, variance="x"),
             "precondition:variance-form")
    _refuses(lambda: G.gamp(A, np.ones(3), G.Channel("awgn"), Prior("gaussian"), iters=0, seed=0),
             "precondition:positive-iters")
    _refuses(lambda: SE.gamp_se(-1, G.Channel("awgn"), Prior("gaussian")), "precondition:alpha")


@slow
def test_gamp_gauss_bernoulli_matches_state_evolution():
    """Gauss-Bernoulli prior (rho = 0.2), AWGN Delta = 1e-4, alpha = 0.5 (above the SE spinodal: one fixed point):
    the final MSE of GAMP at N = 2000 over named seeds 0-7 (30 steps) lies in the 99% Student-t band around the SE
    MSE at the same t (SE run for exactly 30 steps from the uninformative start; 9.05e-5).  Power (planted error: the
    wrong alpha): the SE at alpha = 0.4 and t = 30 (6.1e-4, still converging) lies outside the band, whose half-width
    was 2.0e-5 on the unused seeds 100-107 (mean 9.7e-5)."""
    from replicant.amp._stats import student_band
    ch, pr = G.Channel("awgn", Delta="1e-4"), Prior("gauss_bernoulli", rho="1/5")
    mses = []
    for s in range(8):
        A, y, x0 = G.linear_instance(2000, 0.5, ch, pr, s)
        r = G.gamp(A, y, ch, pr, iters=30, seed=s, teacher=x0)
        mses.append(float((r.x_hat - x0) @ (r.x_hat - x0)) / 2000)
    band = student_band(mses)
    want = SE.gamp_se(0.5, ch, pr, iters=30).trajectory("uninformative")[-1]
    assert abs(band.mean - want) <= band.half_width, (band, want)
    other = SE.gamp_se(0.4, ch, pr, iters=30).trajectory("uninformative")[-1]
    assert abs(band.mean - other) > band.half_width, (band, other)
