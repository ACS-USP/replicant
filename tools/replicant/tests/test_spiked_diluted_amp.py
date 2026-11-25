"""Finite-N Bayes AMP for the diluted-Rademacher spiked Wigner prior: Task 1, the prior itself.

The prior is the unit-variance one of inference/spiked_diluted.py: x in {0, +-a}, a = 1/sqrt(rho), P(0) = 1 - rho.
Its denoiser here is written in closed form and must not import bayes/ or inference/spiked_diluted.py; the tests
import bayes.channels only as a second witness.
"""
import ast
import inspect
import math
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.core import Refusal
from replicant.inference import spiked as S


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


def _posterior_mean(y, g, rho):
    """E[x | y] for y = g x + sqrt(g) G, x in {-a, 0, a}, by mpmath (50 digits)."""
    mp.mp.dps = 50
    a = 1 / mp.sqrt(mp.mpf(rho))
    g, y, r = mp.mpf(g), mp.mpf(y), mp.mpf(rho)
    def lik(x):
        return mp.exp(-(y - g * x)**2 / (2 * g))
    wp, w0, wm = r / 2 * lik(a), (1 - r) * lik(0), r / 2 * lik(-a)
    return float(a * (wp - wm) / (wp + w0 + wm))


def test_diluted_denoiser_matches_channels():
    """F against the mpmath three-atom posterior mean (1e-12), dF against a central difference (h = 1e-6)."""
    for rho in (0.05, 0.08, 0.5, 1.0):
        P = S.diluted_prior(rho)
        a = 1 / math.sqrt(rho)
        for g in (0.1, 1.0, 5.0):
            ys = np.linspace(-10 * a, 10 * a, 41)
            F = P.F(ys, g)
            for y, f in zip(ys, F):
                assert abs(f - _posterior_mean(y, g, rho)) < 1e-12, (rho, g, y)
            h = 1e-6
            num = (P.F(ys + h, g) - P.F(ys - h, g)) / (2 * h)
            dF = P.dF(ys, g)
            err = np.abs(dF - num) / np.maximum(np.abs(num), 1e-3)
            assert err.max() < 1e-6, (rho, g, err.max())


def test_diluted_denoiser_stable_at_extremes():
    P = S.diluted_prior(1e-3)
    a = 1 / math.sqrt(1e-3)
    ys = np.linspace(-50 * a, 50 * a, 201)
    F, dF = P.F(ys, 10.0), P.dF(ys, 10.0)
    assert np.all(np.isfinite(F)) and np.all(np.isfinite(dF))
    assert np.all(np.abs(F) <= a * (1 + 1e-12))
    assert np.all(dF >= 0)


def test_diluted_sampler_moments():
    rho = 0.08
    x = S.diluted_prior(rho).sample(np.random.default_rng(0), 200000)
    assert abs(x.mean()) < 0.01 and abs((x**2).mean() - 1) < 0.01
    assert abs(np.mean(x == 0) - (1 - rho)) < 0.005


def test_diluted_functions_import_nothing_from_se():
    """AST audit of diluted_prior and the helpers it defines: no name resolves to channels, spiked_diluted or bayes."""
    tree = ast.parse(inspect.getsource(S))
    banned = {"channels", "spiked_diluted", "bayes"}
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)
           and (n.name == "diluted_prior" or n.name.startswith("_diluted"))]
    assert any(f.name == "diluted_prior" for f in fns)
    for f in fns:
        for n in ast.walk(f):
            if isinstance(n, ast.Name):
                assert n.id not in banned, (f.name, n.id)
            elif isinstance(n, ast.Attribute):
                assert n.attr not in banned, (f.name, n.attr)
            elif isinstance(n, ast.Import):
                assert not any(banned & set(al.name.split(".")) for al in n.names), f.name
            elif isinstance(n, ast.ImportFrom):
                assert not (banned & set((n.module or "").split("."))), f.name
                assert not any(al.name in banned for al in n.names), f.name


def test_prior_refusals():
    _refuses(lambda: S._prior("diluted_rademacher"), "precondition:rho")
    for rho in (0, 1.5, "x", -0.1, float("nan")):
        _refuses(lambda rho=rho: S._prior("diluted_rademacher", rho), "precondition:rho")
    assert S._prior("diluted_rademacher", 0.08) is S.diluted_prior(0.08)
    assert S._prior("rademacher") is S.PRIORS["rademacher"]
