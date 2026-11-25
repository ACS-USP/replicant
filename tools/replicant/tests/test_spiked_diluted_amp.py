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
    with mp.workdps(50):
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
    """Finite F, dF with no numpy floating-point event (errstate all=raise) for rho in {1e-3, 0.5, 1}, gamma up to
    1e4, |y| up to 50a; |F| <= a, dF >= 0; at rho = 1 the prior is Rademacher and F = tanh(y)."""
    for rho in (1e-3, 0.5, 1.0):
        P = S.diluted_prior(rho)
        a = 1 / math.sqrt(rho)
        ys = np.linspace(-50 * a, 50 * a, 201)
        for g in (1e-3, 1.0, 10.0, 2000.0, 1e4):
            with np.errstate(all="raise"):
                F, dF = P.F(ys, g), P.dF(ys, g)
            assert np.all(np.isfinite(F)) and np.all(np.isfinite(dF)), (rho, g)
            assert np.all(np.abs(F) <= a * (1 + 1e-12)) and np.all(dF >= 0), (rho, g)
            if rho == 1.0:
                assert np.abs(F - np.tanh(ys)).max() < 1e-12, g


def test_diluted_sampler_moments():
    """Tolerances are analytic, 5 sd of the sample moment at n = 200000, rho = 0.08 (not tuned on the seed):
    sd(mean) = 1/sqrt(n); sd(E x^2) = sqrt(Var(x^2)/n) = sqrt((1 - rho)/rho/n) (about 0.038);
    sd(zero fraction) = sqrt(rho (1 - rho)/n)."""
    rho, n = 0.08, 200000
    x = S.diluted_prior(rho).sample(np.random.default_rng(0), n)
    assert abs(x.mean()) < 5 / math.sqrt(n)
    assert abs((x**2).mean() - 1) < 5 * math.sqrt((1 - rho) / rho / n)
    assert abs(np.mean(x == 0) - (1 - rho)) < 5 * math.sqrt(rho * (1 - rho) / n)


_BANNED = {"channels", "spiked_diluted", "bayes"}


def _audit(src, root="diluted_prior"):
    """Offending names in `root` and every module-level function reachable from it (closures included, whatever
    their names): a name or attribute that is banned, or a module-level import alias of a banned module, or an
    import of one inside a body."""
    tree = ast.parse(src)
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    bad_alias = set()
    for n in tree.body:
        if isinstance(n, ast.Import):
            for al in n.names:
                if _BANNED & set(al.name.split(".")):
                    bad_alias.add(al.asname or al.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            mod = set((n.module or "").split("."))
            for al in n.names:
                if mod & _BANNED or al.name in _BANNED:
                    bad_alias.add(al.asname or al.name)
    seen, todo, bad = set(), [root], []
    while todo:
        name = todo.pop()
        if name in seen or name not in funcs:
            continue
        seen.add(name)
        for n in ast.walk(funcs[name]):
            if isinstance(n, ast.Name):
                if n.id in _BANNED or n.id in bad_alias:
                    bad.append(n.id)
                elif n.id in funcs:
                    todo.append(n.id)
            elif isinstance(n, ast.Attribute) and n.attr in _BANNED:
                bad.append(n.attr)
            elif isinstance(n, ast.Import):
                bad += [al.name for al in n.names if _BANNED & set(al.name.split("."))]
            elif isinstance(n, ast.ImportFrom):
                if (_BANNED & set((n.module or "").split("."))) or any(al.name in _BANNED for al in n.names):
                    bad.append(n.module or "")
    return bad


def test_diluted_functions_import_nothing_from_se():
    """AST audit of diluted_prior and everything it reaches, with failing controls for the audit itself."""
    src = inspect.getsource(S)
    assert "def diluted_prior" in src and _audit(src) == []
    local = "def diluted_prior(r):\n    def _diluted_F(y):\n        from ..bayes import channels\n        return channels.x(y)\n    return _diluted_F\n"
    assert _audit(local), "local from-import not flagged"
    alias = ("import replicant.inference.spiked_diluted as SD\n"
             "def diluted_prior(r):\n    return SD.x(r)\n")
    assert _audit(alias), "module-level alias not flagged"
    helper = ("from ..bayes import channels as C\n"
              "def _helper(r):\n    return C.f(r)\n"
              "def diluted_prior(r):\n    return _helper(r)\n")
    assert _audit(helper), "helper reached from diluted_prior not flagged"


def test_prior_refusals():
    _refuses(lambda: S._prior("diluted_rademacher"), "precondition:rho")
    for rho in (0, 1.5, "x", -0.1, float("nan")):
        _refuses(lambda rho=rho: S._prior("diluted_rademacher", rho), "precondition:rho")
    assert S._prior("diluted_rademacher", 0.08) is S.diluted_prior(0.08)
    assert S._prior("rademacher") is S.PRIORS["rademacher"]
