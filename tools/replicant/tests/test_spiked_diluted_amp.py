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
from replicant.testing import slow


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


# ---- Task 2: the planted start and the diluted SE trajectory --------------------------------------------------------
# State point, fixed from the package's own SE (spiked_diluted) before any finite-N test run:
#   rho = 0.08: dyn 0.980175, IT 0.985607, alg 1.009744 (diluted_spinodals, diluted_it_threshold): the hard phase proper
#   is lambda in (0.9856, 1.0097).  lambda = 0.99 (lambda^2 = 0.9801): SE fixed points gamma = 0 (stable), 0.166568
#   (unstable, the middle one), 0.425999 (stable); phase hard.  Weak start gamma0 = 0.05 (below 0.166568), strong
#   gamma0 = 1.  SE from gamma0 = 1 (diluted_se_trajectory): 1, 0.768867, 0.666765, 0.607702, 0.568907 at t = 0..4;
#   m(gamma_4) = 0.552438.
# Why this point and not rho = 0.05, lambda = 0.95 (wider window): pilots at N = 3000, t = 4 on seeds 1000-1039
# (rho = 0.05) and 1000-1019 (rho = 0.08), never 0-9, gave a per-seed sd of gamma_t about 0.095 at both points (driven
# by |x|^2 / N, sd sqrt((1 - rho) / (rho N))), but a no-Onsager offset from SE of 0.137 at rho = 0.05 against 0.340
# (sd 0.063) here: with 10 seeds the control's expected 99% half-width is about 0.065, a margin of 5.  The planted
# run's mean offset from SE was -0.008 (se 0.02): no resolved finite-N bias.  N = 3000, 10 seeds: about 1 s per run.
_RHO, _LAM, _WEAK, _STRONG = 0.08, 0.99, 0.05, 1.0
_N, _SEEDS, _T = 3000, tuple(range(10)), 4


def test_diluted_se_trajectory_reaches_both_branches():
    """At rho = 0.08, lambda = 0.99 (hard phase): from gamma0 = 0.05 the SE trajectory tends to the uninformative
    fixed point (gamma = 0), from gamma0 = 1 to the informative one (0.425999), within 1e-8 at t = 2000."""
    from replicant.inference import spiked_diluted as SD
    se = SD.diluted_state_evolution(_LAM, _RHO)
    assert se.phase == "hard" and se.uninformative.gamma == 0.0
    mid = [f.gamma for f in se.fixed_points if not f.stable]
    assert len(mid) == 1 and _WEAK < mid[0], (mid, _WEAK)
    weak = SD.diluted_se_trajectory(_LAM, _RHO, _WEAK, 2000)
    strong = SD.diluted_se_trajectory(_LAM, _RHO, _STRONG, 2000)
    assert len(weak) == len(strong) == 2001 and weak[0] == _WEAK and strong[0] == _STRONG
    assert abs(weak[-1] - se.uninformative.gamma) < 1e-8, weak[-1]
    assert abs(strong[-1] - se.informative.gamma) < 1e-8, (strong[-1], se.informative.gamma)
    # gamma_{t+1} = lambda^2 m(gamma_t), checked on one step
    assert abs(strong[1] - _LAM**2 * SD.diluted_overlap(_RHO, _STRONG)) < 1e-15
    _refuses(lambda: SD.diluted_se_trajectory(_LAM, _RHO, 0, 5), "precondition:gamma0")
    _refuses(lambda: SD.diluted_se_trajectory(_LAM, _RHO, float("nan"), 5), "precondition:gamma0")
    _refuses(lambda: SD.diluted_se_trajectory(_LAM, _RHO, 1.0, 0), "precondition:positive-iters")


def _planted(reaction):
    return S.spiked_amp_ensemble(_N, _LAM, "diluted_rademacher", seeds=_SEEDS, iters=_T, reaction=reaction, rho=_RHO,
                                 init="planted", gamma0=_STRONG)


def _inside(v, band):
    return abs(v - band.mean) <= band.half_width


def test_planted_amp_tracks_se_trajectory():
    """Planted AMP from gamma0 = 1 at rho = 0.08, lambda = 0.99, N = 3000, seeds 0-9, t = 4 (fixed t: MV Thm 2 is the
    N -> oo statement at fixed t).  The SE gamma_4 = 0.568907 (diluted_se_trajectory) lies in the 99% Student-t band of
    the empirical gamma_t, and the SE overlap m(gamma_4) = E[x F(x^t; gamma_t)] = 0.552438 in the band of q."""
    from replicant.inference import spiked_diluted as SD
    gse = SD.diluted_se_trajectory(_LAM, _RHO, _STRONG, _T)[_T]
    ens = _planted(True)
    assert ens.init == "planted" and ens.gamma0 == _STRONG and len(ens.gamma_t) == len(_SEEDS)
    assert "planted" in ens.note and "plug-in" in ens.note
    assert _inside(gse, ens.bands["gamma_t"]), (gse, ens.bands["gamma_t"])
    assert _inside(SD.diluted_overlap(_RHO, gse), ens.bands["q"]), (SD.diluted_overlap(_RHO, gse), ens.bands["q"])


def test_planted_amp_no_onsager_fails():
    """The control: the same point and seeds with reaction=False; the SE gamma_4 lies outside the gamma_t band."""
    from replicant.inference import spiked_diluted as SD
    gse = SD.diluted_se_trajectory(_LAM, _RHO, _STRONG, _T)[_T]
    b = _planted(False).bands["gamma_t"]
    assert not _inside(gse, b), (gse, b)


def test_existing_priors_unchanged():
    """q lists captured on HEAD 75d283b before the planted start was added (seeds=(0,) alone is refused
    precondition:two-samples by the band, so seeds (0, 1); 15 significant digits)."""
    e = S.spiked_amp_ensemble(400, 1.5, "rademacher", seeds=(0, 1), iters=10)
    assert [float(f"{v:.15g}") for v in e.q] == [0.590866685596371, 0.76416572163114], e.q
    e = S.spiked_amp_ensemble(400, 1.5, "gaussian", seeds=(0, 1), iters=10)
    assert [float(f"{v:.15g}") for v in e.q] == [0.607332003383008, 0.680652909498955], e.q
    assert e.init == "spectral" and e.gamma0 is None


def test_planted_refusals():
    Y, x0 = S.spiked_matrix(50, _LAM, "diluted_rademacher", 0, rho=_RHO)
    kw = dict(snr=_LAM, iters=3, seed=0, rho=_RHO)
    _refuses(lambda: S.spiked_amp(Y, "diluted_rademacher", init="planted", x0=x0, **kw), "precondition:gamma0")
    for bad in (0, -1, float("nan"), float("inf"), True, "x"):
        _refuses(lambda bad=bad: S.spiked_amp(Y, "diluted_rademacher", init="planted", x0=x0, gamma0=bad, **kw),
                 "precondition:gamma0")
    _refuses(lambda: S.spiked_amp(Y, "diluted_rademacher", init="planted", gamma0=1.0, **kw), "precondition:x0")
    _refuses(lambda: S.spiked_amp(Y, "diluted_rademacher", init="planted", gamma0=1.0, x0=x0[:10], **kw),
             "precondition:x0")
    Yg, xg = S.spiked_matrix(50, 2, "gaussian", 0)
    _refuses(lambda: S.spiked_amp(Yg, "gaussian", snr=2, iters=3, seed=0, gamma0=1.0), "precondition:known-parameter")
    _refuses(lambda: S.spiked_amp(Yg, "gaussian", snr=2, iters=3, seed=0, x0=xg), "precondition:known-parameter")
    _refuses(lambda: S.spiked_amp(Yg, "gaussian", snr=0.9, iters=3, seed=0), "precondition:snr-above-one")
    _refuses(lambda: S.spiked_amp(Yg, "gaussian", snr=2, iters=3, seed=0, init="warm"), "precondition:known-init")
    _refuses(lambda: S.spiked_amp_ensemble(50, 2, "gaussian", seeds=(0, 1), iters=3, gamma0=1.0),
             "precondition:known-parameter")
    _refuses(lambda: S.spiked_amp_ensemble(50, 0.9, "gaussian", seeds=(0, 1), iters=3), "precondition:snr-above-one")
    _refuses(lambda: S.spiked_amp_ensemble(50, 0.9, "gaussian", seeds=(0, 1), iters=3, init="planted"),
             "precondition:gamma0")
    # planted runs below lambda = 1 (no eigsh) and its first gamma is the empirical one of x^0
    r = S.spiked_amp(Y, "diluted_rademacher", init="planted", x0=x0, gamma0=1.0, **kw)
    g = np.random.default_rng([0, 4]).normal(size=50)
    assert r.gamma_trace[0] == S._gamma_hat(1.0 * x0 + 1.0 * g)
    ens = S.spiked_amp_ensemble(50, 0.9, "gaussian", seeds=(0, 1), iters=3, init="planted", gamma0=0.5)
    assert isinstance(ens.bands["gamma_t"], S.Band) and len(ens.gamma_t) == 2


def test_planted_start_imports_nothing_from_se():
    """AST audit of the planted start and of the finite-N entry points (everything they reach), with a failing control."""
    src = inspect.getsource(S)
    for root in ("_planted_start", "spiked_amp", "spiked_amp_ensemble", "spiked_matrix"):
        assert f"def {root}(" in src, root
        assert _audit(src, root) == [], (root, _audit(src, root))
    bad = ("def _planted_start(x0, g, s):\n    from .spiked_diluted import diluted_overlap\n    return x0\n"
           "def spiked_amp(Y):\n    return _planted_start(Y, 1, 0)\n")
    assert _audit(bad, "spiked_amp"), "planted start importing spiked_diluted not flagged"


# ---- Task 3: catalog wiring (spiked_wigner, prior=diluted_rademacher, N given) and the two-branch tests -------------
# State points, fixed from the package's own SE (spiked_diluted) before any test-seed run:
#   rho = 0.05: dyn 0.849363, IT 0.864471, alg 1.002700 (diluted_spinodals, diluted_it_threshold).
#   Hard phase proper: lambda = 0.95 in (0.8645, 1.0027): SE fixed points gamma = 0 (stable), 0.0788588 (unstable, the
#   middle one), 0.727772 (stable).  Weak start gamma0 = 0.03 (below 0.0788588), strong gamma0 = 1.
#   SE at t = 4 (diluted_se_trajectory): weak 0.0201559, strong 0.735357; the SE gap 0.715.
#   Negative control: lambda = 0.6 (phase impossible, one SE fixed point gamma = 0; 0.6 / dyn = 0.71, far below dyn).
#   SE at t = 8: weak 8.43655e-06, strong 4.99340e-04 (both decaying to the uninformative fixed point 0).
# Pilots (seeds 100-119, 200-219, 300-319; never the test seeds 0-19), N = 8000, 20 seeds:
#   two-branch, t = 4: separation / summed half-widths 16.7, 16.5, 22.6; each band held its own SE in every block.
#   At N = 3000 (seeds 100-119) the ratio was 10.8 (no margin), so N = 8000; at t >= 15 (N = 3000) weak runs escaped
#   (a seed's |x|^2 / N fluctuation puts the effective lambda above alg), so t = 4.
#   negative control, t = 8: the strong band 5.1e-4 +- 6.5e-5, 4.6e-4 +- 5.0e-5, 5.2e-4 +- 5.7e-5, below the weak
#   start 0.03 by > 50x; each band held its own SE in every block.
#   lambda = 0.7 was not used: at N = 3000 seed 407 lingered at gamma ~ 0.2 for 20 steps (|x|^2 / N = 1.15, effective
#   lambda 0.81, near dyn) and at N = 8000 the strong band of seeds 200-219 missed its SE at t = 8.
#   The ruled "SE values closer than the summed half-widths" form is NOT used for the negative control: in 20 pilot
#   blocks (lambda = 0.6, N = 3000, 10 seeds, t = 12 and t = 22) it held in 4; below dyn both SE values decay
#   geometrically to 0 and the bands shrink with them, so the two transients stay resolved at every fixed t (see
#   the Task 3 report).
_R3, _HARD, _BELOW, _W3 = 0.05, 0.95, 0.6, 0.03
_N3, _SEEDS3 = 8000, tuple(range(20))


def _sw():
    from replicant.catalog import catalog
    from replicant.compose import compose
    return compose(catalog()["spiked_wigner"].spec, "amp")


def _dil(**kw):
    return _sw().run(prior="diluted_rademacher", **kw)


def _band_of(r, key):
    """(mean, half-width) of a catalog band."""
    return r.values[key], r.values[f"{key}_halfwidth99"]


def test_diluted_catalog_planted_runs_below_one():
    """Review focus 3: init=planted at lambda = 0.95 < 1 (hard phase, rho = 0.05) runs on the catalog; gamma_se_t is
    the SE at t from gamma0 (exact-by-literature, fixed t), the amp bands are statistical, NOT final."""
    from replicant.catalog import catalog
    from replicant.inference import spiked_diluted as SD
    st, _ = catalog()["spiked_wigner"].execute("amp", prior="diluted_rademacher", rho=_R3, snr=_HARD, N=400,
                                                init="planted", gamma0=1, seeds=(0, 1, 2), iters=4)
    v = st["values"]
    assert v["init"] == "planted" and v["gamma0"] == 1.0 and v["N"] == 400 and v["seeds"] == [0, 1, 2]
    assert v["iters"] == 4 and 0 <= v["amp_converged_fraction"] <= 1
    assert v["gamma_se_t"] == SD.diluted_se_trajectory(_HARD, _R3, 1.0, 4)[4]
    for k in ("q_amp", "overlap_amp", "mmse_amp", "matrix_mmse_amp", "gamma_amp_t"):
        assert math.isfinite(v[k]) and v[f"{k}_halfwidth99"] > 0, k
    r = _dil(rho=_R3, snr=_HARD, N=400, init="planted", gamma0=1, seeds=(0, 1, 2), iters=4)
    f = r.finality
    assert f["gamma_se_t"].final and f["gamma_se_t"].state == "exact-by-literature", f["gamma_se_t"]
    assert "plug-in" in f["gamma_se_t"].note
    for k in ("q_amp", "overlap_amp", "mmse_amp", "matrix_mmse_amp", "gamma_amp_t"):
        assert not f[k].final and f[k].ansatz == "amp-finite-N", (k, f[k])
    assert "gamma_amp_t" in r.statistical and "gamma_se_t" not in r.statistical, r.statistical


def test_diluted_catalog_refusals():
    """N needs snr and rho; the finite-N knobs need N; gamma0 needs init=planted; the spectral start needs lambda > 1;
    init / gamma0 are refused on the plain priors.  The old refusal name appears in no .py file of the package."""
    from pathlib import Path
    from replicant.catalog import catalog
    e = catalog()["spiked_wigner"]
    ex = lambda **kw: e.execute("amp", **kw)
    d = dict(prior="diluted_rademacher")
    _refuses(lambda: ex(**d, rho=_R3, N=100), "precondition:state-point")
    _refuses(lambda: ex(**d, snr=_HARD, N=100), "precondition:rho")
    _refuses(lambda: ex(**d, rho=_R3, snr=1.5, N=100, gamma0=1), "precondition:known-parameter")
    _refuses(lambda: ex(**d, rho=_R3, snr=1.5, N=100, init="spectral", gamma0=1), "precondition:known-parameter")
    for k, val in (("init", "planted"), ("gamma0", 1), ("seeds", (0, 1)), ("iters", 5)):
        _refuses(lambda k=k, val=val: ex(**d, rho=_R3, snr=_HARD, **{k: val}), "precondition:known-parameter")
    _refuses(lambda: ex(**d, rho=_R3, snr=0.95, N=100), "precondition:snr-above-one")
    _refuses(lambda: ex(**d, rho=_R3, snr=_HARD, N=100, init="planted"), "precondition:gamma0")
    _refuses(lambda: ex(**d, rho=_R3, snr=_HARD, N=100, init="warm", gamma0=1), "precondition:known-parameter")
    _refuses(lambda: ex(**d, rho=_R3, snr=_HARD, N=100, init="warm"), "precondition:known-init")
    for k, val in (("init", "planted"), ("gamma0", 1)):
        _refuses(lambda k=k, val=val: ex(snr=2, N=100, **{k: val}), "precondition:known-parameter")
    old = "unsupported:" + "diluted-finite-n"
    pkg = Path(__file__).resolve().parents[1]
    hits = [str(p) for p in pkg.rglob("*.py") if old in p.read_text()]
    assert hits == [], hits


def test_diluted_spectral_easy_phase():
    """lambda = 1.5 at rho = 0.08 (easy phase, one SE fixed point gamma = 2.18565), spectral start, N = 2000, the
    default seeds 0-7 and iters 30: gamma_se_t (the SE fixed point) lies in the gamma_amp_t band, and it carries MV
    Thm 2's label.  Pilots (seeds 100-107, 200-207): 2.31 +- 0.32, 2.17 +- 0.38."""
    r = _dil(rho=0.08, snr=1.5, N=2000)
    v = r.values
    assert v["init"] == "spectral" and v["gamma0"] is None and v["seeds"] == list(range(8)) and v["iters"] == 30
    assert v["se_phase"] == "easy" and v["gamma_se_t"] == v["se_uninformative_gamma"]
    assert abs(v["gamma_se_t"] - v["gamma_amp_t"]) <= v["gamma_amp_t_halfwidth99"], v
    f = r.finality["gamma_se_t"]
    assert f.state == "exact-by-literature" and "MV Thm 2" in f.note, f
    # in the hard phase the spectral start is refused below lambda = 1 and above it MV's conditions are not met
    hard = _dil(rho=0.08, snr=1.005, N=300, seeds=(0, 1), iters=3)
    assert hard.values["se_phase"] == "hard" and not hard.finality["gamma_se_t"].final, hard.finality["gamma_se_t"]


def _two_starts(snr, t):
    kw = dict(rho=_R3, snr=snr, N=_N3, seeds=_SEEDS3, iters=t, init="planted")
    return _dil(gamma0=_W3, **kw), _dil(gamma0=1, **kw)


@slow
def test_diluted_two_branches_in_the_hard_phase():
    """rho = 0.05, lambda = 0.95 (hard phase proper), N = 8000, seeds 0-19, t = 4 (state points and pilots above the
    Task 3 block).  The weak (gamma0 = 0.03) and strong (gamma0 = 1) planted starts end on different branches: the
    gamma_amp_t band means differ by more than 10x the summed half-widths (margin fixed before the run), and each band
    contains its own gamma_se_t (weak 0.0201559, strong 0.735357)."""
    weak, strong = _two_starts(_HARD, 4)
    (mw, hw), (ms, hs) = _band_of(weak, "gamma_amp_t"), _band_of(strong, "gamma_amp_t")
    assert abs(ms - mw) > 10 * (hw + hs), (mw, hw, ms, hs)
    for r, (m, h) in ((weak, (mw, hw)), (strong, (ms, hs))):
        assert abs(r.values["gamma_se_t"] - m) <= h, (r.values["gamma_se_t"], m, h)
    assert abs(weak.values["gamma_se_t"] - 0.0201559) < 1e-6 and abs(strong.values["gamma_se_t"] - 0.735357) < 1e-6


@slow
def test_diluted_negative_control_below_dyn():
    """rho = 0.05, lambda = 0.6 (far below dyn 0.849), N = 8000, seeds 0-19, t = 8.  Both starts end on the same
    (uninformative) branch: each band contains its own gamma_se_t (weak 8.43655e-06, strong 4.99340e-04), and the
    strong start's whole band lies below the weak start's gamma0 = 0.03, i.e. it has fallen into the basin the weak
    start began in.  The same criterion fails at the hard point (strong band near 0.735), asserted here on the
    library at N = 3000, seeds 0-9, so the control can fail."""
    weak, strong = _two_starts(_BELOW, 8)
    (mw, hw), (ms, hs) = _band_of(weak, "gamma_amp_t"), _band_of(strong, "gamma_amp_t")
    for r, (m, h) in ((weak, (mw, hw)), (strong, (ms, hs))):
        assert abs(r.values["gamma_se_t"] - m) <= h, (r.values["gamma_se_t"], m, h)
    assert ms + hs < _W3, (ms, hs)
    hard = S.spiked_amp_ensemble(3000, _HARD, "diluted_rademacher", seeds=tuple(range(10)), iters=8, rho=_R3,
                                 init="planted", gamma0=1.0).bands["gamma_t"]
    assert not hard.mean + hard.half_width < _W3, hard
