"""Finite-N approximate survey propagation (amp/asp_finite.py) against the ASP state evolution at a fixed t
(amp/asp.py `asp_trajectory`).

The finite-N run on one SK matrix (amp/sk.py `sk_matrix`, Gaussian, from default_rng([seed, 1])) gives the
empirical (q0^t, q1^t); entry t of its traces is compared with entry t of the SE trajectory from the same start.
Bands are 99% Student-t over the named seeds 0-5 (df = 5).  N, the seeds and t were fixed from pilots on the
disjoint seeds 100-105 (recorded in each docstring), before any test seed was run: N = 2000, t = 10, seeds 0-5.
Pilots (seeds 100-105, N in {1000, 2000, 3000}, t in {5, 10, 15, 20}): at every (N, t) the SE was inside both
bands (|mean - SE| / half-width <= 0.66); N = 2000, t = 10 costs ~1.4 s per six-seed ensemble.  m comes from families/sk.py
`sk_1rsb` (a comparison target only: nothing from it feeds amp/asp_finite.py, nor amp/asp.py).
"""
import ast
import functools
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.amp import asp as A
from replicant.amp import asp_finite as F
from replicant.families.sk import sk_1rsb
from replicant.testing import slow

PKG = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(6))
N = 2000
T = 10


def _planted(key, value, fn):
    old = F._HOOKS[key]
    F._HOOKS[key] = value
    try:
        return fn()
    finally:
        F._HOOKS[key] = old
        assert F._HOOKS[key] is old


def _inside(band, x):
    return abs(x - band.mean) <= band.half_width


def _m(beta, h=0):
    return float((sk_1rsb(beta) if h == 0 else sk_1rsb(beta, h)).m)


@functools.lru_cache(maxsize=None)
def _ens(beta, h, n, iters, reaction=True, start=None):
    return F.asp_amp_ensemble(n, beta, _m(beta, h), float(h), seeds=SEEDS, iters=iters, start=start,
                              reaction=reaction)


def test_asp_trajectory_ends_at_fixed_point():
    """t = 0 is the start (default (0.5, 0.95)); the last of 3000 steps of _map is asp_fixed_point's (q0, q1)."""
    m = _m(2)
    tr = A.asp_trajectory(2, m, iters=3000)
    assert len(tr) == 3001 and tr[0] == (0.5, 0.95), tr[:2]
    fp = A.asp_fixed_point(2, m)
    assert abs(tr[-1][0] - fp.q0) < 1e-12 and abs(tr[-1][1] - fp.q1) < 1e-12, (tr[-1], fp)
    zero = A.asp_trajectory(2, m, start=(0.0, 0.9), iters=5)
    assert all(q0 == 0.0 for q0, _ in zero), zero


def _points():
    for beta in (2, 3):
        m = _m(beta)
        for gap in (0.15, 0.3, 0.45):
            yield beta, m, math.sqrt(gap), np.linspace(-3, 3, 25)


def _complement(x, beta, m, s, sign):
    """1 - sign f(x) at h = 0 (sign = +-1 per point, fixed by the centre of the difference), computed without cancellation as <2 / (1 + exp(2 sign(x) Xi))> on the module's
    own nodes (so that its derivative is -sign(x) f' exactly in exact arithmetic).  Near |f| = 1 a central
    difference of f itself loses ~1e-10 absolute to rounding (eps / 1e-6), more than 1e-7 of f' ~ 1e-6 at
    beta = 3, x = 3."""
    zp, lw = F._nodes(F.GH_ORDER)
    sg = sign[:, None]
    tot = num = 0.0
    for sv in (1.0, -1.0):
        xi = beta * (x[:, None] + sv * s * zp)
        W = np.exp(m * (np.abs(xi) + np.log1p(np.exp(-2 * np.abs(xi)))) + lw)   # cosh^m up to a constant 2^-m
        tot = tot + W.sum(axis=1)
        num = num + (W * 2 / (1 + np.exp(2 * sg * xi))).sum(axis=1)
    return num / tot


def test_asp_denoiser_derivative():
    """f' = beta (<sech^2> + m (<tanh^2> - <tanh>^2)) (cosh^m average) against a central difference, h = 1e-6,
    relative error < 1e-7 at beta = 2 and 3, sk_1rsb's m, q1 - q0 in {0.15, 0.3, 0.45}, x in [-3, 3].  The
    difference is taken of 1 - sign(x) f (_complement), the same quadrature without the cancellation near |f| = 1;
    the plain difference of f is checked where it is well conditioned (|f'| > 1e-2)."""
    d = 1e-6
    for beta, m, s, x in _points():
        f, fp, _ = F._denoise(x, beta, m, 0.0, s)
        sg = np.where(x >= 0, 1.0, -1.0)
        assert np.max(np.abs(_complement(x, beta, m, s, sg) - (1 - sg * f))) < 1e-12, (beta, s)
        fd = -sg * (_complement(x + d, beta, m, s, sg) - _complement(x - d, beta, m, s, sg)) / (2 * d)
        rel = np.abs(fd - fp) / np.abs(fp)
        assert np.all(rel < 1e-7), (beta, s, float(rel.max()), x[np.argmax(rel)])
        fd_plain = (F._denoise(x + d, beta, m, 0.0, s)[0] - F._denoise(x - d, beta, m, 0.0, s)[0]) / (2 * d)
        ok = np.abs(fp) > 1e-2
        assert np.all(np.abs(fd_plain - fp)[ok] / np.abs(fp[ok]) < 1e-7), (beta, s)


def test_asp_quadrature_order():
    """The Gauss-Hermite order F.GH_ORDER and twice it agree on f, f' and <tanh^2> to 1e-10."""
    for beta, m, s, x in _points():
        a = F._denoise(x, beta, m, 0.0, s)
        b = F._denoise(x, beta, m, 0.0, s, order=2 * F.GH_ORDER)
        for u, v in zip(a, b):
            assert np.max(np.abs(u - v)) < 1e-10, (beta, s, float(np.max(np.abs(u - v))))


def test_asp_finite_tracks_se():
    """beta = 2, h = 0, m = sk_1rsb(2).m = 0.28184 (q0 = 0.21463, q1 = 0.61920 at the fixed point), default start
    (0.5, 0.95), N = 2000, t = 10: the SE (q0_10, q1_10) = (0.24430, 0.61477) lie inside the bands.  Below the AT
    line the iterates do not converge by t = 10 (nor need they): the target is the SE at t, not the fixed point.
    Pilot (seeds 100-105): q0 0.24840 +- 0.01561, q1 0.61564 +- 0.00586."""
    se = A.asp_trajectory(2, _m(2), iters=T)[T]
    ens = _ens(2, 0, N, T)
    print(f"beta 2: q0 {ens.bands['q0'].mean:.5f} +- {ens.bands['q0'].half_width:.5f} (SE {se[0]:.5f}); "
          f"q1 {ens.bands['q1'].mean:.5f} +- {ens.bands['q1'].half_width:.5f} (SE {se[1]:.5f})")
    assert _inside(ens.bands["q0"], se[0]), (ens.bands, se)
    assert _inside(ens.bands["q1"], se[1]), (ens.bands, se)
    assert len(ens.q0) == len(SEEDS) and ens.N == N and ens.iters == T


@slow
def test_asp_finite_tracks_se_more_points():
    """beta = 3 at h = 0 (m = 0.25672; SE at t = 10: (0.31040, 0.80139)), and beta = 2 at h = 0.3 with that h's
    sk_1rsb m = 0.37564 (SE at t = 10: (0.43955, 0.63992)); N = 2000, t = 10.  Pilot (seeds 100-105): beta 3 q0
    0.31549 +- 0.01773, q1 0.80196 +- 0.00343; h 0.3 q0 0.44646 +- 0.03171, q1 0.63761 +- 0.00982."""
    for beta, h in ((3, 0), (2, "0.3")):
        se = A.asp_trajectory(beta, _m(beta, h), float(h), iters=T)[T]
        ens = _ens(beta, h, N, T)
        assert _inside(ens.bands["q0"], se[0]), (beta, h, ens.bands, se)
        assert _inside(ens.bands["q1"], se[1]), (beta, h, ens.bands, se)


def test_asp_finite_controls_fail():
    """reaction=False misses the SE q0_T; _HOOKS["reweight"] = False (m = 0 in f and f') misses the SE q1_T at
    sk_1rsb's m.  Each gap must exceed 10x the control's half-width (margin fixed before the test seeds ran).
    The m = 0 gap was measured first, on the pilot seeds 100-105 at beta = 2, h = 0, N = 2000, t = 10: q1 0.53060
    +- 0.00236 against the SE 0.61477, 35.7 half-widths, so the beta = 2, h = 0 point (m = 0.282) suffices and the
    larger-m point is not needed (at beta = 2, h = 0.3, m = 0.376, the pilot gap was only 8.1 half-widths, the RS
    average being closer there).  No Onsager term, pilot: q0 0.82295 +- 0.00492 against 0.24430 (118 half-widths)."""
    se = A.asp_trajectory(2, _m(2), iters=T)[T]
    naive = _ens(2, 0, N, T, reaction=False)
    b = naive.bands["q0"]
    assert abs(b.mean - se[0]) > 10 * b.half_width, (b, se)
    rs = _planted("reweight", False, lambda: F.asp_amp_ensemble(N, 2, _m(2), seeds=SEEDS, iters=T))
    b = rs.bands["q1"]
    assert abs(b.mean - se[1]) > 10 * b.half_width, (b, se)


def test_asp_finite_q0_zero_start():
    """h = 0, start (0, 0.9): x^0 = 0 and f = <tanh> is odd in V, so f = 0 identically: every q0 is 0, the q0 band
    is (0, 0) and nothing is NaN.  q1_t is then the same for every seed and equals the SE's q1_t."""
    ens = F.asp_amp_ensemble(400, 2, _m(2), seeds=(0, 1, 2), iters=5, start=(0.0, 0.9))
    se = A.asp_trajectory(2, _m(2), start=(0.0, 0.9), iters=5)
    b0, b1 = ens.bands["q0"], ens.bands["q1"]
    assert b0.mean == 0.0 and b0.half_width == 0.0, b0
    assert all(math.isfinite(v) for v in (b0.stderr, b1.mean, b1.half_width, b1.stderr)), (b0, b1)
    assert all(v == 0.0 for v in ens.q0), ens.q0
    r = F.asp_amp(F.sk_matrix(400, 0), 2, _m(2), iters=5, seed=0, start=(0.0, 0.9))
    assert all(v == 0.0 for v in r.q0_trace), r.q0_trace
    for t in range(6):
        assert abs(r.q1_trace[t] - se[t][1]) < 1e-9, (t, r.q1_trace[t], se[t])


def test_asp_finite_imports_nothing_from_asp():
    """amp/asp_finite.py imports neither amp/asp.py nor families/ (AST audit)."""
    tree = ast.parse((PKG / "amp" / "asp_finite.py").read_text())
    found = []                                  # (absolute-ish module path, imported name)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = ("." * node.level) + (node.module or "")
            found += [(base, a.name) for a in node.names]
        elif isinstance(node, ast.Import):
            found += [(a.name, "") for a in node.names]
    assert found, "no imports parsed"
    bad = [(mod, name) for mod, name in found
           if "famil" in mod or "famil" in name or "sk_1rsb" in name
           or mod.split(".")[-1] == "asp" or (name == "asp" and mod.rstrip(".").split(".")[-1] in ("", "amp"))]
    assert not bad, (bad, found)
