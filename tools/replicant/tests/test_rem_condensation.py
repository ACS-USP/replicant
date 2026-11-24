"""K2: condensation in the REM (KZ Secs. 15.2.4-15.2.5, Ex. 15.2): the participation ratio Y = sum_i w_i^2, its
second moment and the overlap law, added to families/rem.py and the `rem` entry.

Convention: KZ Eq. (15.1), P(E) = N(0, N/2), is families/rem.py's N_scale = 1/2 (beta_c = 2 sqrt(ln 2)); m =
beta_c/beta.  Replica predictions (physics-level): E[Y] = 1 - m (Eq. (15.39)), E[Y^2] = (3 - 5m + 2m^2)/3
(Eq. (15.66)), E P(q) = (1 - m) delta(q - 1) + m delta(q) (Eq. (15.42)).  Independent routes: exact REM sampling
over named seeds (2^N i.i.d. energies, the Gibbs weights, two Gibbs draws per sample), and the exact finite-N
moments by the quadrature E[1/Z^k] = int t^(k-1) E e^(-tZ) dt / (k-1)! (families/rem.py, no replicas).
Finite N: at N = 14, beta = 4 beta_c, |E_N[Y] - (1 - m)| = 0.025 and |E_N[Y^2] - (15.66)| = 0.032 (the quadrature says
so; the test asserts both below 0.035): the sampled band is compared with the formula on band + 0.035.  Planted m -> 1 - m (rem._HOOKS["m"])
moves every prediction by |1 - 2m| = 1/2 at m = 1/4, against a band of about 0.06 + 0.035: power ~ 1.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import mpmath as mp
import numpy as np

from replicant.amp._stats import student_band
from replicant.catalog import catalog, check_ref
from replicant.families import rem as REM
from replicant.testing import slow

S = "0.5"                       # KZ's N_scale
N, SEEDS, SYSTEMATIC = 14, tuple(range(400)), 0.035


def _planted(module, key, value, fn):
    reg = module._HOOKS
    old = reg[key]
    reg[key] = value
    try:
        return fn()
    finally:
        reg[key] = old
        assert module._HOOKS is reg and reg[key] is old


def _flip(m):
    return 1 - m


def _samples(beta, n=N, seeds=SEEDS):
    """(Y, Y^2, 1(two Gibbs draws coincide)) per seed: 2^n energies of variance n/2 from default_rng([seed, 15])."""
    ys, y2, same = [], [], []
    for seed in seeds:
        rng = np.random.default_rng([seed, 15])
        a = -float(beta) * rng.standard_normal(2**n) * math.sqrt(float(S) * n)
        w = np.exp(a - a.max())
        w /= w.sum()
        y = float(w @ w)
        i, j = rng.choice(2**n, size=2, p=w)
        ys.append(y), y2.append(y * y), same.append(float(i == j))
    return student_band(ys), student_band(y2), student_band(same)


def _predictions(beta):
    ov = REM.rem_overlap_distribution(beta, S)
    return float(REM.rem_participation_ratio(beta, S)), float(REM.rem_y2(beta, S)), float(ov[1])


def _agree(bands, preds, systematic):
    return all(abs(b.mean - p) <= b.half_width + systematic for b, p in zip(bands, preds))


def test_rem_condensation_formulas():
    with mp.workdps(30):
        bc = REM.rem_critical_beta(S)
        for k in (mp.mpf(4), mp.mpf(2), mp.mpf("1.25")):
            m = 1 / k
            assert abs(REM.rem_participation_ratio(k * bc, S) - (1 - m)) < mp.mpf(10)**-25        # Eq. (15.39)
            assert abs(REM.rem_y2(k * bc, S) - (3 - 5 * m + 2 * m * m) / 3) < mp.mpf(10)**-25       # Eq. (15.66)
            ov = REM.rem_overlap_distribution(k * bc, S)                                            # Eq. (15.42)
            assert abs(ov[1] - (1 - m)) < mp.mpf(10)**-25 and abs(ov[0] - m) < mp.mpf(10)**-25
            assert REM.rem_y2(k * bc, S) > REM.rem_participation_ratio(k * bc, S)**2                # not self-averaging
        # high temperature (Eq. (15.41)): delta(q), Y -> 0
        assert REM.rem_participation_ratio(bc / 2, S) == 0 and REM.rem_overlap_distribution(bc / 2, S)[0] == 1
    # the rem entry gains the values; the earlier values at the banked point are unchanged
    e = catalog()["rem"]
    v = e.run(beta=3, condensation=True)
    sol = REM.rem_1rsb(3, N_scale="0.5")
    assert v["f"] == sol.f and v["m"] == sol.m and v["e"] == sol.e
    with mp.workdps(30):
        assert abs(v["participation_ratio"] - (1 - mp.mpf(sol.m))) < mp.mpf(10)**-25
    assert v["overlap_p1"] == v["participation_ratio"]
    for ref in e.refs:
        assert check_ref(e, ref).ok, ref.key
    # behind the parameter: the earlier parameter sets give the earlier keys only (and stay final)
    from replicant.compose import compose
    old = compose(e.spec).run(beta=3)
    assert "participation_ratio" not in old.values and old.final
    assert not compose(e.spec).run(beta=3, condensation=True).final        # replica values: not final
    try:
        e.run(beta=3, condensation="yes")
    except Exception as r:
        assert getattr(r, "reason", "") == "precondition:known-parameter", r
    else:
        raise AssertionError("condensation=yes was read")


def test_rem_condensation():
    """Exact REM sampling at N = 14, beta = 4 beta_c (m = 1/4), seeds 0-399: the 99% bands of Y, Y^2 and the
    two-draw coincidence contain the exact finite-N moments, and lie within 0.035 of the replica predictions; the
    planted m -> 1 - m fails."""
    beta = 4 * REM.rem_critical_beta(S)
    bands = _samples(beta)
    ey, ey2 = REM.rem_participation_ratio(beta, S, N=N), REM.rem_y2(beta, S, N=N)
    assert abs(bands[0].mean - ey) <= bands[0].half_width and abs(bands[2].mean - ey) <= bands[2].half_width
    assert abs(bands[1].mean - ey2) <= bands[1].half_width
    preds = _predictions(beta)
    assert abs(ey - preds[0]) < SYSTEMATIC and abs(ey2 - preds[1]) < SYSTEMATIC     # the declared systematic
    assert _agree(bands, preds, SYSTEMATIC), (bands, preds)
    bad = _planted(REM, "m", _flip, lambda: _predictions(beta))
    assert all(not _agree([b], [p], SYSTEMATIC) for b, p in zip(bands, bad)), bad


@slow
def test_rem_finite_n_moments_approach_the_replica_values():
    """The exact finite-N E[Y], E[Y^2] tend to Eqs. (15.39), (15.66) as N grows (slowly near beta_c)."""
    for k in (4, 2, mp.mpf(4) / 3):
        beta = k * REM.rem_critical_beta(S)
        y, y2 = _predictions(beta)[:2]
        gaps = [abs(REM.rem_participation_ratio(beta, S, N=n) - y) for n in (16, 32, 64, 128)]
        gaps2 = [abs(REM.rem_y2(beta, S, N=n) - y2) for n in (16, 32, 64, 128)]
        assert all(a > b for a, b in zip(gaps, gaps[1:])) and gaps[-1] < 0.6 * gaps[0], (k, gaps)
        assert all(a > b for a, b in zip(gaps2, gaps2[1:])), (k, gaps2)
