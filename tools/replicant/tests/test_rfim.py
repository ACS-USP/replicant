"""K1: the fully connected random field Ising model (KZ Ch. 2): meanfield/rfim.py and the `rfim` catalog entry.

Convention: H = -(J N/2) Sbar^2 - sum_i h_i s_i, h_i ~ N(0, Delta) with Delta the variance (KZ p. 37), J = 1 unless
stated.  Banked: two readings of KZ Fig. 2.2.1 (whose horizontal axis is sqrt(Delta), not the Delta of its label).
Independent routes: exact ground states by sorting, the Hubbard-Stratonovich free entropy, brute-force enumeration
at N = 12.  Statistical bands: 99% Student-t over named seeds; power against each plant stated in its test.
"""
import itertools
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from replicant.amp._stats import student_band
from replicant.core import Refusal
from replicant.meanfield import rfim as RF
from replicant.refdata import load

BANK = load("meanfield")
SEEDS = tuple(range(100, 116))


def _planted(module, key, value, fn):
    reg = module._HOOKS
    old = reg[key]
    reg[key] = value
    try:
        return fn()
    finally:
        reg[key] = old
        assert module._HOOKS is reg and reg[key] is old


def _brute(h, beta, J=1.0):
    """(ln Z / N, min H / N) over all 2^N configurations."""
    n = len(h)
    s = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
    e = -J * n * (s.mean(axis=1)) ** 2 / 2 - s @ h
    lz = np.logaddexp.reduce(-beta * e)
    return lz / n, e.min() / n


def test_exact_routes_match_enumeration():
    """At N = 12 the sorted ground state and the Hubbard-Stratonovich ln Z / N equal brute-force enumeration."""
    for seed, law, beta, J in ((1, RF.field_law(Delta="0.5"), 1.5, 1), (2, RF.field_law("bimodal", h0="0.7"), 0.7, 2),
                               (3, RF.field_law(Delta="2"), 3.0, 0.5)):
        h = RF.rfim_sample_fields(12, law, seed)
        phi, e0 = _brute(h, beta, J)
        assert abs(RF.rfim_exact_ground_energy(h, J) - e0) < 1e-12
        assert abs(RF.rfim_exact_phi(h, beta, J) - phi) < 1e-10, (seed, RF.rfim_exact_phi(h, beta, J), phi)


def test_rfim_rs_vs_exact_ground_state():
    """RS e_min (Eq. (2.3)) inside the 99% band of exact sorted ground states (branch Sbar >= 0) at N = 4000 over 16
    named seeds, below and above Delta_c = 2/pi.  Planted Eq. (2.3) with -J m^2/2 lies more than 40 band widths away
    (power ~1: the shift is ~0.6 against a band of ~0.007).  The unrestricted ground state is lower on every seed and
    its mean lies below e_min (the O(N^-1/2) gain of choosing the better ferromagnetic state).  Guerra's bounds
    sandwich the exact free entropy on every sample; a planted +beta J m^2/2 in the lower bound breaks it."""
    for delta in ("0.36", "0.81"):
        law = RF.field_law(Delta=delta)
        hs = [RF.rfim_sample_fields(4000, law, s) for s in SEEDS]
        plus = [RF.rfim_exact_ground_energy(h, 1, "plus") for h in hs]
        full = [RF.rfim_exact_ground_energy(h) for h in hs]
        band = student_band(plus)
        rs = RF.rfim_ground_energy(1, law).e
        assert abs(rs - band.mean) <= band.half_width, (delta, rs, band)
        assert all(f <= p for f, p in zip(full, plus))
        bad = _planted(RF, "emin_sign", -1, lambda: RF.rfim_ground_energy(1, law).e)
        assert abs(bad - band.mean) > 40 * band.half_width, (delta, bad, band)
        if float(delta) < 2 / np.pi:
            assert np.mean(full) < rs - band.half_width
    for n in (60, 240):
        gaps = []
        for s in range(4):
            h = RF.rfim_sample_fields(n, RF.field_law(Delta="0.5"), s)
            lo, up = RF.rfim_guerra_bound(h, 1.5)
            x = RF.rfim_exact_phi(h, 1.5)
            assert lo <= x + 1e-12 and x <= up + 1e-12, (n, s, lo, x, up)
            gaps.append(up - lo)
            bad = _planted(RF, "guerra_quadratic", 1, lambda: RF.rfim_guerra_bound(h, 1.5)[0])
            assert bad > x + 1e-3, (bad, x)
        if n == 240:
            assert np.mean(gaps) < 0.5 * gaps60
        gaps60 = np.mean(gaps)


def test_rfim_rs_and_bank():
    """The RS saddle (Eqs. (2.1)-(2.2)): -dPhi/dbeta = e by a centred difference; the T = 0 limit approaches e_min;
    Theorem 7's variance bound holds on exact samples; the banked Fig. 2.2.1 readings match e_min only with the
    horizontal axis read as sqrt(Delta) (as labelled, Delta, the reading at 0.6 fails)."""
    law = RF.field_law(Delta="0.36")
    a, b = RF.rfim_rs(2.0 - 1e-5, 1, law), RF.rfim_rs(2.0 + 1e-5, 1, law)
    assert abs(-(b.phi - a.phi) / 2e-5 - RF.rfim_rs(2.0, 1, law).e) < 1e-6
    assert abs(RF.rfim_rs(40.0, 1, law).e - RF.rfim_ground_energy(1, law).e) < 2e-3
    phis = [RF.rfim_exact_phi(RF.rfim_sample_fields(200, RF.field_law(Delta="1"), s), 1.0) for s in range(16)]
    assert np.var(phis, ddof=1) <= 1.0 * 1.0 / 200                     # Var Phi_N <= Delta beta^2 / N
    for key, delta in (("rfim_fig221_emin_delta0p36", "0.36"), ("rfim_fig221_emin_delta0p81", "0.81")):
        want = BANK[key]
        assert abs(RF.rfim_ground_energy(1, RF.field_law(Delta=delta)).e - float(want.value)) <= float(want.tol)
    want = BANK["rfim_fig221_emin_delta0p36"]                          # the label's reading: Delta = 0.6
    assert abs(RF.rfim_ground_energy(1, RF.field_law(Delta="0.6")).e - float(want.value)) > 5 * float(want.tol)
    assert abs(RF.rfim_delta_c() - 2 / np.pi) < 1e-15


def test_rfim_entry_and_refusals():
    from replicant.catalog import catalog
    e = catalog()["rfim"]
    v = e.run(Delta="0.36", beta="2", N=300, seeds=[0, 1, 2])
    # the exact finite-N ground energy (Sbar >= 0 state, O(1/N) from the limit) contains the RS T = 0 energy e_min in
    # its 99% band widened by 1/N; a planted e_min of another law (Delta = 1, e_min -0.798, 0.27 away against a
    # band of about 0.14) falls outside it
    band = v["e_ground_plus_halfwidth99"] + 1 / 300
    assert abs(v["e_ground_plus"] - v["e_min"]) <= band, (v["e_ground_plus"], v["e_min"], band)
    wrong = RF.rfim_ground_energy(1, RF.field_law(Delta="1")).e
    assert abs(v["e_ground_plus"] - wrong) > band, (v["e_ground_plus"], wrong, band)
    assert v["phi_guerra_lower"] <= v["phi_exact"] <= v["phi_guerra_upper"]
    assert abs(v["phi"] - v["phi_exact"]) < 0.05
    for bad, reason in (({"law": "cauchy"}, "unsupported:field-law"), ({"Delta": "-1"}, "precondition:Delta"),
                        ({"law": "bimodal", "Delta": "1"}, "precondition:known-parameter")):
        try:
            e.run(**bad)
        except Refusal as r:
            assert r.reason == reason, r
        else:
            raise AssertionError(bad)
    guide = (Path(RF.__file__).parent.parent / "GUIDE.md").read_text()
    assert "| `rfim`" in guide and "`rfim` (replica)" in guide
