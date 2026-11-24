"""I3 catalog entries: ising_triangular, ising_honeycomb and ising2d_anisotropic as runs (gauge-mapped
and frustrated sign patterns, the signed torus, the planted swapped dual couplings, the memo).  Split
from test_lattice_geom.py, whose helpers it reuses; the routes and identities stay there.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp

from test_lattice_geom import _arbs, _cached, _near, _planted, _refusal, _run


# ---- the catalog entries ------------------------------------------------------------------------
def test_triangular_and_honeycomb_entries():
    before = mp.mp.dps
    v = _cached("ising_triangular")
    assert v["phase"] == "T > T_c" and v["duality_identity"]["exact"]
    assert {"minus_beta_f_square_route", "minus_beta_f_pierre"} <= set(v)
    v = _run("ising_triangular", beta=40)
    assert abs(mp.mpf(v["minus_beta_f"]) - 120) < 1e-30 and v["phase"] == "T < T_c"
    v = _cached("ising_honeycomb")
    assert v["star_triangle_identity"]["overlap"] and v["phase"] == "T > T_c"
    assert _refusal(lambda: _run("ising_honeycomb", J1=0)).reason == "precondition:J1"
    assert _refusal(lambda: _run("ising_triangular", J2=0)).reason == "precondition:J2"
    assert mp.mp.dps == before


def test_triangular_antiferromagnet_ground_state_entropy():
    from replicant.lattice import tri_honey as T
    s0 = T.wannier_entropy()
    assert _near(s0, "triangular_af_ground_state_entropy") and s0.rad() < 1e-40
    f1 = T.triangular_free_energy(1, -1, -1, -1)                  # -beta f - beta |e0| -> s0
    assert 0 <= f1.mid() - 1 - s0.mid() < 0.03                     # 0.0218 (beta = 3: slow tier)
    v = _cached("ising_triangular", J1=-1, J2=-1, J3=-1)
    assert "beta_c" not in v and v["ground_state_counts"]["3 x 3"]["frustrated_bonds"] == 9
    assert "minus_beta_f_pierre" in v                  # the second route at v < 0 (Sec. 5.2)
    for key in ("triangular_af_ground_state_entropy_zukovic",
                "triangular_af_ground_state_entropy_pierre_integral"):
        assert _near(s0, key), key


def test_gauge_equivalent_and_frustrated_sign_patterns():
    """J1 J2 J3 > 0 is gauge-equivalent to |J| (its values, critical point and routes); J1 J2 J3 < 0 is
    frustrated and computed with its signs, refused only by the integrator's own stall."""
    v = _run("ising_triangular", beta=5, J1=-1, J2=-1, J3=1)
    w = _run("ising_triangular", beta=5)
    assert v["gauge"].startswith("J = ") and v["minus_beta_f"] == w["minus_beta_f"]
    assert v["beta_c"] == w["beta_c"] and "minus_beta_f_square_route" in v
    v = _run("ising_triangular", beta="0.5", J1=-1, J2=2, J3=1)          # frustrated (beta = 3: slow)
    assert "beta_c" not in v and "gauge" not in v and "minus_beta_f" in v
    v = _run("ising_honeycomb", J1=-1)                                   # bulk: |J|
    assert v["gauge"] and v["minus_beta_f"] == _cached("ising_honeycomb")["minus_beta_f"]
    v = _run("ising2d_anisotropic", J2=-2)
    assert v["gauge"] and v["minus_beta_f"] == _cached("ising2d_anisotropic")["minus_beta_f"]


def test_gauge_mapped_runs_report_the_signed_torus():
    """J = (-1, -1, 1) at beta = 1/5: the bulk is |J|'s, but the odd 3 x 3 torus is not gauge-closed,
    so its value is the signed torus's: ln Z / N = 0.76377012841100205489 by enumeration (|J|:
    0.80510378392); on 4 x 4 the two agree.  Planted: the torus with |J| (torus_signed False)."""
    from replicant import catalog_lattice_common as CC
    from replicant.lattice import geom_tori as G
    from replicant.lattice.ball import certify
    c, E = G.counts("triangular", 3, 3), G.class_sizes("triangular", 3, 3)
    brute = certify(lambda: G.z_of_counts(c, E, _arbs("-0.2", "-0.2", "0.2")).log() / 9)
    with mp.workdps(45):
        assert abs(brute.mid() - mp.mpf("0.76377012841100205489")) < 1e-19

    def signed3():
        v = _run("ising_triangular", beta="1/5", J1=-1, J2=-1, J3=1, L=3)
        with mp.workdps(45):
            return abs(mp.mpf(v["torus_minus_beta_f"]) - brute.mid())
    assert signed3() < 1e-35
    assert _planted(CC, "torus_signed", False, signed3) > 0.04    # |J|'s 0.80510: the check can fail
    v = _run("ising_triangular", beta="1/5", J1=-1, J2=-1, J3=1, L=4)
    assert v["torus_minus_beta_f"] == _cached("ising_triangular")["torus_minus_beta_f"]
    # the square at odd L and the honeycomb: tests/test_lattice_geom_slow.py


def test_planted_swapped_dual_couplings_are_refused():
    from replicant.lattice import anisotropic as A
    r = _planted(A, "dual_swapped", True, lambda: _refusal(lambda: _run("ising2d_anisotropic")))
    assert r.reason == "no-solution:duality-fixed-point", r


def test_memo_is_keyed_on_the_hooks_and_can_be_cleared():
    from replicant.lattice import anisotropic as A
    a = A.free_energy("0.2", 1, 2)
    assert A.free_energy("0.2", 1, 2) is a                               # memoised
    b = _planted(A, "cross_coupling", True, lambda: A.free_energy("0.2", 1, 2))
    assert b is not a and not b.overlaps(a)                              # a plant is a new key
    A.clear_memo()
    c = A.free_energy("0.2", 1, 2)
    assert c is not a and c.overlaps(a)
