"""K3: the random field Ising model on random graphs (KZ Sec. 4.3, Ex. 4.2): the `rfim` entry's route cavity
(catalog_rfim_sparse.py) through factor/population.py and factor/bp.py.

Convention (KZ Ch. 4): H = -J sum_(ij) s_i s_j - sum_i h_i s_i, beta the Ising beta, h_i ~ N(0, Delta) (Delta the
variance); c the degree (regular) or the mean degree (Erdos-Renyi).  Independent routes: the certified Bethe-lattice
ferromagnet in a uniform field (lattice/bethe.py: the homogeneous limit of the population), BP on sampled graphs,
the fully connected entry (route replica, K1) as c -> oo with the edge coupling J/c.
"""
import math
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from replicant import catalog_rfim_sparse as RS
from replicant.amp._stats import t_quantile
from replicant.catalog import catalog, check_ref
from replicant.coloring import potts_bp as PB
from replicant.compose import compose
from replicant.core import Refusal
from replicant.factor import bp as BP
from replicant.factor import population as POP
from replicant.lattice.bethe import bethe_ising_fixed_points
from replicant.meanfield import rfim as RF
from replicant.testing import slow


def _refusal(fn):
    try:
        fn()
    except Refusal as r:
        return r
    raise AssertionError("expected a Refusal")


def test_rfim_population_homogeneous_limits():
    """Deterministic limits of the population code, to 1e-9: a uniform field h on a c-regular graph (every member
    equal) gives the certified Bethe-lattice ferromagnet (phi and magnetization of the 'upper' fixed point); the
    Potts law on a 5-regular graph gives Eq. (6.12) from the planted start and the paramagnetic closed form from
    the uniform start.  The cavity route's Delta = 0 point is KZ's Ex. 4.2.1 (banked: 0.346 at c = 4); a run too
    short to relax (near beta_c, almost no field) is labelled undetermined, never refused; unknown graphs and bad couplings are refused."""
    for c, beta, h in ((4, "1/2", "1/10"), (3, "1", "1/20")):
        law = POP.rfim_law(float(Fraction(beta)), 1.0, lambda rng, n, h=h: np.full(n, float(Fraction(h))))
        r = POP.population_dynamics(law, ("regular", c), pop=100, sweeps=8, burn_in=300, seeds=(0, 1), init="planted")
        ref = next(f for f in bethe_ising_fixed_points(c, beta, h) if f.branch in ("upper", "unique"))
        assert r.status == "converged" and r.phi_stderr < 1e-12, r
        assert abs(r.phi - float(ref.phi.mid())) < 1e-9, (r.phi, ref.phi)
        assert abs((r.marginal[0] - r.marginal[1]) - float(ref.magnetization.mid())) < 1e-9
    for b in (-0.5, -0.7):
        pl = POP.population_dynamics(POP.potts_law(3, b), ("regular", 5), pop=100, sweeps=8, burn_in=300, seeds=(0, 1),
                                     init="planted")
        un = POP.population_dynamics(POP.potts_law(3, b), ("regular", 5), pop=100, sweeps=8, seeds=(0, 1))
        best = max((f for f in PB.potts_ferro_fixed_points(3, 5, b) if f.stable), key=lambda f: f.a)
        assert abs(pl.phi - best.phi) < 1e-9 and abs(un.phi - PB.potts_paramagnetic(3, 5, b).phi) < 1e-12, (b, pl, un)
    e = catalog()["rfim"]
    assert all(check_ref(e, r).ok for r in e.refs if r.route == "cavity")
    v = e.run("cavity", graph="poisson", c=3)
    assert abs(v["beta_c_zero_field"] - math.atanh(1 / 3)) < 1e-15
    res = compose(e.spec, "cavity").run(c=4, beta="9/25", Delta="1/1000000", pop=100, sweeps=8, seeds=[0, 1])
    assert res.values["pd_status"] == "undetermined" and "UNDETERMINED" in res.finality["m_cavity"].note
    assert not res.finality["m_cavity"].final and "m_cavity" in res.statistical
    res = compose(e.spec, "cavity").run(c=4, beta="1/2", Delta="1/10", pop=100, sweeps=8, seeds=[0, 1], N=60)
    assert "approximation" in res.finality["phi_bp"].note, res.finality["phi_bp"]       # loopy sampled graphs
    assert _refusal(lambda: e.run("cavity", graph="lattice")).reason == "unsupported:graph"
    assert _refusal(lambda: e.run("cavity", c=4, J=-1)).reason == "precondition:J"
    assert _refusal(lambda: POP.population_dynamics(POP.potts_law(3, 1), ("regular", 3), pop=10, sweeps=8,
                                                    seeds=(0, 1))).reason == "precondition:population"
    assert _refusal(lambda: POP.population_dynamics(POP.FactorLaw(2, 2, lambda rng, n: np.broadcast_to(
        1 - np.eye(2), (n, 2, 2))), ("regular", 3), pop=100, sweeps=8, seeds=(0, 1))).reason == \
        "unsupported:hard-constraints"


@slow
def test_rfim_population_dynamics():
    """The RFIM on 4-regular graphs (Gaussian fields): the population (planted s = +1) against BP on sampled graphs
    (N = 3000, 4 seeds) for m and Phi, inside t_{0.995, 3} hypot(se_pd, se_bp) + 0.01 (a finite-N allowance), in
    the ordered (beta = 1/2, Delta = 1/10) and the disordered (beta = 1/4) phase.  Power: a planted edge term
    (factor.bp._HOOKS edge_term = 0.5) shifts phi_bp by (c/2) E ln Z_ij ~ 0.3, about 30 times the band, so the
    plant is detected with probability ~ 1.  Then the fully connected limit: on c-regular graphs with coupling
    J/c, Phi and |m| approach the K1 replica route (Delta = 1/4; beta = 3 ordered, beta = 1/2 disordered) as c = 10,
    20, 40, 80: c |dPhi| below 0.6, |d|m|| decreasing to below 0.01, and in the disordered phase a clean 1/c (c dPhi
    about 0.06).  Near the ordering transition (beta = 3/2) the population relaxes slowly at c = 20 and is labelled
    undetermined, so the limit is taken away from it."""
    e = catalog()["rfim"]
    t = t_quantile(0.995, 3)
    for beta, ordered in (("1/2", True), ("1/4", False)):
        v, err = None, None
        res = compose(e.spec, "cavity").run(c=4, beta=beta, Delta="1/10", pop=3000, sweeps=20, N=3000,
                                            seeds=[0, 1, 2, 3])
        v, err = res.values, res.errors
        assert v["pd_status"] == "converged" and v["bp_status"] == "converged", v
        for a, b in (("m_cavity", "m_bp"), ("phi_cavity", "phi_bp")):
            band = t * math.hypot(err[a], err[b]) + 0.01
            assert abs(v[a] - v[b]) < band, (beta, a, v[a], v[b], band)
        assert (v["m_cavity"] > 0.5) if ordered else (abs(v["m_cavity"]) < 0.02), v
        law = RF.field_law("gaussian", "1/10")
        bad = []
        for s in (0, 1, 2, 3):
            old = BP._HOOKS["edge_term"]
            BP._HOOKS["edge_term"] = 0.5
            try:
                bad.append(RS.rfim_graph(3000, 4, "regular", float(Fraction(beta)), 1.0, law, s)[1])
            finally:
                BP._HOOKS["edge_term"] = old
                assert BP._HOOKS["edge_term"] is old
        assert abs(np.mean(bad) - v["phi_cavity"]) > 10 * (t * math.hypot(err["phi_cavity"], err["phi_bp"]) + 0.01)
    for beta in ("3", "1/2"):                         # ordered (|m| = 0.860) and disordered (m = 0) at Delta = 1/4
        rs = RF.rfim_rs(beta, 1, RF.field_law("gaussian", "1/4"))
        gaps = []
        for c in (10, 20, 40, 80):
            v = e.run("cavity", c=c, J=f"1/{c}", beta=beta, Delta="1/4", pop=3000, sweeps=20)
            assert v["pd_status"] == "converged", v
            gaps.append((abs(v["phi_cavity"] - rs.phi), abs(abs(v["m_cavity"]) - abs(rs.m))))
            assert c * gaps[-1][0] < 0.6, (beta, c, gaps)
        assert all(gaps[i + 1][1] < gaps[i][1] or gaps[i + 1][1] < 0.002 for i in range(3)), (beta, gaps)
        assert gaps[-1][0] < 0.004 and gaps[-1][1] < 0.01, (beta, gaps)
        if beta == "1/2":                             # clean 1/c: c dPhi stays in [0.04, 0.08] (about 0.06)
            assert gaps[0][0] > gaps[1][0] > gaps[2][0] > gaps[3][0], gaps
            assert all(0.04 < c * g[0] < 0.08 for c, g in zip((10, 20, 40, 80), gaps)), gaps
