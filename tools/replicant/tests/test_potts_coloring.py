"""K3: Potts / coloring BP on random graphs (KZ Sec. 4.2, Ch. 6): coloring/, the `potts_bethe` and `coloring` entries.

Convention: weight exp(-beta delta_{s s'}), beta > 0 antiferromagnetic, beta < 0 ferromagnetic, beta_Potts =
-2 beta_Ising (checked in both directions on printed numbers).  Independent routes: the certified Bethe-lattice
ferromagnet (lattice/bethe.py, no shared code: an import audit), generic BP (factor/) on sampled graphs, exact
first moments of G(N, M) by brute force.  Planted errors go through coloring.potts_bp._HOOKS, restored with
identity asserts.
"""
import ast
import math
import os
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from replicant.catalog import catalog
from replicant.coloring import first_moment as FM
from replicant.coloring import potts_bp as PB
from replicant.coloring import potts_graph as PG
from replicant.factor import bp as BP
from replicant.factor import graph as FG
from replicant.lattice.bethe import bethe_critical_beta, bethe_ising_fixed_points
from replicant.refdata import check, load
from replicant.testing import slow

BANK = load("factor")
PKG = Path(PB.__file__).resolve().parent.parent


def _planted(module, key, value, fn):
    reg = module._HOOKS
    old = reg[key]
    reg[key] = value
    try:
        return fn()
    finally:
        reg[key] = old
        assert module._HOOKS is reg and reg[key] is old


def _imports(src):
    out = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom):
            out.add(("." * node.level) + (node.module or ""))
        elif isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
    return out


def _forbidden(mods):
    return {m for m in mods if any(x in m.split(".") for x in ("lattice", "cavity", "inference"))}


def test_q2_potts_reproduces_bethe_ising():
    """Potts q = 2 at beta_Potts = -2 beta_Ising against the certified BP fixed points of lattice/bethe.py (k = d):
    the cavity magnetization 2a - 1 = tanh x, Phi_Ising = Phi_Potts - beta_I d/2, by the closed form (6.11)-(6.12)
    and by generic BP on a sampled d-regular graph (Bethe / N and the mean marginal).  factor/ and coloring/ import
    nothing from lattice/, cavity/ or inference/ (AST audit, a planted import is caught); a planted sign in the
    beta map breaks the agreement."""
    files = sorted((PKG / "factor").glob("*.py")) + sorted((PKG / "coloring").glob("*.py"))
    assert len(files) >= 8 and all(not _forbidden(_imports(f.read_text())) for f in files), files
    assert _forbidden(_imports("from ..lattice.bethe import bethe_ising_fixed_points\n")) == {"..lattice.bethe"}
    rng = np.random.default_rng(5)
    for d, bi in ((3, "4/5"), (5, "2/5"), (5, "1/5")):
        b_i = float(Fraction(bi))
        b_p = PB.potts_beta(b_i)
        assert b_p == -2 * b_i and PB.ising_beta(b_p) == b_i                  # both directions
        fps = bethe_ising_fixed_points(d, bi, 0)
        ref = next(f for f in fps if f.branch in ("upper", "unique"))
        x, phi_i, m_i = float(ref.cavity_field.mid()), float(ref.phi.mid()), float(ref.magnetization.mid())
        top = max(PB.potts_ferro_fixed_points(2, d, b_p), key=lambda f: f.a)
        assert abs((2 * top.a - 1) - math.tanh(x)) < 1e-12 and abs(top.phi - b_i * d / 2 - phi_i) < 1e-12, (d, bi)
        assert len(fps) == len(PB.potts_ferro_fixed_points(2, d, b_p))
        g = PG.potts_graph(2, 200, FG.random_regular_edges(200, d, rng), b_p)
        st = BP.bp(g, iters=2000, init=("planted", np.zeros(200, dtype=int)))
        assert st.converged, st.note
        b = BP.marginals(g, st)
        assert abs(BP.bethe_free_entropy(g, st).phi - b_i * d / 2 - phi_i) < 1e-10
        assert abs(np.mean(b[:, 0] - b[:, 1]) - m_i) < 1e-10
        bad = _planted(PB, "ising_map", 0.5, lambda: max(PB.potts_ferro_fixed_points(2, d, PB.potts_beta(b_i)),
                                                           key=lambda f: f.a).phi)
        assert abs(bad - b_i * d / 2 - phi_i) > 1e-3, (d, bi, bad)


def test_bethe_peierls_table():
    """KZ Sec. 4.2: beta_BP for c = 2d, d = 2..5 (printed 3 digits, 0.346 truncated: tol 1e-3) from the Potts q = 2
    stability point (Eq. (6.13), degree c) mapped to the Ising convention; the certified atanh(1/(c - 1)) agrees to
    1e-14; generic BP on a sampled c-regular graph from the planted start orders only above it.  Planted sign of the
    map: every row fails.  The printed Potts-convention numbers of Sec. 6.5.1 map back the other way."""
    rng = np.random.default_rng(3)
    for dim in (2, 3, 4, 5):
        c = 2 * dim
        b_i = PB.ising_beta(PB.potts_beta_stab(2, c - 1))
        assert check(BANK[f"bethe_peierls_beta_dim{dim}"], b_i, label=str(dim)).ok, (dim, b_i)
        assert abs(b_i - float(bethe_critical_beta(c).mid())) < 1e-14
        bad = _planted(PB, "ising_map", 0.5, lambda: PB.ising_beta(PB.potts_beta_stab(2, c - 1)))
        assert not check(BANK[f"bethe_peierls_beta_dim{dim}"], bad, label=str(dim)).ok
        e = FG.random_regular_edges(24, c, rng)
        for f, ordered in ((0.8, False), (1.25, True)):
            g = PG.potts_graph(2, 24, e, PB.potts_beta(f * b_i))
            st = BP.bp(g, iters=3000, init=("planted", np.zeros(24, dtype=int)))
            m = float(np.mean(BP.marginals(g, st)[:, 0])) - 0.5
            assert st.converged and ((m > 1e-3) if ordered else (abs(m) < 1e-8)), (dim, f, m)
    for d, key in ((4, "potts_beta_stab_q2_d4"), (6, "potts_beta_stab_q2_d6")):    # Potts -> Ising -> Potts
        assert check(BANK[key], PB.potts_beta(float(bethe_critical_beta(d).mid())), label=key).ok


def test_potts_first_order_q3():
    """q = 3 on 5-regular graphs: beta_stab < beta_c < beta_s < 0 from Eq. (6.11)-(6.12) (the saddle node at beta_s,
    Phi_ferro = Phi_para at beta_c), both starts of the map converging (status determined); generic BP on a sampled
    5-regular graph from the planted and a random start: ordered below beta_c, paramagnetic above, both branches'
    Bethe / N equal to (6.12) and the paramagnetic closed form; outside the window both starts reach one fixed point.
    Controls: a starved iteration budget makes the scalar points undetermined, and BP with 3 iterations does too;
    q = 2 is second order."""
    t = PB.potts_transitions(3, 5)
    assert t.order == "first" and t.status == "determined", t
    assert abs(t.beta_stab + math.log(2)) < 1e-15 and t.beta_stab < t.beta_c < t.beta_s < 0, t
    assert PB._ferro(3, 5, t.beta_s - 1e-7) is not None and PB._ferro(3, 5, t.beta_s + 1e-7) is None
    f = PB._ferro(3, 5, t.beta_c)
    assert abs(f.phi - PB.potts_paramagnetic(3, 5, t.beta_c).phi) < 1e-10
    und = _planted(PB, "map_iters", 3, lambda: PB.potts_transitions(3, 5))
    assert und.status == "undetermined" and "converged: False" in und.note, und
    assert PB.potts_transitions(2, 5).order == "second"
    n, w = 150, t.beta_s - t.beta_c
    g = lambda b: PG.potts_graph(3, n, FG.random_regular_edges(n, 5, np.random.default_rng(8)), b)
    for b, phase in ((t.beta_c - 0.4 * w, "ordered"), (t.beta_c + 0.4 * w, "paramagnetic-like")):
        r = PG.two_starts(g(b), iters=5000, seed=2)
        assert r.phase == phase, (b, r)
        assert abs(r.phi_planted - PB._ferro(3, 5, b).phi) < 1e-9
        assert abs(r.phi_random - PB.potts_paramagnetic(3, 5, b).phi) < 1e-9
    r = PG.two_starts(g(t.beta_s + 0.003), iters=5000, seed=2)           # past the spinodal: the ferro branch is gone
    assert r.phase == "same" and abs(r.b0_planted - 1 / 3) < 1e-6, r
    r = PG.two_starts(g(t.beta_stab - 0.01), iters=5000, seed=2)         # past beta_stab: the paramagnet is unstable
    assert r.phase == "same" and abs(r.b0_random - 1 / 3) > 0.1, r
    r = PG.two_starts(g(t.beta_c - 0.4 * w), iters=3, seed=2)
    assert r.phase == "undetermined" and "not converged" in r.note, r


def test_coloring_annealed_and_ks():
    """KZ Sec. 6.6: c_KS(q = 3) = 4 (Eq. (6.24)) and c_ann(q = 3) = 5.4 (Eq. (6.17), read on the PDF page: the
    annealed threshold) against the bank, planted formulas failing; the exact first moment of G(N, M) by a
    composition sum equals brute-force enumeration at N = 5, 6; (1/N) ln E Z approaches Phi_ann = Phi_Bethe(1/q)
    at O(1/N) and changes sign between c = 5.3 and 5.5 at N = 120; the paramagnetic s(e) peaks at ln q, e = c/2q."""
    assert check(BANK["coloring_c_ks_q3"], PB.kesten_stigum(3), label="ks").ok
    assert check(BANK["coloring_c_ann_q3"], PB.coloring_annealed_threshold(3), label="ann").ok
    assert not check(BANK["coloring_c_ks_q3"], _planted(PB, "ks_power", 1, lambda: PB.kesten_stigum(3)), label="").ok
    assert not check(BANK["coloring_c_ann_q3"], _planted(PB, "ann_numerator", 1,
                                                         lambda: PB.coloring_annealed_threshold(3)), label="").ok
    assert abs(PB.kesten_stigum(5, "1/2") - (5 - (1 - math.exp(-0.5))) ** 2 / (1 - math.exp(-0.5)) ** 2) < 1e-12
    for q, n, m, beta in ((3, 5, 4, "1/2"), (3, 5, 4, "inf"), (2, 6, 5, "3/2")):
        lz = FM.first_moment_log(q, n, m, beta)
        assert abs(lz - math.log(FM.first_moment_brute(q, n, m, beta))) < 1e-12, (q, n, m, beta)
        ann = n * math.log(q) + m * math.log1p(-(1 - math.exp(-PB.beta_value(beta))) / q)
        assert abs(lz - ann) > 1e-3                               # finite N: not the annealed closed form
    pa = PB.potts_paramagnetic(3, 3, "inf").phi
    gaps = [FM.first_moment_log(3, n, 3 * n // 2, "inf") / n - pa for n in (30, 60, 120)]
    assert all(1.8 < gaps[i] / gaps[i + 1] < 2.2 for i in range(2)) and abs(gaps[-1]) < 0.01, gaps
    assert FM.first_moment_log(3, 120, 318, "inf") > 0 > FM.first_moment_log(3, 120, 330, "inf")     # c = 5.3, 5.5
    small = PB.potts_paramagnetic(4, 6, "1e-6")
    assert abs(small.s - math.log(4)) < 1e-9 and abs(small.e - 6 / 8) < 1e-6
    v = catalog()["coloring"].run("cavity", q=3, c=3)
    assert v["phi_para"] == v["s_para"] and v["c_ks"] == 4.0 and abs(v["c_ann"] - 2 * math.log(3) / math.log(1.5)) < 1e-12


@slow
def test_coloring_bp_kesten_stigum_on_graphs():
    """Damped BP (damping 1/2: the uniform mode of the parallel iteration, eigenvalue -c/(q - 1), is tamed, the
    bulk c~ lambda^2 of Eq. (6.23) is not) on sampled ER graphs at beta = oo, q = 3: back to chi = 1/q at
    c = 3 < c_KS = 4 on every seed; no convergence at c = 5 on any seed (N = 2000)."""
    for s in (0, 1):
        ok, dev = PG.bp_convergence(3, 3.0, "inf", n=2000, seed=s)
        assert ok and dev < 1e-8, (s, dev)
        ok, dev = PG.bp_convergence(3, 5.0, "inf", n=2000, seed=s)
        assert not ok and dev > 0.1, (s, dev)
    v = catalog()["coloring"].run("cavity", q=3, c=5, N=1500, seeds=[0, 1])
    assert v["bp_converged_fraction"] == 0.0


@slow
def test_potts_population_on_er_graphs():
    """ER graphs, q = 3, c = 5 (c~ = c: beta_stab = -ln(1 + 3/4) = -0.5596).  At beta = -0.9 < beta_stab both the
    planted and the random population order ("same" fixed point) with Phi above the paramagnetic closed form; at
    beta = -0.3 the planted population falls back to chi = 1/q and reproduces the closed form within its band.  BP on
    sampled ER graphs (N = 3000, 4 seeds) agrees with the population within 0.01 (finite-N allowance; the joint 99%
    bands are about 1e-3)."""
    e = catalog()["potts_bethe"]
    kw = dict(graph="poisson", c=5, q=3, pop=3000, sweeps=20, N=3000, seeds=[0, 1, 2, 3])
    v = e.run("cavity", beta="-0.9", **kw)
    assert v["pd_status"] == "converged" and v["pd_phase"] == "same" and v["b0_pd_planted"] > 0.8, v
    assert v["phi_pd_planted"] > v["phi_para"] + 0.1 and v["bp_status"] == "converged", v
    assert abs(v["phi_planted_bp"] - v["phi_pd_planted"]) < 0.01, v
    v = e.run("cavity", beta="-0.3", **kw)
    assert v["pd_status"] == "converged" and abs(v["b0_pd_planted"] - 1 / 3) < 1e-3, v
    assert abs(v["phi_pd_planted"] - v["phi_para"]) < 0.01 and abs(v["phi_random_bp"] - v["phi_para"]) < 0.01, v
