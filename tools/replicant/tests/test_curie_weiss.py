"""K1: the Curie-Weiss ferromagnet (KZ Ch. 1, App. 1.B-1.C): meanfield/curie_weiss.py, meanfield/cw_exact.py and the
`curie_weiss` catalog entry.

Convention: H_N = -(1/2N) sum_ij S_i S_j - h sum_i S_i (KZ Eq. (1.1)), J = 1, Phi = lim ln Z_N / N.  Banked:
reference/meanfield.json (cw_beta_c printed; the spinodal read off Fig. 1.B.4, whose printed closed form is an
erratum).  Independent routes: the certified root count, a grid sign-change count bisected in h, the exact
binomial Z_N in arb.  Planted errors go through the modules' _HOOKS, restored with identity asserts.
"""
import ast
import os
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import mpmath as mp
import numpy as np

from replicant.core import Refusal
from replicant.meanfield import curie_weiss as CW
from replicant.meanfield import cw_exact as CX
from replicant.refdata import load

BANK = load("meanfield")
POINTS = (("3/2", "1/20"), ("1/2", "1/5"), ("2", "-1/10"), ("3/2", "0"), ("4/5", "0"))


def _planted(module, key, value, fn):
    reg = module._HOOKS
    old = reg[key]
    reg[key] = value
    try:
        return fn()
    finally:
        reg[key] = old
        assert module._HOOKS is reg and reg[key] is old


def _refusal(fn):
    try:
        fn()
    except Refusal as r:
        return r
    raise AssertionError("expected a Refusal")


def _count_roots(beta, h, grid=np.linspace(-1 + 1e-9, 1 - 1e-9, 40001)):
    f = grid - np.tanh(beta * (grid + h))
    return int(np.sum(np.sign(f[1:]) != np.sign(f[:-1])))


def _numerical_spinodal(beta):
    """The field where the grid sign-change count of m - tanh beta (m + h) drops from 3 to 1 (bisection)."""
    lo, hi = 1e-6, 1.0                       # h = 0 puts m = 0 on a grid node (sign 0)
    assert _count_roots(beta, lo) == 3 and _count_roots(beta, hi) == 1
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _count_roots(beta, mid) == 3 else (lo, mid)
    return (lo + hi) / 2


def test_cw_critical_point_and_spinodal():
    want = BANK["cw_beta_c"]
    with mp.workdps(30):
        assert abs(CW.cw_critical_beta().mid() - mp.mpf(want.value)) <= mp.mpf(want.tol)
    # independent: the certified root count changes at beta = 1 (h = 0)
    assert len(CW.cw_fixed_points("99/100", 0)) == 1 and len(CW.cw_fixed_points("101/100", 0)) == 3
    for beta in ("3/2", "2", "4"):
        closed = float(CW.cw_spinodal(beta).mid())
        assert abs(closed - _numerical_spinodal(float(Fraction(beta)))) < 1e-6, beta
        # the certified root count flips across the closed form
        assert len(CW.cw_fixed_points(beta, f"{closed - 1e-6:.12f}")) == 3
        assert len(CW.cw_fixed_points(beta, f"{closed + 1e-6:.12f}")) == 1
        # planted: the wrong sign of the atanh term must fail
        bad = float(_planted(CW, "spinodal_sign", 1, lambda: CW.cw_spinodal(beta)).mid())
        assert abs(bad - _numerical_spinodal(float(Fraction(beta)))) > 0.5, (beta, bad)
    # the bank's figure reading at beta = 1.5 holds the corrected form, not the printed (erratum) one
    fig = BANK["cw_spinodal_fig1b4_beta1p5"]
    s = np.sqrt(1 - 1 / 1.5)
    printed = np.sqrt((1 / 1.5) * (1 - 1 / 1.5)) - np.arctanh(s) / 1.5
    assert abs(float(CW.cw_spinodal("3/2").mid()) - float(fig.value)) <= float(fig.tol)
    assert abs(printed - float(fig.value)) > 10 * float(fig.tol), printed
    assert _refusal(lambda: CW.cw_spinodal("1")).reason == "precondition:ordered-phase"


def test_cw_three_toolboxes_agree():
    """Theorem 1 (certified), the cavity potential and the field-theoretic saddle give the same Phi and m at every
    point; KZ p. 18's typo (beta h m counted twice in g) planted in the field route breaks the agreement at h != 0."""
    for beta, h in POINTS:
        sol = CW.cw_solution(beta, h)
        with mp.workdps(30):
            phi = sol.phi.mid()
            mc, pc = CW.cw_free_entropy_cavity(beta, h)
            mf, pf = CW.cw_free_entropy_field(beta, h)
            assert abs(pc - phi) < mp.mpf("1e-25") and abs(pf - phi) < mp.mpf("1e-25"), (beta, h, phi, pc, pf)
            m = (sol.selected or sol.fixed_points[-1]).m.mid()
            assert abs(abs(mc) - abs(m)) < mp.mpf("1e-20") and abs(abs(mf) - abs(m)) < mp.mpf("1e-20")
    with mp.workdps(30):
        bad = _planted(CW, "field_h", 2, lambda: CW.cw_free_entropy_field("3/2", "1/20")[1])
        assert abs(bad - CW.cw_free_entropy("3/2", "1/20").mid()) > mp.mpf("1e-3"), bad
    # the magnetization is the h-derivative of Phi (Eq. (1.12)), by a centred difference of the certified Phi
    with mp.workdps(30):
        d = (CW.cw_free_entropy("3/2", "0.0501").mid() - CW.cw_free_entropy("3/2", "0.0499").mid()) / mp.mpf("0.0002")
        assert abs(d / mp.mpf(1.5) - CW.cw_magnetization("3/2", "1/20").mid()) < mp.mpf("1e-7")
    assert _refusal(lambda: CW.cw_magnetization("3/2", 0)).reason == "ambiguous:branch"
    assert CW.cw_magnetization("3/2", 0, "upper").mid() > 0


def _closure(tree, root):
    """Module-level functions / classes reachable from `root` through calls and name references."""
    defs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    seen, todo = set(), [root]
    while todo:
        name = todo.pop()
        if name in seen or name not in defs:
            continue
        seen.add(name)
        todo += [n.id for n in ast.walk(defs[name]) if isinstance(n, ast.Name)]
    return seen


def test_cw_toolboxes_share_no_code():
    """The three routes' call graphs inside curie_weiss.py are disjoint (they share only imported helpers such as
    lattice.ball.rational, which parses the inputs); the walk detects a shared helper in a planted source."""
    tree = ast.parse(Path(CW.__file__).read_text())
    var = _closure(tree, "cw_solution")
    cav = _closure(tree, "cw_free_entropy_cavity")
    fld = _closure(tree, "cw_free_entropy_field")
    assert {"cw_fixed_points", "_roots_at", "_newton"} <= var, var
    assert not var & cav and not var & fld and not cav & fld, (var & cav, var & fld, cav & fld)
    planted = ast.parse("def _h(x):\n    return x\n\ndef a(x):\n    return _h(x)\n\ndef b(x):\n    return _h(x) + 1\n")
    assert _closure(planted, "a") & _closure(planted, "b") == {"_h"}
    # the two rate routes of cw_exact.py are disjoint too (Legendre: _sup_potential, Brent; Gartner-Ellis: its own
    # tilted exponent and golden section)
    tree = ast.parse(Path(CX.__file__).read_text())
    leg, ge = _closure(tree, "_rate_legendre"), _closure(tree, "_rate_gartner_ellis")
    assert {"_sup_potential", "_phi"} <= leg and {"_ge_scgf_sup", "_ge_golden"} <= ge, (leg, ge)
    assert not leg & ge, leg & ge


def test_cw_exact_z_converges():
    """(1/N) ln Z_N -> Phi with N (phi_N - Phi) -> -(1/2) ln(1 - beta (1 - m*^2)) (+ ln 2 at h = 0, beta > 1) at
    O(1/N); the rate at finite N matches the Legendre route at exposed points within Lemma 1-2's window, follows the
    true (non-convex) rate at non-exposed points, and the cavity potential phi~ is not a rate (KZ p. 16)."""
    for beta, h in (("3/2", "1/20"), ("3/2", "-1/20"), ("1/2", "0"), ("3/2", "0")):
        phi, term = CW.cw_free_entropy(beta, h).mid(), CX.cw_finite_size_term(beta, h)
        gaps = []
        for n in (50, 200, 800):
            x = CX.cw_exact(n, beta, h)
            with mp.workdps(30):
                gaps.append(abs(n * (x.phi.mid() - phi) - term))
        assert gaps[2] < gaps[1] < gaps[0] and gaps[2] < 2e-3, (beta, h, gaps)
        assert gaps[0] / gaps[2] > 8, gaps                                   # O(1/N): about 16 over 16x in N
        with mp.workdps(30):                                                 # planted: no finite-size term
            assert abs(800 * (CX.cw_exact(800, beta, h).phi.mid() - phi)) > 0.1
    n = 1000
    bound = 2 * np.log(n * (n + 1)) / n
    for m in (0.9, -0.95, 0.98):                               # exposed at beta = 1.5, h = 0: |m| > m* = 0.8586
        i_n = float(CX.cw_exact_rate(n, str(m), "3/2", 0).mid())
        assert abs(i_n - CX.cw_rate_function(m, "3/2", 0)) < bound, (m, i_n)
    for m in (0.0, 0.5, -0.6):                                 # non-exposed: |m| < m*
        i_n, true = float(CX.cw_exact_rate(n, str(m), "3/2", 0).mid()), CX.cw_ldp_rate(m, "3/2", 0)
        assert CX.cw_rate_function(m, "3/2", 0) < 1e-9 and abs(i_n - true) < bound and true > 0.03, (m, i_n)
    for m, beta, h in ((0.3, "3/2", 0), (0.95, "3/2", "1/20"), (-0.4, "1/2", "1/5"), (0.7, "2", "-1/10")):
        assert abs(CX.cw_rate_function(m, beta, h) - CX.cw_rate_function(m, beta, h, "gartner_ellis")) < 1e-8
    with mp.workdps(30):                                                     # phi~ is not the rate (planted)
        phit = -1.5 * 0.25 / 2 + float(mp.log(2 * mp.cosh(1.5 * 0.5)))
        wrong = float(CW.cw_free_entropy("3/2", 0).mid()) - phit
        i_n = float(CX.cw_exact_rate(n, "0.5", "3/2", 0).mid())
        assert abs(wrong - i_n) > 3 * abs(CX.cw_ldp_rate(0.5, "3/2", 0) - i_n), (wrong, i_n)
    assert _refusal(lambda: CX.cw_exact_rate(10, "0.15", "3/2", 0)).reason == "precondition:magnetization-grid"
    assert _refusal(lambda: CX.cw_rate_function(0.1, "3/2", 0, "saddle")).reason == "unsupported:route"


def test_curie_weiss_entry_routes_and_refusals():
    from replicant.catalog import catalog
    from replicant.compose import compose
    e = catalog()["curie_weiss"]
    rep = e.run("replica", beta="3/2", h="1/20", m="0.5")
    assert rep["branches"] == ["lower", "middle", "upper"] and not rep["stable_middle"]
    with mp.workdps(40):
        assert abs(mp.mpf(rep["phi"]) - rep["phi_field"]) < mp.mpf("1e-25")
    assert rep["rate_ldp"] > rep["rate_legendre"] - 1e-12
    cav = e.run("cavity", beta="3/2", h="1/20", N=400)
    assert abs(mp.mpf(cav["cavity_ratio"]) - cav["phi"]) < 2e-3            # ln Z_(N+1) / Z_N -> Phi
    ex = compose(e.spec, "exact").run(N=50, beta="1/2", h="1/10")
    assert ex.rigor == "certified" and abs(mp.mpf(ex.values["phi_N"]) - CW.cw_free_entropy("1/2", "1/10").mid()) < 0.05
    assert "h_sp" not in e.run("replica", beta="1/2")                       # no spinodal for beta <= 1
    crit = e.run("exact", beta=1, h=0, N=100)                               # finite N is fine at the critical point
    assert "phi_N" in crit and "finite_size_term" not in crit
    with mp.workdps(30):                                                    # beta_c, h = 0: m = 0, Phi = ln 2
        assert abs(mp.mpf(e.run("replica", beta=1, h=0)["phi"]) - mp.log(2)) < mp.mpf("1e-30")
    r = _refusal(lambda: compose(e.spec).run(beta="3/2", field="1"))
    assert r.reason == "precondition:known-parameter"
    guide = (Path(CW.__file__).parent.parent / "GUIDE.md").read_text()
    assert "| `curie_weiss`" in guide and "`curie_weiss` (replica)" in guide
