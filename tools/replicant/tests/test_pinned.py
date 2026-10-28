"""Pinned directions: the joint pair problem (perceptron with fixed external directions).

Default tier: the derived functional, the capacity boundary against the banked paper values and
the catalog perceptron, the support-vector law, the integer-n Hessian check of the stability
sectors (Schur: replicon decoupled from M; the standard block mixes), one RS saddle below
capacity, and planted controls (wrong entropy term, wrong Gram matrix, dropped M-Q mixing) that
must fail.  Slow tier: all five paper states audited against the derived equations, the c average
against the paper's exact margins, the d = 6 .. 30 table.
"""
import functools
import os
import sys

import mpmath as mp
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant import derive_pinned as DP
from replicant import pinned as P
from replicant import pinned_modes as PM
from replicant import refdata
from replicant.agree import derive_rs
from replicant.compose import compose
from replicant.core import Refusal
from replicant.families.perceptron import gardner_capacity
from replicant.pinned_check import hessian_integer_n, sector_matrices
from replicant.testing import slow

REF = refdata.load("pinned")
STATES = ((6, 16), (10, 35), (14, 70), (20, 140), (30, 350))


def _refusal(fn):
    try:
        fn()
    except Refusal as r:
        return r
    raise AssertionError("expected Refusal")


def _ok(key, got):
    res = refdata.check(REF[key], got, label=key)
    assert res.ok, (key, got, REF[key].value, REF[key].tol)


def _d6_point():
    """the d = 6, n = 16, c = 0 state at kappa = 0.9 kappa* (below the capacity 0.30236082064)"""
    return dict(alpha=mp.mpf(14) / 6, d=6, c=0, kappa=mp.mpf("0.9") * mp.mpf("0.3023608206457258"))


@functools.lru_cache(maxsize=None)
def _d6_state():
    """the audited RS saddle of the d = 6 point (computed once for the module)"""
    p = _d6_point()
    return P.state_point(p["alpha"], p["d"], p["c"], p["kappa"], audit=True, dps=12)


@functools.lru_cache(maxsize=None)
def _d6_brute():
    """the integer-n (n = 4) Hessian sectors at the d = 6 saddle (independent code path)"""
    p, sp_ = _d6_point(), _d6_state()
    return sector_matrices(4, hessian_integer_n(4, float(p["alpha"]), p["d"], p["c"],
                                                float(p["kappa"]), float(sp_.q), float(sp_.M)))


# ---- the derivation ---------------------------------------------------------------------
def test_pinned_functional_is_derived_by_the_package_steps():
    spec = DP.pinned_spec()
    eqs, state, trace = derive_rs(spec)
    assert [p.name for p in state["rs_free"].params] == ["q", "M"]
    q, M, c, d, k = (sp.Symbol(x, real=True) for x in ("q", "M", "c", "d", "kappa"))
    phi = state["rs_free"].expr
    # entropic part: (1/2)[ln(1-q) + (q - 2 M^2/(1+c))/(1-q)]  (hand-supplied term, then generic RS / n -> 0)
    ent = phi.subs(sp.Symbol("alpha", positive=True), 0)
    qq = sp.Symbol("q", positive=True)
    want = (sp.log(1 - qq) + (qq - 2 * sp.Symbol("M", real=True)**2 / (1 + c)) / (1 - qq)) / 2
    assert sp.simplify(ent - want) == 0, ent
    assert len(eqs) == 2
    from replicant.agree import default_spec
    assert not DP.is_pinned(default_spec("perceptron")) and DP.is_pinned(spec)


def test_untied_branch_is_refused_by_derivation():
    r = _refusal(P.untied_slack)
    assert r.reason == "no-solution:untied-slack" and "M_2 = M_1*c" in r.detail
    calc = compose(DP.pinned_spec(tied=False))
    assert _refusal(lambda: calc.run(d=6, c=0, n=16)).reason == "no-solution:untied-slack"


def test_composer_accepts_the_pinned_constraint_and_refuses_bad_input():
    calc = compose(DP.pinned_spec())
    assert calc.report.kind == "composed" and calc.report.missing == []
    assert any(a.field == "constraint" for a in calc.report.adapted)
    assert _refusal(lambda: calc.run(d=6, c=1, n=16)).reason == "precondition:independent-pinned-directions"
    assert _refusal(lambda: calc.run(d=6, c=0, n=16, foo=1)).reason == "precondition:known-parameter"
    assert _refusal(lambda: calc.run(c=0, n=16)).reason == "precondition:state-point"
    assert _refusal(lambda: calc.run(d=6, c=0, n=16, alpha=2)).reason == "precondition:consistent-parameters"
    assert _refusal(lambda: P.capacity(0, 6, 0)).reason == "precondition:alpha-range"
    assert _refusal(lambda: P.capacity(2, 0, 0)).reason == "precondition:dimension"
    ising = DP.pinned_spec()
    ising = type(ising)(ising.family, "ising", ising.disorder, ising.constraint, params=ising.params)
    assert _refusal(lambda: derive_rs(ising)).reason == "precondition:pinned-spherical-only"


# ---- acceptance: reduction to the catalog perceptron ------------------------------------------
def test_without_pinned_overlap_it_is_the_catalog_perceptron():
    """M fixed at 0: alpha_c = 1/Psi(-sqrt(d) kappa), the catalog spherical perceptron at the margin
    sqrt(d) kappa, to 1e-15 (and the free-M capacity can only be larger)."""
    for d, kappa in ((6, "0.2"), (14, "0.05"), (10, "-0.1")):
        mp.mp.dps = 30
        got = P.alpha_c(kappa, d, 0, M=0)
        want = mp.mpf(gardner_capacity("spherical", kappa=mp.sqrt(d) * mp.mpf(kappa)).alpha_c)
        assert abs(got - want) < mp.mpf("1e-15"), (d, kappa, got, want)
        assert P.alpha_c(kappa, d, 0) > got
    mp.mp.dps = 15


def test_functional_at_zero_overlap_is_the_catalog_functional():
    """the derived phi(q, M = 0) is the catalog perceptron's phi(q) at kappa_p = sqrt(d) kappa."""
    eqs, state, _ = derive_rs(DP.pinned_spec())
    phi_p = state["rs_free"].expr.subs(sp.Symbol("M", real=True), 0)
    from replicant.agree import default_spec
    phi_c = derive_rs(default_spec("perceptron"))[1]["rs_free"].expr
    pt = {sp.Symbol("alpha", positive=True): sp.Float("1.7"), sp.Symbol("q", positive=True): sp.Float("0.6"),
          sp.Symbol("d", positive=True): 6, sp.Symbol("c", real=True): sp.Float("0.3"),
          sp.Symbol("kappa", real=True): sp.Float("0.1")}
    from replicant.gauss import numeric
    a = numeric(phi_p, pt)
    pc = {s: v for s, v in pt.items() if s.name in ("alpha", "q")}
    pc[sp.Symbol("kappa", real=True)] = sp.sqrt(6) * sp.Float("0.1")
    b = numeric(phi_c, pc)
    assert abs(a - b) < mp.mpf("1e-12"), (a, b)


# ---- acceptance: the paper's m_RS and the support-vector law ---------------------------------------
def test_capacity_is_the_paper_pair_margin():
    for d, n in STATES[:3]:
        cap = P.capacity(mp.mpf(n - 2) / d, d, 0, dps=25)
        _ok(f"m_rs_d{d}_n{n}_c0", cap.kappa)
        _ok(f"a_star_d{d}_n{n}_c0", cap.a)
        _ok(f"h_star_d{d}_n{n}_c0", cap.h)
        assert abs(cap.M - mp.sqrt(mp.mpf(1) / 2) * cap.a) < mp.mpf("1e-14")


def test_support_vector_law_and_the_replicon_counting_identity():
    d, n = 10, 35
    cap = P.capacity(mp.mpf(n - 2) / d, d, 0, dps=25)
    paper_h = mp.mpf(REF["h_star_d10_n35_c0"].value)
    k_paper = (n - 2) * mp.ncdf(-paper_h)            # code/rs_duals.py: k = (n-2) Phi(-h*)
    assert abs(cap.support_vectors() - k_paper) < mp.mpf("1e-6")
    # joint replicon at capacity = 1 - k/d (positive iff fewer than d active constraints)
    assert abs(cap.replicon - (1 - cap.support_vectors() / d)) < mp.mpf("1e-14")
    assert 0 < cap.replicon < 1
    # the catalog slice replicon of the paper, 1 - Phi(-h)/Psi(h), is negative for the same state
    assert 1 - mp.ncdf(-cap.h) / P.psi(cap.h) < 0


def test_capacity_depends_on_the_pair_overlap_through_the_gram_matrix():
    base = P.capacity(mp.mpf(33) / 10, 10, 0, dps=20).kappa
    pos = P.capacity(mp.mpf(33) / 10, 10, mp.mpf("0.5"), dps=20).kappa
    neg = P.capacity(mp.mpf(33) / 10, 10, mp.mpf("-0.5"), dps=20).kappa
    assert neg < base < pos


# ---- the stability sectors -------------------------------------------------------------------
def test_class_formulas_match_the_integer_n_hessian_and_schur():
    p, sp_ = _d6_point(), _d6_state()
    lam, leak, T, tleak = _d6_brute()
    cv = PM.class_values(4, p["alpha"], p["d"], p["c"], p["kappa"], sp_.q, sp_.M, dps=15)
    assert abs(lam - PM.replicon_hess(cv)) < 1e-9 * abs(lam)
    assert leak < 1e-9 * abs(lam)                # the replicon vector does not mix with M (Schur)
    assert tleak < 1e-9 * abs(lam)               # the standard block closes on (e, mu)
    blk = PM.pinned_block(p["alpha"], p["d"], p["c"], p["kappa"], sp_.q, sp_.M, n=4, dps=15, cv=cv)
    for i in range(2):
        for j in range(2):
            assert abs(float(blk.T[i, j]) - T[i, j]) < 1e-9 * abs(T).max()
    assert abs(T[0, 1]) > 1.0 and abs(T[1, 0]) > 1.0     # the mixing is not small


def test_joint_state_point_is_stable_and_not_the_slice_replicon():
    p, sp_ = _d6_point(), _d6_state()
    assert abs(sp_.replicon - mp.mpf("0.475162814705")) < 1e-8
    assert abs(sp_.replicon_derived - sp_.replicon) < 1e-8     # derive_replicon at the audited root
    assert abs(sp_.q - mp.mpf("0.9575464624")) < 1e-8
    assert sp_.longitudinal > 0 and sp_.block.stable and not sp_.block.complex_pair
    assert sp_.stable is True and all(f.state == "stable" for f in sp_.finality().values())
    # the fixed-a slice of the catalog (alpha_eff, -h) at the same capacity point is unstable
    cap = P.capacity(p["alpha"], p["d"], p["c"], dps=15)
    assert 1 - mp.ncdf(-cap.h) / P.psi(cap.h) < 0 < cap.replicon
    assert _refusal(lambda: P.state_point(p["alpha"], 6, 0, cap.kappa * 1.001, audit=False)).reason \
        == "no-solution:above-capacity"


def test_stable_requires_a_positive_replicon():
    """finality follows the computed sign: a negative replicon makes every value's label unstable."""
    import dataclasses
    sp_ = _d6_state()
    bad = dataclasses.replace(sp_, replicon=-abs(sp_.replicon), replicon_derived=None, stable=False)
    assert all(f.state == "unstable" and not f.final for f in bad.finality().values())
    assert all(f.final for f in sp_.finality().values())


# ---- planted controls -----------------------------------------------------------------------
def _planted(key, value, fn):
    old = DP._PLANT.get(key)
    DP._PLANT[key] = value
    try:
        return fn()
    finally:
        if old is None:
            del DP._PLANT[key]
        else:
            DP._PLANT[key] = old


def test_planted_wrong_entropy_term_fails_the_paper_margin():
    """an entropic term blind to the pinned overlap has no interior capacity maximum at all."""
    r = _planted("entropy", "drop_m", lambda: _refusal(lambda: P.capacity(mp.mpf(14) / 6, 6, 0, dps=15)))
    assert r.reason == "no-solution:capacity"
    # and the derived functional changes: phi no longer carries the 2 M^2/(1+c) shift
    eqs = _planted("entropy", "drop_m", lambda: derive_rs(DP.pinned_spec())[0])
    assert "M**2" not in str(eqs[0].lhs.subs(sp.Symbol("alpha", positive=True), 0))


def test_planted_wrong_gram_fails_the_paper_margin_at_nonzero_overlap():
    honest = P.capacity(mp.mpf(33) / 10, 10, mp.mpf("0.5"), dps=20).kappa
    wrong = _planted("gram", "identity", lambda: P.capacity(mp.mpf(33) / 10, 10, mp.mpf("0.5"), dps=20).kappa)
    assert abs(honest - wrong) > 0.02
    # at c = 0 the identity Gram IS the Gram: the control must pass there (it is a c-dependent error)
    c0 = _planted("gram", "identity", lambda: P.capacity(mp.mpf(14) / 6, 6, 0, dps=20).kappa)
    assert abs(c0 - mp.mpf(REF["m_rs_d6_n16_c0"].value)) < 1e-8


def test_planted_dropped_mixing_fails_the_integer_n_check():
    p, sp_ = _d6_point(), _d6_state()
    q, M = sp_.q, sp_.M
    _, _, T, _ = _d6_brute()
    PM._PLANT_MODES["mixing"] = "drop"
    try:
        blk = PM.pinned_block(p["alpha"], p["d"], p["c"], p["kappa"], q, M, n=4, dps=15)
    finally:
        del PM._PLANT_MODES["mixing"]
    assert abs(float(blk.T[0, 1]) - T[0, 1]) > 1.0 and blk.uv == 0
    # the replicon does not use the mixing: dropping it leaves the replicon unchanged (Schur)
    assert PM._PLANT_MODES == {}


def test_pair_overlap_average_one_row_against_the_exact_margins():
    """one row of data/rs_validation.json (d = 10, n = 40) in the default tier, with an 8-node
    Gauss-Jacobi rule for the law of c: the paper code's own c-average to 1e-5, the paper's sampled
    RS mean within sampling noise, the exact margins within the paper's stated bias."""
    from scipy.special import roots_jacobi
    d, n = 10, 40
    x, w = roots_jacobi(8, (d - 3) / 2, (d - 3) / 2)
    w = w / w.sum()
    mean = float(sum(wi * P.capacity(mp.mpf(n - 2) / d, d, mp.mpf(float(c)), dps=15).kappa
                     for wi, c in zip(w, x)))
    assert abs(mean - float(REF["validation_d10_n40_cavg_paper_code"].value)) < 1e-5
    _ok("validation_d10_n40_mean_rs", mean)
    _ok("validation_d10_n40_mean_exact", mean)


# ---- slow tier -----------------------------------------------------------------------------------
@slow
def test_all_paper_states_capacity_and_audited_joint_replicon():
    for d, n in STATES:
        cap = P.capacity(mp.mpf(n - 2) / d, d, 0, dps=30)
        _ok(f"m_rs_d{d}_n{n}_c0", cap.kappa)
        _ok(f"h_star_d{d}_n{n}_c0", cap.h)
        assert cap.replicon > 0 and 1 - mp.ncdf(-cap.h) / P.psi(cap.h) < 0
        for f in (mp.mpf("0.999"), mp.mpf("0.9"), mp.mpf("0.5")):
            sp_ = P.state_point(mp.mpf(n - 2) / d, d, 0, f * cap.kappa, audit=(f == mp.mpf("0.9")))
            assert sp_.stable and sp_.replicon > 0 and sp_.longitudinal > 0 and sp_.block.stable
            if f == mp.mpf("0.9"):
                assert abs(sp_.replicon_derived - sp_.replicon) < 1e-8


@slow
def test_rs_average_over_the_pair_overlap_against_the_exact_margins():
    """E_c[kappa*(c)] over the exact law of c = <v_i, v_j>: equals the paper code's own c-average
    (1e-8), lies within the paper's stated bias of the exact margins and within sampling noise of the
    paper's sampled RS mean."""
    from scipy.special import roots_jacobi
    for d, n in ((10, 40), (14, 80)):
        x, w = roots_jacobi(10, (d - 3) / 2, (d - 3) / 2)
        w = w / w.sum()
        vals = [P.capacity(mp.mpf(n - 2) / d, d, mp.mpf(float(c)), dps=20).kappa for c in x]
        mean = float(sum(wi * v for wi, v in zip(w, vals)))
        _ok(f"validation_d{d}_n{n}_cavg_paper_code", mean)
        _ok(f"validation_d{d}_n{n}_mean_rs", mean)
        _ok(f"validation_d{d}_n{n}_mean_exact", mean)
