"""Perceptron with a margin LAW (families/margin_law.py, derive_margin.py, numerics/margin_perceptron.py).

kappa_l iid per pattern from a stated law, averaged INSIDE the replica integrals.  Checks:
  (a) the degenerate laws (constant, Gaussian with s = 0, one-atom or repeated-atom mixtures)
      reproduce the catalog: capacity to 1e-15, replicon to 1e-8 (observed ~1e-17), the saddle,
      the derived equations symbolically, the derived solve;
  (b) the Gaussian offset collapses to the catalog at q -> 1, alpha_c(law)/(1 - a^2) =
      alpha_c^catalog(kappa~ = -h), for eight (a, h) pairs, kappa~ from -2 to 2 (the banked
      internal check reference/perceptron.json margin_law_collapse);
  (c) independent finite-N sampling: the least-norm storage threshold of patterns and margins
      drawn at N = 60 ... 400, extrapolated, against the RS capacity inside the package's
      band (numerics/finite_size);
  (d) planted controls, each must be caught: a wrong variance of the merged Gaussian, kappa_l
      replaced by its mean (not averaged), kappa averaged per replica (inside the replica
      product).
Other: level (a) against the family numerics (Gaussian, mixture, Ising), the unmerged
two-dimensional audit of the variance q + s^2, the (a) / (b) symbolic agreement, the named
refusals, the CLI / catalog route.
"""
import json
import os
import sys
import tempfile
from contextlib import contextmanager

import mpmath as mp
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant import agree, cli, derive as D, derive_margin as DM
from replicant.catalog import catalog, check_ref
from replicant.compose import compose
from replicant.core import Refusal
from replicant.derive_families import ALPHA, KAPPA
from replicant.derive_replicon import derived_replicon
from replicant.families import margin_law as ML
from replicant.families.perceptron import _ising_q, _gauss, gardner_capacity, gardner_entropy_point
from replicant.numerics import margin_perceptron as MP
from replicant.solve_derived import solve_derived_rs
from replicant.spec import ModelSpec
from replicant.testing import slow

DPS = 20
AH = [(0.3, -0.5), (0.6, 0.4), (0.8, 1.2), (0.1, -2.0), (0.5, 2.0), (0.9, -1.0), (0.0, 0.3),
      (0.7, -0.3)]          # kappa~ = -h: +0.5, -0.4, -1.2, 2, -2, 1, -0.3, 0.3


def _spec(**params):
    return ModelSpec("perceptron", params.pop("variables", "spherical"), "gaussian",
                     DM.CONSTRAINT, None, params)


def _refusal(fn):
    try:
        fn()
    except Refusal as r:
        return r
    raise AssertionError("no Refusal")


@contextmanager
def planted(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


# ---- (a) the degenerate law is the catalog -------------------------------------------------
def test_degenerate_laws_reproduce_the_catalog_capacity_and_replicon():
    with mp.workdps(DPS):
        for k in (-2, -0.5, 0, 0.7, 2):
            cat = gardner_capacity("spherical", k)
            for law in (ML.constant(k), ML.gaussian(k, 0), ML.mixture([1], [k]),
                        ML.mixture([0.25, 0.75], [k, k])):
                c = ML.law_capacity(law)
                assert abs(c.alpha_c - cat.alpha_c) < 1e-15, (k, law)
                assert abs(c.replicon - cat.replicon) < 1e-8, (k, law)
                assert c.stable == cat.stable
        S0, p0 = gardner_entropy_point("0.9", "spherical", "0.5")
        for law in (ML.constant("0.5"), ML.gaussian("0.5", 0), ML.mixture([0.5, 0.5], [0.5, 0.5])):
            S, p = ML.law_entropy_point("0.9", law)
            assert abs(S - S0) < 1e-14 and abs(p.q - p0.q) < 1e-14
            assert abs(p.replicon - p0.replicon) < 1e-8


def test_degenerate_law_derivation_is_the_catalog_derivation():
    cat = agree.derived_rs_equations(agree.default_spec("perceptron"))
    for params in ({"kappa_law": "constant"}, {"kappa_law": "gaussian", "kappa_s": 0},
                   {"kappa_law": "mixture", "kappa_weights": (1,), "kappa_values": (KAPPA,)}):
        if params["kappa_law"] == "mixture":
            continue            # a mixture atom is a number; the symbolic kappa is the catalog's
        mine = agree.derived_rs_equations(_spec(**params, kappa=KAPPA, alpha=ALPHA))
        assert agree.compare_equations(mine, cat, route="symbolic").agree, params
    with mp.workdps(DPS):
        a = solve_derived_rs(agree.default_spec("perceptron"), alpha="0.5", kappa="0.5", dps=DPS)
        b = solve_derived_rs(_spec(kappa_law="constant"), alpha="0.5", kappa="0.5", dps=DPS)
        assert abs(a.values["q"] - b.values["q"]) < 1e-18
        assert abs(a.free_energy - b.free_energy) < 1e-18
        assert abs(a.replicon - b.replicon) < 1e-8
        assert a.stable == b.stable is True


# ---- (b) the Gaussian offset collapses to the catalog at q -> 1 ----------------------------
def test_gaussian_offset_collapses_to_the_catalog_capacity():
    assert len(AH) >= 6 and min(-h for _, h in AH) < 0 < max(-h for _, h in AH)
    with mp.workdps(DPS):
        for a, h in AH:
            law = ML.offset_law(a, h)
            factor, kt = ML.collapse(law)
            assert abs(factor - (1 - mp.mpf(str(a))**2)) < 1e-18 and abs(kt + mp.mpf(str(h))) < 1e-18
            cat = gardner_capacity("spherical", -mp.mpf(str(h))).alpha_c
            got = ML.law_capacity(law).alpha_c / (1 - mp.mpf(str(a))**2)      # alpha_eff = alpha/(1-a^2)
            assert abs(got - cat) < 1e-15, (a, h, got, cat)
            assert ML.collapse_residual(law) < 1e-15, (a, h)


def test_replicon_at_capacity_does_not_collapse():
    """The capacity collapses, the replicon does not: 1 - alpha_c H(-kappa~) with the law's
    alpha_c = (1 - a^2) alpha_c^catalog, not the catalog's 1 - alpha_c^cat H(-kappa~)."""
    with mp.workdps(DPS):
        law = ML.offset_law(0.6, 0.4)
        c, cat = ML.law_capacity(law), gardner_capacity("spherical", "-0.4")
        assert abs(c.replicon - (1 - c.alpha_c * ML._H(mp.mpf("0.4")))) < 1e-15
        assert abs(c.replicon - cat.replicon) > 0.1


def test_the_collapse_is_a_banked_internal_check():
    e = catalog()["perceptron"]
    refs = [r for r in e.refs if r.key == "margin_law_collapse"]
    assert len(refs) == 1
    out = check_ref(e, refs[0])
    assert out.ok, out


# ---- the family numerics against the unmerged two-dimensional audit ------------------------
def test_merged_variance_matches_the_unmerged_two_dimensional_integral():
    with mp.workdps(DPS):
        for law in (ML.gaussian("0.5", "0.8"), ML.gaussian("-1", "1.7"),
                    ML.mixture([0.3, 0.7], ["-0.5", 1])):
            for q in ("0.2", "0.6"):
                a = ML._alpha_of(law, mp.mpf(q))
                assert abs(a / ML.alpha_unmerged(law, q) - 1) < 1e-10, (law, q)


def test_saddle_alpha_tends_to_the_capacity_at_q_to_1():
    with mp.workdps(DPS):
        for law in (ML.gaussian("0.5", "0.8"), ML.mixture([0.5, 0.5], [0, "1.5"])):
            ac = ML.law_capacity(law).alpha_c
            assert abs(ML._alpha_of(law, 1 - mp.mpf("1e-9")) / ac - 1) < 1e-6


# ---- level (a) against level (b) -----------------------------------------------------------
def _level_a_failures(spec, run, family_law, alpha, tol=1e-9, dps=15):
    """Derived RS q, entropy and replicon against the family numerics at the same load."""
    with mp.workdps(dps):
        try:
            sol = solve_derived_rs(spec, alpha=alpha, dps=dps, **run)
        except Refusal as r:                     # equations with no saddle where the family has one
            return [f"derived solve refused: {r.reason}"]
        S, pt = ML.law_entropy_point(alpha, family_law)
        fails = []
        for name, got, want in (("q", sol.values["q"], pt.q), ("S", sol.free_energy, S),
                                ("replicon", sol.replicon, pt.replicon)):
            if abs(got - want) > tol:
                fails.append(f"{name}: derived {mp.nstr(got, 12)} vs family {mp.nstr(want, 12)}")
        return fails


GAUSS = (_spec(kappa_law="gaussian"), {"kappa": "0.5", "kappa_s": "0.8"},
         lambda: ML.gaussian("0.5", "0.8"), "0.4")
MIXED = (_spec(kappa_law="mixture", kappa_weights=(0.3, 0.7), kappa_values=(-1.0, 0.5)), {},
         lambda: ML.mixture([0.3, 0.7], [-1, "0.5"]), "0.5")


def test_derived_gaussian_law_matches_the_family():
    spec, run, law, alpha = GAUSS
    assert not _level_a_failures(spec, run, law(), alpha)


def test_derived_mixture_matches_the_family_with_atoms_at_or_below_zero():
    """Atoms kappa <= 0 are numbers, where sympy rewrites erfc(-y) as 2 - erfc(y) at
    construction (_derived_num._unflip undoes it); without it the solve finds no root."""
    spec, run, law, alpha = MIXED
    assert not _level_a_failures(spec, run, law(), alpha)


@slow
def test_derived_mixture_with_an_atom_exactly_at_zero_matches_the_family():
    zero = _spec(kappa_law="mixture", kappa_weights=(0.5, 0.5), kappa_values=(0, 1.5))
    assert not _level_a_failures(zero, {}, ML.mixture([0.5, 0.5], [0, "1.5"]), "0.3", 1e-12, 20)


def test_derived_and_family_equations_agree_symbolically_and_a_wrong_variance_does_not():
    spec = _spec(kappa_law="gaussian", kappa=KAPPA, kappa_s=DM.KAPPA_S, alpha=ALPHA)
    a = agree.derived_rs_equations(spec)
    assert agree.compare_equations(a, ML.rs_equations_symbolic()).agree
    q, ks = sp.Symbol("q", positive=True), DM.KAPPA_S
    bad = [sp.Eq(e.lhs, e.rhs.xreplace({sp.sqrt(q + ks**2): sp.sqrt(q) + ks}))
           for e in ML.rs_equations_symbolic()]
    assert not agree.compare_equations(a, bad).agree


def test_derived_replicon_uses_the_merged_split_at_a_given_saddle():
    spec, run, law, alpha = GAUSS
    with mp.workdps(DPS):
        S, pt = ML.law_entropy_point(alpha, law())
        rep = derived_replicon(_spec(kappa_law="gaussian"), {"q": pt.q, "alpha": alpha,
                                                          **{k: mp.mpf(v) for k, v in run.items()}})
        assert abs(rep.value - pt.replicon) < 1e-12 and "gauss-covariance" in rep.route


@slow
def test_ising_law_equations_match_an_independent_hand_formula():
    """The level-(a) hook is variable-blind: the Ising law is derived with the same pattern
    term and the Ising site.  Independent target: alpha(qhat) = qhat (1-q)/E_kappa E_u E(x)^2
    and S = -qhat(1-q)/2 + E ln 2cosh + alpha E ln H, with q(qhat) from perceptron._ising_q
    and the law integrals of margin_law (the merge audited above)."""
    law = ML.gaussian("0.2", "0.5")
    with mp.workdps(DPS):
        qhat = mp.mpf("1.3")
        q, omq = _ising_q(qhat)
        T = ML._TLaw(law, q, omq)
        alpha = qhat * omq / T.eh2()
        g = mp.sqrt(qhat)
        S = (-qhat * omq / 2 + _gauss(lambda z: mp.log(2 * mp.cosh(g * z)), [-30 / g, 0, 30 / g])
             + alpha * T.elnh())
        sol = solve_derived_rs(_spec(kappa_law="gaussian", variables="ising"), alpha=alpha,
                               kappa="0.2", kappa_s="0.5", dps=DPS)
        assert abs(sol.values["q"] - q) < 1e-12 and abs(sol.free_energy - S) < 1e-12
        deg = solve_derived_rs(_spec(kappa_law="constant", variables="ising"), alpha="0.3",
                               kappa="0.2", dps=DPS)
        cat = solve_derived_rs(agree.default_spec("perceptron_ising"), alpha="0.3", kappa="0.2",
                               dps=DPS)
        assert abs(deg.free_energy - cat.free_energy) < 1e-18 and abs(deg.replicon - cat.replicon) < 1e-8


def test_composer_runs_the_law_spec_and_reports_the_adapted_step():
    spec = _spec(kappa_law="gaussian", kappa=KAPPA, kappa_s=DM.KAPPA_S, alpha=ALPHA)
    calc = compose(spec)
    r = calc.report
    assert r.kind == "composed" and r.nearest == "perceptron"
    assert [a.step for a in r.adapted] == ["average-disorder"]
    assert r.adapted[0].now == DM.CONSTRAINT
    res = calc.run(alpha="0.4", kappa="0.5", kappa_s="0.8", dps=15)
    assert res.stable is True and abs(res.values["q"] - ML.law_entropy_point(
        "0.4", ML.gaussian("0.5", "0.8"))[1].q) < 1e-12


# ---- named refusals ------------------------------------------------------------------------
def test_unsupported_laws_are_refused_by_name():
    bad = [dict(kappa_law="lognormal"), dict(kappa_law="gaussian", kappa_s=-1),
           dict(kappa_law="gaussian"),                                       # needs kappa_s
           dict(kappa_law="gaussian", kappa_s="inf"), dict(kappa_law="gaussian", kappa_s=True),
           dict(kappa_law="constant", kappa_s=1),                            # not read: refused
           dict(kappa_law="mixture", kappa=1, kappa_weights=(1,), kappa_values=(1,)),
           dict(kappa_law="mixture", kappa_weights=(0.6, 0.6), kappa_values=(1, 2)),
           dict(kappa_law="mixture", kappa_weights=(1.5, -0.5), kappa_values=(1, 2)),
           dict(kappa_law="mixture", kappa_weights=(0.5, 0.5), kappa_values=(1,)),
           dict(kappa_law="mixture", kappa_weights=(), kappa_values=()),
           dict(kappa_law="mixture", kappa_weights=(0.5, 0.5)),              # no values
           dict(kappa_law="mixture", kappa_weights=(1,), kappa_values=("nan",)),
           dict(kappa_law="mixture", kappa_weights=(1 / 65,) * 65, kappa_values=(1,) * 65),
           dict(kappa_s=1), dict(kappa_values=(1,))]                         # no kappa_law
    for p in bad:
        r = _refusal(lambda: ML.parse_margin_law(p))
        assert r.reason == "precondition:supported-margin-law", (p, r.reason)
    assert ML.parse_margin_law({"kappa": 1}) is None          # the catalog's constant margin
    assert _refusal(lambda: ML.offset_law(1.0, 0)).reason == "precondition:supported-margin-law"
    assert _refusal(lambda: ML.collapse(ML.mixture([1], [0]))).reason == \
        "precondition:supported-margin-law"
    for p in bad:      # the same refusals through the level-(a) derivation
        if p.get("kappa_law") and p["kappa_law"] != "gaussian" or p.get("kappa_law") == "gaussian" \
                and p.get("kappa_s") is not None:
            r = _refusal(lambda: agree.derive_rs(_spec(**p)))
            assert r.reason == "precondition:supported-margin-law", (p, r.reason)
    assert _refusal(lambda: agree.derive_rs(_spec())).reason == "precondition:supported-margin-law"


def test_weight_must_be_a_function_of_lambda_minus_kappa():
    g = sp.Lambda((DM._LAM, DM._KL), sp.Heaviside(DM._LAM - 2 * DM._KL))
    assert _refusal(lambda: DM.shift_form(g)).reason == "precondition:margin-shift-form"
    assert DM.shift_form(DM.weight())(0.3) == 1


def test_other_constraints_and_1rsb_are_refused_by_name():
    r = _refusal(lambda: compose(ModelSpec("perceptron", "spherical", "gaussian", "margin law")))
    assert r.reason == "missing-step:constraint:margin law"
    assert "margin kappa law" in r.detail
    spec = _spec(kappa_law="gaussian", kappa=KAPPA, kappa_s=DM.KAPPA_S, alpha=ALPHA)
    r = _refusal(lambda: compose(spec, ansatz="1rsb").run(alpha="0.4", kappa="0.5",
                                                          kappa_s="0.8", dps=15))
    assert r.reason.startswith("not-implemented:1rsb")


def test_catalog_route_and_cli():
    e = catalog()["perceptron"]
    p = cli.parse_param
    _, res = cli.build_result("perceptron", "replica", dict([
        p("kappa_law=gaussian"), p("kappa=-0.5"), p("kappa_s=0.75"), p("alpha=1.0")]))
    with mp.workdps(DPS):
        want = ML.law_capacity(ML.gaussian("-0.5", "0.75")).alpha_c
    assert abs(mp.mpf(res.values["alpha_c"]) - want) < 1e-14
    assert mp.mpf(res.values["collapse_residual"]) < 1e-13 and res.stable is True
    assert p("kappa_weights=0.25,0.75")[1] == ("0.25", "0.75")
    spec, out = cli.build_result("perceptron", "replica", dict([p("kappa_law=mixture"),
                                 p("kappa_weights=0.5,0.5"), p("kappa_values=0,1.5")]))
    assert abs(mp.mpf(out.values["alpha_c"]) - mp.mpf("0.53660260410596949399")) < 1e-14
    r = _refusal(lambda: cli.build_result("perceptron", "cavity", {"kappa_law": "gaussian"}))
    assert r.reason == "precondition:known-parameter"
    r = _refusal(lambda: cli.build_result("perceptron", "replica", dict([
        p("kappa_law=gaussian"), p("kappa_values=1,2")])))
    assert r.reason == "precondition:supported-margin-law"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "spec.json")
        json.dump({"family": "perceptron", "variables": "spherical", "disorder": "gaussian",
                   "constraint": "margin kappa law",
                   "params": {"kappa_law": "gaussian", "kappa": None, "kappa_s": None,
                              "alpha": None}}, open(path, "w"))
        spec = cli.load_spec(path)[1]
        assert spec.params["kappa_s"] == DM.KAPPA_S
        r = compose(spec).run(alpha="0.4", kappa="0.5", kappa_s="0.8", dps=15)
        assert r.stable is True


# ---- (c) independent finite-N sampling -----------------------------------------------------
def _sampled_band(law, sizes, samples, seed=20261028):
    return MP.capacity_finite_size(law, sizes, samples, seed)


def test_least_norm_point_is_exact_on_a_tiny_problem():
    import numpy as np
    G = np.array([[1.0, 0.0], [0.0, 1.0]])
    x, n2 = MP.min_norm_point(G, np.array([1.0, 2.0]))
    assert abs(n2 - 5) < 1e-12 and np.allclose(x, [1, 2])
    assert MP.min_norm_point(np.array([[1.0], [-1.0]]), np.array([1.0, 1.0])) is None


def _brackets_and_excludes_planted(law, sizes, samples):
    ac = float(ML.law_capacity(law).alpha_c)
    fs = _sampled_band(law, sizes, samples)
    assert not fs.excludes(ac), (law.describe(), ac, fs.e_inf, fs.band, fs.variants)
    with mp.workdps(DPS):
        mean_only = float(ML.law_capacity(ML.constant(law.mean())).alpha_c)   # kappa not averaged
    assert fs.excludes(mean_only), (law.describe(), mean_only, fs.e_inf, fs.band)
    return fs


def test_sampling_brackets_the_rs_capacity_and_excludes_the_planted_values():
    """Mixture kappa in {0, 1.5} (RS capacity 0.5366) and the Gaussian offset (a, h) = (0.3,
    -0.5), kappa~ = 0.5 (0.8747): convex, ball = sphere, so the least-norm threshold is the
    sphere's."""
    sizes, samples = (60, 120, 240), {60: 80, 120: 40, 240: 20}
    _brackets_and_excludes_planted(ML.mixture([0.5, 0.5], [0, "1.5"]), sizes, samples)
    _brackets_and_excludes_planted(ML.gaussian("0.5", "0.8"), sizes, samples)


@slow
def test_sampling_brackets_the_rs_capacity_at_larger_sizes_and_for_negative_kappa_tilde():
    """Gaussian margin 0.5 + 0.8 g at N up to 320; and (a, h) = (0.6, 0.4), kappa~ = -0.4,
    alpha_c = 2.54 > 2: the threshold is the ball's (minimum-norm point, an upper bound of the
    sphere's), RS replicon 0.126 > 0 there."""
    _brackets_and_excludes_planted(ML.gaussian("0.5", "0.8"), (80, 160, 320),
                                   {80: 150, 160: 80, 320: 40})
    law = ML.offset_law(0.6, 0.4)
    assert ML.law_capacity(law).replicon > 0.1 and ML.law_capacity(law).alpha_c > 2
    _brackets_and_excludes_planted(law, (60, 120, 200), {60: 100, 120: 50, 200: 30})


# ---- (d) planted controls: each must be caught ---------------------------------------------
def _merged_variance_failures(law, q="0.5"):
    """Levels (b): the unmerged 2-D integral and the q -> 1 limit against the capacity."""
    with mp.workdps(DPS):
        fails = []
        if abs(ML._alpha_of(law, mp.mpf(q)) / ML.alpha_unmerged(law, q) - 1) > 1e-6:
            fails.append("merged variance != unmerged 2-D integral")
        if abs(ML._alpha_of(law, 1 - mp.mpf("1e-9")) / ML.law_capacity(law).alpha_c - 1) > 1e-4:
            fails.append("alpha(q -> 1) != capacity")
        return fails


def test_planted_wrong_merged_variance_is_caught_at_level_b():
    law = ML.gaussian("0.5", "0.8")
    assert not _merged_variance_failures(law)
    for wrong in (lambda q, s: mp.sqrt(q) + s, lambda q, s: mp.sqrt(q + s), lambda q, s: mp.sqrt(q)):
        with planted(ML, "_merged_std", wrong):
            assert len(_merged_variance_failures(law)) == 2, wrong
    with planted(ML, "_merged_std", lambda q, s: mp.sqrt(q + s * s)):
        assert not _merged_variance_failures(law)


def _wrong_variance_caught(wrong):
    spec, run, law, alpha = GAUSS
    with planted(DM, "merge_independent", wrong):
        return _level_a_failures(spec, run, law(), alpha)       # q, S, replicon move or no saddle


def test_planted_wrong_merged_variance_is_caught_at_level_a():
    spec, run, law, alpha = GAUSS
    assert not _level_a_failures(spec, run, law(), alpha)
    sqrt_q_plus_s = lambda cs: sum(abs(sp.sympify(c)) for c in cs)       # sqrt(q) + s
    assert _wrong_variance_caught(sqrt_q_plus_s), "wrong merged variance went undetected"


@slow
def test_planted_wrong_merged_variance_q_plus_s_is_caught_at_level_a():
    q_plus_s = lambda cs: sp.sqrt(sp.sympify(cs[0])**2 + sp.sympify(cs[1]))         # q + s
    assert _wrong_variance_caught(q_plus_s), "wrong merged variance went undetected"


def test_planted_kappa_not_averaged_is_caught_at_level_b():
    law = ML.offset_law(0.6, 0.4)
    assert ML.collapse_residual(law) < 1e-13
    with mp.workdps(DPS):
        mean = law.mean()
    with planted(ML.MarginLaw, "components",
                 lambda self: ((mp.mpf(1), mean, mp.mpf(0)),)):             # kappa_l -> its mean
        assert ML.collapse_residual(law) > 0.1


def test_planted_kappa_not_averaged_is_caught_at_level_a():
    orig = DM.components

    def mean_only(law):                    # kappa_l replaced by its mean inside the integrals
        comps = orig(law)
        return [(sp.S(1), sp.Add(*(w * m for w, m, _ in comps)), sp.S(0))]
    spec, run, law, alpha = GAUSS
    assert not _level_a_failures(spec, run, law(), alpha)
    with planted(DM, "components", mean_only):
        assert _level_a_failures(spec, run, law(), alpha), "kappa replaced by its mean undetected"


@slow
def test_planted_kappa_not_averaged_is_caught_at_level_a_for_the_mixture():
    orig = DM.components
    mean_only = lambda law: [(sp.S(1), sp.Add(*(w * m for w, m, _ in orig(law))), sp.S(0))]
    spec, run, law, alpha = MIXED
    with planted(DM, "components", mean_only):
        assert _level_a_failures(spec, run, law(), alpha), "kappa replaced by its mean undetected"


def test_planted_kappa_averaged_per_replica_is_caught_at_level_a():
    """The margin average taken inside the replica product ([E_kappa E_z g]^n instead of
    E_kappa [E_z g]^n): a different n -> 0 limit (ln of the mean, not the mean of ln)."""
    def annealed(e, ctx):
        g, cov, nn, law = e.args
        q = ctx.free_of_indices(ctx.rs(cov), "covariance")
        G, mean = DM.shift_form(g), 0
        for w, m, s in DM.components(law):
            c = DM.merge_independent([sp.sqrt(q), s])
            mean += w * D.gauss.expect_closed(G(c * D._T + sp.sqrt(1 - q) * D._Z - m), D._Z, [])
        return sp.log(D.gauss.GaussE(mean**nn, D._T))
    spec, run, law, alpha = MIXED
    assert not _level_a_failures(spec, run, law(), alpha)
    rules = list(D.RS_RULES)
    D.RS_RULES[:] = [(DM.LnGaussProdLaw, annealed)]
    try:
        fails = _level_a_failures(spec, run, law(), alpha)
    finally:
        D.RS_RULES[:] = rules
    assert fails, "kappa averaged per replica went undetected"
