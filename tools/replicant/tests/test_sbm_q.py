"""K4: the stochastic block model with q groups (KZ Ch. 10; DKMZ, arXiv:1109.3041).

Convention (inference/sbm_q.py): edge probability c_ab / N, c_aa = c_in, c_ab = c_out, groups of expected size 1/q;
mean degree c = (c_in + (q - 1) c_out)/q; epsilon = c_out / c_in.  The q = 2 entry (inference/sbm.py) has a/N
inside and b/N across: c_in = a, c_out = b, c = (a + b)/2.  Bank: reference/sbm.json.
"""
import ast
import math
import os
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.amp._stats import student_band
from replicant.catalog import catalog, check_ref
from replicant.compose import compose
from replicant.core import Refusal
from replicant.inference import sbm as B
from replicant.inference import sbm_q as SQ
from replicant.inference import sbm_q_pd as PD
from replicant.refdata import check, load
from replicant.testing import slow

BANK = load("sbm")
PKG = Path(__file__).resolve().parents[1]
NEW = ("sbm_q.py", "sbm_q_pd.py", "sbm_learn.py", "nonbacktracking.py")


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


def _imports(src):
    out = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom):
            out.add(("." * node.level) + (node.module or "") + "".join(f":{a.name}" for a in node.names))
        elif isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
    return out


def _uses_sbm(mods):
    """Imports that reach inference/sbm.py: `from . import sbm`, `from .sbm import ...`, `...inference.sbm`."""
    return {m for m in mods if any(part in ("sbm", ".sbm", "inference.sbm") or part.endswith(".sbm")
                                   for part in m.replace(":", " ").split())}


def _as_q(g):
    return SQ.from_edges(g.N, g.edges, (g.labels == -1).astype(int))


# ---- q = 2 against inference/sbm.py (independent code) ------------------------------------------------
def test_q2_matches_existing_sbm():
    """sbm_q_bp at q = 2 on the graphs of sbm.sbm_sample reproduces sbm.sbm_bp (N = 1500, a = 5.5, b = 0.5, seed 3):
    the converged marginals agree to 1e-6 up to the global flip, the overlaps are equal, the field difference
    h_0 - h_1 is +-dh, from the random and from the planted start.  The new modules import nothing from
    inference/sbm.py (AST audit; a planted import is caught).  Planted error: the non-edge field dropped in sbm_q
    (_HOOKS["field"] = 0) moves the marginals by more than 1e-4 (measured: see report)."""
    for f in NEW:
        assert not _uses_sbm(_imports((PKG / "inference" / f).read_text())), f
    assert _uses_sbm(_imports("from . import sbm\n")) and _uses_sbm(_imports("from .sbm import _u\n"))
    assert _uses_sbm(_imports("from ..inference.sbm import sbm_bp\n"))
    assert not _uses_sbm(_imports("from . import sbm_q as SQ\n"))
    g = B.sbm_sample(1500, 5.5, 0.5, 3)
    ref = B.sbm_bp(g, 5.5, 0.5, iters=400, seed=0)
    assert ref.converged and ref.overlap > 0.5
    gq = _as_q(g)

    def gap(r):
        d = [np.max(np.abs(ref.marginals - r.marginals[:, k])) for k in (0, 1)]
        k = int(np.argmin(d))
        return d[k], (r.field[0] - r.field[1]) * (1 if k == 0 else -1)

    for init in ("random", "planted"):
        r = SQ.sbm_q_bp(gq, 2, 5.5, 0.5, init=init, iters=600, seed=0, damping=0.0)
        d, dh = gap(r)
        assert r.converged and d < 1e-6 and abs(r.overlap - ref.overlap) < 1e-12, (init, d, r.overlap)
        assert abs(dh - ref.field) < 1e-6, (dh, ref.field)
    saved = SQ._HOOKS["field"]
    try:
        SQ._HOOKS["field"] = 0.0
        bad = SQ.sbm_q_bp(gq, 2, 5.5, 0.5, init="random", iters=600, seed=0, damping=0.0)
    finally:
        SQ._HOOKS["field"] = saved
    assert SQ._HOOKS["field"] is saved
    assert gap(bad)[0] > 1e-4, gap(bad)


def test_conventions_map_both_ways():
    """(a, b) -> (c_in, c_out) = (a, b): the KS margins agree (q = 2: tau = (a - b)^2/(2(a + b)) =
    (c_in - c_out)^2/(q^2 c) with c = (a + b)/2); the KS point at c = 3 gives the same a/b = c_in/c_out = 1/eps_c
    (bank 3.73) from both modules, and from (c_in_ks, c_out_ks) the mean degree (c_in + c_out)/2 is back at 3.
    Wrong maps fail: reading b as the total cross degree (c_out = b/(q - 1) is the same at q = 2, so the slip is
    tested at q = 4, c = 16 against the bank 2.33) and reading c as c_in + c_out."""
    with mp.workdps(30):
        for a, b in ((5.5, 0.5), (4, 2), (1, 6)):
            assert abs(SQ.sbm_q_kesten_stigum(2, a, b).margin - B.sbm_kesten_stigum(a, b).margin) < mp.mpf("1e-25")
            assert SQ.mean_degree(2, a, b) == mp.mpf(a + b) / 2
        L, L2 = SQ.sbm_q_ks_line(2, 3), B.sbm_ks_line(3)
        assert abs(L.ratio - L2.ratio) < mp.mpf("1e-25") and check(BANK["inverse_eps_c_q2_c3"], L.ratio, label="x").ok
        assert abs((L.c_in + L.c_out) / 2 - 3) < mp.mpf("1e-25")
        L4 = SQ.sbm_q_ks_line(4, 16)
        assert check(BANK["inverse_eps_c_q4_c16"], L4.ratio, label="q4").ok and abs(L4.ratio - mp.mpf(28) / 12) < 1e-25
        assert abs(SQ.mean_degree(4, L4.c_in, L4.c_out) - 16) < mp.mpf("1e-25")
        # slip 1: c_out taken as the total cross degree (q - 1) c_out
        slip = SQ.sbm_q_kesten_stigum(4, L4.c_in, 3 * L4.c_out)
        assert abs(slip.margin - 1) > 0.1
        # slip 2: c read as c_in + c_out (the q = 2 sum) at q = 4
        wrong = (16 + 3 * 2 * mp.sqrt(8)) / (16 - 2 * mp.sqrt(8))
        assert not check(BANK["inverse_eps_c_q4_c16"], wrong, label="slip").ok
    k = SQ.sbm_q_kesten_stigum(5, 0, 20)
    assert k.mean_degree == 16 and k.lambda2 == mp.mpf(-1) / 4 and k.margin == 1 and not k.detectable


def test_sampler_blocks_and_degrees():
    """q = 3, c_in = 2, c_out = 6, N = 3000 (seed 0): edges per block within 4 binomial standard deviations of
    c_ab n_a n_b N (n_a the sampled sizes), mean degree near (c_in + 2 c_out)/3; edges distinct, i < j; the
    sampler is a function of the seed."""
    N, C = 3000, SQ.affinity(3, 2, 6)
    g = SQ.sbm_q_sample(N, 3, 2, 6, 0)
    E, lab = g.edges, g.labels
    assert np.all(E[:, 0] < E[:, 1]) and len(np.unique(E[:, 0] * N + E[:, 1])) == len(E)
    sz = np.bincount(lab, minlength=3)
    for a in range(3):
        for b in range(a, 3):
            got = int(np.sum(((lab[E[:, 0]] == a) & (lab[E[:, 1]] == b)) | ((lab[E[:, 0]] == b) & (lab[E[:, 1]] == a))))
            pairs = sz[a] * (sz[a] - 1) / 2 if a == b else sz[a] * sz[b]
            want = pairs * C[a, b] / N
            assert abs(got - want) < 4 * math.sqrt(want), (a, b, got, want)
    assert abs(2 * len(E) / N - 14 / 3) < 0.2
    assert np.array_equal(SQ.sbm_q_sample(N, 3, 2, 6, 0).edges, E)
    assert not np.array_equal(SQ.sbm_q_sample(N, 3, 2, 6, 1).edges, E)


def test_bethe_at_the_factorized_point():
    """At chi = n (exact fixed point of the degree-balanced model, h = C n = c) the Bethe free entropy (10.23) equals
    the closed form (M/N) ln c - c/2 on the same graph, which is KZ Eq. (10.31) -(c/2)(1 - ln c) when 2M/N = c;
    the BP result labels it an approximation on the loopy sampled graph.  Planted error: dropping the + c/2 term
    of (10.23) is off by c/2."""
    g = SQ.sbm_q_sample(800, 4, 1, 3, 2)
    C, n = SQ.affinity(4, 1, 3), np.full(4, 0.25)
    chi = np.full((2 * len(g.edges), 4), 0.25)
    got = SQ.bethe(g, n, C, chi, n @ C)
    c = float(n @ C @ n)
    assert abs(got - SQ.phi_para(g, n, C)) < 1e-12 and abs(got - (len(g.edges) / 800 * math.log(c) - c / 2)) < 1e-12
    M = c * 800 / 2
    assert abs(M / 800 * math.log(c) - c / 2 - (-(c / 2) * (1 - math.log(c)))) < 1e-12
    assert abs((got - c / 2) - SQ.phi_para(g, n, C)) > 1
    r = SQ.sbm_q_bp(g, 4, 1, 3, init="random", iters=300, seed=0)
    assert r.converged and r.phi_label.startswith("approximation") and abs(r.phi - r.phi_para) < 1e-9


def test_q5_first_order_thresholds():
    """q = 5, c_in = 0 (KZ Fig. 10.4.3, DKMZ Fig. 3: c_d = 12.84, c_c = 13.23, c_l = 16).  Default tier, every
    number computed:
    - c_l, two routes: the KS bisection (sbm_q_ks_degree) equals the bank 16; and population dynamics: one sweep
      from ("tilt", 1e-4) multiplies the planted magnetisation by c lambda_2^2 (derived in sbm_q_pd.py; c / 16
      here), measured over 6 independent named seeds (5000 members each) at c = 12 and 17; each band contains
      c / 16, and the root of the line through the two measured rates is in the propagated 99% band of 16.
    - c_d: the planted start decays to the paramagnet at c = 12.3 and stays informative at 13.0 (1000 members):
      12.3 < c_d < 13.0.
    - c_c: Delta Phi of the planted branch at 14.0 is positive with a 99% Student-t band over 4 independent seeds
      (each the mean of 20 measured sweeps; seeds independent, so the sweep autocorrelation does not enter), so
      c_c < 14; and c_c > 12.3, because at 12.3 there is no informative branch (the planted start decays, Delta
      Phi = 0) and c_c is a zero of Delta Phi on that branch, which first exists at c_d.
    - the random start decays at 14 (phase III: the pair disagrees below c_l).
    The banked c_d, c_c lie in these computed brackets.  Planted error: the mean degree read as c_out
    (_HOOKS["degree"] = 0.8) makes the planted start decay at 13.0 and the growth rate at 17 fall to 0.85, outside
    its band.  The located values are in the slow test."""
    c_l = SQ.sbm_q_ks_degree(5, 0)
    assert check(BANK["c_l_q5_cin0"], c_l, label="c_l").ok
    rate = lambda c: [PD.population(5, 0, c, init=("tilt", 1e-4), pop=5000, sweeps=1, measure=1, seed=s).m / 1e-4
                      for s in range(6)]
    r12, r17 = student_band(rate(12.0)), student_band(rate(17.0))
    assert abs(r12.mean - 12 / 16) < r12.half_width and abs(r17.mean - 17 / 16) < r17.half_width, (r12, r17)
    slope = (r17.mean - r12.mean) / 5
    root = 12 + (1 - r12.mean) / slope
    err = math.hypot(r12.half_width, r17.half_width) / slope * 2
    assert abs(root - float(BANK["c_l_q5_cin0"].value)) < err, (root, err)
    run = lambda c, init, seed=0, **kw: PD.population(5, 0, c, init=init, pop=1000, seed=seed, **kw)
    lo, mid = run(12.3, "planted", sweeps=100, measure=5), run(13.0, "planted", sweeps=100, measure=5)
    assert lo.m < 1e-3 and abs(lo.phi - lo.phi_para) < 1e-9 and PD.INFORMATIVE < 0.5 < mid.m, (lo.m, mid.m)
    hi = [run(14.0, "planted", seed=s, sweeps=60, measure=20) for s in range(4)]
    bd = student_band([r.phi - r.phi_para for r in hi])
    assert all(r.m > 0.5 for r in hi) and bd.mean - bd.half_width > 0, bd
    assert abs(run(14.0, "random", sweeps=60, measure=5).m) < 1e-3
    c_d, c_c = float(BANK["c_d_q5_cin0"].value), float(BANK["c_c_q5_cin0"].value)
    assert 12.3 < c_d < 13.0 and 12.3 < c_c < 14.0
    saved = PD._HOOKS["degree"]
    try:
        PD._HOOKS["degree"] = 0.8
        bad = run(13.0, "planted", sweeps=100, measure=5)
        bad_rate = student_band(rate(17.0))
    finally:
        PD._HOOKS["degree"] = saved
    assert PD._HOOKS["degree"] is saved
    assert bad.m < PD.INFORMATIVE and abs(bad_rate.mean - 17 / 16) > bad_rate.half_width, bad_rate


@slow
def test_q5_thresholds_located():
    """The catalog run at q = 5, c_in = 0 (population dynamics, 3000 members, 400 sweeps per c_d probe, seeds 0-3;
    ~90 s): c_d and c_c reproduce the bank in the 99% Student-t band of compare() (stderr over seeds, tol of the
    printed digits); c_l = 16.  Power: the slip of the degree hook (0.8) moves c_d to about 12.84 / 0.8 = 16 (the
    planted start decays on the whole default bracket: undetermined), far outside the band."""
    e = catalog()["sbm"]
    for r in e.refs:
        if r.key in ("c_d_q5_cin0", "c_c_q5_cin0", "c_l_q5_cin0"):
            out = check_ref(e, r)
            assert out.ok, out
    T = PD.sbm_q_transitions(5, 0, seed=0)
    assert T.status == "determined" and abs(T.c_l - 16) < 1e-12
    saved = PD._HOOKS["degree"]
    try:
        PD._HOOKS["degree"] = 0.8
        bad = PD.sbm_q_transitions(5, 0, seed=0, steps=3)
    finally:
        PD._HOOKS["degree"] = saved
    assert PD._HOOKS["degree"] is saved
    assert bad.status == "undetermined" and bad.c_d is None, bad


@slow
def test_finite_n_bp_pair_shows_the_phases():
    """N = 10^4, q = 5, c_in = 0 (seed 1): the planted / random BP pair on one sampled graph per c.  c = 12: both
    starts reach the factorized point (phase I).  c = 13: planted informative (overlap > 0.5) with Delta Phi < 0,
    random factorized-like (overlap < 0.1, |Delta Phi| < 1e-3) (phase II).  c = 14: planted informative with
    Delta Phi > 0, random as at 13 (phase III).  c = 17: both informative with the same Delta Phi (phase IV).
    Non-converged runs fail the test (measured: see report)."""
    for c, phase in ((12.0, "I"), (13.0, "II"), (14.0, "III"), (17.0, "IV")):
        g = SQ.sbm_q_sample(10000, 5, 0, 5 * c / 4, 1)
        pl = SQ.sbm_q_bp(g, 5, 0, 5 * c / 4, init="planted", iters=1500, seed=1)
        rd = SQ.sbm_q_bp(g, 5, 0, 5 * c / 4, init="random", iters=1500, seed=1)
        dp, dr = pl.phi - pl.phi_para, rd.phi - rd.phi_para
        assert pl.converged and rd.converged, (c, pl.note, rd.note)
        if phase == "I":
            assert pl.overlap < 0.1 and rd.overlap < 0.1 and abs(dp) < 1e-6, (c, pl.overlap, dp)
        elif phase in ("II", "III"):
            assert pl.overlap > 0.5 and rd.overlap < 0.1 and abs(dr) < 1e-3, (c, pl.overlap, rd.overlap, dr)
            assert (dp < 0) if phase == "II" else (dp > 0), (c, dp)
        else:
            assert pl.overlap > 0.9 and rd.overlap > 0.9 and abs(dp - dr) < 1e-6 and dp > 0, (c, dp, dr)


# ---- catalog --------------------------------------------------------------------------------------------
def test_catalog_q_path_and_old_parameters():
    """With q: the KS margin (c_in < c_out accepted), the q = 4, c = 16 KS point claims the bank 2.33, finite-N BP
    values are statistical with the approximation label.  Without q: the E6b parameter sets give the same values
    as inference/sbm.py, and the refusals are unchanged (a < b with N is still unsupported:disassortative)."""
    e = catalog()["sbm"]
    run = lambda **p: compose(e.spec, "cavity").run(**p)
    r = run(q=4, c=16)
    assert check(BANK["inverse_eps_c_q4_c16"], r.values["inverse_eps_ks_q"], label="q4").ok
    assert r.finality["inverse_eps_ks_q"].state == "not-computed" and not r.final
    assert [x for x in e.refs if x.key == "inverse_eps_c_q4_c16" and not x.slow and check_ref(e, x).ok]
    d = run(q=5, c_in="0", c_out="25")
    assert d.values["ks_detectable_q"] and d.values["mean_degree"] == 20 and d.values["eps"] is None
    old = run(c=3)
    assert old.values["inverse_eps_ks"] == B.sbm_ks_line(3).ratio and "inverse_eps_ks_q" not in old.values
    ab = run(a="5.5", b="0.5")
    assert ab.values["ks_margin"] == B.sbm_kesten_stigum("5.5", "0.5").margin and ab.final
    _refuses(lambda: run(a="0.5", b="5.5", N=500), "unsupported:disassortative")
    fin = run(q=3, c_in="9", c_out="1", N=600, seeds=(0, 1), iters=300)
    assert fin.values["q_bp_status"] == "converged" and "q_bp_overlap" in fin.statistical and not fin.final
    assert "approximation" in fin.values["q_bp_phi_label"] and fin.values["q_bp_overlap"] > 0.5
    assert "approximation" in fin.finality["q_bp_delta_phi"].note
    FIN = {"q": 3, "c_in": "5", "c_out": "1", "N": 500}
    for bad, reason in (({"c_in": "5", "b": "1"}, "precondition:known-parameter"),
                        ({"q": 3, "a": "5", "b": "1"}, "precondition:known-parameter"),
                        ({"init": "planted", "a": "5", "b": "1"}, "precondition:known-parameter"),
                        ({**FIN, "init": "cold"}, "precondition:known-parameter"),
                        ({**FIN, "learn": "maybe"}, "precondition:known-parameter"),
                        ({"q": 3, "c_in": "5", "c_out": "1", "seeds": (0, 1)}, "precondition:known-parameter"),
                        ({**FIN, "pop": 10}, "precondition:known-parameter"),
                        ({**FIN, "learn": True, "init": "planted"}, "precondition:known-parameter"),
                        ({**FIN, "learn": "true", "damping": 0.3}, "precondition:known-parameter"),
                        ({"q": 3}, "precondition:state-point"),
                        ({"q": 3, "c": 5, "c_in": 1}, "precondition:state-point"),
                        ({"q": 3, "c_in": "5", "N": 500}, "precondition:state-point"),
                        ({"q": 1, "c": 5}, "precondition:group-count"), ({"q": 3, "c_in": "-1", "c_out": "2"},
                                                                        "precondition:affinity")):
        _refuses(lambda: run(**bad), reason)


def test_sbm_q_refusals_are_named():
    g = SQ.sbm_q_sample(200, 3, 2, 6, 0)
    _refuses(lambda: SQ.sbm_q_bp(g, 3, 2, 6, init="cold", iters=5, seed=0), "precondition:init")
    _refuses(lambda: SQ.sbm_q_bp(g, 3, 2, 6, init="random", iters=0, seed=0), "precondition:positive-iters")
    _refuses(lambda: SQ.sbm_q_bp(g, 3, 2, 6, init="random", iters=5, seed=0, damping=1), "precondition:damping")
    _refuses(lambda: SQ.sbm_q_bp("g", 3, 2, 6, init="random", iters=5, seed=0), "precondition:sbm-graph")
    _refuses(lambda: SQ.sbm_q_bp(g, 2, 2, 6, init="random", iters=5, seed=0), "precondition:sbm-graph")
    _refuses(lambda: SQ.sbm_q_sample(5, 3, 2, 6, 0), "precondition:size")
    _refuses(lambda: SQ.sbm_q_sample(20, 3, 30, 6, 0), "precondition:edge-probability")
    _refuses(lambda: SQ.sbm_q_sample(200, 3, 0, 0, 0), "precondition:affinity")
    _refuses(lambda: SQ.sbm_q_sample(200, 3, 2, 6, 0, sizes=(0.5, 0.5)), "precondition:group-sizes")
    _refuses(lambda: SQ.sbm_q_ks_line(3, "0.5"), "no-transition:ks-line")
    _refuses(lambda: PD.population(5, 0, 13, init="cold", pop=10, sweeps=2, measure=1, seed=0), "precondition:init")
    _refuses(lambda: PD.population(5, 0, 13, init="planted", pop=10, sweeps=2, measure=3, seed=0),
             "precondition:population")
    _refuses(lambda: SQ.from_edges(3, [[0, 0]], [0, 1, 2]), "precondition:sbm-graph")
