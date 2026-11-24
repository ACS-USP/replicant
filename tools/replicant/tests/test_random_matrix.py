"""K1: random-matrix spectra (KZ Ch. 3, Ex. 3.1): rmt/stieltjes.py and the `random_matrix` catalog entry.

Convention: A = (G + G^T)/sqrt(2N); Sigma = X X^T / N with X of size M x N, alpha = M/N; S(z) = (1/N) sum
1/(lambda_i - z).  Banked (reference/meanfield.json): the semicircle sqrt(4 - x^2)/(2 pi) and its edge, the MP edges
(1 +- sqrt alpha)^2, as printed (values: their evaluations).  Independent routes: the cavity recursion (separate
code), eigvalsh of sampled matrices over named seeds, and learning/ridge.py's Marchenko-Pastur transform.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import mpmath as mp
import numpy as np

from replicant.amp._stats import t_quantile
from replicant.core import Refusal
from replicant.refdata import load
from replicant.rmt import stieltjes as RM

BANK = load("meanfield")
SEEDS = tuple(range(6))


def _planted(module, key, value, fn):
    reg = module._HOOKS
    old = reg[key]
    reg[key] = value
    try:
        return fn()
    finally:
        reg[key] = old
        assert module._HOOKS is reg and reg[key] is old


def _bin_masses(kind, cuts, alpha=None):
    """Mass of the replica density on each bin (16-point Gauss-Legendre per bin)."""
    x, w = np.polynomial.legendre.leggauss(16)
    out = []
    for a, b in zip(cuts, cuts[1:]):
        pts = (b - a) / 2 * x + (a + b) / 2
        out.append(sum(wi * (b - a) / 2 * float(RM.density(kind, f"{p:.17g}", alpha=alpha).mid())
                       for p, wi in zip(pts, w)))
    return np.array(out)


def test_spectra_vs_eigvalsh():
    """Bank values by both routes; the eigvalsh histogram (N = 400, 6 named seeds) inside a per-bin band of
    t_{0.995, 5} stderr + 1/N (an assumed finite-N systematic, O(1/N) in the bulk; observed deviations are below
    0.002); a planted semicircle radius 2.2 leaves the band in at least 8 of the 10 bins and by more than 4 band
    widths in one (power ~1: the shifts are 0.016-0.032 against bands of about 0.006).  Wishart: the extreme
    eigenvalues sit at the MP edges within 5 Tracy-Widom scales (about 0.2 here), and alpha = 2 has exactly M - N zero eigenvalues (atom 1 - 1/alpha)."""
    with mp.workdps(30):
        for key, route, fn in (("semicircle_edge", "replica", lambda r: RM.edges("wigner", r)[1].mid()),
                               ("semicircle_edge", "cavity", lambda r: RM.edges("wigner", r)[1].mid()),
                               ("semicircle_density_0", "replica", lambda r: RM.density("wigner", "0").mid()),
                               ("semicircle_density_0", "cavity", lambda r: RM.density("wigner", "0", "cavity")),
                               ("semicircle_density_1", "cavity", lambda r: RM.density("wigner", "1", "cavity")),
                               ("mp_edge_upper_a0p5", "replica", lambda r: RM.edges("wishart", r, "1/2")[1].mid()),
                               ("mp_edge_lower_a0p25", "cavity", lambda r: RM.edges("wishart", r, "1/4")[0].mid())):
            want = BANK[key]
            assert abs(fn(route) - mp.mpf(want.value)) <= mp.mpf(want.tol), (key, route)
    cuts = np.linspace(-2.5, 2.5, 11)
    counts = np.array([np.histogram(RM.sample_spectrum("wigner", 400, s), cuts)[0] / 400 for s in SEEDS])
    mean, se = counts.mean(axis=0), counts.std(axis=0, ddof=1) / np.sqrt(len(SEEDS))
    band = t_quantile(0.995, len(SEEDS) - 1) * se + 1 / 400
    want = _bin_masses("wigner", cuts)
    assert np.all(np.abs(mean - want) <= band), (mean, want, band)
    bad = _planted(RM, "edge", mp.mpf("2.2"), lambda: _bin_masses("wigner", cuts))
    assert np.max(np.abs(mean - bad) / band) > 4 and np.sum(np.abs(mean - bad) > band) >= 8, (mean, bad, band)
    lo, hi = RM.edges("wishart", alpha="1/2")
    ev = [RM.sample_spectrum("wishart", 400, s, alpha="1/2") for s in SEEDS]
    M, N = 200, 400                  # the Tracy-Widom scale of lambda_max: a tolerance, not banked
    tol = 5 * (np.sqrt(M) + np.sqrt(N)) * (1 / np.sqrt(M) + 1 / np.sqrt(N)) ** (1 / 3) / N
    assert all(abs(e[-1] - float(hi.mid())) < tol and abs(e[0] - float(lo.mid())) < tol for e in ev)
    wrong = RM.edges("wishart", alpha="2")[1]                       # alpha read as 1/gamma: the edge of alpha = 2
    assert all(abs(e[-1] - float(wrong.mid())) > 1 for e in ev)
    ev2 = RM.sample_spectrum("wishart", 150, 0, alpha="2")
    assert int(np.sum(np.abs(ev2) < 1e-9)) == 300 - 150 and RM.atom_at_zero("wishart", alpha="2") == 0.5


def test_cavity_edges_are_located_on_the_moebius_matrix():
    """The cavity edges come from certified root isolation of trace^2 - 4 det on the recursion's Moebius matrix,
    not from a closed form: they enclose the bank values to their tolerance, and a planted variance factor 1.21 in
    the matrix (hook cavity_scale) moves them to the planted law's edges (Wigner +-2.2; Wishart alpha = 1/4 read as
    0.3025: (1 -+ 0.55)^2), far outside the bank tolerance, while the replica edges do not move."""
    with mp.workdps(40):
        for key, kind, a, side in (("semicircle_edge", "wigner", None, 1), ("mp_edge_lower_a0p25", "wishart", "1/4", 0),
                                   ("mp_edge_upper_a0p25", "wishart", "1/4", 1),
                                   ("mp_edge_lower_a0p5", "wishart", "1/2", 0),
                                   ("mp_edge_upper_a0p5", "wishart", "1/2", 1)):
            b = RM.edges(kind, "cavity", alpha=a)[side]
            assert abs(b.mid() - mp.mpf(BANK[key].value)) <= mp.mpf(BANK[key].tol) and b.digits >= 30, (key, b)
        bad = _planted(RM, "cavity_scale", mp.mpf("1.21"), lambda: (RM.edges("wigner", "cavity"),
                                                                    RM.edges("wishart", "cavity", alpha="1/4")))
        assert abs(bad[0][1].mid() - mp.mpf("2.2")) < mp.mpf("1e-25"), bad
        assert abs(bad[1][0].mid() - mp.mpf("0.2025")) < mp.mpf("1e-25"), bad
        assert abs(bad[1][1].mid() - mp.mpf("2.4025")) < mp.mpf("1e-25"), bad
        assert abs(RM.edges("wigner")[1].mid() - 2) == 0


def test_replica_and_cavity_routes_agree():
    """S(z) off the axis and the density on it by the two routes (no shared code); the cavity edge (parabolic
    Moebius map) equals the replica edge (discriminant root)."""
    with mp.workdps(30):
        for z in (0.3 + 0.2j, -1.7 + 0.01j, 2.5 + 0.5j):
            assert abs(complex(RM.stieltjes_wigner(z).mid()) - complex(RM.stieltjes_wigner_cavity(z))) < 1e-14
            assert abs(complex(RM.stieltjes_mp(z, "1/3").mid()) - complex(RM.stieltjes_mp_cavity(z, "1/3"))) < 1e-14
        for x, a in (("0.7", "1/3"), ("1.9", "2")):
            assert abs(RM.density("wishart", x, alpha=a).mid() - RM.density("wishart", x, "cavity", alpha=a)) < 1e-20
        for kind, a in (("wigner", None), ("wishart", "3/7")):
            r, c = RM.edges(kind, alpha=a), RM.edges(kind, "cavity", alpha=a)
            assert all(x.overlaps(y) for x, y in zip(r, c))
    assert RM.density("wigner", "2.5").mid() == 0
    try:
        RM.stieltjes_wigner_cavity(1.0)
    except Refusal as r:
        assert r.reason == "precondition:upper-half-plane"
    else:
        raise AssertionError("the cavity recursion needs Im z > 0")


def test_wishart_alpha_is_ridge_gamma():
    """Convention: KZ Ch. 3's alpha = M/N is learning/ridge.py's gamma = p/n (both: matrix dimension over number of
    columns): S_KZ(-lam; alpha) = m_ridge(-lam; gamma = alpha), in both directions, and alpha -> 1/alpha (the
    regression chapters' map) does not reproduce it."""
    from replicant.learning.ridge import stieltjes_mp as ridge_m
    with mp.workdps(30):
        for a, lam in (("1/2", "0.3"), ("3", "1.2")):
            kz = RM.stieltjes_mp(f"-{lam}", a).real.mid()
            assert abs(kz - ridge_m(mp.mpf(eval_frac(a)), mp.mpf(lam))[0]) < mp.mpf("1e-20")
            assert abs(kz - ridge_m(1 / mp.mpf(eval_frac(a)), mp.mpf(lam))[0]) > mp.mpf("1e-2")
        g = mp.mpf("0.25")
        assert abs(ridge_m(g, mp.mpf("0.5"))[0] - RM.stieltjes_mp("-0.5", "1/4").real.mid()) < mp.mpf("1e-20")


def eval_frac(s):
    from fractions import Fraction
    f = Fraction(s)
    return mp.mpf(f.numerator) / f.denominator


def test_random_matrix_entry():
    from replicant.catalog import catalog
    from replicant.compose import compose
    e = catalog()["random_matrix"]
    v = compose(e.spec).run(kind="wishart", alpha="2", x="1", y="0.1")
    assert v.rigor == "certified" and v.values["atom_zero"].startswith("0.5")
    c = e.run("cavity", kind="wigner", x="0.5", y="0.1")
    assert abs(mp.mpf(c["stieltjes_im"]) - RM.stieltjes_wigner(0.5 + 0.1j).imag.mid()) < 1e-14
    for bad, reason in (({"kind": "goe"}, "unsupported:ensemble"), ({"kind": "wishart"}, "precondition:alpha"),
                        ({"alpha": "1/2"}, "precondition:known-parameter"), ({"y": "0.1"}, "precondition:state-point")):
        try:
            e.run(**bad)
        except Refusal as r:
            assert r.reason == reason, (bad, r)
        else:
            raise AssertionError(bad)
    guide = (Path(RM.__file__).parent.parent / "GUIDE.md").read_text()
    assert "| `random_matrix`" in guide and "`random_matrix` (replica, cavity)" in guide
