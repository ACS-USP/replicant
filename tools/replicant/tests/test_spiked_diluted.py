"""K5: the diluted-Rademacher spiked Wigner model and its first-order transition (KZ Ex. 8.2; Lesieur, Krzakala &
Zdeborova, J. Stat. Mech. (2017) 073403, arXiv:1701.00858 (LKZ); Antenucci, Franz, Urbani & Zdeborova, PRX 9,
011020 (2019), arXiv:1805.05857 (AFUZ)); and the Ch. 11 dense spin-glass game as the Rademacher spike.

Convention (inference/spiked_diluted.py): replicant's prior is the unit-variance one, x in {0, +-1/sqrt(rho)} with
P(0) = 1 - rho; LKZ (Eq. (33)) and the notes (Ex. 8.2, model 2) take x in {0, +-1} (variance rho) with
Y = x x^T / sqrt(N) + sqrt(Delta) xi.  Rescaling x by sqrt(rho): lambda^2 = rho^2 / Delta, i.e. LKZ's horizontal
axis Delta / rho^2 is 1 / lambda^2.  In the notes' units (Y = sqrt(lambda_notes / N) x x^T + xi, x in {0, +-1}):
lambda_notes = 1 / Delta, so lambda = rho sqrt(lambda_notes).  Bank: reference/spiked.json.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.amp._stats import student_band
from replicant.bayes import channels as CH
from replicant.core import Refusal
from replicant.inference import spiked as S
from replicant.inference import spiked_diluted as SD
from replicant.inference import spin_glass_game as G
from replicant.refdata import check, load
from replicant.testing import slow

BANK = load("spiked")
RHO = 0.08     # AFUZ Fig. 1 and LKZ Fig. 3 / Fig. 6 (bottom right)


def _ok(key, got):
    return check(BANK[key], got, label=key).ok


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


def _planted(key, value, fn):
    old = SD._HOOKS[key]
    SD._HOOKS[key] = value
    SD.clear_caches()
    try:
        return fn()
    finally:
        SD._HOOKS[key] = old
        SD.clear_caches()
        assert SD._HOOKS[key] is old


def test_diluted_scalar_channel_against_channels_sparse_prior():
    """The diluted prior lives in bayes/channels.py (family `diluted`, amplitude Sqrt(1/rho)); its fast path
    channels.scalar_moments equals channels' general mixture route overlaps() for the same prior written as
    sparse(rho = 1/4, a = 2), and, at rho = 0.08 (irrational amplitude), an independent mpmath quadrature of
    rho a E[F(gamma a + sqrt(gamma) Z)] with the closed-form 3-atom F.  The slope m'(gamma) = E[Var(X|y)^2]
    (Guo-Shamai-Verdu) matches a central difference, and dI/dgamma = (1 - m)/2 (I-MMSE) with I = h(Y) - h(Z).
    Control: the unnormalised prior (x in {0, +-1}, variance rho: the notes' and LKZ's) breaks the agreement."""
    import mpmath as mp
    pr = CH.Prior("sparse", rho="1/4", a=2)
    assert CH.Prior("diluted", rho="0.08").second_moment == 1
    for g in (0.05, 0.4, 1.3, 4.0):
        m_ch = CH.overlaps(pr, g)["m"]
        assert abs(SD.diluted_overlap(0.25, g) - m_ch) < 1e-13, (g, SD.diluted_overlap(0.25, g), m_ch)
        h = 1e-4 * g
        fd = (SD.diluted_overlap(0.25, g + h) - SD.diluted_overlap(0.25, g - h)) / (2 * h)
        assert abs(SD.diluted_overlap_slope(0.25, g) - fd) < 1e-7, (g, fd)
        di = (SD.diluted_mutual_info(0.25, g + h) - SD.diluted_mutual_info(0.25, g - h)) / (2 * h)
        assert abs(di - (1 - SD.diluted_overlap(0.25, g)) / 2) < 1e-7, (g, di)
    with mp.workdps(20):
        a = 1 / mp.sqrt(mp.mpf("0.08"))
        for g in (mp.mpf("0.0444450374617"), mp.mpf("0.28797173862")):
            e = mp.exp(-g * a * a / 2)
            F = lambda y: mp.mpf("0.08") * a * mp.sinh(a * y) * e / (mp.mpf("0.92") + mp.mpf("0.08") * mp.cosh(a * y) * e)
            want = mp.mpf("0.08") * a * mp.quad(lambda z: mp.npdf(z) * F(g * a + mp.sqrt(g) * z), [-mp.inf, 0, mp.inf])
            assert abs(SD.diluted_overlap(RHO, float(g)) - want) < 1e-14, (g, want)
    assert abs(SD.diluted_overlap(RHO, 1e-6) / 1e-6 - 1) < 1e-4      # unit variance: m ~ gamma at 0
    bad = _planted("unit_variance", False, lambda: SD.diluted_overlap(0.25, 0.4))
    assert abs(bad - CH.overlaps(pr, 0.4)["m"]) > 1e-2, bad


def test_diluted_rademacher_first_order():
    """rho = 0.08: two stable SE fixed points in the hard phase, the two spinodals and the IT threshold, against AFUZ's
    printed Delta / rho^2 (Delta_dyn ~ 1.041, Delta_IT ~ 1.0295, Delta_alg ~ 0.9805; Delta_c = rho^2: LKZ Eq. (193))
    and the phase labels at three state points.  Planted control: the unnormalised amplitude moves the spinodals."""
    sp = SD.diluted_spinodals(RHO)
    assert _ok("diluted_rho0p08_delta_dyn_over_rho2", 1 / sp.snr_dyn**2), sp
    assert _ok("diluted_rho0p08_delta_alg_over_rho2", 1 / sp.snr_alg**2), sp
    lam_it = SD.diluted_it_threshold(RHO)
    assert _ok("diluted_rho0p08_delta_it_over_rho2", 1 / lam_it**2), lam_it
    # LKZ Fig. 6, rho = 0.08: Delta_alg < Delta_c < Delta_IT < Delta_dyn
    assert sp.snr_dyn < lam_it < 1 < sp.snr_alg
    hard = S.spiked_state_evolution(1 / math.sqrt(1.02), prior="diluted_rademacher", rho=RHO)
    assert hard.phase == "hard", hard.note
    stable = [f for f in hard.fixed_points if f.stable]
    assert len(stable) == 2 and stable[0].gamma == 0 and stable[1].gamma > 0.3, hard.fixed_points
    assert hard.uninformative.gamma == 0 and hard.informative.gamma == stable[1].gamma
    assert abs(hard.q - stable[1].gamma * 1.02) < 1e-14 and hard.bayes and len(hard.fixed_points) == 3
    meta = S.spiked_state_evolution(1 / math.sqrt(1.035), prior="diluted_rademacher", rho=RHO)
    assert meta.phase == "impossible" and meta.metastable and meta.q == 0, meta.note
    easy = S.spiked_state_evolution(1 / math.sqrt(0.97), prior="diluted_rademacher", rho=RHO)
    assert easy.phase == "easy" and not easy.metastable and easy.q > 0.3, easy.note
    assert len([f for f in easy.fixed_points if f.stable]) == 1
    bad = _planted("unit_variance", False, lambda: SD.diluted_spinodals(RHO))
    assert bad is None or abs(1 / bad.snr_dyn**2 - 1.041) > 0.01, bad


@slow
def test_diluted_lkz_orderings():
    """(slow: three more rho values.)  LKZ Fig. 6: rho = 0.0909 has Delta_dyn < Delta_c; rho = 0.0863 has
    Delta_IT < Delta_c < Delta_dyn; rho = 0.097
    (above rho_tri) has a single stable fixed point at every Delta (no spinodals)."""
    a = SD.diluted_spinodals(0.0909)
    assert a is not None and a.snr_dyn > 1, a                  # Delta_dyn / rho^2 < 1
    b, it = SD.diluted_spinodals(0.0863), SD.diluted_it_threshold(0.0863)
    assert b.snr_dyn < 1 < it, (b, it)                         # Delta_IT < Delta_c < Delta_dyn
    assert SD.diluted_spinodals(0.097) is None


def test_diluted_tricritical_point():
    """LKZ Fig. 7 caption: the spinodals and the IT line join at (Delta_tri = 0.008935, Delta_tri / rho_tri^2 = 0.9612,
    rho_tri = 0.09641) for the Rademacher-Bernoulli prior; computed here as the rho where D(gamma) = m(gamma)/gamma
    stops being non-monotone (fixed points are D(gamma) = 1/lambda^2)."""
    tri = SD.diluted_tricritical()
    assert _ok("diluted_rho_tri", tri.rho), tri
    assert _ok("diluted_delta_tri_over_rho2", 1 / tri.snr**2), tri
    assert _ok("diluted_delta_tri", tri.rho**2 / tri.snr**2), tri


def test_diluted_spinodals_next_to_the_tricritical_point():
    """Within ~1e-7 below rho_tri the 361-point grid misses the two zeros of s = gamma m' - m; they are now bracketed
    on either side of the maximum of s, so the spinodals exist (gamma_alg < gamma_dyn, snr_alg > snr_dyn) instead of a
    silent None.  At rho_tri itself (|max s| < 1e-10) first-order vs continuous is labelled undetermined by the
    entry, never reported first_order False; just above rho_tri it is continuous."""
    rt = SD.diluted_tricritical().rho
    for d in (1e-7, 1e-9):
        sp = SD.diluted_spinodals(rt - d)
        assert sp is not None and sp.gamma_alg < sp.gamma_dyn and sp.snr_alg > sp.snr_dyn, (d, sp)
        assert SD.diluted_first_order(rt - d) == "first-order"
    assert SD.diluted_first_order(rt) == "undetermined" and SD.diluted_first_order(rt + 1e-7) == "continuous"
    from replicant.catalog import catalog
    v = catalog()["spiked_wigner"].run("amp", prior="diluted_rademacher", rho=rt)
    assert v.get("first_order_status") == "undetermined" and "first_order" not in v, v
    assert catalog()["spiked_wigner"].run("amp", prior="diluted_rademacher", rho="0.08")["first_order"] is True


def test_diluted_undetermined_not_refused():
    """A run whose SE iterations do not converge (iteration cap far below what the spinodal's slowing needs) keeps the
    fixed points but labels the Bayes values undetermined (never a Refusal)."""
    r = SD.diluted_state_evolution(1 / math.sqrt(1.01), RHO, max_iter=3)
    assert r.phase == "undetermined" and r.q is None and r.gamma is None and not r.bayes, r.note
    assert len(r.fixed_points) == 3


def test_diluted_conventions_both_ways():
    """lambda (replicant, unit-variance prior) <-> LKZ's Delta (x in {0, +-1}) <-> the notes' lambda_notes = 1/Delta:
    lambda^2 = rho^2 / Delta = rho^2 lambda_notes.  One number both ways: AFUZ's Delta_IT = 1.0295 rho^2 at rho = 0.08,
    and LKZ's Delta_c = rho^2 (Eq. (193)) is replicant's computed SE instability lambda = 1."""
    d = 1.0295 * RHO**2
    lam = SD.snr_from_lkz(d, RHO)
    assert abs(lam - 1 / math.sqrt(1.0295)) < 1e-15 and abs(SD.lkz_from_snr(lam, RHO) - d) < 1e-15
    assert abs(SD.snr_from_notes(1 / d, RHO) - lam) < 1e-15 and abs(SD.notes_from_snr(lam, RHO) - 1 / d) < 1e-12
    assert abs(SD.diluted_overlap_slope(RHO, 0.0) - 1) < 1e-12            # slope of the SE map at gamma = 0 is lambda^2
    assert abs(SD.lkz_from_snr(1.0, RHO) - RHO**2) < 1e-15                 # Delta_c = rho^2 <-> lambda_c = 1
    _refuses(lambda: S.spiked_state_evolution(2, prior="gaussian", rho=0.5), "precondition:known-parameter")
    _refuses(lambda: S.spiked_state_evolution(2, prior="diluted_rademacher"), "precondition:rho")
    _refuses(lambda: S.spiked_state_evolution(2, prior="diluted_rademacher", rho=1.5), "precondition:rho")


def _ser(x):
    import mpmath as mp
    if isinstance(x, mp.mpf):
        return "mpf:" + mp.nstr(x, 30)
    if isinstance(x, float):
        return "f:" + repr(x)
    if isinstance(x, (list, tuple)):
        return [_ser(y) for y in x]
    if isinstance(x, dict):
        return {k: _ser(v) for k, v in x.items()}
    return x


def test_existing_routes_match_58953a7():
    """Golden values recorded with the code of commit 58953a7 (before K5; `git archive 58953a7` into a temporary
    directory, the same runs): spiked_wigner's gaussian and rademacher priors (threshold, SE, finite-N AMP) and the
    sk amp route (SE, finite-N AMP) give bit-identical values on HEAD (tests/golden_58953a7.json).  Since the spectral
    first-step fix (f_{-1} = x^0 / lambda, MV App. A.2) the spiked_wigner finite-N AMP keys (`*_amp`,
    `*_halfwidth99`, `amp_converged_fraction`) are reproduced bit-identically with the private hook
    S._spectral_first_step = False (the pre-fix f_{-1} = 0); with the fix every other key is still bit-identical and
    the finite-N q_amp differs (the new values are pinned in test_spiked_diluted_amp.test_existing_priors_spectral_literals)."""
    import json
    from pathlib import Path
    from replicant.catalog import catalog
    from replicant.compose import compose
    cat = catalog()
    finite = lambda k: k.endswith("_amp") or k.endswith("_halfwidth99") or k == "amp_converged_fraction"
    for g in json.loads((Path(__file__).parent / "golden_58953a7.json").read_text()):
        run = lambda: _ser(compose(cat[g["entry"]].spec, "amp").run(**g["params"]).values)
        got = run()
        if g["entry"] == "spiked_wigner" and "N" in g["params"]:
            saved = S._spectral_first_step
            try:
                S._spectral_first_step = False
                old = run()
            finally:
                S._spectral_first_step = saved
            assert old == g["values"], (g["entry"], g["params"])
            assert {k: v for k, v in got.items() if not finite(k)} == {k: v for k, v in g["values"].items()
                                                                       if not finite(k)}, g["params"]
            assert got["q_amp"] != g["values"]["q_amp"], (got["q_amp"], g["params"])
        else:
            assert got == g["values"], (g["entry"], g["params"])


def test_diluted_catalog_snr_parsing_and_finality():
    """The diluted path parses snr like the plain path (a rational string is accepted, a bool is a named refusal);
    q_se and matrix_mmse_se carry LM Thm 1, while overlap_se and mmse_se (vector quantities no fetched theorem covers,
    and not AMP's performance in the hard phase) are not-computed."""
    from replicant.catalog import catalog
    from replicant.compose import compose
    e = catalog()["spiked_wigner"]
    r = compose(e.spec, "amp").run(prior="diluted_rademacher", rho="0.08", snr="1/2")
    assert r.values["se_phase"] == "impossible" and r.values["q_se"] == 0, r.values
    assert r.finality["q_se"].state == "exact-by-literature" and r.finality["matrix_mmse_se"].state == "exact-by-literature"
    for k in ("overlap_se", "mmse_se"):
        assert r.finality[k].state == "not-computed" and "LM Thm 1 covers" in r.finality[k].note, r.finality[k]
    plain = compose(e.spec, "amp").run(snr="2")
    assert plain.finality["overlap_se"].state == "exact-by-literature"        # MV Thm 2 on the plain path, unchanged
    _refuses(lambda: compose(e.spec, "amp").run(prior="diluted_rademacher", rho="0.08", snr=True), "precondition:snr")
    _refuses(lambda: S.spiked_state_evolution("x", prior="diluted_rademacher", rho=RHO), "precondition:snr")


# ---- Ch. 11: the dense spin-glass game ---------------------------------------------------------------------------
def test_dense_spin_glass_game_is_rademacher_spike():
    """KZ Eqs. (11.27)-(11.28): all pairs observed, J_ij = S_i S_j with probability 1 - p,
    p = 1/(1 + e^{2 beta/sqrt N}).
    E[J_ij | S] = tanh(beta/sqrt N) S_i S_j ~ beta S_i S_j / sqrt N and Var J_ij -> 1, so J is the notes' spiked Wigner
    (8.1) with sqrt(lambda_notes) = beta: lambda_LM = beta^2 and replicant's lambda = beta (DERIVED here, not a printed
    value).  Checks: (i) the notes' two-parameter SE (11.59)-(11.60) iterated from m = q = 1, coded here without
    Nishimori, equals spiked_wigner's rademacher q_se at lambda = beta; (ii) the top eigenvalue of J / sqrt(N) over
    seeds 0-5 at N = 400, beta = 1.5 sits at lambda + 1/lambda (notes Eq. (12.129) with lambda_notes = beta^2) within
    the 99% Student-t band; planted lambda = beta^2 (the lambda_LM slip) puts it at 2.694 against 2.167, outside."""
    assert G.game_snr(1.5) == 1.5 and G.game_snr_lm(1.5) == 2.25

    def notes_se(beta, iters=400):
        z = np.linspace(-12.0, 12.0, 4801)        # trapezoid (Gauss-Hermite misses tanh's poles at large beta)
        w = np.exp(-z**2 / 2)
        w = w / w.sum()
        m = q = 1.0
        for _ in range(iters):
            t = np.tanh(beta**2 * m + beta * math.sqrt(q) * z)
            m, q = float(w @ t), float(w @ t**2)
        return m, q
    for beta in (1.2, 1.5, 2.0):
        m, q = notes_se(beta)
        se = S.spiked_state_evolution(G.game_snr(beta), "rademacher")
        assert abs(q - se.q) < 1e-9 and abs(m - se.q) < 1e-9, (beta, m, q, se.q)
    beta, seeds = 1.5, range(6)
    tops = []
    for s in seeds:
        J = G.game_matrix(400, beta, s)
        tops.append(float(np.linalg.eigvalsh(J / math.sqrt(400))[-1]))
    band = student_band(tops)
    lam = G.game_snr(beta)
    assert abs(band.mean - (lam + 1 / lam)) <= band.half_width, (band, lam + 1 / lam)
    wrong = G.game_snr_lm(beta)
    assert abs(band.mean - (wrong + 1 / wrong)) > band.half_width, band
    assert all(abs(x) == 1 for x in G.game_matrix(30, beta, 0)[np.triu_indices(30, 1)])
