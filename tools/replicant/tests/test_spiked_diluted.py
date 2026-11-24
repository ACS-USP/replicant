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
    """m(gamma) = E[X E[X|y]] of the closed-form denoiser equals bayes/channels.py's mixture route for the same prior
    (sparse, rho = 1/4, a = 2: unit variance) at several gamma; the slope m'(gamma) = E[Var(X|y)^2] (Guo-Shamai-Verdu,
    mmse' = -E Var^2) matches a central difference; dI/dgamma = (1 - m)/2 (I-MMSE) with I from the log-partition
    formula.  Planted control: the amplitude 1/rho (variance 1/rho, the unnormalised slip) breaks the agreement."""
    pr = CH.Prior("sparse", rho="1/4", a=2)
    for g in (0.05, 0.4, 1.3, 4.0):
        m_ch = CH.overlaps(pr, g)["m"]
        assert abs(SD.diluted_overlap(0.25, g) - m_ch) < 1e-12, (g, SD.diluted_overlap(0.25, g), m_ch)
        h = 1e-4 * g
        fd = (SD.diluted_overlap(0.25, g + h) - SD.diluted_overlap(0.25, g - h)) / (2 * h)
        assert abs(SD.diluted_overlap_slope(0.25, g) - fd) < 1e-7, (g, fd)
        di = (SD.diluted_mutual_info(0.25, g + h) - SD.diluted_mutual_info(0.25, g - h)) / (2 * h)
        assert abs(di - (1 - SD.diluted_overlap(0.25, g)) / 2) < 1e-7, (g, di)
    assert abs(SD.diluted_overlap(RHO, 1e-6) / 1e-6 - 1) < 1e-4      # unit variance: m ~ gamma at 0
    bad = _planted("amplitude_power", -1.0, lambda: SD.diluted_overlap(0.25, 0.4))
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
    hard = S.spiked_state_evolution(1 / math.sqrt(1.01), prior="diluted_rademacher", rho=RHO)
    assert hard.phase == "hard", hard.note
    stable = [f for f in hard.fixed_points if f.stable]
    assert len(stable) == 2 and stable[0].gamma == 0 and stable[1].gamma > 0.3, hard.fixed_points
    assert hard.uninformative.gamma == 0 and hard.informative.gamma == stable[1].gamma
    assert abs(hard.q - stable[1].gamma * 1.01) < 1e-14 and hard.bayes and len(hard.fixed_points) == 3
    meta = S.spiked_state_evolution(1 / math.sqrt(1.035), prior="diluted_rademacher", rho=RHO)
    assert meta.phase == "impossible" and meta.metastable and meta.q == 0, meta.note
    easy = S.spiked_state_evolution(1 / math.sqrt(0.97), prior="diluted_rademacher", rho=RHO)
    assert easy.phase == "easy" and not easy.metastable and easy.q > 0.3, easy.note
    assert len([f for f in easy.fixed_points if f.stable]) == 1
    bad = _planted("amplitude_power", -1.0, lambda: SD.diluted_spinodals(RHO))
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


def test_existing_priors_unchanged():
    """The gaussian and rademacher SE (no rho) are the E6b results: q = 1 - 1/lambda^2 at lambda = 2 as banked."""
    se = S.spiked_state_evolution(2)
    assert type(se) is S.SEResult and _ok("q_gauss_snr2", se.q)
    assert S.spiked_state_evolution("1.5", "rademacher") == S.spiked_state_evolution("1.5", "rademacher", rho=None)


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
