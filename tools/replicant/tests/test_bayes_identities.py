"""K2: the Nishimori, Stein and I-MMSE identities as validators (KZ Secs. 7.2.4-7.2.5; bayes/identities.py).

I-MMSE convention, from Guo, Shamai & Verdu, IEEE Trans. Inf. Theory 51, 1261 (2005), arXiv:cs/0412108 (read):
Theorem 1, Eq. (15), for Y = sqrt(snr) X + N (Eq. (3)), N ~ N(0, 1), information in nats:
d I / d snr = mmse(snr) / 2.  KZ Theorem 14 (Eq. (7.42)) states the same with snr = 1/Delta.
The validators return a CheckResult (ok, got, want, tol, label) and never raise for a result off the Nishimori
line.  Planted errors go through identities._HOOKS, restored with identity asserts.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from replicant.amp import gamp_se as GS
from replicant.bayes import channels as CH
from replicant.bayes import identities as ID
from replicant.catalog import catalog


def _planted(module, key, value, fn):
    reg = module._HOOKS
    old = reg[key]
    reg[key] = value
    try:
        return fn()
    finally:
        reg[key] = old
        assert module._HOOKS is reg and reg[key] is old


PRIORS = (CH.Prior("gaussian"), CH.Prior("rademacher"), CH.Prior("gauss_bernoulli"),
          CH.Prior("sparse", rho="1/4", a=2))
GRID = (0.25, 1.0, 3.0)


def test_immse_holds_and_a_plant_fails():
    """dI/dsnr = mmse/2 (GSV Theorem 1, nats) for four priors on a grid, by Richardson-extrapolated central
    differences of the mutual information (log p(y) quadrature) against the MMSE (posterior-variance quadrature):
    two separate integrals.  A planted factor 1 in place of 1/2 must fail at every prior."""
    for P in PRIORS:
        r = ID.check_immse(P, GRID)
        assert r.ok, (P.name, r)
        assert "I-MMSE" in r.label
    for P in PRIORS:
        bad = _planted(ID, "immse_factor", 1, lambda: ID.check_immse(P, GRID))
        assert not bad.ok, (P.name, bad)
    # the GSV Gaussian-input example: I = log(1 + snr)/2 (Eq. (11)), mmse = 1/(1 + snr) (Eq. (13))
    g = CH.Prior("gaussian")
    assert abs(CH.mutual_information(g, 3.0) - 0.5 * math.log(4)) < 1e-13 and abs(CH.mmse(g, 3.0) - 0.25) < 1e-13


def test_nishimori_on_existing_entries():
    """q = m on spiked_wigner state evolution and on teacher_student Bayes (GAMP state evolution): the scalar channel
    each one ends in (snr = gamma* for spiked, snr = r for the perceptron) gives E[x eta] = E[eta^2], and that common
    value is the entry's own overlap q.  A mismatched prior is flagged off the Nishimori line, never raised."""
    v = catalog()["spiked_wigner"].run(snr="1.5", prior="rademacher")
    sp = ID.scalar_overlaps(CH.Prior("rademacher"), v["gamma_se"])
    r = ID.check_nishimori(sp)
    assert r.ok and r.label == "on the Nishimori line", r
    assert ID.check_nishimori({"m": sp["m"], "q": v["q_se"]}, tol=1e-9).ok      # SE's q is the scalar m
    t = catalog()["teacher_student"].run("amp", alpha=2)
    se = GS.perceptron_state_evolution(2)
    assert se.overlap == t["q_se"]
    ov = ID.scalar_overlaps(CH.Prior("gaussian"), se.r)
    assert ID.check_nishimori(ov).ok and ID.check_nishimori({"m": t["q_se"], "q": ov["q"]}, tol=1e-9).ok
    # mismatched prior: Gaussian data denoised as if +-1 (tanh) -> m != q, reported, not raised (the reverse,
    # +-1 data through the linear Gaussian denoiser, stays on the line: both priors have unit variance)
    off = ID.scalar_overlaps(CH.Prior("gaussian"), v["gamma_se"], model=CH.Prior("rademacher"))
    r = ID.check_nishimori(off)
    assert not r.ok and r.label == "not on the Nishimori line" and abs(float(r.got) - float(r.want)) > 1e-3, r
    # MMSE = q0 - m on the line (KZ Sec. 7.2.4)
    assert abs(CH.mmse(CH.Prior("rademacher"), v["gamma_se"]) - (sp["q0"] - sp["m"])) < 1e-12


def test_stein_holds_and_a_plant_fails():
    """E[g(X)(X - mu)] = sigma^2 E[g'(X)] (KZ Lemma 7) for X ~ N(mu, sigma^2); a wrong derivative fails, and so does a
    planted variance (sigma in place of sigma^2)."""
    cases = ((np.tanh, lambda x: 1 - np.tanh(x)**2, 0.3, 2.0), (np.sin, np.cos, -1.0, 0.5),
             (lambda x: x**3, lambda x: 3 * x**2, 0.7, 1.5))
    for g, dg, mu, s2 in cases:
        assert ID.check_stein(g, dg, mu, s2).ok, (mu, s2)
        assert not ID.check_stein(g, lambda x: 2 * dg(x), mu, s2).ok
        assert not _planted(ID, "stein_power", 0.5, lambda: ID.check_stein(g, dg, mu, s2)).ok


def test_nishimori_check_reports_missing_keys_by_refusal_not_by_a_pass():
    from replicant.core import Refusal
    try:
        ID.check_nishimori({"q": 0.5})
    except Refusal as r:
        assert r.reason == "precondition:overlaps", r
    else:
        raise AssertionError("a result without m was checked")
