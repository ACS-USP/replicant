"""The identities of Bayes-optimal inference as validators (KZ Secs. 7.2.4-7.2.5).

Each check returns a refdata.CheckResult (ok, got, want, tol, label) and never raises because an identity fails: a
result off the Nishimori line is reported as such (label "not on the Nishimori line"), not refused.  Malformed input
(a result without the overlaps) is a Refusal.

  check_nishimori(result, tol)   q = m (KZ Theorem 13 with f = x^(1) x^(2): E<x>^2 = E<x> x*), result a mapping
                                 or object with m = E[x* <x>] and q = E[<x>^2]
  scalar_overlaps(prior, snr, model=None)   {m, q, q0} of the scalar channel (bayes/channels.overlaps)
  check_stein(g, dg, mu, sigma2)  E[g(X)(X - mu)] = sigma^2 E[g'(X)], X ~ N(mu, sigma^2) (KZ Lemma 7), by quadrature
  check_immse(prior, snr_grid)   dI/dsnr = mmse/2: Guo, Shamai & Verdu, IEEE Trans. Inf. Theory 51, 1261 (2005),
                                 arXiv:cs/0412108 (read), Theorem 1, Eq. (15), for Y = sqrt(snr) X + N, N ~ N(0, 1),
                                 information in nats (natural log); KZ Theorem 14, Eq. (7.42), with snr = 1/Delta.
                                 The derivative is a Richardson-extrapolated central difference of
                                 channels.mutual_information (an integral of log p(y)), compared with channels.mmse
                                 (an integral of the posterior variance): two separate quadratures.

_HOOKS (planted-error hooks, tests only): "immse_factor" (1/2), "stein_power" (1: sigma^2 enters to this power).
"""
import math
from fractions import Fraction

import numpy as np

from ..core import Refusal
from ..refdata import CheckResult
from . import channels as CH

_HOOKS = {"immse_factor": Fraction(1, 2), "stein_power": 1}


def _get(result, key):
    if isinstance(result, dict):
        v = result.get(key)
    else:
        v = getattr(result, key, None)
    if v is None:
        raise Refusal("precondition:overlaps", f"a Nishimori check needs m (overlap with the truth) and q (overlap "
                      f"of two posterior samples); {key!r} is missing from {result!r}")
    return float(v)


def check_nishimori(result, tol=1e-9):
    m, q = _get(result, "m"), _get(result, "q")
    ok = abs(q - m) <= tol
    return CheckResult(ok, repr(q), repr(m), repr(tol), "on the Nishimori line" if ok else
                       "not on the Nishimori line")


def scalar_overlaps(prior, snr, model=None):
    return CH.overlaps(prior, snr, model)


def check_stein(g, dg, mu=0.0, sigma2=1.0, tol=1e-10):
    """Both sides by the trapezoid rule over X = mu + sigma u, |u| <= 12 (exponentially accurate for analytic g)."""
    s2 = float(sigma2)
    if not s2 > 0:
        raise Refusal("precondition:sigma2", f"sigma2 = {sigma2!r}: need a variance > 0")
    x = float(mu) + math.sqrt(s2) * CH._Z
    lhs = float(CH._WZ @ (g(x) * (x - float(mu))))
    rhs = s2 ** _HOOKS["stein_power"] * float(CH._WZ @ dg(x))
    return CheckResult(abs(lhs - rhs) <= tol * max(1.0, abs(rhs)), repr(lhs), repr(rhs), repr(tol),
                       "Stein's lemma E[g(X)(X - mu)] = sigma^2 E[g'(X)]")


def _dI(prior, s, h):
    D = lambda k: (CH.mutual_information(prior, s + k) - CH.mutual_information(prior, s - k)) / (2 * k)
    return (4 * D(h / 2) - D(h)) / 3


def check_immse(prior, snr_grid, tol=1e-8, h=1e-3):
    """max over the grid of |dI/dsnr - mmse/2| <= tol (each snr > h)."""
    grid = [float(s) for s in snr_grid]
    if not grid or min(grid) <= h:
        raise Refusal("precondition:snr", f"snr grid {snr_grid!r}: need snr > {h} (central differences)")
    got = [_dI(prior, s, h) for s in grid]
    want = [float(_HOOKS["immse_factor"]) * CH.mmse(prior, s) for s in grid]
    err = max(abs(a - b) for a, b in zip(got, want))
    return CheckResult(bool(err <= tol), repr(np.round(got, 12).tolist()), repr(np.round(want, 12).tolist()),
                       repr(tol), f"I-MMSE dI/dsnr = mmse/2 (GSV Theorem 1, nats): max deviation {err:.2g}")


__all__ = ["check_immse", "check_nishimori", "check_stein", "scalar_overlaps"]
