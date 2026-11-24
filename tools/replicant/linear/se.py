"""Bayes-optimal state evolution of GAMP (linear/gamp.py) for a Channel x Prior pair (K6; KZ notes (12.141)-(12.144),
Sec. 16.3 (16.36)-(16.49) on the Nishimori line).

On the Nishimori line the four SE overlaps reduce to one, m = E[x0 x_hat] = E[x_hat^2] (KZ (12.141)-(12.144) with
m_u = q_u, m_omega = q_omega), rho = E x0^2, V = rho - m (the MSE):
    q_hat = alpha E_{omega, y}[g_out(omega, y, V)^2],   omega ~ N(0, m), z | omega ~ N(omega, V), y ~ P_out(y | z)
    m'    = rho - mmse(prior, snr = q_hat)            (the K2 scalar channel: E[x eta] = E[eta^2] = rho - mmse)
Channels (q_hat derived here, not taken from amp/gamp_se.py):
    awgn(Delta):  q_hat = alpha/(Delta + V)                                               (KZ (16.76)-(16.78))
    sign:         sum_y Phi(u_y) g^2 = phi(t)^2/(V Phi(t) Phi(-t)), t = omega/sqrt V, so
                  q_hat = (alpha/V) E_t[phi(t)^2/(Phi(t) Phi(-t))],  t ~ N(0, m/V)  (trapezoid h = 0.005, in units
                  of the standard deviation when it is < 1, else on |t| <= 14 with the analytically normalised Gaussian
                  weight: the integrand decays like t phi(t), the weight does not).
Two starts (the ruling on first-order transitions): "uninformative" m^0 = (E x0)^2 (GAMP's x_hat = E x0) and
"informative" m^0 = rho (1 - 1e-8) (V^0 = 1e-8 rho; m^0 = rho exactly with Delta = 0 would be the absorbing exact-
recovery point).  Each start runs until |m^{t+1} - m^t| < tol or the MSE falls below 1e-14 rho (recorded as exact
recovery, MSE 0), at most 20000 steps; `iters` instead runs exactly that many steps.  status: "unique" (both converged
to one fixed point), "two-fixed-points" (both converged, MSE differing by > 1e-6 rho: a first-order region), or
"undetermined" (a start did not converge: no fixed-point claim).  `overlap` / `mse` are the uninformative start's
(what GAMP reaches from its initialisation).  Deterministic N -> oo predictions, physics-level (the GLM SE is Rangan's
claim and BKMMZ's Claim 1; no theorem applied here).  _HOOKS["qhat"] (1.0) scales q_hat (planted errors only).
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.special import log_ndtr

from ..bayes.channels import mmse
from ..core import Refusal
from .gamp import Channel, _prior, prior_moments

_HOOKS = {"qhat": 1.0}
_U = np.linspace(-12.0, 12.0, 4801)
_WU = np.exp(-_U**2 / 2)
_WU /= _WU.sum()
_T = np.linspace(-14.0, 14.0, 5601)
_LOG_H = -_T**2 - math.log(2 * math.pi) - log_ndtr(_T) - log_ndtr(-_T)     # log phi^2/(Phi Phi-bar) on _T
_MAX = 20000


def _sign_qhat(alpha, m, V):
    s = math.sqrt(m / V)
    if s < 1:
        t = s * _U
        h = np.exp(-t**2 - math.log(2 * math.pi) - log_ndtr(t) - log_ndtr(-t))
        e = float(_WU @ h)
    else:
        w = np.exp(-_T**2 / (2 * s * s)) * (_T[1] - _T[0]) / (s * math.sqrt(2 * math.pi))
        e = float(w @ np.exp(_LOG_H))         # the analytic normalisation: the integrand, not the weight, is cut
    return alpha * e / V


def qhat(alpha, channel, m, rho):
    """alpha E[g_out^2] at overlap m (module docstring); inf at V = 0 for awgn with Delta = 0."""
    V = rho - m
    if channel.name == "awgn":
        d = channel.params["Delta"] + V
        out = math.inf if d <= 0 else alpha / d
    elif V <= 0:
        out = math.inf
    else:
        out = _sign_qhat(alpha, m, V)
    return _HOOKS["qhat"] * out


def _alpha(alpha):
    try:
        a = float(alpha)
    except (TypeError, ValueError):
        a = math.nan
    if not (math.isfinite(a) and a > 0):
        raise Refusal("precondition:alpha", f"alpha={alpha!r}: need a finite alpha = M/N > 0")
    return a


@dataclass
class GampSE:
    alpha: float
    channel: str
    prior: str
    rho: float
    starts: dict                     # start -> {"mse": [...] trajectory, "converged": bool}
    status: str                      # unique | two-fixed-points | undetermined
    overlap: float = math.nan        # m of the uninformative start (nan when undetermined)
    mse: float = math.nan
    mse_informative: float = math.nan
    label: str = "deterministic: N -> oo state-evolution prediction (quadrature), physics-level"
    note: str = ""

    def trajectory(self, start="uninformative"):
        return self.starts[start]["mse"]


def _run(alpha, channel, prior, rho, m0, iters, tol):
    m, tr = m0, [rho - m0]
    steps = iters or _MAX
    converged = False
    for _ in range(steps):
        q = qhat(alpha, channel, m, rho)
        new = rho if math.isinf(q) else rho - mmse(prior, q)
        if rho - new < 1e-14 * rho:
            new = rho
        tr.append(rho - new)
        done = abs(new - m) < tol or new == rho
        m = new
        if done and iters is None:
            converged = True
            break
    if iters is not None:
        converged = len(tr) > 2 and (abs(tr[-1] - tr[-2]) < tol or tr[-1] == 0)
    return {"mse": tr, "converged": converged}


def gamp_se(alpha, channel, prior, *, iters=None, tol=1e-13) -> GampSE:
    """Bayes-optimal SE from both starts (module docstring)."""
    a = _alpha(alpha)
    prior = _prior(prior)
    if not isinstance(channel, Channel):
        raise Refusal("precondition:known-channel", f"channel={channel!r}: need a linear.gamp.Channel")
    if iters is not None and (isinstance(iters, bool) or not isinstance(iters, int) or iters < 1):
        raise Refusal("precondition:positive-iters", f"iters={iters!r} is not a positive int")
    mean, _ = prior_moments(prior)
    rho = float(prior.second_moment)
    starts = {"uninformative": _run(a, channel, prior, rho, mean * mean, iters, tol),
              "informative": _run(a, channel, prior, rho, rho * (1 - 1e-8), iters, tol)}
    mu, mi = starts["uninformative"]["mse"][-1], starts["informative"]["mse"][-1]
    if not all(s["converged"] for s in starts.values()):
        status = "undetermined"
    else:
        status = "two-fixed-points" if abs(mu - mi) > 1e-6 * rho else "unique"
    note = (f"Bayes SE, {channel!r}, {prior!r}, alpha = {a}: uninformative MSE {mu:.6g}, informative {mi:.6g} "
            f"({status})")
    if status == "undetermined":
        return GampSE(a, channel.name, repr(prior), rho, starts, status, note="UNDETERMINED: " + note)
    return GampSE(a, channel.name, repr(prior), rho, starts, status, rho - mu, mu, mi, note=note)


__all__ = ["GampSE", "gamp_se", "qhat"]
