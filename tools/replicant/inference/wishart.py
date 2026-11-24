"""The rank-one spiked Wishart (bipartite) model: two-overlap Bayes state evolution, rectangular Bayes AMP, and the BBP
edge of the top singular value (KZ Sec. 8.1 and Sec. 12.3.3, Eqs. (12.87)-(12.100); Lesieur, Krzakala & Zdeborova
(LKZ), J. Stat. Mech. (2017) 073403, arXiv:1701.00858, Eqs. (195)-(198), (248); Baik, Ben Arous & Peche (BBP), Ann.
Probab. 33, 1643 (2005), arXiv:math/0403022, Eqs. (4)-(5) and Thm 1.1).

Convention (replicant, as inference/spiked.py scales the Wigner case):
    Y = (lambda / n) u v^T + W,   Y and W n x m,   W_ij ~ N(0, 1/n),   alpha = m / n,
u in R^n, v in R^m i.i.d. from unit-variance priors (bayes/channels.py names or Prior objects; E x^2 = 1 is checked).
  - The notes' (12.87): W_notes = sqrt(lambda_notes / n) u v^T + xi, n x m, alpha = m/n: W_notes = sqrt(n) Y with
    lambda_notes = lambda^2 (the package-wide lambda_notes = lambda_LM = lambda^2).  The notes' Sec. 8.1 writes the
    same model with u in R^M, v in R^N and 1/N; Ch. 12's alpha is (length of v) / (length of u) = columns / rows,
    unrelated to the Ch. 3 / ridge ratios of K1.
  - LKZ: Y_LKZ = U V^T / sqrt(N) + sqrt(Delta) xi, U in R^N, V in R^M, alpha = M/N: Y_LKZ = sqrt(N Delta) Y with
    lambda = 1/sqrt(Delta) (lkz_delta_from_snr, snr_from_lkz_delta).
  - BBP: with v Gaussian the m columns of sqrt(n) Y are i.i.d. N(0, I_n + lambda^2 uhat uhat^T) (uhat = u/|u|, |u|^2
    = n to leading order): N = n variables, M = m samples, gamma^2 = M/N = alpha, l_1 = 1 + lambda^2, and S = (1/M)
    X X^T = (n/m) Y Y^T, so sigma_1(Y)^2 = alpha lambda_1(S).  BBP's threshold l_1 = 1 + gamma^(-1) (Thm 1.1 (36)) is
        lambda_c = alpha^(-1/4)     (lambda_notes,c = alpha^(-1/2); LKZ (198): Delta_c = sqrt(alpha)),
    the edge ((1 + gamma)/gamma)^2 (Eq. (4)) is sigma_edge = 1 + sqrt(alpha), and above it the outlier
    l_1 (1 + gamma^(-2) / (l_1 - 1)) (Thm 1.1 (39)) is sigma_1 = sqrt((1 + lambda^2)(alpha lambda^2 + 1)) / lambda.
    BBP prove this for complex samples and conjecture it (Sec. 1.3) for real ones; the real case is not proved in a
    paper fetched here.  For alpha < 1 transposing Y maps (lambda, alpha) to (lambda sqrt(alpha), 1/alpha) and the
    formulas are invariant (u Gaussian then plays v's role).

State evolution (Bayes-optimal; LKZ (195)-(196), (248); the notes' (12.97)-(12.100) with U* and V* swapped back, see
below).  With m(snr) = E[x E[x | sqrt(snr) x + z]] of a unit-variance prior (bayes/channels.py's channel):
    m_u^t = m_U(alpha lambda^2 m_v^t),    m_v^{t+1} = m_V(lambda^2 m_u^t),
derived here: the u-input of AMP is lambda Y vhat = lambda^2 alpha m_v U + lambda sqrt(alpha q_v) Z and q = m on the
Nishimori line.  The uninformative point m = 0 loses stability at alpha lambda^4 m_U'(0) m_V'(0) = 1
(wishart_se_threshold computes it from numerical slopes; = alpha^(-1/4) for unit variances).  Misprint: the notes'
(12.97)-(12.98) put V* in g_u's argument and (12.99)-(12.100) U* in g_v's; the signal reaching u is U* and the one
reaching v is V* (u^{t+1} = W g_v / sqrt(n) carries u* (v*. g_v)/n), as in LKZ (195)-(196).

AMP (the notes' (12.88)-(12.89) in replicant's scaling; A = lambda Y has entries of variance lambda^2 / n):
    h_u^t = lambda Y vhat^t - b_u^t uhat^{t-1},  uhat^t = f_u(h_u^t),  b_u^t = lambda^2 alpha <f_v'(h_v^t)>_m,
    h_v^{t+1} = lambda Y^T uhat^t - b_v^t vhat^t,  vhat^{t+1} = f_v(h_v^{t+1}),  b_v^t = lambda^2 <f_u'(h_u^t)>_n,
f(h) = E[x | y = h / sigma] at snr = mu^2 / sigma^2 and f'(h) = channels.denoiser_dy(...)(h / sigma) / sigma, with the
Bayes plug-in sigma_u^2 = mu_u = lambda^2 alpha |vhat|^2 / m and sigma_v^2 = mu_v = lambda^2 |uhat|^2 / n (Nishimori:
q = m), so the AMP does not call the SE.  The start vhat^0 is supplied (informative_start: E[v | sqrt(s0) v + z] with
fresh z, independent of W; the SE then starts at m_v^0 = m_V(s0)).  Finite-N runs are statistical, never final.

_HOOKS (planted errors, tests only): "threshold_power" (-1/4), "onsager_scale" (1), "alpha_in_u_channel" (True).
"""
import math
from dataclasses import dataclass

import numpy as np

from ..bayes import channels as CH
from ..core import Refusal
from ..csp import _checks

_HOOKS = {"threshold_power": -0.25, "onsager_scale": 1.0, "alpha_in_u_channel": True}


def _alpha(a):
    if isinstance(a, bool):
        raise Refusal("precondition:alpha", f"alpha = {a!r}: need a finite number > 0 (alpha = m/n)")
    try:
        v = float(a)
    except (TypeError, ValueError):
        raise Refusal("precondition:alpha", f"alpha = {a!r}: need a finite number > 0 (alpha = m/n)") from None
    if not (math.isfinite(v) and v > 0):
        raise Refusal("precondition:alpha", f"alpha = {a!r}: need a finite number > 0 (alpha = m/n)")
    return v


def _snr(s):
    if isinstance(s, bool):
        raise Refusal("precondition:snr", f"snr = {s!r}: lambda must be a finite number >= 0")
    try:
        v = float(s)
    except (TypeError, ValueError):
        raise Refusal("precondition:snr", f"snr = {s!r}: lambda must be a finite number >= 0") from None
    if not (math.isfinite(v) and v >= 0):
        raise Refusal("precondition:snr", f"snr = {s!r}: lambda must be a finite number >= 0")
    return v


def prior(p):
    """A channels.Prior from a name or a Prior; refused unless E x^2 = 1 (the conventions above assume it)."""
    pr = p if isinstance(p, CH.Prior) else CH.Prior(p)
    if pr.second_moment != 1:
        raise Refusal("precondition:unit-variance", f"{pr!r} has E x^2 = {pr.second_moment}; the spiked Wishart "
                      "conventions (lambda_c = alpha^(-1/4), BBP's l_1 = 1 + lambda^2) need unit-variance priors")
    return pr


# ---- conventions and the BBP edge -----------------------------------------------------------------------------------
def lkz_delta_from_snr(snr):
    return 1 / _snr(snr)**2


def snr_from_lkz_delta(delta):
    return 1 / math.sqrt(float(delta))


def notes_from_snr(snr):
    return _snr(snr)**2


def snr_from_notes(lam_notes):
    return math.sqrt(float(lam_notes))


def bbp_threshold(alpha):
    """lambda_c = alpha^(-1/4) (BBP Thm 1.1 (36) with gamma^2 = alpha, l_1 = 1 + lambda^2; LKZ (198))."""
    return _alpha(alpha) ** _HOOKS["threshold_power"]


def bbp_top_singular(snr, alpha):
    """N -> oo limit of sigma_1(Y): 1 + sqrt(alpha) up to lambda_c, sqrt((1 + l^2)(alpha l^2 + 1)) / l above."""
    lam, a = _snr(snr), _alpha(alpha)
    if lam <= bbp_threshold(a):
        return 1 + math.sqrt(a)
    return math.sqrt((1 + lam * lam) * (a * lam * lam + 1)) / lam


def bbp_edge_scale(n, alpha):
    """The scale of sigma_1(Y)'s fluctuations at the edge: BBP Eq. (5) (real null case) gives lambda_1(S) - edge of
    order (1 + gamma)^(4/3) / (gamma M^(2/3)) with N = min(n, m) <= M = max(n, m), gamma^2 = M/N; sigma_1(Y) =
    sqrt(M lambda_1(S) / n) converts it."""
    a = _alpha(alpha)
    N, M = sorted((n, a * n))
    g = math.sqrt(M / N)
    lam1 = ((1 + g) / g) ** 2
    return math.sqrt(M / n) * (1 + g) ** (4 / 3) / (g * M ** (2 / 3)) / (2 * math.sqrt(lam1))


# ---- state evolution ----------------------------------------------------------------------------------------------
def _m(pr, s):
    """m(snr) = E[x eta(y)] = E[eta(y)^2] (Nishimori) for y = sqrt(snr) x + z: sum over components of a trapezoid
    over z (channels' grid); no 1 - mmse cancellation, so m ~ snr is accurate as snr -> 0."""
    if s == 0:
        return 0.0
    mean, _ = CH.denoiser(pr, s)
    return float(sum(w * (CH._WZ @ mean(y)**2) for w, y in CH._y_nodes(pr, s)))


def _slope0(pr):
    D = lambda h: _m(pr, h) / h
    hs = (1e-4, 5e-5, 2.5e-5)
    r1 = [2 * D(hs[i + 1]) - D(hs[i]) for i in range(2)]
    return (4 * r1[1] - r1[0]) / 3


def wishart_se_threshold(alpha, prior_u, prior_v):
    """lambda where (0, 0) stops being stable: alpha lambda^4 m_U'(0) m_V'(0) = 1, slopes by Richardson extrapolation
    (an independent route to alpha^(-1/4))."""
    return (_alpha(alpha) * _slope0(prior(prior_u)) * _slope0(prior(prior_v))) ** -0.25


@dataclass(frozen=True)
class WishartSE:
    snr: float
    alpha: float
    prior_u: str
    prior_v: str
    m_u: float
    m_v: float
    trajectory: tuple      # ((nan, m_v^0), (m_u^0, m_v^1), (m_u^1, m_v^2), ...)
    converged: bool
    iterations: int
    note: str


def wishart_se(snr, alpha, prior_u, prior_v, *, iters=None, init_snr=None, tol=1e-14, max_iter=20000) -> WishartSE:
    """Bayes SE from m_v^0 = 1 (the informative start, E v^2) or m_v^0 = m_V(init_snr); `iters` fixes the number of
    steps (the trajectory for an AMP comparison), else it runs to |step| <= tol or max_iter (converged False)."""
    lam, a = _snr(snr), _alpha(alpha)
    pu, pv = prior(prior_u), prior(prior_v)
    for k, v in (("iters", iters), ("max_iter", max_iter)):
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 1):
            raise Refusal("precondition:positive-iters", f"{k} = {v!r} is not a positive int")
    mv = 1.0 if init_snr is None else _m(pv, _snr(init_snr))
    traj, cu = [(math.nan, mv)], a if _HOOKS["alpha_in_u_channel"] else 1.0
    mu, ok, t = math.nan, False, 0
    for t in range(1, (iters or max_iter) + 1):
        mu = _m(pu, cu * lam * lam * mv)
        new = _m(pv, lam * lam * mu)
        ok = abs(new - mv) <= tol and abs(mu - traj[-1][0]) <= tol if t > 1 else False
        mv = new
        traj.append((mu, mv))
        if ok and iters is None:
            break
    note = (f"Bayes SE ({'informative start m_v^0 = 1' if init_snr is None else f'm_v^0 = m_V({init_snr})'}), "
            f"{t} steps: m_u = {mu:.12g}, m_v = {mv:.12g}" + ("" if ok or iters else "; NOT converged"))
    return WishartSE(lam, a, pu.name, pv.name, mu, mv, tuple(traj), bool(ok), t, note)


# ---- finite N --------------------------------------------------------------------------------------------------------
def wishart_matrix(n, alpha, snr, prior_u, prior_v, seed):
    """(Y, u, v): W from default_rng([seed, 1]), u from [seed, 2], v from [seed, 3]; m = alpha n must be an int."""
    a, lam, seed = _alpha(alpha), _snr(snr), _checks.seed(seed)
    m = a * n if not isinstance(n, bool) and isinstance(n, int) else math.nan
    if not (n >= 10 and abs(m - round(m)) < 1e-9 and round(m) >= 2):
        raise Refusal("precondition:size", f"n = {n!r}, alpha = {alpha!r}: need an int n >= 10 with alpha n an "
                      "int >= 2")
    m = int(round(m))
    pu, pv = prior(prior_u), prior(prior_v)
    W = np.random.default_rng([seed, 1]).normal(size=(n, m)) / math.sqrt(n)
    u = CH.sample(pu, np.random.default_rng([seed, 2]), n)
    v = CH.sample(pv, np.random.default_rng([seed, 3]), m)
    return lam / n * np.outer(u, v) + W, u, v


def informative_start(x, prior_x, init_snr, seed):
    """E[x | sqrt(s0) x + z], z from default_rng([seed, 7]): a start correlated with x and independent of W."""
    s0 = _snr(init_snr)
    mean, _ = CH.denoiser(prior(prior_x), s0)
    x = np.asarray(x, dtype=float)
    return mean(math.sqrt(s0) * x + np.random.default_rng([_checks.seed(seed), 7]).normal(size=x.size))


@dataclass
class WishartAMP:
    u_hats: list           # uhat^t, t = 0..iters-1
    v_hats: list           # vhat^t, t = 0..iters (vhat^0 the start)
    snr: float
    alpha: float
    increments: list       # |vhat^{t+1} - vhat^t|^2 / m
    finite: bool           # False: an iterate went non-finite (the run stops there; its values are not usable)
    converged: bool        # finite and the last increment <= tol
    label: str = "statistical"


def _den(pr, sig2, h):
    """(f(h), f'(h)) of the Bayes denoiser for h = mu x + sigma z with mu = sigma^2 (snr = sigma^2); a non-finite
    sigma^2 gives non-finite output (caught by wishart_amp)."""
    if not math.isfinite(sig2):
        return np.full_like(h, math.nan), np.full_like(h, math.nan)
    if sig2 <= 0:
        return np.zeros_like(h), np.zeros_like(h)
    sig = math.sqrt(sig2)
    mean, _ = CH.denoiser(pr, sig2)
    return mean(h / sig), CH.denoiser_dy(pr, sig2)(h / sig) / sig


def wishart_amp(Y, prior_u, prior_v, *, snr, iters, v_init, tol=1e-6) -> WishartAMP:
    """Rectangular Bayes AMP on Y (module docstring).  A non-finite iterate stops the run with finite False;
    converged means finite with the last increment |vhat^{t+1} - vhat^t|^2 / m <= tol (never a Refusal)."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2 or min(Y.shape) < 2 or not np.all(np.isfinite(Y)):
        raise Refusal("precondition:matrix", f"Y has shape {Y.shape} or non-finite entries; need a finite n x m array")
    n, m = Y.shape
    a, lam = m / n, _snr(snr)
    pu, pv = prior(prior_u), prior(prior_v)
    if isinstance(iters, bool) or not isinstance(iters, int) or iters < 1:
        raise Refusal("precondition:positive-iters", f"iters = {iters!r} is not a positive int")
    vh = np.asarray(v_init, dtype=float)
    if vh.shape != (m,):
        raise Refusal("precondition:start", f"v_init has shape {vh.shape}; need ({m},)")
    sc, l2 = _HOOKS["onsager_scale"], lam * lam
    u_old, b_u = np.zeros(n), 0.0
    us, vs, incs, finite = [], [vh], [], True
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(iters):
            uh, du = _den(pu, l2 * a * float(vh @ vh) / m, lam * (Y @ vh) - sc * b_u * u_old)
            b_v = l2 * float(np.mean(du))
            vh_new, dv = _den(pv, l2 * float(uh @ uh) / n, lam * (Y.T @ uh) - sc * b_v * vh)
            b_u = l2 * a * float(np.mean(dv))
            if not (np.all(np.isfinite(uh)) and np.all(np.isfinite(vh_new)) and math.isfinite(b_u)):
                finite = False
                break
            incs.append(float((vh_new - vh) @ (vh_new - vh)) / m)
            u_old, vh = uh, vh_new
            us.append(uh)
            vs.append(vh)
    ok = finite and bool(incs) and incs[-1] <= tol
    return WishartAMP(us, vs, lam, a, incs, finite, ok)


__all__ = ["WishartAMP", "WishartSE", "bbp_edge_scale", "bbp_threshold", "bbp_top_singular", "informative_start",
           "lkz_delta_from_snr", "notes_from_snr", "prior", "snr_from_lkz_delta", "snr_from_notes", "wishart_amp",
           "wishart_matrix", "wishart_se", "wishart_se_threshold"]
