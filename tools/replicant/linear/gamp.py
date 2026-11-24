"""GAMP for generalized linear estimation with a Channel x Prior interface (K6; KZ notes Sec. 12.6, 12.8, Ch. 16).

Model (KZ (12.169)-(12.170), Ch. 16): x0 in R^N i.i.d. from a bayes/channels.py Prior (K2), A in R^{M x N} with
i.i.d. N(0, 1/N) entries, alpha = M/N, z = A x0, y_mu ~ P_out(y | z_mu).  Channels:
    awgn(Delta)   y = z + sqrt(Delta) xi:   g_out = (y - omega)/(Delta + V),  dg/domega = -1/(Delta + V)   (KZ (16.75);
                                            Rangan Table I)
    sign          y = sign(z) (noiseless):  Z_out = Phi(u), u = y omega/sqrt V, g_out = y phi(u)/(sqrt V Phi(u)),
                                            dg/domega = -g (omega/V + g)  (derived here from d/du (phi/Phi) =
                                            -u phi/Phi - (phi/Phi)^2)
g_out(omega, y, V) = d/domega ln int dz P_out(y|z) N(z; omega, V) (KZ (12.171)).  The input side is the K2 denoiser:
the tilted measure P(x) exp(-(x - R)^2/(2 Sigma)) is the scalar channel y' = sqrt(snr) x + z at snr = 1/Sigma and
y' = R/sqrt(Sigma), so (f_a, f_v) = channels.denoiser(prior, 1/Sigma) at y' (KZ (12.172)).
Iteration (scalar-variance GAMP: Rangan, arXiv:1010.5141, Alg. 1 with scalar variances (his Sec. IV "Algorithm 2"
simplification); KZ Alg. 3, (16.76)-(16.81)), from x_hat = E x, v = Var x, g = 0:
    V      = <v>
    omega  = A x_hat - onsager * V g                     (onsager = _HOOKS["onsager"], 1; a planted error changes it)
    g, dg  = g_out(omega, y, V), its omega-derivative
    1/Sigma = -(1/N) sum_mu dg_mu        ("derivative": Rangan's tau_r, A_ij^2 -> 1/N)
            or (1/N) sum_mu g_mu^2       ("nishimori": equal in expectation on the Nishimori line; amp/gamp.py's form)
    R      = x_hat + Sigma A^T g
    x_hat, v = f_a, f_v of the prior at (Sigma, R).
Gaussian prior and awgn: at a fixed point g (Delta + V) = y - A x_hat + V g, so g = (y - A x_hat)/Delta and
x_hat = Sigma' A^T g with Sigma' from the prior, i.e. (I + A^T A/Delta) x_hat = A^T y/Delta: the exact posterior mean
whatever the scalar Sigma (tests/test_linear_gamp.py).
Status: finite-N iterates are statistical (one instance; no error bar); Bayes-optimal only on the Nishimori line
(x0 from `prior`, y from `channel`).  If <v> falls below 1e-12 or 1/Sigma underflows to 0 (the marginals collapsed,
e.g. exact recovery), the iterate is kept fixed and the note says so (a float64 fixed point).  A non-finite V,
1/Sigma, g_out or iterate is a divergence: the run stops, returns its last finite iterate with finite=False and
converged=False (never frozen, never converged).  `seed`
names the instance (linear_instance draws it); the iteration itself is deterministic.
"""
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import log_ndtr

from ..bayes.channels import Prior, denoiser, sample
from ..core import Refusal

_HOOKS = {"onsager": 1.0}
_V_FLOOR = 1e-12
_SQ2PI = math.sqrt(2 * math.pi)
_PARAMS = {"awgn": {"Delta": "1"}, "sign": {}}


class Channel:
    """A named output channel P_out(y | z) with its parameters (awgn: Delta >= 0 as a float)."""
    __slots__ = ("name", "params")

    def __init__(self, name, **p):
        if name not in _PARAMS:
            raise Refusal("precondition:known-channel", f"channel={name!r}; known: {', '.join(_PARAMS)}")
        unknown = set(p) - set(_PARAMS[name])
        if unknown:
            raise Refusal("precondition:known-parameter",
                          f"channel {name!r} takes {sorted(_PARAMS[name]) or 'no parameters'}, not {sorted(unknown)}")
        params = {**_PARAMS[name], **{k: v for k, v in p.items() if v is not None}}
        if name == "awgn":
            try:
                d = float(params["Delta"])
            except (TypeError, ValueError):
                d = math.nan
            if not (math.isfinite(d) and d >= 0):
                raise Refusal("precondition:Delta", f"Delta={params['Delta']!r}: need a finite noise variance >= 0")
            params["Delta"] = d
        self.name, self.params = name, params

    def gout(self, omega, y, V):
        """(g_out, d g_out / d omega) at variance V > 0 (module docstring)."""
        if self.name == "awgn":
            d = self.params["Delta"] + V
            return (y - omega) / d, np.full_like(omega, -1 / d)
        sv = math.sqrt(V)
        u = y * omega / sv
        g = y * np.exp(-u * u / 2 - log_ndtr(u)) / (_SQ2PI * sv)
        return g, -g * (omega / V + g)

    def sample(self, z, rng):
        if self.name == "awgn":
            return z + math.sqrt(self.params["Delta"]) * rng.standard_normal(z.shape)
        return np.where(z >= 0, 1.0, -1.0)

    def check_labels(self, y):
        if self.name == "sign" and not np.all(np.abs(y) == 1):
            raise Refusal("precondition:sign-labels", "labels must be +-1 on the (noiseless) sign channel")

    def __repr__(self):
        return f"Channel({self.name!r}{''.join(f', {k}={v}' for k, v in self.params.items())})"


def _prior(prior):
    if not isinstance(prior, Prior):
        raise Refusal("precondition:known-prior", f"prior={prior!r}: need a bayes.channels.Prior")
    return prior


def prior_moments(prior):
    """(E x, Var x) of a channels Prior (floats)."""
    mean = sum(float(w) * float(mu) for w, mu, _ in prior.components)
    return mean, float(prior.second_moment) - mean * mean


def linear_instance(N, alpha, channel, prior, seed):
    """(A, y, x0) from numpy.random.default_rng([seed, 6]): M = round(alpha N), A_ij ~ N(0, 1/N)."""
    if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or N < 2:
        raise Refusal("precondition:size", f"N={N!r} must be an int >= 2")
    M = int(round(float(alpha) * N))
    if not (float(alpha) > 0 and M >= 1):
        raise Refusal("precondition:alpha", f"alpha={alpha} gives M = {M} measurements")
    rng = np.random.default_rng([int(seed), 6])
    x0 = sample(_prior(prior), rng, int(N))
    A = rng.standard_normal((M, int(N))) / math.sqrt(N)
    return A, channel.sample(A @ x0, rng), x0


@dataclass
class GAMPRun:
    x_hat: np.ndarray
    v: np.ndarray
    overlap_trace: list          # x_hat . x0 / N per step (None without a teacher)
    mse_trace: list              # |x_hat - x0|^2 / N per step (None without a teacher)
    converged: bool              # last increment <(x_hat^T - x_hat^{T-1})^2> < tol
    increments: list = field(default_factory=list)
    frozen: bool = False
    finite: bool = True          # False: an iterate went non-finite and the run stopped (never converged)
    alpha: float = 0.0
    N: int = 0
    seed: int = 0
    iters: int = 0
    label: str = "statistical"
    note: str = ""


def gamp(A, y, channel, prior, *, iters, seed, teacher=None, variance="derivative", tol=1e-10) -> GAMPRun:
    """`iters` GAMP steps (module docstring) on A (M x N, entries of variance 1/N), measurements y."""
    prior = _prior(prior)
    if not isinstance(channel, Channel):
        raise Refusal("precondition:known-channel", f"channel={channel!r}: need a linear.gamp.Channel")
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or min(A.shape) < 1 or not np.all(np.isfinite(A)):
        raise Refusal("precondition:matrix", f"A has shape {A.shape}; need a finite M x N matrix")
    y = np.asarray(y, dtype=float)
    if y.shape != (A.shape[0],):
        raise Refusal("precondition:label-count", f"{y.shape} measurements for {A.shape[0]} rows")
    channel.check_labels(y)
    if variance not in ("derivative", "nishimori"):
        raise Refusal("precondition:variance-form", f"variance={variance!r}: derivative | nishimori")
    if isinstance(iters, bool) or not isinstance(iters, int) or iters < 1:
        raise Refusal("precondition:positive-iters", f"iters={iters!r} is not a positive int")
    M, N = A.shape
    mean, var = prior_moments(prior)
    xh, v, g = np.full(N, mean), np.full(N, var), np.zeros(M)
    if teacher is not None:
        teacher = np.asarray(teacher, dtype=float)
        if teacher.shape != (N,):
            raise Refusal("precondition:teacher-length", f"teacher has shape {teacher.shape}; need ({N},)")
    ov = None if teacher is None else [float(xh @ teacher) / N]
    ms = None if teacher is None else [float((xh - teacher) @ (xh - teacher)) / N]
    incs, frozen, finite = [], False, True
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(iters):
            V = float(v.mean())
            lam = 0.0
            if math.isfinite(V) and V >= _V_FLOOR:
                omega = A @ xh - _HOOKS["onsager"] * V * g
                g_new, dg = channel.gout(omega, y, V)
                lam = float(-dg.sum() if variance == "derivative" else g_new @ g_new) / N
            if not (math.isfinite(V) and math.isfinite(lam)) or (lam > 0 and not np.all(np.isfinite(g_new))):
                finite = False                 # diverged: stop, never report it converged
                break
            if not (lam > 0 and V >= _V_FLOOR):
                frozen = True                  # collapsed marginals (V or 1/Sigma underflow): a float64 fixed point
                incs.append(0.0)
            else:
                R = xh + A.T @ g_new / lam
                mean_fn, var_fn = denoiser(prior, lam)
                ys = R * math.sqrt(lam)
                new, v_new = mean_fn(ys), var_fn(ys)
                if not (np.all(np.isfinite(new)) and np.all(np.isfinite(v_new))):
                    finite = False
                    break
                g, v = g_new, v_new
                incs.append(float((new - xh) @ (new - xh)) / N)
                xh = new
            if teacher is not None:
                ov.append(float(xh @ teacher) / N)
                ms.append(float((xh - teacher) @ (xh - teacher)) / N)
    note = (f"statistical (finite N = {N}, one instance, seed {seed}; never certified); Bayes-optimal only on the "
            f"Nishimori line ({prior!r}, {channel!r}); variance form {variance}")
    if frozen:
        note += "; the variances collapsed (<v> < 1e-12 or 1/Sigma underflowed to 0): the iterate was kept fixed"
    if not finite:
        note += (f"; NON-FINITE iterates at step {len(incs) + 1}: stopped, the last finite iterate is returned, "
                 "NOT converged")
    conv = finite and bool(incs) and incs[-1] < tol
    return GAMPRun(xh, v, ov, ms, conv, incs, frozen, finite, M / N, N, seed, iters, note=note)


__all__ = ["Channel", "GAMPRun", "gamp", "linear_instance", "prior_moments"]
