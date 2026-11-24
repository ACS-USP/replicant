"""The LASSO: soft-threshold AMP, its state evolution and the calibration threshold <-> lambda, proximal gradient, and
an exact coordinate-descent solver (K6; KZ notes Sec. 12.7, 16.6.2 (Alg. 4), 18.1).

Convention (the notes'): A in R^{M x N}, i.i.d. N(0, 1/N) entries, alpha = M/N, y = A x0 + sqrt(Delta) xi (`sigma2` =
Delta), x0 i.i.d. from a bayes/channels.py Prior (default Gauss-Bernoulli (1 - rho) delta_0 + rho N(0, 1)),
    LASSO:  x_hat = argmin (1/2)|y - A x|^2 + lam |x|_1.
Bayati & Montanari (IEEE TIT 58, 1997 (2012), arXiv:1008.2581, "BM"; fetched) use A_BM = A/sqrt(alpha) (variance 1/M),
delta = alpha and the same cost; dividing the notes' cost by alpha maps it to theirs with sigma_BM^2 = Delta/alpha and
lam_BM = lam/alpha.  The soft threshold is ST(u; theta) = sign(u) max(|u| - theta, 0) (KZ (16.84)-(16.86); (16.86) as
printed reads "R + lambda Sigma if R < lambda Sigma": the condition is R < -lambda Sigma).

AMP (KZ Alg. 4, (16.90)-(16.95)), a = v = 0, g = 0:  V = <v>, omega = A a - V g, Sigma = (1 + V)/alpha,
g = (y - omega)/(1 + V), R = a + A^T (y - omega)/alpha, a = ST(R; lam Sigma), v = Sigma 1{|R| > lam Sigma}.
Calibration (derived here; the notes do not print it).  At a fixed point V = Sigma b with b = <1{|R| > theta}>, so
Sigma = (1 + Sigma b)/alpha, i.e. Sigma = 1/(alpha - b), and the threshold is theta = lam Sigma:
    lam = theta (alpha - b)                                                        (*)
and y - omega = (y - A a)/(1 - b/alpha), so a = ST(a + A^T (y - A a)/(alpha - b); theta): the proximal-gradient fixed
point with eta = 1/(alpha - b) (KZ (12.154)), i.e. the LASSO KKT conditions with lam = theta/eta.  (*) is BM's
Eq. (1.13)
/ (1.15), lam_BM = theta (1 - b/delta), under the map above.  It is checked at finite N against coordinate descent
(tests/test_lasso.py), never fitted.
State evolution (KZ (18.13)-(18.14); BM (1.7)): R = x0 + tau Z with tau^2 = (E + Delta)/alpha, E the MSE;
    E' = mse(tau, theta) = E[(ST(X0 + tau Z; theta) - X0)^2],  b = P(|X0 + tau Z| > theta),
both in closed form per prior component (Gaussian partial moments; an atom is a component of variance 0).  The fixed
point at a given lam follows BM: for kappa = theta/tau above kappa_min ((1 + k^2) Phi(-k) - k phi(k) = alpha/2; 0 when
alpha > 1), tau_*^2(kappa) is the unique root of tau^2 = (Delta + mse(tau, kappa tau))/alpha (BM Prop. 1.3, brentq),
lam(kappa) = kappa tau_* (alpha - b) (BM (1.15) mapped), and kappa(lam) by brentq (BM Cor. 1.7: unique).
`calibrate(theta, ...)` takes theta as this kappa (the threshold in units of the effective noise tau, BM's alpha,
DMM's lambda) and returns lam.  Risk = E[(ST - X0)^2] = alpha tau^2 - Delta (BM Cor. 1.6).
Finality: BM Thm 1.5 / Cor. 1.6 prove the risk for Gaussian A, Delta > 0, lam > 0, P(X0 != 0) > 0 (exact-by-literature
there); lam = 0, alpha > 1 is least squares (HMRT, the ridgeless risk Delta/(alpha - 1)).  Finite-N runs are
statistical.  _HOOKS["slope"] (1.0) scales the soft-threshold slope b wherever it enters (the SE calibration and AMP's
Onsager term v): planted errors only.
"""
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from ..bayes.channels import Prior
from ..core import Refusal

_HOOKS = {"slope": 1.0}
_SQ2PI = math.sqrt(2 * math.pi)


def soft_threshold(u, theta):
    return np.sign(u) * np.maximum(np.abs(u) - theta, 0.0)


def _phi(a):
    return 0.0 if math.isinf(a) else math.exp(-a * a / 2) / _SQ2PI


def _pm(lo, hi):
    """(P, E[U 1], E[U^2 1]) of U ~ N(0, 1) on (lo, hi)."""
    p0 = float(ndtr(hi) - ndtr(lo))
    fl, fh = _phi(lo), _phi(hi)
    lf = 0.0 if math.isinf(lo) else lo * fl
    hf = 0.0 if math.isinf(hi) else hi * fh
    return p0, fl - fh, p0 + lf - hf


def _sq(c, d, m):
    """E[(c U + d)^2 1] from the partial moments m = (P, E U, E U^2)."""
    return c * c * m[2] + 2 * c * d * m[1] + d * d * m[0]


def mse_b(prior, tau, theta):
    """(E[(ST(X0 + tau Z; theta) - X0)^2], P(|X0 + tau Z| > theta)) in closed form (module docstring).
    Component (w, mu, v): Y = mu + s U, s^2 = v + tau^2, X0 | Y ~ N(mu + c (Y - mu), v tau^2/s^2), c = v/s^2."""
    err = b = 0.0
    for w, mu, v in prior.components:
        w, mu, v = float(w), float(mu), float(v)
        s = math.sqrt(v + tau * tau)
        c, omc = v / (s * s), tau * tau / (s * s)          # c and 1 - c (no cancellation as tau -> 0)
        hi, lo = (theta - mu) / s, (-theta - mu) / s
        up, dn, mid = _pm(hi, math.inf), _pm(-math.inf, lo), _pm(lo, hi)
        e = _sq(omc * s, -theta, up) + _sq(omc * s, theta, dn) + _sq(-c * s, -mu, mid)
        err += w * (e + v * tau * tau / (s * s))
        b += w * (up[0] + dn[0])
    return err, b


def _num(x, name, lo=0.0, strict=False):
    try:
        v = float(x)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v) or v < lo or (strict and v == lo):
        raise Refusal(f"precondition:{name}", f"{name}={x!r}: need a finite number {'>' if strict else '>='} {lo}")
    return v


def _get_prior(rho, prior):
    if prior is not None:
        if rho is not None:
            raise Refusal("precondition:known-parameter", "give rho (Gauss-Bernoulli) or a prior, not both")
        if not isinstance(prior, Prior):
            raise Refusal("precondition:known-prior", f"prior={prior!r}: need a bayes.channels.Prior")
        return prior
    r = _num(rho, "rho", strict=True)
    if r > 1:
        raise Refusal("precondition:rho", f"rho={rho!r}: the non-zero fraction must lie in (0, 1]")
    return Prior("gauss_bernoulli", rho=str(rho))


@dataclass
class LassoSE:
    alpha: float
    lam: float
    sigma2: float
    prior: str
    risk: float              # E = lim |x_hat - x0|^2 / N
    tau2: float              # effective noise (E + Delta)/alpha
    threshold: float         # theta = kappa tau
    kappa: float             # theta / tau (BM's alpha)
    b: float                 # P(|X0 + tau Z| > theta): the fraction of non-zeros of x_hat
    converged: bool = True
    label: str = "deterministic: N -> oo state-evolution fixed point (closed-form Gaussian moments)"
    note: str = ""


def _tau2(prior, a, D, kappa):
    """tau_*^2(kappa): root of (D + mse(tau, kappa tau))/a - tau^2 above D/a (module docstring)."""
    f = lambda t2: (D + mse_b(prior, math.sqrt(t2), kappa * math.sqrt(t2))[0]) / a - t2
    lo = D / a
    hi = max(2 * lo, 1.0)
    while f(hi) > 0:
        hi *= 2
        if hi > 1e16:
            raise Refusal("no-solution:bracket", f"no tau root for kappa = {kappa} (below kappa_min?)")
    return brentq(f, lo, hi, xtol=1e-15, rtol=1e-14)


def _kappa_min(a):
    if a > 1:
        return 0.0
    return brentq(lambda k: (1 + k * k) * ndtr(-k) - k * _phi(k) - a / 2, 0.0, 20.0, xtol=1e-15)


def _at_kappa(prior, a, D, kappa):
    t2 = _tau2(prior, a, D, kappa)
    t = math.sqrt(t2)
    err, b = mse_b(prior, t, kappa * t)
    b *= _HOOKS["slope"]
    return t2, err, b, kappa * t * (a - b)


def calibrate(theta, alpha, rho, sigma2, *, prior=None):
    """lam(theta) with theta the threshold in units of tau (BM's alpha): lam = theta tau_* (alpha - b) (module
    docstring, (*))."""
    a, D = _num(alpha, "alpha", strict=True), _num(sigma2, "sigma2", strict=True)
    k = _num(theta, "theta")
    pr = _get_prior(rho, prior)
    if k <= _kappa_min(a) and not (a > 1 and k == 0):
        raise Refusal("precondition:theta", f"theta={theta}: need theta > kappa_min(alpha) = {_kappa_min(a):.6g}")
    return _at_kappa(pr, a, D, k)[3]


def lasso_se(alpha, rho, lam, sigma2, *, prior=None) -> LassoSE:
    """The LASSO's SE fixed point at penalty lam (module docstring): risk, tau^2, threshold, kappa, b."""
    a, D, lm = _num(alpha, "alpha", strict=True), _num(sigma2, "sigma2", strict=True), _num(lam, "lambda")
    pr = _get_prior(rho, prior)
    kmin = _kappa_min(a)
    if lm == 0:
        if a <= 1:
            raise Refusal("precondition:lambda", f"lam = 0 needs alpha > 1 (least squares); alpha = {a}")
        k = 0.0
    else:
        lo = kmin + 1e-3 if a <= 1 else 0.0
        f = lambda k: _at_kappa(pr, a, D, k)[3] - lm
        while a <= 1 and f(lo) >= 0:              # lam(kappa) -> -oo as kappa -> kappa_min+ (BM Prop. 1.4)
            if lo - kmin < 1e-12:
                raise Refusal("no-solution:bracket", f"lam(kappa) >= {lm} down to kappa_min + 1e-12")
            lo = kmin + (lo - kmin) / 10
        hi = kmin + 1.0
        while f(hi) < 0:
            hi = 2 * hi + 1
        k = brentq(f, lo, hi, xtol=1e-14, rtol=1e-13)
    t2, err, b, got = _at_kappa(pr, a, D, k)
    return LassoSE(a, lm, D, repr(pr), err, t2, k * math.sqrt(t2), k, b,
                   note=f"BM calibration: kappa = {k:.10g} (kappa_min {kmin:.6g}), lam(kappa) = {got:.12g}")


def lasso_se_iterate(alpha, rho, lam, sigma2, *, prior=None, iters=None, tol=1e-14) -> LassoSE:
    """The state evolution of KZ Alg. 4 itself (the amp route), from its start a = v = 0: E^0 = E x0^2, V^0 = 0,
        tau_t^2 = (E^t + Delta)/alpha, Sigma_t = (1 + V^t)/alpha, theta_t = lam Sigma_t,
        (E^{t+1}, b_t) = mse_b(tau_t, theta_t),  V^{t+1} = Sigma_t b_t,
    iterated until |E^{t+1} - E^t| and |V^{t+1} - V^t| < tol (at most 20000 steps; `iters`: exactly that many).  Its
    fixed point is lasso_se's (Sigma = 1/(alpha - b) is the calibration (*)); `converged` False is reported, never
    refused."""
    a, D, lm = _num(alpha, "alpha", strict=True), _num(sigma2, "sigma2", strict=True), _num(lam, "lambda")
    pr = _get_prior(rho, prior)
    E, V, conv = float(pr.second_moment), 0.0, False
    for t in range(iters or 20000):
        t2 = (E + D) / a
        S = (1 + V) / a
        newE, b = mse_b(pr, math.sqrt(t2), lm * S)
        newV = S * _HOOKS["slope"] * b
        done = abs(newE - E) < tol and abs(newV - V) < tol
        E, V = newE, newV
        if not (math.isfinite(E) and math.isfinite(V)):
            break
        if done:
            conv = True
            if iters is None:
                break
    t2 = (E + D) / a
    S = (1 + V) / a
    _, b = mse_b(pr, math.sqrt(t2), lm * S)
    return LassoSE(a, lm, D, repr(pr), E, t2, lm * S, lm * S / math.sqrt(t2), b, conv,
                   label="deterministic: N -> oo state evolution of KZ Alg. 4, iterated",
                   note=f"{t + 1} SE steps from a = v = 0; {'converged' if conv else 'NOT converged'}")


# ---- finite N ----------------------------------------------------------------------------------------------------
def _check(A, y, lam):
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)
    if A.ndim != 2 or y.shape != (A.shape[0],) or not (np.all(np.isfinite(A)) and np.all(np.isfinite(y))):
        raise Refusal("precondition:matrix", f"A {A.shape}, y {y.shape}: need a finite M x N matrix and M values")
    return A, y, _num(lam, "lambda")


@dataclass
class LassoRun:
    x: np.ndarray
    converged: bool
    iters: int
    threshold: float = math.nan      # AMP: theta = lam Sigma
    Sigma: float = math.nan
    b: float = math.nan              # AMP: <1{|R| > theta}>
    kkt_residual: float = math.nan
    increments: list = field(default_factory=list)
    label: str = "statistical (finite N, one instance; never certified)"
    note: str = ""


def lasso_amp(A, y, lam, *, iters=2000, tol=1e-26) -> LassoRun:
    """KZ Alg. 4 (module docstring) until the mean squared increment < tol (at most `iters` steps)."""
    A, y, lm = _check(A, y, lam)
    M, N = A.shape
    alpha = M / N
    a, v, g = np.zeros(N), np.zeros(N), np.zeros(M)
    incs, conv = [], False
    for _ in range(iters):
        V = float(v.mean())
        omega = A @ a - V * g
        S = (1 + V) / alpha
        g = (y - omega) / (1 + V)
        R = a + A.T @ (y - omega) / alpha
        th = lm * S
        new = soft_threshold(R, th)
        on = np.abs(R) > th
        v = S * _HOOKS["slope"] * on
        incs.append(float((new - a) @ (new - a)) / N)
        a = new
        if not math.isfinite(incs[-1]):
            break
        if incs[-1] < tol:
            conv = True
            break
    b = float(on.mean())
    return LassoRun(a, conv, len(incs), th, S, b, kkt_residual(A, y, a, lm), incs,
                    note=f"KZ Alg. 4 at N = {N}, alpha = {alpha:.4g}, lam = {lm}: threshold {th:.6g}, b = {b:.4g}")


def kkt_residual(A, y, x, lam):
    """max_j distance of a_j^T (y - A x) from lam d|x_j| (the LASSO subgradient condition)."""
    c = A.T @ (y - A @ x)
    nz = x != 0
    r = np.where(nz, np.abs(c - lam * np.sign(x)), np.maximum(np.abs(c) - lam, 0.0))
    return float(r.max())


def proximal_gradient(A, y, lam, *, eta=None, iters=10000, tol=1e-14, x0=None) -> LassoRun:
    """KZ Sec. 12.7.2-12.7.3: x <- ST(x + eta A^T (y - A x); eta lam), eta = 1/|A|_2^2 by default (a descent step)."""
    A, y, lm = _check(A, y, lam)
    if eta is None:
        eta = 1 / float(np.linalg.norm(A, 2)) ** 2
    x = np.zeros(A.shape[1]) if x0 is None else np.array(x0, dtype=float)
    incs, conv = [], False
    for _ in range(iters):
        new = soft_threshold(x + eta * (A.T @ (y - A @ x)), eta * lm)
        incs.append(float(np.max(np.abs(new - x))))
        x = new
        if incs[-1] < tol:
            conv = True
            break
    return LassoRun(x, conv, len(incs), kkt_residual=kkt_residual(A, y, x, lm), increments=incs[-5:],
                    note=f"proximal gradient, eta = {eta:.4g}")


def lasso_exact(A, y, lam, *, tol=1e-11, max_sweeps=20000) -> LassoRun:
    """Coordinate descent on the Gram matrix (x_j <- ST(x_j + c_j/G_jj; lam/G_jj), c = A^T (y - A x) kept updated),
    full sweeps alternating with sweeps over the active set; stops when a full sweep moves no coordinate by more than
    tol and the KKT residual, recomputed from scratch, is < 10 tol.  Exact to the stated residual (a convex problem:
    the KKT conditions certify the minimiser up to that residual)."""
    A, y, lm = _check(A, y, lam)
    Gm = A.T @ A
    d = np.diag(Gm).copy()
    if np.any(d <= 0):
        raise Refusal("precondition:matrix", "a zero column: the coordinate step is undefined")
    N = A.shape[1]
    x = np.zeros(N)
    c = A.T @ y
    sweeps, conv = 0, False

    def sweep(idx):
        big = 0.0
        for j in idx:
            old = x[j]
            u = old + c[j] / d[j]
            new = math.copysign(max(abs(u) - lm / d[j], 0.0), u)
            if new != old:
                x[j] = new
                c[:] -= Gm[:, j] * (new - old)
                big = max(big, abs(new - old))
        return big

    while sweeps < max_sweeps:
        sweeps += 1
        if sweep(range(N)) < tol:
            kkt = kkt_residual(A, y, x, lm)
            if kkt < 10 * tol:
                conv = True
                break
            c[:] = A.T @ (y - A @ x)
            continue
        while sweeps < max_sweeps:
            sweeps += 1
            if sweep(np.flatnonzero(x)) < tol:
                break
    kkt = kkt_residual(A, y, x, lm)
    return LassoRun(x, conv, sweeps, kkt_residual=kkt, note=f"coordinate descent: {sweeps} sweeps, KKT {kkt:.2g}")


__all__ = ["LassoRun", "LassoSE", "calibrate", "kkt_residual", "lasso_amp", "lasso_exact", "lasso_se",
           "lasso_se_iterate", "mse_b",
           "proximal_gradient", "soft_threshold"]
