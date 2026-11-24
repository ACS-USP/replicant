"""The Gaussian scalar channel, its priors and their Bayes-optimal denoisers (KZ Ch. 7, Sec. 7.2).

Convention (one for the package; K5 and K6 plug these denoisers into AMP):
    y = sqrt(snr) x + z,   z ~ N(0, 1),   x ~ prior,
Guo, Shamai & Verdu, IEEE Trans. Inf. Theory 51, 1261 (2005), arXiv:cs/0412108 (read), Eq. (3).  KZ write
y_i = x* + sqrt(Delta) z_i, i = 1..N (Eq. (7.4)): their sum is sufficient, and sum_i y_i / sqrt(N Delta) is this
channel at snr = N / Delta (one observation: snr = 1/Delta, the "inverse noise" of KZ Theorem 14).
inference/spiked.py's y' = gamma X + sqrt(gamma) G is sqrt(gamma) times this channel at snr = gamma.

Priors are finite mixtures of point masses and Gaussians, components (w_k, mu_k, v_k) with exact rational entries
(v_k = 0: an atom).  A prior is added in one place, _FAMILIES: name -> (parameter defaults, components(**p)).
    gaussian(var=1)                   N(0, var)
    rademacher()                      (delta_{+1} + delta_{-1}) / 2
    gauss_bernoulli(rho=1/2, var=1)   (1 - rho) delta_0 + rho N(0, var)        (KZ Sec. 7.2.1, item 3: 1/2, 1)
    sparse(rho=1/2, a=1)              (1 - rho) delta_0 + (rho/2)(delta_a + delta_{-a}), the diluted Rademacher
                                      prior; variance rho a^2 (a = 1/sqrt(rho) for unit variance: pick rho with
                                      a rational, e.g. rho = 1/4, a = 2)
    diluted(rho=1/2)                  the unit-variance diluted Rademacher prior (K5): sparse with a = 1/sqrt(rho),
                                      the amplitude kept exact as Sqrt(1/rho) (irrational for rho = 0.08); its
                                      square is the rational 1/rho, so second_moment is exactly 1
Posterior: component k has evidence N(y; sqrt(snr) mu_k, 1 + snr v_k) and, given k, x is Gaussian with mean
mu_k + v_k sqrt(snr) (y - sqrt(snr) mu_k) / (1 + snr v_k) and variance v_k / (1 + snr v_k).

  Prior(name, **p)                      .components, .second_moment (q0 = E x^2)
  denoiser(prior, snr) -> (mean_fn, var_fn)   numpy-vectorised posterior mean (KZ's eta, Eq. (7.22)) and variance
  denoiser_dy(prior, snr) -> fn         d E[x|y] / dy = sqrt(snr) Var[x|y] (the Onsager derivative, K5)
  map_estimate(prior, y, snr)           argmax of the posterior (atoms only, or a single Gaussian)
  mmse(prior, snr), mutual_information(prior, snr)       binary64 trapezoid quadrature over y (two integrals:
                                        E Var[x|y], and h(Y) = -E log p(y))
  overlaps(prior, snr, model=None)      {m = E x eta, q = E eta^2, q0}: data from `prior`, denoised with `model`
  scalar_moments(prior, snr, slope=False)  fast path (atom-only priors, snr a float or 1-d array): m = E x eta and,
                                        with slope, dm/dsnr = E Var[x|y]^2 (Guo-Shamai-Verdu: mmse' = -E Var^2)
  mmse_ball, information_ball           certified (arb) for gaussian (GSV Eqs. (11), (13)) and rademacher (GSV
                                        Eqs. (17), (18), via bounds/gauss.py); other priors are refused
  posterior_ball(prior, data, Delta)    KZ Eqs. (7.5)-(7.10) for N observations, in arb: mean, var, weights
  DATASETS                              KZ's printed data (7.6), (7.8), (7.11) (inputs, not targets)
"""
import math
from fractions import Fraction

import numpy as np

from ..core import Refusal
from ..lattice.ball import arb_q, ball_of, baller, rational

# trapezoid over z in [-12, 12]: exponentially accurate for these analytic integrands up to snr ~ 1e3 (the poles of
# tanh(sqrt(snr) y) sit at |Im y| = pi / (2 sqrt(snr)))
_Z = np.linspace(-12.0, 12.0, 4801)
_WZ = np.exp(-_Z**2 / 2) / math.sqrt(2 * math.pi) * (_Z[1] - _Z[0])
_WZ = _WZ / _WZ.sum()


def _rho(p):
    r = rational(p, "rho", positive=False)
    if not 0 < r <= 1:
        raise Refusal("precondition:rho", f"rho = {p!r}: the non-zero fraction must lie in (0, 1]")
    return r


def _var(v):
    return rational(v, "var")


class Sqrt:
    """sign * sqrt(radicand), radicand a positive Fraction: an exact atom position whose square is rational."""
    __slots__ = ("sign", "radicand")

    def __init__(self, radicand, sign=1):
        self.radicand, self.sign = Fraction(radicand), sign

    def __float__(self):
        return self.sign * math.sqrt(self.radicand)

    def __neg__(self):
        return Sqrt(self.radicand, -self.sign)

    def __mul__(self, other):
        if isinstance(other, Sqrt) and other.radicand == self.radicand:
            return self.sign * other.sign * self.radicand
        return NotImplemented

    def __gt__(self, other):
        return float(self) > float(other)

    def __lt__(self, other):
        return float(self) < float(other)

    def __eq__(self, other):
        return isinstance(other, Sqrt) and (self.sign, self.radicand) == (other.sign, other.radicand)

    def __hash__(self):
        return hash((self.sign, self.radicand))

    def arb(self):
        return self.sign * arb_q(self.radicand).sqrt()

    def __repr__(self):
        return f"{'-' if self.sign < 0 else ''}sqrt({self.radicand})"


def _arb(x):
    return x.arb() if isinstance(x, Sqrt) else arb_q(x)


_FAMILIES = {
    "gaussian": ({"var": 1}, lambda var: [(Fraction(1), Fraction(0), _var(var))]),
    "rademacher": ({}, lambda: [(Fraction(1, 2), Fraction(1), Fraction(0)), (Fraction(1, 2), Fraction(-1), Fraction(0))]),
    "gauss_bernoulli": ({"rho": Fraction(1, 2), "var": 1},
                        lambda rho, var: [(1 - _rho(rho), Fraction(0), Fraction(0)), (_rho(rho), Fraction(0), _var(var))]),
    "sparse": ({"rho": Fraction(1, 2), "a": 1},
               lambda rho, a: [(1 - _rho(rho), Fraction(0), Fraction(0)), (_rho(rho) / 2, rational(a, "a"), Fraction(0)),
                               (_rho(rho) / 2, -rational(a, "a"), Fraction(0))]),
    "diluted": ({"rho": Fraction(1, 2)},
                lambda rho: [(1 - _rho(rho), Fraction(0), Fraction(0)),
                             (_rho(rho) / 2, Sqrt(1 / _rho(rho)), Fraction(0)),
                             (_rho(rho) / 2, -Sqrt(1 / _rho(rho)), Fraction(0))]),
}


class Prior:
    """A named prior with its parameters (exact rationals) and its mixture components."""
    __slots__ = ("name", "params", "components")

    def __init__(self, name, **p):
        if not isinstance(name, str) or name not in _FAMILIES:
            raise Refusal("precondition:known-prior", f"prior = {name!r}; known: {', '.join(_FAMILIES)}")
        defaults, build = _FAMILIES[name]
        unknown = set(p) - set(defaults)
        if unknown:
            raise Refusal("precondition:known-parameter", f"prior {name!r} takes {sorted(defaults) or 'no parameters'}"
                          f", not {sorted(unknown)}")
        params = {k: rational(v, k) for k, v in {**defaults, **{k: v for k, v in p.items() if v is not None}}.items()}
        comps = [(w, mu, v) for w, mu, v in build(**params) if w > 0]
        self.name, self.params, self.components = name, params, tuple(comps)

    def var(self):
        """The Gaussian component's variance (the `var` parameter; 1 for priors without one)."""
        return self.params.get("var", Fraction(1))

    @property
    def second_moment(self):
        return sum(w * (mu * mu + v) for w, mu, v in self.components)

    def __repr__(self):
        return f"Prior({self.name!r}, {', '.join(f'{k}={v}' for k, v in self.params.items())})"


def _snr(s):
    q = rational(s, "snr", positive=False)
    if q < 0:
        raise Refusal("precondition:snr", f"snr = {s!r}: need snr >= 0")
    return q


def _arrays(prior):
    w, mu, v = (np.array([float(c[i]) for c in prior.components]) for i in range(3))
    return w, mu, v


def _posterior(prior, y, s, arrays=None):
    """(responsibilities r[..., k], component means m[..., k], component variances c[k]) at snr s (float);
    `arrays` = _arrays(prior), passed by denoiser() so its closures convert the Fractions once, not per call."""
    w, mu, v = arrays if arrays is not None else _arrays(prior)
    y = np.asarray(y, dtype=float)[..., None]
    rs = math.sqrt(s)
    d = 1 + s * v
    logev = np.log(w) - 0.5 * np.log(d) - (y - rs * mu)**2 / (2 * d)
    logev -= logev.max(axis=-1, keepdims=True)
    r = np.exp(logev)
    r /= r.sum(axis=-1, keepdims=True)
    return r, mu + v * rs * (y - rs * mu) / d, v / d


def denoiser(prior, snr):
    """(mean_fn, var_fn): y (any numpy shape) -> E[x | y], Var[x | y] for y = sqrt(snr) x + z."""
    s = float(_snr(snr))
    arr = _arrays(prior)

    def mean_fn(y):
        r, m, _ = _posterior(prior, y, s, arr)
        return (r * m).sum(axis=-1)

    def var_fn(y):
        r, m, c = _posterior(prior, y, s, arr)
        mean = (r * m).sum(axis=-1)
        return (r * (m * m + c)).sum(axis=-1) - mean * mean
    return mean_fn, var_fn


def denoiser_dy(prior, snr):
    """y -> d E[x | y] / dy for y = sqrt(snr) x + z.  Differentiating E[x|y] = int x p(x) e^{sqrt(snr) x y} dx / Z(y)
    under the integral gives sqrt(snr) (E[x^2|y] - E[x|y]^2): the snr-scaled posterior variance (K5's Onsager
    terms; tested against a central difference in tests/test_wishart.py)."""
    rs = math.sqrt(float(_snr(snr)))
    _, var_fn = denoiser(prior, snr)
    return lambda y: rs * var_fn(y)


def map_estimate(prior, y, snr):
    """argmax_x P(x | y).  Atoms only: the atom of largest posterior mass; one Gaussian: its posterior mean.  A
    prior mixing atoms and densities has no mode (a mass and a density do not compare): refused."""
    s = float(_snr(snr))
    atoms = [c[2] == 0 for c in prior.components]
    r, m, _ = _posterior(prior, y, s)
    if all(atoms):
        return np.take(_arrays(prior)[1], r.argmax(axis=-1))
    if len(atoms) == 1:
        return m[..., 0]
    raise Refusal("unsupported:map-mixed-measure", f"{prior!r} mixes point masses and densities: the posterior has "
                  "no mode (use the optimal decision between components, KZ Theorem 12)")


def _y_nodes(prior, s):
    """[(weight_k, y nodes)] for E over y = sqrt(s) x + z: component k gives y = sqrt(s) mu_k + sqrt(1 + s v_k) z."""
    return [(float(w), math.sqrt(s) * float(mu) + math.sqrt(1 + s * float(v)) * _Z) for w, mu, v in prior.components]


def mmse(prior, snr):
    """E Var[x | y] (binary64 quadrature)."""
    s = float(_snr(snr))
    _, var_fn = denoiser(prior, s)
    return float(sum(w * (_WZ @ var_fn(y)) for w, y in _y_nodes(prior, s)))


def _log_py(prior, s, y):
    w, mu, v = _arrays(prior)
    d = 1 + s * v
    t = np.log(w) - 0.5 * np.log(2 * math.pi * d) - (y[:, None] - math.sqrt(s) * mu)**2 / (2 * d)
    top = t.max(axis=1)
    return top + np.log(np.exp(t - top[:, None]).sum(axis=1))


def mutual_information(prior, snr):
    """I(X; Y) in nats = h(Y) - (1/2) log(2 pi e), h(Y) = -E log p(y) (binary64 quadrature)."""
    s = float(_snr(snr))
    if s == 0:
        return 0.0
    hy = -sum(w * (_WZ @ _log_py(prior, s, y)) for w, y in _y_nodes(prior, s))
    return float(hy - 0.5 * math.log(2 * math.pi * math.e))


# the fast path's grid: trapezoid h = 0.01 on [-10, 10] (agrees with overlaps()' h = 0.005 on [-12, 12] to 1e-13,
# tests/test_spiked_diluted.py)
_ZF = np.linspace(-10.0, 10.0, 2001)
_WF = np.exp(-_ZF**2 / 2)
_WF = _WF / _WF.sum()


def scalar_moments(prior, snr, slope=False):
    """m(snr) = E[x eta(y)] (and, with slope, dm/dsnr = E[Var(x|y)^2]) for y = sqrt(snr) x + z and an atom-only prior,
    vectorised over a 1-d array of snr.  A prior symmetric under x -> -x is averaged over its x >= 0 atoms only (the
    mirror atoms contribute equally); without slope the atom at 0 is skipped (it contributes x eta = 0)."""
    w, mu, v = _arrays(prior)
    if np.any(v > 0):
        raise Refusal("unsupported:moments-prior", f"{prior!r} has a Gaussian component; scalar_moments is the "
                      "atom-only fast path (use overlaps)")
    s = np.atleast_1d(np.asarray(snr, dtype=float))
    if not np.all(np.isfinite(s)) or np.any(s < 0):
        raise Refusal("precondition:snr", f"snr = {snr!r}: need finite snr >= 0")
    sym = sorted(zip(w, mu)) == sorted(zip(w, -mu))
    keep = [k for k in range(len(w)) if (not sym or mu[k] >= 0) and (slope or mu[k] != 0)]
    rs = np.sqrt(s)[:, None, None]
    lw = np.log(w)
    m = np.zeros(s.size)
    sl = np.zeros(s.size)
    for k in keep:
        f = w[k] * (2 if sym and mu[k] > 0 else 1)
        y = rs[:, :, 0] * mu[k] + _ZF[None, :]
        L = lw + rs * mu * y[..., None] - (rs * mu)**2 / 2
        L -= L.max(axis=-1, keepdims=True)
        p = np.exp(L)
        p /= p.sum(axis=-1, keepdims=True)
        eta = p @ mu
        m += f * mu[k] * (eta @ _WF)
        if slope:
            sl += f * ((p @ (mu * mu) - eta * eta)**2 @ _WF)
    if np.ndim(snr) == 0:
        return (float(m[0]), float(sl[0])) if slope else float(m[0])
    return (m, sl) if slope else m


def overlaps(prior, snr, model=None):
    """{m = E[x eta(y)], q = E[eta(y)^2], q0 = E x^2}: x from `prior`, eta the posterior mean under `model`
    (default: the same prior, the Nishimori line)."""
    s = float(_snr(snr))
    mean_fn, _ = denoiser(model or prior, s)
    wu = _WZ[::20] / _WZ[::20].sum()          # the x-rule of a Gaussian component: h = 0.1 on |u| <= 12
    m = q = 0.0
    for w, mu, v in prior.components:
        # x = mu + sqrt(v) u (u = 0 for an atom), y = sqrt(s) x + z: a tensor rule over (u, z)
        x = float(mu) + (math.sqrt(float(v)) * _Z[::20, None] if v > 0 else np.zeros((1, 1)))
        eta = mean_fn(math.sqrt(s) * x + _Z[None, :])
        wx = wu if v > 0 else np.ones(1)
        m += float(w) * (wx @ (x * eta) @ _WZ)
        q += float(w) * (wx @ (eta * eta) @ _WZ)
    return {"m": float(m), "q": float(q), "q0": float(prior.second_moment)}


# ---- certified values (arb) ---------------------------------------------------------------------------------------
def _certified(prior):
    if prior.name not in ("gaussian", "rademacher"):
        raise Refusal("unsupported:certified-prior", f"{prior!r}: certified MMSE and I exist for gaussian (closed "
                      "form) and rademacher (GSV Eqs. (17)-(18) by acb.integral); this prior has float quadrature only")


def _ball(fn):
    with baller().hygiene.ctx_guard(prec=256):
        return ball_of(fn())


def mmse_ball(prior, snr):
    """gaussian: var / (1 + snr var) (GSV Eq. (13) scaled); rademacher: 1 - E tanh^2(snr + sqrt(snr) z) (GSV
    Eq. (17), whose E tanh equals E tanh^2 by the Nishimori identity, KZ Theorem 13)."""
    _certified(prior)
    s = _snr(snr)

    def f():
        if prior.name == "gaussian":
            v = arb_q(prior.var())
            return v / (1 + arb_q(s) * v)
        from ..bounds.gauss import expect
        a = arb_q(s)
        return 1 - expect("tanh2", a.sqrt(), a)
    return _ball(f)


def information_ball(prior, snr):
    """gaussian: (1/2) log(1 + snr var) (GSV Eq. (11)); rademacher: snr - E log cosh(snr + sqrt(snr) z) (GSV
    Eq. (18))."""
    _certified(prior)
    s = _snr(snr)

    def f():
        a = arb_q(s)
        if prior.name == "gaussian":
            return (1 + a * arb_q(prior.var())).log() / 2
        from ..bounds.gauss import expect
        return a - expect("lncosh", a.sqrt(), a)
    return _ball(f)


DATASETS = {   # KZ Eqs. (7.6), (7.8), (7.11), as printed (the "]" after 1.11920946 in (7.11) is a typo)
    "7.6": ("1.04431591", "2.55352006", "1.43665582", "1.37069702", "0.77697312"),
    "7.8": ("0.04724576", "1.26855971", "-0.19887457", "1.09534511", "-1.46442807", "0.44767123", "2.6244575",
            "1.94488421", "0.58953688", "0.572018"),
    "7.11": ("0.99978688", "1.7956116", "-0.43158072", "3.07234211", "1.11920946", "0.53248943", "0.80011329",
             "-0.52783428", "0.40378413", "-0.00223177"),
}


def posterior_ball(prior, data, Delta):
    """The posterior of x given N observations y_i = x + sqrt(Delta) z_i, in arb (the decimal data exact):
    {mean, var, weights (per component), p_plus (mass on x > 0), p_nonzero (mass off x = 0), N} as Balls."""
    if isinstance(data, str) or not data:
        raise Refusal("precondition:data", f"data = {data!r}: need a non-empty list of numbers (or a KZ data set)")
    d = rational(Delta, "Delta")
    with baller().hygiene.ctx_guard(prec=256):
        from flint import arb
        ys = [arb_q(rational(y, "data", positive=False)) for y in data]
        n = len(ys)
        ybar, nv = sum(ys) / n, arb_q(d) / n
        logs, means, vars_ = [], [], []
        for w, mu, v in prior.components:
            tot = arb_q(v) + nv
            logs.append(arb_q(w).log() - tot.log() / 2 - (ybar - _arb(mu))**2 / (2 * tot))
            means.append(_arb(mu) + arb_q(v) * (ybar - _arb(mu)) / tot)
            vars_.append(arb_q(v) * nv / tot)
        top = max(logs, key=lambda x: x.mid())
        ex = [(x - top).exp() for x in logs]
        r = [e / sum(ex) for e in ex]
        mean = sum(ri * mi for ri, mi in zip(r, means))
        var = sum(ri * (mi * mi + ci) for ri, mi, ci in zip(r, means, vars_)) - mean * mean
        plus = sum((ri for ri, (_, mu, v) in zip(r, prior.components) if mu > 0 and v == 0), arb(0))
        nonzero = sum((ri for ri, (_, mu, v) in zip(r, prior.components) if mu != 0 or v > 0), arb(0))
        return {"mean": ball_of(mean), "var": ball_of(var), "weights": tuple(ball_of(x) for x in r),
                "p_plus": ball_of(plus), "p_nonzero": ball_of(nonzero), "N": n}


__all__ = ["DATASETS", "Prior", "Sqrt", "denoiser", "denoiser_dy", "scalar_moments", "information_ball", "map_estimate",
           "mmse", "mmse_ball",
           "mutual_information", "overlaps", "posterior_ball"]
