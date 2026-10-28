"""Independent integer-n check of the pinned-direction stability sectors (no class formulas).

  hessian_integer_n(n, alpha, d, c, kappa, q, M)  -> numpy Hessian of Phi_n over
                                                     (Q_ab, a < b ; M^1..M^n) at the RS point
  sector_matrices(n, hessian)                     -> (replicon_eig, leak, T, T_leak)

For an integer n the entropy Hessian is assembled from the matrix identity
(1/2)[tr(K^-1 d2K) - tr(K^-1 dK K^-1 dK)] with explicit K = Q - a a^T, and the energy Hessian from
the SLOT RULE alone: every derivative of Z = E_{t ~ N(Q)} prod_a Theta(H^a - t_a) with respect to
Q_ab is d^2/dt_a dt_b, with respect to H^a is -d/dt_a (Price's theorem), so a mixed derivative is
E_z[prod_slots (-1)^{h_a} P_{k_a}(z) P^(n - #slots)] with the RS split; float64 quadrature.  The
projection onto the replicon irrep and the standard irrep is done on explicit vectors, so the
class formulas of pinned_modes and the Schur statements (replicon decoupled from M; standard
block closed) are both tested against it.  A TEST/VALIDATION helper: float64, n >= 4.
"""
import itertools

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from .core import Refusal
from .derive_pinned import _PLANT as _PLANT_DERIVE


def _slot_derivs(z, q, H):
    s = np.sqrt(1 - q)
    u = (H - np.sqrt(q) * z) / s
    ph = norm.pdf(u)
    return norm.cdf(u), -ph / s, -u * ph / s**2, (1 - u * u) * ph / s**3     # P, P1, P2, P3


def _energy_hessian(n, alpha, d, c, kappa, q, a, pairs):
    qp = np.sqrt((1 + (0 if _PLANT_DERIVE.get("gram") == "identity" else c)) / 2)
    H, b = np.sqrt(d) * (qp * a - kappa), np.sqrt(d) * qp
    nv = len(pairs) + n

    def expect(power, ks):
        def f(z):
            P = _slot_derivs(z, q, H)
            v = P[0]**power
            for k in ks:
                v *= P[k]
            return norm.pdf(z) * v
        return quad(f, -14, 14, epsabs=1e-14, epsrel=1e-12, limit=500)[0]

    def inc(v):
        if v < len(pairs):
            i, j = pairs[v]
            return {i: 1, j: 1}, 1.0
        return {v - len(pairs): 1}, -b          # H derivative: -d/dt, chain dH/da = b
    cache = {}

    def Z(vs):
        ks, sgn = {}, 1.0
        for v in vs:
            slots, s = inc(v)
            sgn *= s
            for k, cnt in slots.items():
                ks[k] = ks.get(k, 0) + cnt
        key = tuple(sorted(ks.values()))
        if key not in cache:
            cache[key] = expect(n - len(ks), list(key))
        return sgn * cache[key]
    Z0 = expect(n, [])
    Z1 = np.array([Z([v]) for v in range(nv)])
    Hs = np.zeros((nv, nv))
    for i in range(nv):
        for j in range(i, nv):
            Hs[i, j] = Hs[j, i] = alpha * (Z([i, j]) / Z0 - Z1[i] * Z1[j] / Z0**2)
    return Hs


def _entropy_hessian(n, q, a, pairs):
    nv = len(pairs) + n
    if _PLANT_DERIVE.get("entropy") == "drop_m":
        a = 0.0
    K = (1 - q) * np.eye(n) + (q - a * a) * np.ones((n, n))
    Ki = np.linalg.inv(K)
    one = np.ones(n)
    B = []
    for i, j in pairs:
        m = np.zeros((n, n))
        m[i, j] = m[j, i] = 1
        B.append(m)
    for g in range(n):
        e = np.zeros(n)
        e[g] = 1
        B.append(-a * (np.outer(e, one) + np.outer(one, e)))

    def C(vi, vj):
        m = np.zeros((n, n))
        if vi >= len(pairs) and vj >= len(pairs):
            g, h = vi - len(pairs), vj - len(pairs)
            if g == h:
                m[g, g] = -2
            else:
                m[g, h] = m[h, g] = -1
        return m
    H = np.zeros((nv, nv))
    for i in range(nv):
        for j in range(nv):
            H[i, j] = 0.5 * (np.trace(Ki @ C(i, j)) - np.trace(Ki @ B[i] @ Ki @ B[j]))
    if _PLANT_DERIVE.get("entropy") == "drop_m":      # entropy does not depend on M at all
        H[len(pairs):, :] = 0
        H[:, len(pairs):] = 0
    return H


def hessian_integer_n(n, alpha, d, c, kappa, q, M):
    """Hessian of Phi_n in the variables (Q_ab for a < b, then a^1..a^n), a = M / q_p, at the RS
    point.  Float64; n >= 4 so that the replicon irrep exists."""
    if not (isinstance(n, int) and n >= 4):
        raise Refusal("precondition:replicon-sector", f"n = {n}: the integer-n check needs n >= 4")
    qp = np.sqrt((1 + c) / 2)
    a = M / qp
    pairs = list(itertools.combinations(range(n), 2))
    return (_entropy_hessian(n, q, a, pairs)
            + _energy_hessian(n, alpha, d, c, kappa, q, a, pairs)), pairs


def sector_matrices(n, hess_pairs):
    """(replicon eigenvalue, replicon leak into M, T, standard-block leak): the Hessian applied
    to explicit irrep vectors.  `leak` is the norm of the M-components (and of the rest of the Q
    components) of H applied to the replicon vector; T[:, :] the 2 x 2 operator on (e, mu)."""
    Hs, pairs = hess_pairs
    npair = len(pairs)
    # replicon vector: eps_12 = eps_34 = 1, eps_13 = eps_24 = -1 (every row sums to 0)
    rep = np.zeros(npair)
    for k, p in enumerate(pairs):
        rep[k] = {(0, 1): 1, (2, 3): 1, (0, 2): -1, (1, 3): -1}.get(p, 0)
    x = np.concatenate([rep, np.zeros(n)])
    y = Hs @ x
    lam = (y @ x) / (x @ x)
    leak = np.linalg.norm(y - lam * x)
    eps = np.zeros(n)
    eps[0], eps[1] = 1, -1
    vq = np.array([eps[i] + eps[j] for i, j in pairs])
    x1, x2 = np.concatenate([vq, np.zeros(n)]), np.concatenate([np.zeros(npair), eps])
    y1, y2 = Hs @ x1, Hs @ x2

    def coef(y):
        return (y[:npair] @ vq) / (vq @ vq), (y[npair:] @ eps) / (eps @ eps)
    c1, c2 = coef(y1), coef(y2)
    T = np.array([[c1[0], c2[0]], [c1[1], c2[1]]])
    r1 = y1 - np.concatenate([c1[0] * vq, c1[1] * eps])
    r2 = y2 - np.concatenate([c2[0] * vq, c2[1] * eps])
    return lam, leak, T, max(np.abs(r1).max(), np.abs(r2).max())
