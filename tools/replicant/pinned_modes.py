"""Stability sectors of the RS saddle of the tied pinned-direction pair problem.

  class_values(n, alpha, d, c, kappa, q, M)  -> ClassValues   (Hessian entries by S_n class)
  replicon_hess(cv, n)                       -> P - 2 Q + R
  replicon_lambda(...)                       -> lambda_R = -(1-q)^2 (P - 2Q + R)  at n -> 0
  pinned_block(...)                          -> Block(T, eigenvalues, ...)  the (anomalous, M) block

Variables.  Q_ab (a < b) and, per replica, the pinned overlap M^a = <u^a, v_i> = <u^a, v_j> (the
tied common mode; a^a = M^a / q_p, q_p^2 = (1+c)/2, so (M^a)^T G^-1 M^b = a^a a^b).  The
replicated functional per d is
      Phi_n(Q, M) = (1/2) ln det[Q_ab - a^a a^b]  +  alpha ln E_{t ~ N(0,Q)} prod_a Theta(H^a - t_a),
      H^a = sqrt(d)(M^a - kappa).
At the RS point (Q_ab = q, M^a = M) S_n acts on the fluctuations (dQ_ab, dM^a).  The pair
variables split into the trivial, the standard (n-1 dim, Sum_b dQ_ab != 0) and the replicon
(n(n-3)/2 dim) irreps, the M^a into the trivial and the standard one.  Schur's lemma then gives:
  * the REPLICON irrep occurs once, in the Q sector: M^a does NOT mix with it.  Its eigenvalue
    is the Q-Q Hessian at fixed M, P - 2Q + R, the same function of (q, H) as the perceptron's
    (the entropic part is -1/(1-q)^2, the energy part alpha E[((ln P)'')^2]); the pinned
    direction enters only through the saddle (q, M) and through H;
  * the STANDARD irrep occurs twice (dQ_ab = e_a + e_b with Sum e = 0, and Sum dM^a = 0): the
    antisymmetric M fluctuations MIX with the anomalous Q mode.  On the coefficients (e, mu) the
    Hessian acts as the 2 x 2 matrix
        T = [[ lam2,  U - V ], [ (n-2)(U - V),  W - X ]],
        lam2 = P + (n-4) Qc - (n-3) R,
    with the class values P (same pair), Qc (one shared index), R (disjoint) of the Q-Q block,
    U, V (dQ_ab against dM^c with c in / not in {a, b}) and W, X (dM^a against dM^a / dM^b).
    The trivial irreps give the longitudinal block (the derived `mode` of solve_derived).
  * the difference mode M_i^a - M_j^a is not a smooth direction: the energy depends on
    min(M_i, M_j), a concave kink, so the tie is a corner maximum, first-order stable; it is not
    in T (stated, not computed).
Class values.  Entropy (RS, K = (1-q) I + (q - a^2) 1 1^T, K^-1 = k1 I + k2 1 1^T; hand-derived
from (1/2)[tr(K^-1 d2K) - tr(K^-1 dK K^-1 dK)], verified against the integer-n Hessian of
pinned_check.hessian_integer_n):
      P = -k2^2 - (k1+k2)^2,  Qc = -k2^2 - k2 (k1+k2),  R = -2 k2^2,
      U = a ks (k1 + 2 k2),   V = 2 a ks k2,
      W = -(k1+k2) - a^2 ks (ks + n (k1+k2)),  X = -k2 - a^2 ks (ks + n k2),
      k1 = 1/(1-q), ks = k1 + n k2 = 1/(1-q+n(q-a^2)).
Energy (Price's theorem on the slot functions P_k = d^k/dh^k Phi((H-h)/sqrt(1-q)) at h = sqrt(q) z;
every derivative with respect to H^a is -d/dt_a; <.>_n the replica-measure average with weight
P^n / E P^n, R_k = P_k / P):
      P = alpha (<R2^2> - A^2),  Qc = alpha (<R2 R1^2> - A^2),  R = alpha (<R1^4> - A^2),
      U = alpha b (-<R2 R1> + A B),  V = alpha b (-<R1^3> + A B),
      W = alpha b^2 (<R2> - B^2),  X = alpha b^2 (A - B^2),   A = <R1^2>, B = <R1>, b = sqrt(d) q_p.
Sign convention.  lambda_R = -(1-q)^2 (P-2Q+R) > 0 is RS replicon-stable (the package's); the
block's eigenvalues e of T use the same orientation: Hessian-negative (Re e < 0) is stable, and
`modes = -e` is reported (positive = stable).  Complex eigenvalues (possible at n = 0 since
n - 2 < 0 makes T non-symmetric) are reported as such; the n -> 0 criterion for the standard sector
is the analytic continuation of the integer-n one, not a theorem (physics-level).
Planted controls (derive_pinned._PLANT, plus `_PLANT_MODES`): "mixing": "drop" zeroes U - V.
"""
from dataclasses import dataclass

import mpmath as mp

from .core import Refusal
from .derive_pinned import _PLANT as _PLANT_DERIVE
from .solve import gauss_expect

_PLANT_MODES = {}


def _qp2(c):
    """q_p^2 = (1 + c)/2 (the tied common mode); the planted identity Gram drops c."""
    return (1 + (0 if _PLANT_DERIVE.get("gram") == "identity" else c)) / 2


@dataclass(frozen=True)
class ClassValues:
    n: object
    ent: dict
    en: dict

    def total(self, key):
        return self.ent[key] + self.en[key]


@dataclass(frozen=True)
class Block:
    n: object
    T: object
    eigenvalues: tuple
    modes: tuple
    complex_pair: bool
    stable: bool          # every Re(eigenvalue) < 0
    lam2: object
    uv: object
    wx: object


def _slot_ratios(q, H, z):
    """R1, R2, R3 = P_k / P at h = sqrt(q) z for P(h) = Phi((H - h)/s), s = sqrt(1-q)."""
    s = mp.sqrt(1 - q)
    u = (H - mp.sqrt(q) * z) / s
    G = mp.npdf(u) / mp.ncdf(u)
    return -G / s, -u * G / s**2, (1 - u * u) * G / s**3, u, mp.ncdf(u)


def _energy_classes(n, alpha, b, q, H, dps):
    def avg(f):
        def body(z):
            R1, R2, R3, u, P = _slot_ratios(q, H, z)
            return P**n * f(R1, R2, R3)
        return gauss_expect(body, dps=dps)
    Z0 = avg(lambda r1, r2, r3: 1)
    N = lambda f: avg(f) / Z0
    A, B1 = N(lambda r1, r2, r3: r1**2), N(lambda r1, r2, r3: r1)
    return {
        "P": alpha * (N(lambda r1, r2, r3: r2**2) - A**2),
        "Qc": alpha * (N(lambda r1, r2, r3: r2 * r1**2) - A**2),
        "R": alpha * (N(lambda r1, r2, r3: r1**4) - A**2),
        "U": alpha * b * (-N(lambda r1, r2, r3: r2 * r1) + A * B1),
        "V": alpha * b * (-N(lambda r1, r2, r3: r1**3) + A * B1),
        "W": alpha * b**2 * (N(lambda r1, r2, r3: r2) - B1**2),
        "X": alpha * b**2 * (A - B1**2)}


def _entropy_classes(n, q, a):
    k1 = 1 / (1 - q)
    ks = 1 / (1 - q + n * (q - a * a))
    k2 = (ks - k1) / n if n != 0 else -(q - a * a) / (1 - q)**2
    cls = {"P": -k2**2 - (k1 + k2)**2, "Qc": -k2**2 - k2 * (k1 + k2), "R": -2 * k2**2,
           "U": a * ks * (k1 + 2 * k2), "V": 2 * a * ks * k2,
           "W": -(k1 + k2) - a * a * ks * (ks + n * (k1 + k2)),
           "X": -k2 - a * a * ks * (ks + n * k2)}
    if _PLANT_DERIVE.get("entropy") == "drop_m":      # entropy blind to the pinned overlap
        return _entropy_classes_no_m(n, q)
    return cls


def _entropy_classes_no_m(n, q):
    k1 = 1 / (1 - q)
    ks = 1 / (1 - q + n * q)
    k2 = (ks - k1) / n if n != 0 else -q / (1 - q)**2
    return {"P": -k2**2 - (k1 + k2)**2, "Qc": -k2**2 - k2 * (k1 + k2), "R": -2 * k2**2,
            "U": mp.mpf(0), "V": mp.mpf(0), "W": mp.mpf(0), "X": mp.mpf(0)}


def class_values(n, alpha, d, c, kappa, q, M, *, dps=20) -> ClassValues:
    """Hessian entries of Phi_n at the RS point, by S_n class; n real (n = 0: the replica limit)."""
    with mp.workdps(dps + 5):
        alpha, d, c, kappa, q, M = (mp.mpf(x) for x in (alpha, d, c, kappa, q, M))
        if not 0 < q < 1:
            raise Refusal("precondition:overlap-range", f"q = {mp.nstr(q, 6)} not in (0, 1)")
        if abs(c) >= 1:
            raise Refusal("precondition:independent-pinned-directions",
                          f"|c| = {mp.nstr(abs(c), 6)} >= 1: the Gram matrix is singular")
        qp = mp.sqrt(_qp2(c))
        a = M / qp
        H = mp.sqrt(d) * (M - kappa)
        en = _energy_classes(n, alpha, mp.sqrt(d) * qp, q, H, dps)
        ent = _entropy_classes(n, q, a)
        return ClassValues(n, ent, en)


def replicon_hess(cv: ClassValues):
    t = cv.total
    return t("P") - 2 * t("Qc") + t("R")


def replicon_lambda(alpha, d, c, kappa, q, M, *, n=0, dps=20, cv=None):
    """lambda_R = -(1-q)^2 (P - 2Q + R) at the joint saddle (positive: replicon-stable).  The
    mixing with M is absent by Schur (module docstring); this is the Q-Q block at fixed M."""
    cv = cv or class_values(n, alpha, d, c, kappa, q, M, dps=dps)
    with mp.workdps(dps + 5):
        return -(1 - mp.mpf(q))**2 * replicon_hess(cv)


def pinned_block(alpha, d, c, kappa, q, M, *, n=0, dps=20, cv=None) -> Block:
    """The standard-irrep block T (anomalous Q mode and antisymmetric M mode, mixed)."""
    cv = cv or class_values(n, alpha, d, c, kappa, q, M, dps=dps)
    with mp.workdps(dps + 5):
        t = cv.total
        n_ = mp.mpf(n)
        lam2 = t("P") + (n_ - 4) * t("Qc") - (n_ - 3) * t("R")
        uv = 0 if _PLANT_MODES.get("mixing") == "drop" else t("U") - t("V")
        wx = t("W") - t("X")
        T = mp.matrix([[lam2, uv], [(n_ - 2) * uv, wx]])
        tr, det = lam2 + wx, lam2 * wx - (n_ - 2) * uv**2
        disc = tr**2 - 4 * det
        root = mp.sqrt(disc) if disc >= 0 else mp.sqrt(-disc) * 1j
        eig = ((tr + root) / 2, (tr - root) / 2)
        cplx = disc < 0
        return Block(n, T, eig, tuple(-e for e in eig), cplx,
                     all(mp.re(e) < 0 for e in eig), lam2, uv, wx)


def longitudinal_mode(alpha, d, c, kappa, q, M, *, dps=20):
    """The trivial-irrep (uniform) mode of the extra order parameter: mode = -(phi_aa - phi_aq^2 /
    phi_qq), phi the RS per-replica functional in (q, M), q stationary (solve_derived_
    stability's `mode`, positive = stable; its generic route divides by an underflowing erfc on this
    functional, so the Hessian is written here from L = ln Phi, L' = G(u), L'' = -G (u + G))."""
    with mp.workdps(dps + 5):
        alpha, d, c, kappa, q, M = (mp.mpf(x) for x in (alpha, d, c, kappa, q, M))
        qp = mp.sqrt(_qp2(c))
        a, b = M / qp, mp.sqrt(d) * qp
        if _PLANT_DERIVE.get("entropy") == "drop_m":
            a_ent, ent_aa, ent_aq = mp.mpf(0), mp.mpf(0), mp.mpf(0)
        else:
            a_ent, ent_aa, ent_aq = a, -1 / (1 - q), -a / (1 - q)**2
        H = b * a - mp.sqrt(d) * kappa
        s = mp.sqrt(1 - q)

        def avg(f):
            def body(z):
                u = (H - mp.sqrt(q) * z) / s
                G = mp.npdf(u) / mp.ncdf(u)
                uq = (H - mp.sqrt(q) * z) / (2 * s**3) - z / (2 * mp.sqrt(q) * s)
                # d_q of uq (derivative of the two terms)
                uqq = (z / (4 * q * mp.sqrt(q) * s) - z / (2 * mp.sqrt(q) * s**3)
                       + 3 * (H - mp.sqrt(q) * z) / (4 * s**5))
                ua, uaq = b / s, b / (2 * s**3)
                L1, L2 = G, -G * (u + G)
                return f(L1, L2, ua, uq, uaq, uqq)
            return gauss_expect(body, dps=dps)
        e_aa = alpha * avg(lambda L1, L2, ua, uq, uaq, uqq: L2 * ua**2)
        e_aq = alpha * avg(lambda L1, L2, ua, uq, uaq, uqq: L2 * ua * uq + L1 * uaq)
        e_qq = alpha * avg(lambda L1, L2, ua, uq, uaq, uqq: L2 * uq**2 + L1 * uqq)
        ent_qq = 1 / (2 * (1 - q)**2) + (q - a_ent**2) / (1 - q)**3
        p_aa, p_aq, p_qq = ent_aa + e_aa, ent_aq + e_aq, ent_qq + e_qq
        return -(p_aa - p_aq**2 / p_qq) / qp**2       # in the order parameter M = q_p a
