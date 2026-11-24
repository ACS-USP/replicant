"""Denoising a sparse vector (KZ Sec. 7.3, Theorem 15, App. 7.A-7.B).

Model (KZ Eqs. (7.51)-(7.54)): x* is one of the d = 2^N unit basis vectors, y = x* + sqrt(Delta/N) z, z ~ N(0, I_d).
The likelihood ratio is
    Z = 2^-N sum_i exp(-N/(2 Delta) + (N/Delta) delta_{i,i*} + sqrt(N/Delta) z_i),
and Phi(Delta) = lim (1/N) E log Z.  Theorem 15: f(Delta) = 1/(2 Delta) - log 2 for Delta <= Delta_c, 0 above,
Delta_c = 1/(2 log 2) (KZ Eqs. (7.57)-(7.58); the extraction's "1/2 log 2" is 1/(2 log 2), as Eq. (7.92) and the
statement of Delta_c on p. 126 confirm).  By I-MMSE (Eq. (7.56)) q = 2 dPhi/d(1/Delta): q = 1, MMSE = 0 below
Delta_c; q = 0, MMSE = 1 above.  For a noise variance sigma^2 = Delta/N the threshold is sigma_c^2 = 1/(2 log d).
App. 7.B (replica, RS): the two saddles give log 2 - 1/(2 Delta) (all replicas on i*, Eq. (7.92), written there as
a free energy F = -f) and 0 (Eq. (7.93)); f is the larger.

REM route (independent of Theorem 15): the d - 1 wrong components are 2^N i.i.d. terms exp(sqrt(N/Delta) z_i), a
REM (families/rem.py) Z_REM = sum exp(-beta E_i), E_i ~ N(0, N_scale N), at N_scale = 1/2 and beta = sqrt(2/Delta).
(1/N) log Z -> max(1/(2 Delta) - log 2, -log 2 - 1/(2 Delta) + phi_REM(beta)), phi_REM = -beta f_REM.  The REM
transition beta_c = 2 sqrt(log 2) is Delta_c = 2/beta_c^2.

  delta_c(), sigma_c2(d)
  sparse_vector_free_entropy(Delta, d=None)   d None: Theorem 15 (mpmath, 30 digits); d = 2^N: the exact (1/N) E log Z
                                              at that N, by E log Z = int du [exp(-e^u) - E exp(-e^u Z)] with
                                              E exp(-t Z) a product over the independent terms (binary64 trapezoid)
  sparse_vector_mmse(Delta)                   0 below Delta_c, 1 above (I-MMSE on Theorem 15)
  rem_parameters(Delta), free_entropy_via_rem(Delta)
  sample_log_z(Delta, N, seed)                (1/N) log Z of one sample (tests: named seeds)

_HOOKS (planted errors, tests only): "beta_factor" (2: beta^2 = factor/Delta), "planted_copies" (1: the planted
term's multiplicity in the finite-d quadrature).
"""
import math

import mpmath as mp
import numpy as np

from ..core import Refusal
from ..families import rem as REM
from ..lattice.ball import rational

_HOOKS = {"beta_factor": 2, "planted_copies": 1}
_DPS = 30


def _delta(Delta):
    return mp.mpf(rational(Delta, "Delta").numerator) / rational(Delta, "Delta").denominator


def delta_c():
    with mp.workdps(_DPS):
        return 1 / (2 * mp.log(2))


def sigma_c2(d):
    if not (isinstance(d, int) and d >= 2):
        raise Refusal("precondition:dimension", f"d = {d!r}: need an int >= 2")
    with mp.workdps(_DPS):
        return 1 / (2 * mp.log(d))


def _n_of(d):
    if not (isinstance(d, int) and d >= 2 and d & (d - 1) == 0 and d <= 2**40):
        raise Refusal("precondition:dimension", f"d = {d!r}: need d = 2^N, 1 <= N <= 40")
    return d.bit_length() - 1


def sparse_vector_free_entropy(Delta, d=None):
    if d is not None:
        return _finite_d(Delta, _n_of(d))
    with mp.workdps(_DPS):
        D = _delta(Delta)
        return max(1 / (2 * D) - mp.log(2), mp.mpf(0))


def sparse_vector_mmse(Delta):
    with mp.workdps(_DPS):
        return 0 if _delta(Delta) < delta_c() else 1


def rem_parameters(Delta):
    """(beta, N_scale) of the REM formed by the d - 1 wrong components."""
    with mp.workdps(_DPS):
        return mp.sqrt(_HOOKS["beta_factor"] / _delta(Delta)), mp.mpf("0.5")


def free_entropy_via_rem(Delta):
    with mp.workdps(_DPS):
        D = _delta(Delta)
        beta, s = rem_parameters(Delta)
        null = -mp.log(2) - 1 / (2 * D) - beta * REM.rem_free_energy(beta, N_scale=s)
        return max(1 / (2 * D) - mp.log(2), null)


_ZG = np.linspace(-12.0, 12.0, 1201)
_WG = np.exp(-_ZG**2 / 2)
_WG /= _WG.sum()


def _finite_d(Delta, N):
    D = float(_delta(Delta))
    sig, ln2 = math.sqrt(N / D), math.log(2)
    c_star, c_null = N / (2 * D) - N * ln2, -N / (2 * D) - N * ln2      # log of each term's prefactor, 2^-N absorbed
    span = abs(c_star) + abs(c_null) + 12 * sig + N * ln2 + 40
    u = np.arange(-span, 40.0, 0.05)

    def h(c):     # E[1 - exp(-e^u a)], log a = c + sig z
        with np.errstate(over="ignore"):
            return (-np.expm1(-np.exp(np.minimum(u[:, None] + c + sig * _ZG[None, :], 700)))) @ _WG
    copies = _HOOKS["planted_copies"]
    with np.errstate(divide="ignore"):         # h = 1 exactly: log G = -inf, G = 0 (u far above log Z)
        logG = (copies * np.log1p(-h(c_star)) if copies else 0.0) + (2**N - 1) * np.log1p(-h(c_null))
    integrand = np.expm1(-np.exp(u)) - np.expm1(logG)
    return float(integrand.sum() * (u[1] - u[0]) / N)


def sample_log_z(Delta, N, seed):
    """(1/N) log Z of one draw: z from default_rng([seed, 7]), component 0 planted."""
    D = float(_delta(Delta))
    z = np.random.default_rng([seed, 7]).standard_normal(2**N)
    la = -N / (2 * D) + math.sqrt(N / D) * z
    la[0] += N / D
    top = la.max()
    return float((top + math.log(np.exp(la - top).sum()) - N * math.log(2)) / N)


__all__ = ["delta_c", "free_entropy_via_rem", "rem_parameters", "sample_log_z", "sigma_c2", "sparse_vector_free_entropy",
           "sparse_vector_mmse"]
