"""The spin-glass game of KZ Ch. 11 in its dense limit, as the Rademacher spiked Wigner model (DERIVED here).

KZ Sec. 11.1, Eqs. (11.1)-(11.3): N people carry S_i = +-1; asked about a pair, they answer J_ij = S_i S_j, a lie
(J_ij = -S_i S_j) with probability p.  Sec. 11.3, Eq. (11.27): all pairs are asked and p = 1/(1 + e^{2 beta/sqrt(N)}),
so the posterior is exp((beta/sqrt N) sum_{i<j} J_ij S_i S_j) / Z (11.28).

Derivation (not printed in the notes).  E[J_ij | S] = (1 - 2p) S_i S_j = tanh(beta/sqrt N) S_i S_j = beta S_i S_j/sqrt N
+ O(N^{-3/2}) and Var[J_ij | S] = 1 - tanh^2(beta/sqrt N) -> 1, so J = (beta/sqrt N) S S^T + xi with unit-variance
(non-Gaussian, +-1-valued) noise: the notes' spiked Wigner model (8.1) Y = sqrt(lambda_notes / N) x x^T + xi with
x = S (the Rademacher prior) and sqrt(lambda_notes) = beta.  Hence lambda_LM = lambda_notes = beta^2 and, with
replicant's Y = (lambda/N) x x^T + W (inference/spiked.py; J / sqrt N has this form), lambda = beta.  The Gaussian
noise of (8.1) is replaced by +-1 noise of the same variance: the output-channel universality of LKZ Sec. I C (and the
notes' Eq. (11.30), whose transition beta = 1 is the Rademacher lambda_c = 1).  Checked in tests/test_spiked_diluted.py
against the notes' own SE (11.59)-(11.60) and the top eigenvalue of J / sqrt(N).

  game_snr(beta) -> lambda = beta;  game_snr_lm(beta) -> lambda_LM = beta^2
  game_matrix(N, beta, seed) -> J (N x N, +-1 off the diagonal, 0 on it): S from default_rng([seed, 2]), the lies from
                                default_rng([seed, 1]).
"""
import math

import numpy as np

from ..core import Refusal
from ..csp import _checks


def _beta(beta):
    if isinstance(beta, bool):
        raise Refusal("precondition:beta", f"beta = {beta!r}: need a number >= 0")
    try:
        b = float(beta)
    except (TypeError, ValueError):
        raise Refusal("precondition:beta", f"beta = {beta!r}: need a number >= 0") from None
    if not (math.isfinite(b) and b >= 0):
        raise Refusal("precondition:beta", f"beta = {beta!r}: need a finite number >= 0")
    return b


def game_snr(beta):
    """replicant's spiked-Wigner lambda of the dense game: lambda = beta (derived, module docstring)."""
    return _beta(beta)


def game_snr_lm(beta):
    """Lelarge-Miolane's (= the notes') lambda of the dense game: beta^2."""
    return _beta(beta)**2


def game_matrix(N, beta, seed):
    """J_ij = S_i S_j, flipped with probability p = 1/(1 + e^{2 beta/sqrt N}) (KZ (11.27)), symmetric, zero diagonal."""
    if isinstance(N, bool) or not isinstance(N, int) or N < 10:
        raise Refusal("precondition:size", f"N = {N!r} must be an int >= 10")
    b, seed = _beta(beta), _checks.seed(seed)
    p = 1 / (1 + math.exp(2 * b / math.sqrt(N)))
    S = np.random.default_rng([seed, 2]).choice(np.array([-1.0, 1.0]), size=N)
    lie = np.random.default_rng([seed, 1]).random((N, N)) < p
    J = np.triu(np.where(lie, -1.0, 1.0) * np.outer(S, S), 1)
    return J + J.T


__all__ = ["game_matrix", "game_snr", "game_snr_lm"]
