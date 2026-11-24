"""The exact first moment E_G Z_G(beta) of the Potts / coloring partition function over G(N, M) (KZ Sec. 6.6).

G(N, M): M distinct edges drawn uniformly among the P = N(N - 1)/2 pairs.  Z_G = sum_s exp(-beta #monochromatic
edges).  For a coloring with color counts n_1..n_q there are P_mono = sum_c n_c(n_c - 1)/2 monochromatic pairs,
so exactly (hypergeometric in the number j of monochromatic edges):
    E_G Z = sum_{n_1 + .. + n_q = N} N!/prod n_c! sum_j C(P_mono, j) C(P - P_mono, M - j) e^{-beta j} / C(P, M).
KZ's annealed form q^N (e^-beta/q + 1 - 1/q)^M replaces P_mono/P by 1/q (balanced colorings): the two agree in
(1/N) ln at N -> oo with c = 2M/N fixed (beta > 0), Phi_ann = ln q + (c/2) ln(1 - theta/q).  The brute-force
route enumerates every graph of G(N, M) and every coloring (tiny N only).

  first_moment_log(q, n, m, beta) -> ln E Z (the composition sum, lgamma and log-sum-exp)
  first_moment_brute(q, n, m, beta) -> E Z (enumeration; refused above 2e6 graph-coloring pairs)
"""
import itertools
import math

import numpy as np

from ..core import Refusal
from .potts_bp import _q, beta_value


def _lc(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _compositions(n, q):
    if q == 1:
        yield (n,)
        return
    for k in range(n + 1):
        for rest in _compositions(n - k, q - 1):
            yield (k,) + rest


def _lse(xs):
    mx = max(xs)
    return mx + math.log(math.fsum(math.exp(x - mx) for x in xs)) if mx > -math.inf else -math.inf


def first_moment_log(q, n, m, beta):
    q, b = _q(q), beta_value(beta)
    P = n * (n - 1) // 2
    if not (isinstance(n, int) and isinstance(m, int) and n >= 2 and 0 <= m <= P):
        raise Refusal("precondition:graph", f"G(N, M) needs ints N >= 2 and 0 <= M <= N(N-1)/2, got {n}, {m}")
    lb = -b if b != math.inf else -math.inf
    terms = []
    for comp in _compositions(n, q):
        pm = sum(k * (k - 1) // 2 for k in comp)
        lmult = math.lgamma(n + 1) - sum(math.lgamma(k + 1) for k in comp)
        js = [j for j in range(max(0, m - (P - pm)), min(pm, m) + 1) if j == 0 or lb > -math.inf]
        if js:
            terms.append(lmult + _lse([_lc(pm, j) + _lc(P - pm, m - j) + j * (lb if j else 0.0) for j in js]))
    return _lse(terms) - _lc(P, m)


def first_moment_brute(q, n, m, beta):
    q, b = _q(q), beta_value(beta)
    pairs = list(itertools.combinations(range(n), 2))
    n_graphs = math.comb(len(pairs), m)
    if n_graphs * q ** n > 2_000_000:
        raise Refusal("precondition:enumeration-size", f"{n_graphs} graphs x {q}^{n} colorings")
    conf = np.indices((q,) * n).reshape(n, -1)
    mono = np.array([conf[i] == conf[j] for i, j in pairs], dtype=float)        # (P, q^N)
    w = math.exp(-b) if b != math.inf else 0.0
    total = 0.0
    for g in itertools.combinations(range(len(pairs)), m):
        k = mono[list(g)].sum(0)
        total += float(np.sum(w ** k))
    return total / n_graphs


__all__ = ["first_moment_brute", "first_moment_log"]
