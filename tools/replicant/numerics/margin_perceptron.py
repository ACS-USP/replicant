"""Finite-N storage threshold of the spherical perceptron with per-pattern margins (E11 style).

  min_norm_point(G, h)                       -> (x, |x|^2) the least-norm x with G x >= h, or None
  storage_threshold(xi, kappa)               -> P*, the largest P such that the first P patterns
                                                with their margins can be stored in |J|^2 <= N
  capacity_finite_size(law, Ns, samples, seed, *, conf=0.99) -> FSResult
                                                per-sample P*/N, extrapolated in N^-omega

Model (families/margin_law.py): J.J = N, J.xi^l/sqrt(N) >= kappa_l, xi^l iid N(0, I_N), kappa_l
iid from a MarginLaw, independent of the patterns.  Independent of every replica formula: it
shares no code with them.

Feasibility in the ball.  C = {J: xi J/sqrt(N) >= kappa} is a polyhedron, "stored" means
C meets the ball |J|^2 <= N, i.e. min_C |J|^2 <= N (convex, exact).  It is the problem the
sphere asks whenever C is unbounded in a direction r with xi r > 0 (a.s. for P < 2N): J0 + t r is
in C for all t >= 0, the norms of C fill [min_C |J|, oo) by connectedness, so C meets the sphere
|J|^2 = N iff it meets the ball, whatever the signs of the kappa_l.  That covers every law with
alpha_c < 2 (alpha_c = 1/E D(kappa) < 2 iff E D(kappa) > 1/2, e.g. every kappa~ >= 0).  For
alpha_c >= 2 (kappa~ < 0) the ball threshold is only an UPPER bound on the sphere's (sphere
feasible implies ball feasible; deciding the sphere needs a maximum-norm search, not convex, not
implemented): that is the exact minimum-norm statement, and the threshold compared with the RS
alpha_c there is the ball's.  Adding a pattern only adds a constraint, so feasibility is monotone in P
and P* is found by bisection on nested prefixes.

Least-norm point.  Lawson-Hanson's least-distance programming (Solving Least Squares Problems,
1974, Ch. 23): with E = [G^T; h^T] and f = (0, ..., 0, 1), solve min_{u >= 0} |E u - f| by NNLS
(scipy.optimize.nnls); r = E u - f; x_j = -r_j/r_{N+1} (j <= N) if r_{N+1} != 0, else G x >= h
has no solution.

Extrapolation and band: numerics/finite_size._extrapolate (weighted least squares of the sample
means in N^-omega, omega = 2/3, band = 99% t-band + the largest shift over the variant fits).
Statistical and finite-size: never certified.
"""
import math

import numpy as np
from scipy.optimize import nnls

from ..core import Refusal
from . import finite_size as FS


def min_norm_point(G, h):
    """(x, |x|^2) minimising |x| subject to G x >= h (G: P x N), or None if infeasible."""
    P, N = G.shape
    E = np.vstack([G.T, h[None, :]])
    f = np.zeros(N + 1)
    f[N] = 1.0
    u, _ = nnls(E, f, maxiter=50 * (P + N))
    r = E @ u - f
    if abs(r[N]) < 1e-12:
        return None
    x = -r[:N] / r[N]
    return x, float(x @ x)


def sample_problem(law, N, P, rng):
    """P patterns (rows ~ N(0, I_N)) and P margins from the law."""
    xi =rng.standard_normal((P, N))
    comps = [(float(w), float(m), float(s)) for w, m, s in law.components()]
    which = rng.choice(len(comps), size=P, p=[c[0] for c in comps])
    m = np.array([c[1] for c in comps])[which]
    s = np.array([c[2] for c in comps])[which]
    return xi, m + s * rng.standard_normal(P)


def stored(xi, kappa, N, P):
    """True iff the first P patterns can be stored in the ball (module docstring)."""
    G, h = xi[:P] / math.sqrt(N), kappa[:P]
    res = min_norm_point(G, h)
    return res is not None and res[1] <= N * (1 + 1e-9)


def storage_threshold(xi, kappa):
    """Largest P such that the first P constraints can be stored: bisection on the monotone
    predicate.  Refuses when even all of xi can be stored (draw more patterns)."""
    Pmax, N = xi.shape
    if stored(xi, kappa, N, Pmax):
        raise Refusal("precondition:enough-patterns",
                      f"all {Pmax} patterns are storable at N = {N}: draw more")
    lo, hi = 1, Pmax          # stored(lo), not stored(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if stored(xi, kappa, N, mid) else (lo, mid)
    return lo


def capacity_finite_size(law, Ns, samples, seed, *, conf=0.99, load=1.6):
    """Per-sample P*/N for N in Ns (at least 3 distinct sizes), `samples` runs per size (an int
    or {N: int}), named seed; patterns drawn up to load * alpha_c * N + 10 (a sample that stores
    them all is refused precondition:enough-patterns: raise `load`)."""
    from ..families.margin_law import law_capacity
    ac = float(law_capacity(law).alpha_c)
    Ns, per = FS._check_sizes(Ns, samples, 3)
    per_sample, diag = {}, {}
    for N in Ns:
        rng = np.random.default_rng([seed, N])
        pmax = int(load * ac * N) + 10
        vals = []
        for _ in range(per[N]):
            xi, kappa = sample_problem(law, N, pmax, rng)
            vals.append(storage_threshold(xi, kappa) / N)
        per_sample[N] = np.array(vals)
        diag[N] = f"{per[N]} samples, P*/N in [{min(vals):.3f}, {max(vals):.3f}]"
    return FS._extrapolate(Ns, per_sample, diag, (("3/4", 0.75), ("1", 1.0)), conf,
                           f"storage threshold of {law.describe()}")
