"""Level (a): the perceptron family with PINNED (external, fixed) directions.

Model.  u on the sphere |u| = 1 in R^d; k = 2 fixed unit vectors v_1, v_2 with Gram matrix
G = [[1, c], [c, 1]] (the pair {i, j} of the random-documents problem); N = alpha d iid
Gaussian patterns xi_l (entries N(0, 1), so <u, xi_l> ~ N(0, 1) for |u| = 1).  A solution u
satisfies, for every pattern l and both a = 1, 2,
      <u, v_a> - <u, xi_l> / sqrt(d)  >=  kappa .
Since xi_l is shared by the two constraints of a pattern, they read t_l <= sqrt(d) (min_a M_a - kappa)
with t_l = <u, xi_l>, M_a = <u, v_a>: only the smaller overlap enters the energy.  In the
catalog convention of derive_families (g(lambda) = Theta(lambda - kappa_p), lambda = -t), this is
      kappa_p = sqrt(d) (kappa - M_bind),   M_bind = min_a M_a .
d is the dimension (the replica functional is per d, N / d = alpha); the paper's reading of the
O(1) threshold h = sqrt(d)(M - kappa) at finite d is the one used here.

Constraints (spec.constraint), both spherical variables only:
  "margin kappa pinned directions"          TIED: M_1 = M_2 = M, one extra order parameter `M`.
  "margin kappa pinned directions untied"   constraint a = 1 binds (M_1 <= M_2): the energy reads
                                            M_1 only; extras `M_1`, `M_2`.  The RS stationarity of
                                            M_2 is entropic alone (M_2 = c M_1), which contradicts
                                            M_2 >= M_1 whenever |c| < 1 and M_1 > 0: pinned.untied_slack
                                            refuses this branch by name (no-solution:untied-slack).
Entropic term (derived here, written into the formal LogDet).  With the pinned components of
replica alpha fixed to M^alpha = (<u^alpha, v_a>)_a, the remaining d - k coordinates of u^alpha live
in the orthogonal complement with squared norm 1 - M^T G^-1 M and overlaps
Q_ab - (M^a)^T G^-1 M^b, so
      (1/d) ln vol = (1/2) ln det [ Q_ab - (M^a)^T G^-1 M^b ]_{a, b = 1..n} + O(1/d),
RS: (n-1) ln(1-q) + ln(1 - q + n (q - m2)) over 2, m2 = M^T G^-1 M; n -> 0 per replica:
      (1/2) [ ln(1-q) + (q - m2) / (1-q) ] .
(A single replica, n = 1, gives (1/2) ln(1 - m2): the volume of the slice <u, v> fixed of the sphere.)
The pinned components themselves are fixed by delta functions: no integral over them remains.

What is HAND-SUPPLIED: this entropic term (not mechanised, as for the spherical volume), the
shared-pattern reduction to min_a M_a, and the weight with the RS value of M inside it (the
energy is written with a replica-independent M: exact at the RS point, which is where every step
below evaluates; the replica-dependent M^alpha enters only the Hessian across replicas, done in
pinned_modes.py).  Everything after the hook (RS ansatz, n -> 0, stationarity, solver, replicon
of the Q-sector) is the generic steps.
Planted controls (test hooks, `_PLANT`): "entropy": "drop_m" (the entropic term ignores M),
"gram": "identity" (G = I, the overlap c ignored).
"""
import sympy as sp

from . import derive as D
from .core import Refusal

D_DIM = sp.Symbol("d", positive=True)
C_OVERLAP = sp.Symbol("c", real=True)

TIED = "margin kappa pinned directions"
UNTIED = "margin kappa pinned directions untied"
CONSTRAINTS = (TIED, UNTIED)

# formal marker passed as the site's `field`: Pinned(m2_aa, m2_ab) = (M^a)^T G^-1 M^a, (M^a)^T G^-1 M^b
Pinned = sp.Function("Pinned")

_PLANT = {}


def is_pinned(spec):
    return spec.constraint in CONSTRAINTS


def check_spec(spec):
    if spec.variables != "spherical":
        raise Refusal("precondition:pinned-spherical-only",
                      f"pinned directions are derived for spherical variables (the entropic term "
                      f"is the spherical volume with the pinned components fixed); got "
                      f"{spec.variables!r}")


def _params(spec):
    p = spec.params
    return p.get("d", D_DIM), p.get("c", C_OVERLAP)


def gram_inverse_form(c, tied, x1, x2):
    """(x^a)^T G^-1 (x^b) for x^a = (x1_a, x2_a) (tied: both components equal x1_a), as an
    expression in the two replicas' values; returned as a function (ma, mb) -> expression."""
    if _PLANT.get("gram") == "identity":
        c = 0
    if tied:
        return lambda a, b: 2 * x1[a] * x1[b] / (1 + c)
    return lambda a, b: (x1[a] * x1[b] - c * (x1[a] * x2[b] + x2[a] * x1[b])
                         + x2[a] * x2[b]) / (1 - c**2)


def pinned_average(zn):
    """alpha * LnGaussProd(g, q_ab, n) with g(lambda) = Theta(lambda - sqrt(d)(kappa - M_bind)):
    the hand-supplied weight, the Gaussian-field statement asserted exactly as for the perceptron."""
    from .derive_families import ALPHA, KAPPA, PatternAverage, _LAM, _symbolic_n
    _symbolic_n(zn)
    spec = zn.spec
    check_spec(spec)
    kappa, alpha = spec.params.get("kappa", KAPPA), spec.params.get("alpha", ALPHA)
    d, _ = _params(spec)
    binding = sp.Symbol("M" if spec.constraint == TIED else "M_1", real=True)
    g = sp.Lambda(_LAM, sp.Heaviside(_LAM - sp.sqrt(d) * (kappa - binding)))
    note = ("Gaussian patterns (hook asserts, not mechanised): lambda_a = -u^a.xi, jointly Gaussian "
            "with covariance Q_ab (symmetric law); both constraints of a pattern share xi, so only "
            f"{binding} = min_a <u, v_a> enters; P = alpha d iid patterns -> alpha ln E_lambda "
            f"prod_a g(lambda_a), g = {g.expr} (hand-supplied; M taken at its RS value, replica "
            "independent)")
    return PatternAverage(spec, zn.n, alpha, g, note)


def pinned_order_parameters(avg, n, sum_pairs):
    """Energy alpha LnGaussProd + the pinned geometry marker for the spherical site + the extra
    replica-indexed order parameters (M, or M_1 and M_2)."""
    from .derive_families import _gauss_prod_reduction
    spec = avg.spec
    tied = spec.constraint == TIED
    _, c = _params(spec)
    if tied:
        Mb = sp.IndexedBase("M", real=True)
        x1 = x2 = {D._a: Mb[D._a], D._b: Mb[D._b]}
        extra = {"M": Mb}
    else:
        M1, M2 = sp.IndexedBase("M_1", real=True), sp.IndexedBase("M_2", real=True)
        x1 = {D._a: M1[D._a], D._b: M1[D._b]}
        x2 = {D._a: M2[D._a], D._b: M2[D._b]}
        extra = {"M_1": M1, "M_2": M2}
    form = gram_inverse_form(c, tied, x1, x2)
    m2_aa, m2_ab = form(D._a, D._a), form(D._a, D._b)
    if _PLANT.get("entropy") == "drop_m":
        m2_aa = m2_ab = sp.S(0)
    return _gauss_prod_reduction(avg, n, field=Pinned(m2_aa, m2_ab), extra=extra,
                                 missing="O(1/d) (the pinned components, saddle-point prefactor)")


def pinned_site(n, q, field):
    """(1/d) ln vol of the spherical replicas with pinned components fixed: (1/2) ln det
    [Q_ab - m2_ab] (module docstring), formal until rs_ansatz."""
    if isinstance(n, int):
        raise Refusal("precondition:symbolic-replica-number",
                      "pinned directions are derived at symbolic n only (the explicit-n volume "
                      "has no registered pinned trace; the integer-n Hessian is "
                      "pinned_modes.hessian_integer_n)")
    m2_aa, m2_ab = field.args
    return D.LogDet(1 - m2_aa, q[D._a, D._b] - m2_ab, n) / 2


def pinned_spec(*, tied=True, params=None):
    """The spec of the pair problem (parameters left symbolic unless given)."""
    from .derive_families import ALPHA, KAPPA
    from .spec import ModelSpec
    base = {"alpha": ALPHA, "kappa": KAPPA, "d": D_DIM, "c": C_OVERLAP}
    base.update(params or {})
    return ModelSpec("perceptron", "spherical", "gaussian", TIED if tied else UNTIED,
                     params=base)
