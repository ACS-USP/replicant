"""The Curie-Weiss ferromagnet (Krzakala & Zdeborova, lecture notes, June 9 2024 -- "KZ" -- Ch. 1, App. 1.B, 1.C).

Convention (KZ Eq. (1.1)-(1.2)): spins S_i = +-1, H_N = -(1/2N) sum_{i,j} S_i S_j - h sum_i S_i (the double sum
includes i = j, so H_N = -N (Sbar^2/2 + h Sbar), Sbar = (1/N) sum_i S_i), J = 1, beta > 0 and h rational;
Phi_N = (1/N) ln Z_N is the free entropy density, f = -Phi / beta.  Theorem 1 (Eq. (1.7)):
    Phi(beta, h) = max_m phi(m),   phi(m) = H(m) + beta m^2/2 + beta h m,   H the binary entropy (1.3).
Three toolboxes give Phi, each in its own code (tests/test_curie_weiss.py checks the call graphs are disjoint):
  variational (Theorem 1; the factorised Gibbs ansatz Eq. (1.17)): every root of phi'(m) = beta (m + h) -
      atanh m, CERTIFIED in arb: phi' decreases, increases on (-s, s), decreases (s = sqrt(1 - 1/beta) when
      beta > 1), so the certified signs of phi'(+-s) say which monotone interval holds a root; each root is
      bracketed by bisection and enclosed by interval Newton (unique in its ball), so the list is complete.
      Branches "unique" or "lower" < "middle" < "upper"; jacobian = beta (1 - m^2) of the saddle iteration
      m <- tanh beta (m + h) (App. 1.B), stable (a local maximum of phi) iff < 1; at beta = 1, h = 0 the
      jacobian is exactly 1 and m = 0 is still the maximum (phi = ln 2 - m^4/12 + O(m^6)): labelled stable.
  cavity (Sec. 1.4, Eq. (1.23); Theorem 6 of App. 1.C): Phi = max_m phi~(m), phi~ = -beta m^2/2 +
      ln 2 cosh beta (m + h), by iterating m <- tanh beta (m + h) from m0 = +-1 (mpmath).
  field-theoretic (Sec. 1.5): Phi = extr_{m, mhat} g, g = beta m^2/2 + beta h m + mhat m + ln 2 + ln cosh mhat,
      by a two-dimensional Newton solve of grad g = 0 (mpmath), the largest g over the stationary points.
Spinodal (App. 1.B; KZ p. 33 prints h_sp = +-sqrt((1/beta)(1 - 1/beta)) -+ (1/beta) atanh sqrt(1 - 1/beta), an
ERRATUM: the double root of m = tanh beta (m + h) is at beta (1 - m^2) = 1, giving)
    h_sp(beta) = sqrt(1 - 1/beta) - (1/beta) atanh sqrt(1 - 1/beta),   beta > 1,
which matches the dashed spinodals of KZ Fig. 1.B.4 at beta = 1.5 (about +-0.14; the printed form gives 0.032).
Selection (cw_solution): branch given -> it; one root -> it; h > 0 -> "upper", h < 0 -> "lower" (the global
maximum, certified against the other local maximum); h = 0 with three roots -> Phi is the common value of the
two maxima and the magnetization is refused ambiguous:branch until a branch is given.

  cw_fixed_points(beta, h) -> (CWFixedPoint, ...)        cw_solution(beta, h, branch=None) -> CWSolution
  cw_magnetization(beta, h=0, branch=None) -> Ball        cw_free_entropy(beta, h) -> Ball
  cw_spinodal(beta) -> Ball                               cw_critical_beta() -> Ball (= 1)
  cw_free_entropy_cavity(beta, h), cw_free_entropy_field(beta, h) -> (m, Phi) mpf (30 digits)
Finite N and the rate functions: meanfield/cw_exact.py.
_HOOKS (planted-error hooks): "spinodal_sign" (-1, the sign of the atanh term of h_sp); "field_h" (1, the weight
of beta h m in g: KZ p. 18's "g(m) = phi(beta, m) + beta h m" counts it twice, a typo the agreement test catches).
"""
from dataclasses import dataclass

import mpmath as mp

from ..core import Refusal
from ..lattice.ball import Ball, arb_q, ball_of, baller, certify, rational

DIGITS = 40
_HOOKS = {"spinodal_sign": -1, "field_h": 1}
_PREC0, _MAX_PREC = 192, 1 << 12


def _beta(beta):
    return rational(beta, "beta")


def _field(h):
    return rational(h, "h", positive=False)


# ---- variational route (Theorem 1), certified ---------------------------------------------------------
@dataclass(frozen=True)
class CWFixedPoint:
    branch: str
    m: Ball               # magnetization (a root of phi')
    phi: Ball             # phi(m), the variational potential at the root
    jacobian: Ball        # beta (1 - m^2)
    stable: bool          # jacobian < 1: a local maximum of phi


@dataclass(frozen=True)
class CWSolution:
    beta: object
    h: object
    fixed_points: tuple
    selected: CWFixedPoint | None
    phi: Ball             # Phi(beta, h) = max_m phi(m)

    @property
    def magnetization(self):
        if self.selected is None:
            raise Refusal("ambiguous:branch", "h = 0 above beta_c = 1: the two maxima +-m* coexist; pass "
                          "branch='lower' or 'upper'")
        return self.selected.m


class _Var:
    def __init__(self, bq, hq):
        self.b, self.h = arb_q(bq), arb_q(hq)

    def d(self, m):
        return self.b * (m + self.h) - m.atanh()

    def dd(self, m):
        return self.b - 1 / (1 - m * m)

    def phi(self, m):
        p, q = (1 + m) / 2, (1 - m) / 2
        return -(p * p.log() + q * q.log()) + self.b * m * m / 2 + self.b * self.h * m


def _bisect(v, lo, hi):
    """A point near the root of v.d on [lo, hi], d(lo) and d(hi) of opposite (decided) signs."""
    from flint import arb
    up = v.d(lo) > 0
    for _ in range(_PREC0 + 64):
        mid = (lo + hi) / 2
        f = v.d(mid)
        if f.contains(0) or (hi - lo).upper() < arb(2) ** (-_PREC0):
            return mid.mid()
        if (f > 0) == up:
            lo = mid
        else:
            hi = mid
    return ((lo + hi) / 2).mid()


def _newton(v, x0):
    """Interval Newton (Moore): an arb X with N(X) inside X, enclosing the unique root of v.d near x0."""
    from flint import arb, ctx
    x = arb(x0)
    for _ in range(8):
        x = (x - v.d(x) / v.dd(x)).mid()
    r = arb(2) ** (-(ctx.prec // 2))
    for _ in range(40):
        X = arb(x, r)
        N = x - v.d(arb(x)) / v.dd(X)
        if X.contains_interior(N):
            return N
        r *= 4
    raise Refusal("no-solution:certification", f"interval Newton did not contract around {x0}")


def _roots_at(bq, hq):
    """[(branch, m ball)] at the current precision, or None when a sign at +-s is undecided."""
    from flint import arb
    v = _Var(bq, hq)
    edge = 2 * (-2 * (v.b * (1 + abs(v.h)) + 1)).exp()          # d(-1 + edge) > 0 > d(1 - edge)
    lo, hi = -1 + edge, 1 - edge
    if bq <= 1:
        if hq == 0:
            return [("unique", arb(0))], v                         # phi' is odd at h = 0: m = 0 exactly
        return [("unique", _newton(v, _bisect(v, lo, hi)))], v
    s = (1 - 1 / v.b).sqrt()
    dl, du = v.d(-s), v.d(s)
    if dl.contains(0) or du.contains(0):
        return None, v
    roots = []
    if dl < 0:
        roots.append(("lower", _newton(v, _bisect(v, lo, -s))))
        if du > 0:
            roots.append(("middle", arb(0) if hq == 0 else _newton(v, _bisect(v, -s, s))))
    if du > 0:
        roots.append(("upper", _newton(v, _bisect(v, s, hi))))
    return ([("unique", roots[0][1])] if len(roots) == 1 else roots), v


def cw_fixed_points(beta, h=0):
    """Every root of the Curie-Weiss saddle equation at (beta, h), certified, in increasing m."""
    bq, hq = _beta(beta), _field(h)
    hyg, prec = baller().hygiene, _PREC0
    while prec <= _MAX_PREC:
        with hyg.ctx_guard(prec=prec):
            roots, v = _roots_at(bq, hq)
            if roots is not None:
                out = []
                for name, m in roots:
                    jac = v.b * (1 - m * m)
                    crit = bq == 1 and hq == 0          # m = 0, jacobian exactly 1: phi = ln 2 - m^4/12 + ...
                    if not (jac < 1 or jac > 1 or crit):
                        break
                    balls = [ball_of(x) for x in (m, v.phi(m), jac)]
                    if min(b.digits for b in balls) < DIGITS:
                        break
                    out.append(CWFixedPoint(name, *balls, bool(crit or jac < 1)))
                else:
                    return tuple(out)
        prec *= 2
    raise Refusal("undetermined:spinodal", f"beta = {beta}, h = {h}: the root count or a stability sign is "
                  f"undecided at {_MAX_PREC} bits (at the spinodal |h| = h_sp(beta)?)")


def cw_solution(beta, h=0, branch=None) -> CWSolution:
    fps = cw_fixed_points(beta, h)
    names, hq = [f.branch for f in fps], _field(h)
    if branch is not None:
        if branch not in names:
            raise Refusal("precondition:branch", f"branch = {branch!r}: the roots here are {', '.join(names)}")
        sel = fps[names.index(branch)]
    elif len(fps) == 1:
        sel = fps[0]
    else:
        sel = None if hq == 0 else fps[names.index("upper" if hq > 0 else "lower")]
    if len(fps) == 3:
        lo, up = fps[0].phi, fps[2].phi
        if hq != 0 and not (up.arb > lo.arb if hq > 0 else lo.arb > up.arb):
            raise Refusal("no-solution:audit", f"h = {h}: the maximum aligned with h ({up if hq > 0 else lo}) "
                          f"is not certified above the other ({lo if hq > 0 else up})")
        if hq == 0 and not lo.overlaps(up):
            raise Refusal("no-solution:audit", f"h = 0: phi(+-m*) differ ({lo}, {up}); phi is even")
    glob = fps[0] if len(fps) == 1 else (fps[2] if hq >= 0 else fps[0])
    return CWSolution(beta, h, fps, sel, glob.phi)


def cw_magnetization(beta, h=0, branch=None) -> Ball:
    return cw_solution(beta, h, branch).magnetization


def cw_free_entropy(beta, h=0) -> Ball:
    return cw_solution(beta, h).phi


def cw_thermo(fp: CWFixedPoint, beta, h=0):
    """(f, e, s) at a root m (Eq. (1.30)): f = -phi/beta, e = -m (m/2 + h), s = H(m) = phi - beta(m^2/2 + h m)."""
    bq, hq = _beta(beta), _field(h)
    with baller().hygiene.ctx_guard(prec=4 * DIGITS + 128):
        b, hb, m, phi = arb_q(bq), arb_q(hq), fp.m.arb, fp.phi.arb
        e = -m * (m / 2 + hb)
        return ball_of(-phi / b), ball_of(e), ball_of(phi + b * e)


def cw_critical_beta() -> Ball:
    """beta_c: the paramagnetic root m = 0 of m = tanh(beta m) is stable while its Jacobian beta (1 -
    tanh(0)^2) < 1 (App. 1.B): beta_c = 1 / (1 - tanh(0)^2), certified."""
    with baller().hygiene.ctx_guard(prec=4 * DIGITS + 128):
        return ball_of(1 / (1 - arb_q(0).tanh() ** 2))           # exact: ball_of, not certify (inf digits)


def cw_spinodal(beta) -> Ball:
    """h_sp(beta) = sqrt(1 - 1/beta) - (1/beta) atanh sqrt(1 - 1/beta) (corrected form; see the docstring)."""
    bq = _beta(beta)
    if not bq > 1:
        raise Refusal("precondition:ordered-phase", f"beta = {beta}: a spinodal exists only for beta > 1")
    sg = _HOOKS["spinodal_sign"]

    def thunk():
        b = arb_q(bq)
        s = (1 - 1 / b).sqrt()
        return s + sg * s.atanh() / b
    return certify(thunk)


# ---- cavity route (Sec. 1.4, App. 1.C) ----------------------------------------------------------------
def cw_free_entropy_cavity(beta, h=0, dps=30):
    """(m, Phi): the stable fixed points of m <- tanh beta (m + h) reached from m0 = +1 and -1, then
    Phi = max phi~(m), phi~ = -beta m^2/2 + ln 2 cosh beta (m + h) (Eq. (1.23))."""
    with mp.workdps(dps + 10):
        bq, hq = rational(beta, "beta"), rational(h, "h", positive=False)
        b, hh = mp.mpf(bq.numerator) / bq.denominator, mp.mpf(hq.numerator) / hq.denominator
        best = None
        for m0 in (1, -1):
            m = mp.mpf(m0)
            for _ in range(20000):
                new = mp.tanh(b * (m + hh))
                if abs(new - m) < mp.mpf(10) ** (-dps - 5):
                    m = new
                    break
                m = new
            else:
                m = mp.findroot(lambda x: x - mp.tanh(b * (x + hh)), m)
            val = -b * m * m / 2 + mp.log(2 * mp.cosh(b * (m + hh)))
            if best is None or val > best[1]:
                best = (m, val)
        return +best[0], +best[1]


# ---- field-theoretic route (Sec. 1.5) -----------------------------------------------------------------
def cw_free_entropy_field(beta, h=0, dps=30):
    """(m, Phi): the stationary points of g(m, mhat) found by Newton from a grid of starts on the curve
    m = -tanh mhat; Phi = the largest g among them (Sec. 1.5: extr_{m, mhat} g)."""
    with mp.workdps(dps + 10):
        bq, hq = rational(beta, "beta"), rational(h, "h", positive=False)
        b, hh = mp.mpf(bq.numerator) / bq.denominator, mp.mpf(hq.numerator) / hq.denominator
        c = _HOOKS["field_h"]
        g = lambda m, k: b * m * m / 2 + c * b * hh * m + k * m + mp.log(2) + mp.log(mp.cosh(k))
        eqs = [lambda m, k: b * m + c * b * hh + k, lambda m, k: m + mp.tanh(k)]
        jac = lambda m, k: [[b, 1], [1, 1 / mp.cosh(k) ** 2]]
        found = []
        for i in range(-9, 10, 2):
            m0 = mp.mpf(i) / 10
            try:
                m, k = mp.findroot(eqs, (m0, -mp.atanh(m0)), J=jac, tol=mp.mpf(10) ** (-2 * dps), verify=False,
                                   maxsteps=400)              # linear convergence at beta = 1, h = 0 (singular J)
            except (ValueError, ZeroDivisionError):
                continue
            if abs(m) < 1 and max(abs(f(m, k)) for f in eqs) < mp.mpf(10) ** (-dps):
                found.append((g(m, k), m))
        if not found:
            raise Refusal("no-solution:solver", f"no stationary point of g at beta = {beta}, h = {h}")
        val, m = max(found, key=lambda t: t[0])
        return +m, +val


__all__ = ["CWFixedPoint", "CWSolution", "cw_critical_beta", "cw_thermo", "cw_fixed_points", "cw_free_entropy",
           "cw_free_entropy_cavity", "cw_free_entropy_field", "cw_magnetization", "cw_solution", "cw_spinodal"]
