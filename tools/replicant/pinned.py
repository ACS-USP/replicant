"""The joint pair problem with pinned directions: capacity boundary, state points, stability.

  capacity(alpha, d, c)                    -> PinnedCapacity   (the q -> 1 limit of the derived
                                                                functional: kappa*, M*, h*, k)
  alpha_c(kappa, d, c, M=None)             -> mpf              (max over M, or at fixed M)
  state_point(alpha, d, c, kappa)          -> StatePoint       (RS saddle below capacity, audited,
                                                                with the joint replicon and sectors)
  untied_slack(c, M1=...)                  -> Refusal          (the untied RS branch is inconsistent)
  support_vector_count(cap, n)             -> k = (n - 2) Phi(-h*)

Model and conventions: derive_pinned.py (u on the sphere, d the dimension, N = alpha d iid
Gaussian patterns, pinned unit vectors with Gram [[1, c], [c, 1]], constraint
<u, v_a> - <u, xi>/sqrt(d) >= kappa for a = 1, 2; tied M_1 = M_2 = M).
Derived RS functional (the package's own steps, `derive_pinned.pinned_spec()`):
      phi(q, M) = (1/2) [ln(1-q) + (q - m2)/(1-q)] + alpha E_t ln Phi((H - sqrt(q) t)/sqrt(1-q)),
      m2 = M^T G^-1 M = 2 M^2 / (1+c),  H = sqrt(d)(M - kappa).
Capacity boundary.  As q -> 1, ln Phi(u) -> -u^2/2 for u < 0, so
      phi = C(M) / (2 (1-q)) + O(ln(1-q)),   C(M) = 1 - m2(M) - alpha Psi(H(M)),
      Psi(h) = E[(t-h)_+^2] = (h^2+1) Phi(-h) - h phi(h).
The volume vanishes where C(M) = 0, and M is stationary: C'(M) = 0.  Equivalently kappa* is
the maximum over M of kappa(M) defined by alpha Psi(sqrt(d)(M-kappa)) = 1 - m2(M): with
M = q_p a (q_p^2 = (1+c)/2) this is the paper's m_RS = max_a [a q_p - h_a/sqrt(d)],
alpha Psi(h_a) = 1 - a^2 (code/rs_pair.py), here DERIVED from the functional above.  At that
point the replicon of the Q sector is lambda_R = 1 - alpha Phi(-h*) = 1 - k/d with
k = N Phi(-h*) the number of active constraints (violated by the unconstrained cavity field):
the support-vector law of code/rs_duals.py and an isostatic-type counting identity.
Refusals (named): precondition:alpha-range, precondition:dimension,
precondition:independent-pinned-directions (|c| >= 1), no-solution:capacity (the maximisation over
M has no interior stationary point), no-solution:above-capacity (kappa >= kappa*),
no-solution:state-point (no RS root from the seeds), no-solution:untied-slack,
precondition:rs-saddle (the audit against the derived equations fails).
Physics-level: the n -> 0 continuation and the RS ansatz are not proved.
"""
from dataclasses import dataclass

import mpmath as mp
import numpy as np
from scipy.optimize import root
from scipy.special import log_ndtr
from scipy.stats import norm

from . import derive_pinned as DP
from .catalog import from_flag
from .core import Refusal
from .pinned_modes import (Block, class_values, longitudinal_mode, pinned_block,
                           replicon_lambda)

CITE = "replicant level (a): pinned-direction perceptron (this package)"


def psi(h):
    """Psi(h) = E[(t-h)_+^2]."""
    return (h * h + 1) * mp.ncdf(-h) - h * mp.npdf(h)


def lam(h):
    """E[(t-h)_+] = phi(h) - h Phi(-h)."""
    return mp.npdf(h) - h * mp.ncdf(-h)


def _check(alpha, d, c):
    if not (alpha > 0 and mp.isfinite(alpha)):
        raise Refusal("precondition:alpha-range", f"alpha = {alpha}: need a finite alpha > 0")
    if not (d > 0 and mp.isfinite(d)):
        raise Refusal("precondition:dimension", f"d = {d}: need a finite d > 0")
    if not abs(c) < 1:
        raise Refusal("precondition:independent-pinned-directions",
                      f"|c| = {abs(c)} >= 1: the Gram matrix [[1, c], [c, 1]] is singular")


def _qp2(c):
    return (1 + (0 if DP._PLANT.get("gram") == "identity" else c)) / 2


def m2_of(M, c):
    """M^T G^-1 M for the tied overlaps: M^2 / q_p^2 (the entropic term's q-shift)."""
    if DP._PLANT.get("entropy") == "drop_m":
        return M * 0
    return M * M / _qp2(c)


def dm2_of(M, c):
    """d m2 / dM."""
    return M * 0 if DP._PLANT.get("entropy") == "drop_m" else 2 * M / _qp2(c)


@dataclass(frozen=True)
class PinnedCapacity:
    alpha: object
    d: object
    c: object
    kappa: object          # kappa*: the maximal margin, = m_RS(c; n, d) of the paper
    M: object              # M* = q_p a*
    a: object              # a* = M* / q_p, the weight of the pair direction e
    h: object              # h* = sqrt(d)(M* - kappa*): the threshold of the Gaussian cavity field
    replicon: object       # lambda_R = 1 - alpha Phi(-h*) (q -> 1 limit of the Q-sector replicon)
    active_over_d: object  # k / d = alpha Phi(-h*) (active constraints per dimension)

    def support_vectors(self, n=None):
        """k = N Phi(-h*) with N = alpha d (= n - 2 for n documents)."""
        return self.alpha * self.d * mp.ncdf(-self.h)


def capacity(alpha, d, c, *, dps=30) -> PinnedCapacity:
    """The capacity boundary of the joint problem (module docstring): the maximal margin
    kappa* at load alpha = N/d in dimension d for a pair of overlap c, from C(M) = C'(M) = 0."""
    with mp.workdps(dps):
        alpha, d, c = mp.mpf(alpha), mp.mpf(d), mp.mpf(c)
        _check(alpha, d, c)
        qp = mp.sqrt(_qp2(c))
        sd = mp.sqrt(d)
        Mmax = qp if DP._PLANT.get("entropy") != "drop_m" else mp.mpf(3)

        def h_of(M):
            rhs = 1 - m2_of(M, c)
            if rhs <= 0:
                return None
            lo, hi = mp.mpf(-40), mp.mpf(40)          # alpha Psi(h) - rhs decreases in h
            for _ in range(32):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if alpha * psi(mid) > rhs else (lo, mid)
            h = (lo + hi) / 2
            for _ in range(8):                         # Newton, Psi' = -2 Lambda
                h = h + (alpha * psi(h) - rhs) / (2 * alpha * lam(h))
            return h

        def kappa_of(M):
            h = h_of(M)
            return None if h is None else M - h / sd
        grid = [Mmax * (k + mp.mpf("0.5")) / 60 for k in range(60)]
        vals = [(kappa_of(M), M) for M in grid]
        vals = [(k, M) for k, M in vals if k is not None]
        k0, M0 = max(vals)
        if M0 in (grid[0], grid[-1]):
            raise Refusal("no-solution:capacity",
                          f"kappa(M) has no interior maximum on (0, {mp.nstr(Mmax, 6)}): "
                          "no stationary point of C(M) at C(M) = 0")

        def g(M):                     # d kappa / dM = 1 - h'(M)/sqrt(d), h' = m2'/(-alpha Psi'(h))
            h = h_of(M)
            return 1 - dm2_of(M, c) / (2 * sd * alpha * lam(h))
        M = mp.findroot(g, (M0, M0 + Mmax / 600), solver="secant",
                        tol=mp.mpf(10)**(-dps + 8), verify=False)
        h = h_of(M)
        kappa = M - h / sd
        return PinnedCapacity(alpha, d, c, kappa, M, M / qp, h, 1 - alpha * mp.ncdf(-h),
                              alpha * mp.ncdf(-h))


def alpha_c(kappa, d, c, M=None, *, dps=30):
    """The largest load with a solution at margin kappa: max_M (1 - m2(M)) / Psi(sqrt(d)(M - kappa)),
    or its value at the fixed overlap M (M = 0: the catalog spherical perceptron with the margin
    sqrt(d) kappa, alpha_c = 1 / Psi(-sqrt(d) kappa))."""
    with mp.workdps(dps):
        kappa, d, c = mp.mpf(kappa), mp.mpf(d), mp.mpf(c)
        _check(mp.mpf(1), d, c)
        sd = mp.sqrt(d)
        f = lambda x: (1 - m2_of(x, c)) / psi(sd * (x - kappa))
        if M is not None:
            return f(mp.mpf(M))
        qp = mp.sqrt(_qp2(c))
        grid = [qp * (k + mp.mpf("0.5")) / 200 for k in range(200)]
        top = max(grid, key=f)
        M = mp.findroot(lambda x: mp.diff(f, x), (top, top + qp / 600), solver="secant",
                        tol=mp.mpf(10)**(-dps + 8), verify=False)
        return f(M)


# ---- state points below capacity -----------------------------------------------------

_GH = None


def _gh():
    global _GH
    if _GH is None:
        x, w = np.polynomial.hermite_e.hermegauss(200)
        _GH = (x, w / np.sqrt(2 * np.pi))
    return _GH


def _float_equations(v, alpha, d, kappa, qp, m2c):
    """(1-q)^2 d phi / dq and (1-q) d phi / dM of the derived functional in (q, a = M / qp):
    float64, 200-node Gauss-Hermite (the root is re-decided by the audit)."""
    q, a = v
    if not 0 < q < 1:
        return [1e3, 1e3]
    z, w = _gh()
    H = np.sqrt(d) * (qp * a - kappa)
    s = np.sqrt(1 - q)
    u = (H - np.sqrt(q) * z) / s
    G = np.exp(norm.logpdf(u) - log_ndtr(u))
    dudq = 0.5 * (H - np.sqrt(q) * z) / s**3 - z / (2 * np.sqrt(q) * s)
    e_a = -m2c * a / (1 - q) + alpha * np.sqrt(d) * qp / s * np.sum(w * G)
    e_q = (q - m2c * a * a) / (2 * (1 - q)**2) + alpha * np.sum(w * G * dudq)
    return [e_q * (1 - q)**2, e_a * (1 - q)]


def _refine_equations(v, alpha, d, kappa, qp, m2c):
    """The same two equations by adaptive quadrature (the polish of the Gauss-Hermite root)."""
    from scipy.integrate import quad
    q, a = v
    if not 0 < q < 1:
        return [1e3, 1e3]
    H = np.sqrt(d) * (qp * a - kappa)
    s = np.sqrt(1 - q)

    def E(f):
        return quad(lambda z: norm.pdf(z) * f(z), -14, 14, epsabs=1e-15, epsrel=1e-14,
                    limit=500)[0]
    u = lambda z: (H - np.sqrt(q) * z) / s
    G = lambda z: np.exp(norm.logpdf(u(z)) - log_ndtr(u(z)))
    dudq = lambda z: 0.5 * (H - np.sqrt(q) * z) / s**3 - z / (2 * np.sqrt(q) * s)
    e_a = -m2c * a / (1 - q) + alpha * np.sqrt(d) * qp / s * E(G)
    e_q = (q - m2c * a * a) / (2 * (1 - q)**2) + alpha * E(lambda z: G(z) * dudq(z))
    return [e_q * (1 - q)**2, e_a * (1 - q)]


@dataclass(frozen=True)
class StatePoint:
    alpha: object
    d: object
    c: object
    kappa: object
    q: object
    M: object
    a: object
    h: object
    replicon: object            # lambda_R from the class values (n -> 0)
    replicon_derived: object    # lambda_R of derive_replicon at the audited saddle, or None
    longitudinal: object        # mode of the uniform fluctuation of M (q stationary)
    block: Block                # the standard-irrep block (anomalous Q mode mixed with dM)
    residual: object            # |d phi| max at the root (float solve)
    stable: object              # True / False; see finality
    note: str

    def finality(self):
        """Per-value Stability: `stable` only if the computed replicon is positive (and no
        computed longitudinal or standard-sector mode has the wrong sign)."""
        rep = self.replicon_derived if self.replicon_derived is not None else self.replicon
        note = (f"joint replicon {mp.nstr(rep, 8)} (Q sector at the joint saddle, no mixing with "
                f"M by Schur); longitudinal M-mode {mp.nstr(self.longitudinal, 6)}; "
                f"standard-sector modes {[mp.nstr(m, 6) for m in self.block.modes]}")
        return {k: from_flag(self.stable, "rs", CITE, note) for k in ("q", "M", "replicon")}


def state_point(alpha, d, c, kappa, *, audit=True, dps=20, seeds=None) -> StatePoint:
    """The tied RS saddle of the derived functional at kappa < kappa*, with the stability
    sectors.  The root is found in float64 from seeds (continuation from the capacity point), then
    (audit=True) verified to be a root of the DERIVED RS equations by an independent quadrature
    and the replicon re-computed by derive_replicon.derived_replicon at that point."""
    with mp.workdps(dps):
        alpha_, d_, c_, kappa_ = (mp.mpf(x) for x in (alpha, d, c, kappa))
        _check(alpha_, d_, c_)
        cap = capacity(alpha_, d_, c_, dps=dps)
        if not kappa_ < cap.kappa:
            raise Refusal("no-solution:above-capacity",
                          f"kappa = {mp.nstr(kappa_, 8)} >= kappa* = {mp.nstr(cap.kappa, 12)}: "
                          "no RS saddle with q < 1 (the volume is empty)")
        qp = float(mp.sqrt(_qp2(c_)))
        m2c = 0.0 if DP._PLANT.get("entropy") == "drop_m" else 1.0
        found = None
        for q0 in tuple(seeds or (0.9, 0.5, 0.97, 0.3, 0.995)):
            for a0 in (float(cap.a), 0.5 * float(cap.a)):
                try:
                    sol = root(_float_equations, [q0, a0],
                               args=(float(alpha_), float(d_), float(kappa_), qp, m2c),
                               method="hybr", tol=1e-14)
                except (ValueError, FloatingPointError, ZeroDivisionError):
                    continue
                q, a = sol.x
                if sol.success and 0 < q < 1 and a > 0 and max(abs(sol.fun)) < 1e-11:
                    found = root(_refine_equations, sol.x, args=(float(alpha_), float(d_),
                                                                 float(kappa_), qp, m2c),
                                 method="hybr", tol=1e-15)
                    break
            if found is not None:
                break
        if found is None:
            raise Refusal("no-solution:state-point",
                          f"no RS root (q, a) found from the seeds at alpha = {alpha}, d = {d}, "
                          f"c = {c}, kappa = {kappa}")
        q, a = (mp.mpf(float(x)) for x in found.x)
        M = a * mp.sqrt(_qp2(c_))
        cv = class_values(0, alpha_, d_, c_, kappa_, q, M, dps=dps)
        rep = replicon_lambda(alpha_, d_, c_, kappa_, q, M, dps=dps, cv=cv)
        blk = pinned_block(alpha_, d_, c_, kappa_, q, M, dps=dps, cv=cv)
        lon = longitudinal_mode(alpha_, d_, c_, kappa_, q, M, dps=dps)
        rep_der, notes = None, []
        if audit:
            from .derive_replicon import derived_replicon
            r = derived_replicon(DP.pinned_spec(),
                                 {"alpha": alpha_, "kappa": kappa_, "d": d_, "c": c_, "q": q, "M": M},
                                 dps=min(dps, 15), saddle_tol="1e-8")
            rep_der = r.value
            notes.append(f"root audited against the derived equations; derived replicon "
                         f"{mp.nstr(rep_der, 10)} ({r.route})")
        use = rep_der if rep_der is not None else rep
        flags = [use > 0, lon > 0, blk.stable]
        note = "; ".join(notes + [("standard-sector eigenvalues complex (n - 2 < 0 makes T "
                                   "non-symmetric); real parts used") if blk.complex_pair else
                                  "standard-sector eigenvalues real"])
        return StatePoint(alpha_, d_, c_, kappa_, q, M, a, mp.sqrt(d_) * (M - kappa_), rep, rep_der,
                          lon, blk, max(abs(found.fun)), all(flags), note)


# ---- the untied branch ----------------------------------------------------------------

def untied_slack():
    """The untied RS branch (constraint 1 binds, M_1 <= M_2) is inconsistent: the derived
    stationarity of M_2 contains no energy term, so it fixes M_2 = c M_1, which violates M_2 >= M_1
    for |c| < 1 and M_1 > 0.  Derived from the untied functional by the package's steps; always
    raises."""
    import sympy as sp

    from .agree import derive_rs
    eqs, state, _ = derive_rs(DP.pinned_spec(tied=False))
    M1, M2 = (p for p in state["rs_free"].params if p.name in ("M_1", "M_2"))
    eq2 = [e for e, p in zip(eqs, state["rs_free"].params) if p is M2][0]
    expr = eq2.lhs - eq2.rhs
    from .gauss import GaussE
    if expr.atoms(GaussE):
        raise Refusal("precondition:untied-structure",
                      f"d phi / d M_2 = {expr} holds an energy term: not entropic alone")
    sol = sp.solve(expr, M2)
    raise Refusal("no-solution:untied-slack",
                  f"the RS stationarity of the non-binding overlap is entropic alone "
                  f"({sp.simplify(expr)} = 0 gives M_2 = {sol[0] if sol else '?'}), which violates "
                  f"the non-binding condition M_2 >= M_1 for |c| < 1 and M_1 > 0: no untied RS "
                  f"saddle; the tied branch is the only consistent one")


# ---- the composer's run path -----------------------------------------------------------

_RUN_PARAMS = frozenset({"alpha", "n", "d", "c", "kappa", "audit", "dps"})


def run_composed(calc, require_final, params):
    """Result of a composed pinned-direction spec (compose.ComposedCalculation._run_composed
    routes here: the generic solve-derived search does not converge on this two-unknown
    functional in practical time, so the state point is located in float64 and audited against the
    derived equations, module docstring).  Parameters: d, c, and alpha or n (alpha = (n-2)/d);
    with kappa the RS saddle below capacity and its stability sectors, else the capacity
    boundary only.  The untied spec is refused no-solution:untied-slack."""
    import sympy as sp

    from .agree import derive_rs
    from .compose_result import Result, _finalise
    from .core import TraceEntry

    spec = calc.spec
    if spec.constraint == DP.UNTIED:
        untied_slack()
    p = {**{k: v for k, v in spec.params.items()
            if not (isinstance(v, sp.Basic) and v.free_symbols)}, **params}
    unknown = sorted(set(p) - _RUN_PARAMS)
    if unknown:
        raise Refusal("precondition:known-parameter",
                      f"the pinned-direction route does not read {', '.join(unknown)}; "
                      f"accepted: {', '.join(sorted(_RUN_PARAMS))}")
    missing = [k for k in ("d", "c") if k not in p] + ([] if ("alpha" in p or "n" in p)
                                                         else ["alpha (or n)"])
    if missing:
        raise Refusal("precondition:state-point", f"need {', '.join(missing)}")
    if "alpha" in p and "n" in p:
        raise Refusal("precondition:consistent-parameters", "give alpha or n, not both")
    dps = int(p.get("dps", 20))
    d, c = mp.mpf(str(p["d"])), mp.mpf(str(p["c"]))
    alpha = mp.mpf(str(p["alpha"])) if "alpha" in p else (mp.mpf(str(p["n"])) - 2) / d
    DP.check_spec(spec)
    eqs, state, trace = derive_rs(spec)
    with mp.workdps(dps):
        cap = capacity(alpha, d, c, dps=dps)
        values = {"kappa_star": cap.kappa, "a_star": cap.a, "M_star": cap.M, "h_star": cap.h,
                  "k_active": cap.support_vectors(), "k_over_d": cap.active_over_d,
                  "replicon_capacity": cap.replicon}
        cap_note = (f"capacity boundary (q -> 1 limit of the derived functional): joint replicon "
                    f"1 - k/d = {mp.nstr(cap.replicon, 8)}; the longitudinal and standard-sector "
                    "modes diverge at q = 1 and are computed below capacity (give kappa)")
        fin = {k: from_flag(cap.replicon > 0, "rs", CITE, cap_note) for k in values}
        notes = ["physics-level (replica n -> 0 and the RS ansatz are not proved)", cap_note]
        if "kappa" in p:
            sp_ = state_point(alpha, d, c, mp.mpf(str(p["kappa"])), audit=bool(p.get("audit", True)),
                              dps=dps)
            values.update({"q": sp_.q, "M": sp_.M, "a": sp_.a, "h": sp_.h,
                           "replicon": sp_.replicon_derived if sp_.replicon_derived is not None
                           else sp_.replicon, "longitudinal_mode": sp_.longitudinal,
                           "standard_modes": tuple(sp_.block.modes), "stable": sp_.stable})
            fin.update(sp_.finality())
            fin.update({k: sp_.finality()["q"] for k in ("a", "h", "longitudinal_mode",
                                                        "standard_modes")})
            notes.append(sp_.note)
    trace.add(TraceEntry("solve-pinned", CITE,
                         f"capacity kappa* = {mp.nstr(cap.kappa, 12)}"
                         + (f"; RS saddle q = {mp.nstr(values['q'], 10)}, M = "
                            f"{mp.nstr(values['M'], 10)}; stable {values['stable']}"
                            if "q" in values else "")))
    res = Result(values, trace, notes=notes, route="replica", report=calc.report,
                 params=dict(p), spec=spec)
    return _finalise(res, fin, require_final)
