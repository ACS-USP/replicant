"""Catalog entries of KZ Ch. 1-3 (K1): `curie_weiss` (routes replica, cavity, exact), `rfim` (route replica; K3's
route cavity, the RFIM on random graphs, lives in catalog_rfim_sparse.py) and `random_matrix` (routes replica,
cavity; kind wigner | wishart).

Source: F. Krzakala & L. Zdeborova, "Statistical Physics Methods in Optimization and Machine Learning" (lecture
notes, version of June 9 2024), read as the PDF and its text: "KZ".  Each entry states its convention in its
module (meanfield/curie_weiss.py, meanfield/rfim.py, rmt/stieltjes.py).  Ball values are reported as their
midpoint (40 digits) with an upper bound of |value - x| over the ball in `errors`.
Run parameters:
  curie_weiss (replica)  beta (rational > 0, default 3/2), h (rational, default 0), branch, m (a rate point)
  curie_weiss (cavity)   beta, h, N (the exact cavity ratio ln Z_{N+1} - ln Z_N)
  curie_weiss (exact)    N (1 .. 20000, default 200), beta, h, m (a grid magnetization: the finite-N rate)
  rfim (replica)         J (> 0, default 1), law (gaussian | bimodal), Delta (variance, default 1/2) or h0,
                         beta (the RS state point), N (exact sorted ground states, and with beta the
                         Hubbard-Stratonovich free entropy and Guerra's bounds), seeds (default 0-15)
  rfim (cavity)          catalog_rfim_sparse.PARAMS (graph, d | c, J, law, Delta, h0, beta, pop, sweeps, seeds, N, iters)
  random_matrix          kind (wigner (default) | wishart), alpha (wishart: M/N), x (density point), y (> 0:
                         S(x + i y))
"""
import mpmath as mp

from .amp._stats import student_band
from . import catalog_rfim_sparse as RS
from .catalog import Out, RefCheck, Stability, entry, fstep, has
from .core import Refusal
from .csp import _checks
from .lattice.ball import arb_q, ball_of, baller, rational
from .meanfield import curie_weiss as CW
from .meanfield import cw_exact as CX
from .meanfield import rfim as RF
from .rmt import stieltjes as RM
from .spec import ModelSpec

KZ = ("Krzakala & Zdeborova, Statistical Physics Methods in Optimization and Machine Learning, lecture notes "
      "(June 9 2024)")
_SEEDS = tuple(range(16))


def _put(**balls):
    vals, errs = {}, {}
    for k, b in balls.items():
        vals[k], errs[k] = b.value_and_error(40)
    return vals, errs


def _lit(where, note=""):
    return Stability("exact-by-literature", "exact", f"{KZ}, {where}", note)


def _phys(ansatz, where, note):
    return Stability("not-computed", ansatz, f"{KZ}, {where}", note)


# ---- curie_weiss -----------------------------------------------------------------------------------------
def _bh(p):
    return p.get("beta", "3/2"), p.get("h", 0)


def _crit(p):
    v, e = _put(beta_c=CW.cw_critical_beta())
    return Out(v, "beta_c: the paramagnetic root's Jacobian beta (1 - tanh(0)^2) reaches 1", errors=e)


def _spin(p):
    v, e = _put(h_sp=CW.cw_spinodal(_bh(p)[0]))
    return Out(v, "h_sp = sqrt(1 - 1/beta) - atanh(sqrt(1 - 1/beta))/beta (KZ p. 33 corrected)", errors=e)


def _variational(p):
    beta, h = _bh(p)
    sol = CW.cw_solution(beta, h, p.get("branch"))
    balls, meta = {"phi": sol.phi}, {"branches": [f.branch for f in sol.fixed_points]}
    for f in sol.fixed_points:
        balls.update({f"m_{f.branch}": f.m, f"phi_{f.branch}": f.phi, f"jacobian_{f.branch}": f.jacobian})
        meta[f"stable_{f.branch}"] = f.stable
    if sol.selected is not None:
        fe, en, s = CW.cw_thermo(sol.selected, beta, h)
        balls.update(magnetization=sol.selected.m, f=fe, e=en, s=s)
    v, e = _put(**balls)
    return Out({**v, **meta}, f"{len(sol.fixed_points)} root(s) ({', '.join(meta['branches'])}) of phi'(m) = 0, "
               "certified" + ("" if sol.selected else "; h = 0 above beta_c: magnetization needs a branch"), errors=e)


def _field(p):
    m, phi = CW.cw_free_entropy_field(*_bh(p))
    return Out({"phi_field": phi, "m_field": m}, f"extr g(m, mhat) by 2D Newton: Phi = {mp.nstr(phi, 15)}")


def _rate(p):
    beta, h = _bh(p)
    m = float(rational(p["m"], "m", positive=False))
    return Out({"rate_legendre": CX.cw_rate_function(m, beta, h, "legendre"),
                "rate_gartner_ellis": CX.cw_rate_function(m, beta, h, "gartner_ellis"),
                "rate_ldp": CX.cw_ldp_rate(m, beta, h)}, f"rate functions of Sbar at m = {m}")


def _cavity(p):
    beta, h = _bh(p)
    m, phi = CW.cw_free_entropy_cavity(beta, h)
    key = "abs_magnetization" if rational(h, "h", positive=False) == 0 else "magnetization"
    return Out({"phi": phi, key: abs(m) if key == "abs_magnetization" else m}, "max phi~ over the stable fixed "
               "points of m <- tanh beta (m + h) from m0 = +1 and -1 (at h = 0 the two maxima are +-m: |m| reported)")


def _ratio(p):
    beta, h = _bh(p)
    n = p["N"]
    a, b = CX.cw_exact_z(n + 1, beta, h), CX.cw_exact_z(n, beta, h)
    with baller().hygiene.ctx_guard(prec=256):
        v, e = _put(cavity_ratio=ball_of(a.arb.log() - b.arb.log()))
    return Out({**v, "N": n}, f"ln Z_(N+1) - ln Z_N at N = {n} (KZ Eqs. (1.19)-(1.21))", errors=e)


def _exact(p):
    beta, h = _bh(p)
    n = p.get("N", 200)
    x = CX.cw_exact(n, beta, h)
    v, e = _put(phi_N=x.phi, magnetization_N=x.magnetization, abs_magnetization_N=x.abs_magnetization,
                energy_N=x.energy, chi_N=x.chi)
    if p.get("m") is not None:
        v2, e2 = _put(rate_N=CX.cw_exact_rate(n, p["m"], beta, h))
        v, e = {**v, **v2}, {**e, **e2}
    meta = {"N": n}
    try:
        meta["finite_size_term"] = CX.cw_finite_size_term(beta, h)
    except Refusal as r:
        if r.reason != "precondition:off-critical":
            raise
    return Out({**v, **meta}, f"binomial sum over the {n + 1} magnetizations in arb (Lemma 2 window audited)",
               errors=e)


def curie_weiss():
    T1, T6 = "Theorem 1, Eq. (1.7)", "App. 1.C, Theorem 6"
    rep = [fstep("cw-critical-point", f"{KZ}, App. 1.B (beta_c = 1)", _crit),
           fstep("cw-spinodal", f"{KZ}, App. 1.B, p. 33 (erratum corrected)", _spin,
                 when=lambda p: rational(_bh(p)[0], "beta") > 1, skip="beta <= 1: no spinodal"),
           fstep("cw-variational", f"{KZ}, Sec. 1.1, {T1}; Sec. 1.3, Eq. (1.17)", _variational),
           fstep("cw-field-theory", f"{KZ}, Sec. 1.5", _field),
           fstep("cw-rate-functions", f"{KZ}, Eq. (1.8), Secs. 1.2.2-1.2.3", _rate, when=has("m"), skip="no m")]
    cav = [fstep("cw-cavity", f"{KZ}, Sec. 1.4, Eq. (1.23); {T6}", _cavity),
           fstep("cw-cavity-ratio", f"{KZ}, Eqs. (1.19)-(1.21)", _ratio, when=has("N"), skip="no N given")]
    exa = [fstep("cw-exact-finite-n", f"{KZ}, Sec. 1.1, Lemmas 1-2, Eqs. (1.2), (1.4), (1.6), (1.11)", _exact)]
    refs = [RefCheck("meanfield", "cw_beta_c", lambda v: v["beta_c"]),
            RefCheck("meanfield", "cw_spinodal_fig1b4_beta1p5", lambda v: v["h_sp"], params={"beta": "3/2"})]
    br = lambda k: (lambda v, b=k: Stability("stable" if v.get(f"stable_{b}") else "unstable", "theorem1",
                                              f"{KZ}, App. 1.B", "computed Jacobian beta (1 - m^2) vs 1"))
    fin = {"beta_c": _lit("App. 1.B (phi = ln 2 + (beta - 1) m^2/2 + O(m^3))"),
           "h_sp": _phys("theorem1", "App. 1.B", "the double root of the saddle equation, certified; derived here "
                         "(KZ's printed form is an erratum): not a literature value"),
           "phi": _lit(f"{T1}; cavity route: {T6}"), "magnetization": _lit(f"Eq. (1.12); cavity route: {T6}"), "abs_magnetization": _lit(T6),
           "f": _lit("Eq. (1.30)"), "e": _lit("Eq. (1.30)"),
           "s": _lit("Eq. (1.30)"), "rate_ldp": _lit("Eq. (1.8)"),
           "phi_field": _phys("saddle", "Sec. 1.5", "complex saddle point (heuristic)"),
           "m_field": _phys("saddle", "Sec. 1.5", "complex saddle point (heuristic)"),
           "rate_legendre": _phys("legendre", "Sec. 1.2.2", "the convex envelope: the true rate at exposed points only"),
           "rate_gartner_ellis": _phys("legendre", "Sec. 1.2.3, Theorem 3 (informal)", "the convex envelope"),
           "cavity_ratio": _lit("Eq. (1.21)", "exact at this N (certified ball)"),
           **{k: _lit("Eqs. (1.2), (1.4), (1.11)", "exact at this N (certified ball)") for k in (
               "phi_N", "magnetization_N", "abs_magnetization_N", "energy_N", "chi_N", "rate_N")}}
    for b in ("unique", "lower", "middle", "upper"):
        fin.update({f"{o}_{b}": br(b) for o in ("m", "phi", "jacobian")})
    meta = ("branches", "N", "finite_size_term") + tuple(f"stable_{b}" for b in ("unique", "lower", "middle", "upper"))
    return entry("curie_weiss", ModelSpec("curie_weiss", "ising", "none", constraint="fully connected ferromagnet"),
              KZ, {"replica": rep, "cavity": cav, "exact": exa}, refs,
              {"replica": "Theorem 1 (rigorous) certified in arb, with the field-theoretic saddle (heuristic) beside it",
               "cavity": "the cavity potential phi~ (Sec. 1.4), rigorous by Theorem 6 (App. 1.C); mpmath, 30 digits",
               "exact": "finite N: the binomial sum in arb, exact; Lemma 2's window audited"},
              "the Curie-Weiss ferromagnet: three toolboxes for Phi(beta, h), its spinodal, the rate functions, and the "
              "exact finite-N binomial sum", finality=fin, meta=meta,
              parameters={"replica": ("beta", "h", "branch", "m"), "cavity": ("beta", "h", "N"),
                          "exact": ("N", "beta", "h", "m")}, rigor={"exact": "certified"})


# ---- rfim ------------------------------------------------------------------------------------------------
def _law(p):
    return RF.field_law(p.get("law", "gaussian"), p.get("Delta", "1/2" if p.get("law", "gaussian") == "gaussian"
                                                         else None), p.get("h0"))


def _rf_state(p):
    law = _law(p)
    return Out({"law": law.kind, "law_scale": law.scale, "J": float(RF._pos(p.get("J", 1), "J"))},
               f"field law {law.kind} ({'Delta' if law.kind == 'gaussian' else 'h0'} = {law.scale}), J = {p.get('J', 1)}")


def _rf_t0(p):
    law, j = _law(p), p.get("J", 1)
    g = RF.rfim_ground_energy(j, law)
    v = {"e_min": g.e, "m_t0": g.m}
    if law.kind == "gaussian":
        v["delta_c_t0"] = RF.rfim_delta_c(j)
    return Out(v, f"T = 0: e_min = {g.e:.9g} at m = {g.m:.9g} (KZ Eq. (2.3))")


def _rf_rs(p):
    rs = RF.rfim_rs(p["beta"], p.get("J", 1), _law(p))
    return Out({"m": rs.m, "phi": rs.phi, "f": rs.f, "e": rs.e}, f"RS at beta = {p['beta']}: m = {rs.m:.9g}, "
               f"Phi = {rs.phi:.12g} (stationary points {', '.join(f'{x:.6g}' for x in rs.stationary)})")


def _rf_sample(p):
    law, j, n = _law(p), p.get("J", 1), p["N"]
    seeds = _checks.seeds(p.get("seeds", _SEEDS))
    hs = [RF.rfim_sample_fields(n, law, s) for s in seeds]
    plus = student_band([RF.rfim_exact_ground_energy(h, j, "plus") for h in hs])
    full = student_band([RF.rfim_exact_ground_energy(h, j) for h in hs])
    v = {"e_ground_plus": plus.mean, "e_ground": full.mean, "e_ground_plus_halfwidth99": plus.half_width,
         "N": n, "seeds": list(seeds)}
    err = {"e_ground_plus": plus.stderr, "e_ground": full.stderr}
    if p.get("beta") is not None:
        lo, ex, up = [], [], []
        for h in hs:
            a, b = RF.rfim_guerra_bound(h, p["beta"], j)
            x = RF.rfim_exact_phi(h, p["beta"], j)
            if not a - 1e-9 <= x <= b + 1e-9:
                raise Refusal("no-solution:audit", f"Guerra's sandwich fails: {a} <= {x} <= {b}")
            lo.append(a), ex.append(x), up.append(b)
        for k, xs in (("phi_guerra_lower", lo), ("phi_exact", ex), ("phi_guerra_upper", up)):
            bd = student_band(xs)
            v[k], err[k] = bd.mean, bd.stderr
    return Out(v, f"exact ground states by sorting over seeds {list(seeds)} at N = {n}", errors=err,
               statistical=tuple(k for k in err),
               notes=("statistical: exact finite-N samples over named seeds (99% Student-t); e_ground picks the "
                      "better of the two ferromagnetic states and lies O(N^-1/2) below e_min, e_ground_plus "
                      "(Sbar >= 0) converges at O(1/N)",))


def rfim():
    steps = [fstep("rfim-state", f"{KZ}, Ch. 2, p. 37", _rf_state),
             fstep("rfim-ground-energy", f"{KZ}, Sec. 2.2.3, Eq. (2.3); Ex. 2.2", _rf_t0),
             fstep("rfim-rs", f"{KZ}, Sec. 2.2, Eqs. (2.1)-(2.2)", _rf_rs, when=has("beta"), skip="no beta given"),
             fstep("rfim-exact-samples", f"{KZ}, Ex. 2.3 (exact ground states by sorting); Sec. 2.3, Ex. 2.2 "
                   "(Guerra's bounds)", _rf_sample, when=has("N"), skip="no N given")]
    refs = [RefCheck("meanfield", "rfim_fig221_emin_delta0p36", lambda v: v["e_min"], params={"Delta": "0.36"}),
            RefCheck("meanfield", "rfim_fig221_emin_delta0p81", lambda v: v["e_min"], params={"Delta": "0.81"}),
            RefCheck("factor", "bethe_peierls_beta_dim2", lambda v: v["beta_c_zero_field"], params={"d": 4},
                     route="cavity")]
    rs = lambda k: _phys("rs", "Sec. 2.2", "replica-symmetric saddle; KZ Sec. 2.3 argues it exact by interpolation "
                         "(not a numbered theorem): NOT final")
    st = Stability("not-computed", "finite-N", "meanfield/rfim.py", "statistical: named seeds, never final")
    fin = {**{k: rs(k) for k in ("m", "phi", "f", "e", "e_min", "m_t0")},
           "delta_c_t0": _phys("rs", "Ex. 2.3", "derived here (2 J^2/pi), not printed by KZ"),
           **{k: st for k in ("e_ground", "e_ground_plus", "phi_guerra_lower", "phi_exact", "phi_guerra_upper")},
           **RS.FINALITY}
    return entry("rfim", ModelSpec("rfim", "ising", "gaussian", constraint="fully connected, random fields"), KZ,
                 {"replica": steps, "cavity": RS.cavity_steps()}, refs,
                 {"replica": "replica RS (physics-level); exact finite-N checks statistical",
                  "cavity": "random graphs: population dynamics and BP on sampled graphs (statistical, named seeds)"},
                 "the random field Ising model: fully connected (RS free entropy, T = 0 ground energy, exact sorted "
                 "ground states, Guerra's bounds) and on random graphs (cavity: population dynamics, BP)", finality=fin,
                 meta=("law", "law_scale", "J", "N", "seeds", "e_ground_plus_halfwidth99") + RS.META,
                 parameters={"replica": ("J", "law", "Delta", "h0", "beta", "N", "seeds"), "cavity": RS.PARAMS})


# ---- random_matrix ---------------------------------------------------------------------------------------
def _rm_args(p):
    kind = p.get("kind", "wigner")
    return kind, p.get("alpha")


def _rm(route):
    def compute(p):
        kind, a = _rm_args(p)
        lo, hi = RM.edges(kind, route, alpha=a)
        balls = {"edge_lower": lo, "edge_upper": hi}
        vals = {"kind": kind}
        if kind == "wishart":
            with baller().hygiene.ctx_guard(prec=128):
                balls["atom_zero"] = ball_of(arb_q(RM.atom_at_zero(kind, alpha=a)))
        if p.get("x") is not None:
            d = RM.density(kind, p["x"], route, alpha=a)
            if route == "replica":
                balls["density"] = d
            else:
                vals["density"] = d
        if p.get("y") is not None:
            y = float(rational(p["y"], "y"))
            if p.get("x") is None:
                raise Refusal("precondition:state-point", "y is the imaginary part of z = x + i y: give x")
            z = complex(float(rational(p["x"], "x", positive=False)), y)
            if route == "replica":
                s = RM.stieltjes_wigner(z) if kind == "wigner" else RM.stieltjes_mp(z, a)
                with baller().hygiene.ctx_guard(prec=256):
                    balls.update(stieltjes_re=ball_of(s.real), stieltjes_im=ball_of(s.imag))
            else:
                s = RM.stieltjes_wigner_cavity(z) if kind == "wigner" else RM.stieltjes_mp_cavity(z, a)
                vals.update(stieltjes_re=s.real, stieltjes_im=s.imag)
        v, e = _put(**balls)
        return Out({**vals, **v}, f"{kind}: support [{lo.render(12)}, {hi.render(12)}]", errors=e)
    return compute


def random_matrix():
    rep = [fstep("rm-replica", f"{KZ}, Sec. 3.2 (p. 53), Ex. 3.1", _rm("replica"))]
    cav = [fstep("rm-cavity", f"{KZ}, Sec. 3.3; Ex. 3.1 (the Wishart recursion derived here)", _rm("cavity"))]
    R = lambda key, get, route="replica", **pr: RefCheck("meanfield", key, get, params=pr, route=route)
    refs = [R("semicircle_edge", lambda v: v["edge_upper"]), R("semicircle_edge", lambda v: v["edge_upper"], "cavity"),
            R("semicircle_density_0", lambda v: v["density"], x="0"),
            R("semicircle_density_0", lambda v: v["density"], "cavity", x="0"),
            R("semicircle_density_1", lambda v: v["density"], x="1"),
            R("mp_edge_lower_a0p25", lambda v: v["edge_lower"], kind="wishart", alpha="1/4"),
            R("mp_edge_upper_a0p25", lambda v: v["edge_upper"], kind="wishart", alpha="1/4"),
            R("mp_edge_lower_a0p5", lambda v: v["edge_lower"], kind="wishart", alpha="1/2"),
            R("mp_edge_upper_a0p5", lambda v: v["edge_upper"], "cavity", kind="wishart", alpha="1/2")]
    law = _phys("rs", "Sec. 3.2, Ex. 3.1", "replica / cavity derivation (physics-level); the semicircle and "
                "Marchenko-Pastur laws are theorems (Wigner 1958, Marchenko & Pastur 1967) not fetched in this build")
    fin = {k: law for k in ("edge_lower", "edge_upper", "density", "stieltjes_re", "stieltjes_im")}
    fin["atom_zero"] = Stability("exact-by-literature", "exact", f"{KZ}, Ex. 3.1 item 4", "rank X X^T <= N: at "
                                 "least M - N zero eigenvalues, a fraction 1 - 1/alpha")
    return entry("random_matrix", ModelSpec("random_matrix", "real", "gaussian", constraint="Wigner / Wishart spectrum"),
                 KZ, {"replica": rep, "cavity": cav}, refs,
                 {"replica": "the RS saddle's root, certified in acb (a ball of the closed form)",
                  "cavity": "the N -> N+1 recursion iterated 2^120 times (mpmath, Im z = 1e-30 for the density)"},
                 "spectra of Wigner and Wishart random matrices: Stieltjes transform, density, edges",
                 finality=fin, meta=("kind",), parameters={"replica": ("kind", "alpha", "x", "y"),
                                                           "cavity": ("kind", "alpha", "x", "y")},
                 rigor={"replica": "certified"})


def meanfield_entries():
    return [curie_weiss(), rfim(), random_matrix()]


__all__ = ["curie_weiss", "meanfield_entries", "random_matrix", "rfim"]
