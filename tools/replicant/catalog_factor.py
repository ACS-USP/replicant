"""Catalog entries of KZ Ch. 4-6 (K3): `potts_bethe` and `coloring` (route cavity); the `rfim` cavity route is in
catalog_rfim_sparse.py (imported by catalog_meanfield.rfim).

Source: F. Krzakala & L. Zdeborova, lecture notes (June 9 2024), "KZ".  Convention (coloring/potts_bp.py): weight
exp(-beta delta_{s_i s_j}); beta > 0 antiferromagnetic (coloring), beta < 0 ferromagnetic; beta_Potts =
-2 beta_Ising; c the mean degree, d the degree of a d-regular graph, c~ the mean excess degree (d - 1; c for ER).
Run parameters:
  potts_bethe (cavity)  graph (regular (default) | poisson), d (regular, default 5), c (poisson), q (default 3),
                        beta (default -0.66), pop, sweeps (poisson: the population, default 2000 x 20),
                        N, seeds (BP on sampled graphs from the planted and a random start), iters
  coloring (cavity)     q (default 3), c (ER mean degree, default 3), beta (> 0 or inf, default inf),
                        N, seeds, iters, damping (BP on sampled ER graphs from random messages: the KS test; damping
                        default chosen above the synchronous uniform-mode bound, coloring/potts_graph.py; a given
                        damping at or below it labels the fraction undetermined)
Closed forms and scalar roots are floats (physics-level); finite-N BP and population dynamics are statistical over
named seeds (never final); a first-order point needs both starts converged, otherwise it is undetermined.
"""
import math

import numpy as np

from .amp._stats import student_band
from . import catalog_coloring as K7
from .catalog import Out, RefCheck, Stability, entry, fstep, has
from .coloring import potts_bp as PB
from .coloring import potts_graph as PG
from .core import Refusal
from .csp import _checks
from .factor import graph as FG
from .factor import population as POP
from .spec import ModelSpec

KZ = ("Krzakala & Zdeborova, Statistical Physics Methods in Optimization and Machine Learning, lecture notes "
      "(June 9 2024)")
_SEEDS = (0, 1, 2, 3)


def _phys(where, note):
    return Stability("not-computed", "rs", f"{KZ}, {where}", note)


def _stat(status_key, where, label_key=None):
    """Finality of a statistical value; with label_key, the run's Bethe labels (factor.bp: exact on a tree factor
    graph, an approximation on a loopy one) go into the note."""
    def f(v):
        und = v.get(status_key) not in (None, "converged", "determined")
        lab = "; ".join(v.get(label_key, ())) if label_key else ""
        return Stability("not-computed", "finite-N", f"{KZ}, {where}", "statistical over named seeds: NOT final"
                         + (f"; Bethe free entropy on sampled graphs: {lab}" if lab else "")
                         + (f"; UNDETERMINED: {v.get(status_key + '_note', '')}" if und else ""))
    return f


def _graph(p):
    g = p.get("graph", "regular")
    if g == "regular":
        d = p.get("d", 5)
        if isinstance(d, bool) or not isinstance(d, int) or d < 2:
            raise Refusal("precondition:d", f"d = {d!r}: an int degree >= 2")
        return g, d, float(d), d - 1.0
    if g == "poisson":
        c = float(PB.beta_value(p.get("c", 3)))
        if not 0 < c < math.inf:
            raise Refusal("precondition:c", f"c = {p.get('c')!r}: a finite mean degree > 0")
        return g, c, c, c
    raise Refusal("unsupported:graph", f"graph = {g!r}: regular | poisson")


# ---- potts_bethe -----------------------------------------------------------------------------------------
def _pb_para(p):
    g, par, c, ct = _graph(p)
    q, beta = p.get("q", 3), p.get("beta", "-0.66")
    pa = PB.potts_paramagnetic(q, c, beta)
    v = {"phi_para": pa.phi, "e_para": pa.e, "s_para": pa.s, "beta_stab": PB.potts_beta_stab(q, ct),
         "graph": g, "q": q, "c_tilde": ct}
    if q == 2:
        v["beta_stab_ising"] = PB.ising_beta(v["beta_stab"])
    return Out(v, f"paramagnetic chi = 1/q on mean degree {c}: Phi = {pa.phi:.12g}; beta_stab = {v['beta_stab']:.12g}")


def _pb_ferro(p):
    _, d, _, _ = _graph(p)
    q, beta = p.get("q", 3), p.get("beta", "-0.66")
    fps = PB.potts_ferro_fixed_points(q, d, beta)
    stable = [f for f in fps if f.stable]
    if not stable:
        sl = ", ".join(f"{f.kind} a = {f.a:.6g} (slope {f.slope:.4g})" for f in fps)
        return Out({"phase": "undetermined", "phase_note": f"no stable root of Eq. (6.11) ({sl}): the homogeneous "
                    "ansatz (6.9) has no stable fixed point here (antiferromagnetic beta on a d-regular graph: the "
                    "paramagnet is unstable to a mode outside the ansatz); phi and a_eq are not reported",
                    "fixed_points": [(f.kind, f.a, f.stable) for f in fps]},
                   "no stable fixed point of the homogeneous ansatz: phase undetermined")
    best = max(stable, key=lambda f: f.phi)
    v = {"phi": best.phi, "a_eq": best.a, "phase": best.kind, "fixed_points": [(f.kind, f.a, f.stable) for f in fps]}
    ferro = [f for f in stable if f.kind == "ferromagnetic"]
    if ferro:
        f = max(ferro, key=lambda f: f.a)
        v.update(a_ferro=f.a, phi_ferro=f.phi)
    return Out(v, f"{len(fps)} root(s) of Eq. (6.11); the stable root of largest Phi (6.12) is {best.kind}, "
               f"a = {best.a:.12g}")


def _pb_trans(p):
    _, d, _, _ = _graph(p)
    t = PB.potts_transitions(p.get("q", 3), d)
    v = {"transition_order": t.order, "transition_status": t.status, "transition_status_note": t.note}
    if t.beta_s is not None:
        v.update(beta_s=t.beta_s, beta_c=t.beta_c)
    return Out(v, f"{t.order}-order: beta_stab = {t.beta_stab:.10g}" + (
        f", beta_c = {t.beta_c:.10g}, beta_s = {t.beta_s:.10g}" if t.beta_s is not None else "") + f" ({t.status})")


def _pb_pop(p):
    _, c, _, _ = _graph(p)
    q, beta = p.get("q", 3), PB.beta_value(p.get("beta", "-0.66"))
    seeds = _checks.seeds(p.get("seeds", _SEEDS))
    law = POP.potts_law(q, beta)
    kw = dict(pop=p.get("pop", 2000), sweeps=p.get("sweeps", 20), seeds=seeds)
    pl = POP.population_dynamics(law, ("poisson", c), init="planted", **kw)
    rd = POP.population_dynamics(law, ("poisson", c), init="random", **kw)
    ok = pl.status == rd.status == "converged"
    v = {"phi_pd_planted": pl.phi, "phi_pd_random": rd.phi, "b0_pd_planted": float(pl.marginal[0]),
         "pd_status": "converged" if ok else "undetermined",
         "pd_status_note": f"planted: {pl.note}; random: {rd.note}"}
    if ok:
        gap = pl.phi - rd.phi
        v["pd_phase"] = ("same" if abs(gap) <= pl.phi_halfwidth99 + rd.phi_halfwidth99 else
                         "ordered" if gap > 0 else "paramagnetic-like")
    return Out(v, f"population dynamics on ER(c = {c}): planted Phi = {pl.phi:.6g}, random Phi = {rd.phi:.6g}",
               errors={"phi_pd_planted": pl.phi_stderr, "phi_pd_random": rd.phi_stderr,
                       "b0_pd_planted": float(pl.marginal_stderr[0])},
               statistical=("phi_pd_planted", "phi_pd_random", "b0_pd_planted"))


def _sample_edges(p, n, rng):
    g, par, _, _ = _graph(p)
    return FG.random_regular_edges(n, par, rng) if g == "regular" else FG.erdos_renyi_edges(n, par, rng)


def _pb_graph(p):
    q, beta, n = p.get("q", 3), p.get("beta", "-0.66"), p["N"]
    seeds = _checks.seeds(p.get("seeds", _SEEDS))
    runs = [PG.two_starts(PG.potts_graph(q, n, _sample_edges(p, n, np.random.default_rng(s)), beta),
                          iters=p.get("iters", 3000), seed=s + 1) for s in seeds]
    bad = [f"seed {s}: {r.note}" for s, r in zip(seeds, runs) if r.phase == "undetermined"]
    labels = sorted({r.label_planted for r in runs} | {r.label_random for r in runs})
    v = {"N": n, "seeds": list(seeds), "phases_bp": [r.phase for r in runs], "bethe_labels_bp": labels,
         "bp_status": "undetermined" if bad else "converged", "bp_status_note": "; ".join(bad)}
    err = {}
    for k in ("phi_planted", "phi_random"):
        b = student_band([getattr(r, k) for r in runs])
        v[f"{k}_bp"], err[f"{k}_bp"] = b.mean, b.stderr
    return Out(v, f"BP on {len(seeds)} sampled graphs (N = {n}) from the planted and a random start: phases "
               f"{v['phases_bp']}", errors=err, statistical=tuple(err))


def potts_bethe():
    reg = lambda p: p.get("graph", "regular") == "regular"
    steps = [fstep("potts-paramagnetic", f"{KZ}, Sec. 6.4; Eqs. (6.13), (6.22)", _pb_para),
             fstep("potts-ferro-fixed-points", f"{KZ}, Sec. 6.5, Eqs. (6.9)-(6.12)", _pb_ferro,
                   when=lambda p: reg(p) and PB.beta_value(p.get("beta", "-0.66")) < math.inf,
                   skip="the homogeneous ansatz (6.9) needs a d-regular graph and a finite beta"),
             fstep("potts-transitions", f"{KZ}, Secs. 6.5.1-6.5.2", _pb_trans, when=reg, skip="ER: no scalar ansatz"),
             fstep("potts-population", f"{KZ}, Sec. 4.3.2 (population dynamics), Sec. 6.5", _pb_pop,
                   when=lambda p: not reg(p) and p.get("pop") is not None, skip="regular graph or no pop given"),
             fstep("potts-bp-graphs", f"{KZ}, Secs. 5.2, 6.5 (BP on one graph)", _pb_graph, when=has("N"),
                   skip="no N given")]
    R = lambda key, get, **pr: RefCheck("factor", key, get, params=pr, route="cavity")
    refs = [R(f"bethe_peierls_beta_dim{k}", lambda v: v["beta_stab_ising"], q=2, d=2 * k, beta="-1/10")
            for k in (2, 3, 4, 5)]
    refs += [R("potts_beta_stab_q2_d4", lambda v: v["beta_stab"], q=2, d=4, beta="-1/10"),
             R("potts_beta_stab_q2_d6", lambda v: v["beta_stab"], q=2, d=6, beta="-1/10"),
             R("potts_beta_stab_q2_d5_fig651", lambda v: v["beta_stab"], q=2, d=5, beta="-1/10")]
    bp = _phys("Secs. 6.4-6.5", "BP fixed point of the cavity equations (physics-level; KZ Eq. (6.14) cites "
               "Dembo-Montanari(-Sly-Sun) for beta < 0, not fetched in this build)")
    fo = lambda v: Stability("not-computed", "rs", f"{KZ}, Sec. 6.5.2", "a first-order point: both starts of Eq. (6.11) "
                             "must converge" + ("" if v.get("transition_status") == "determined" else
                                                f"; UNDETERMINED: {v.get('transition_status_note', '')}"))
    fin = {**{k: bp for k in ("phi_para", "e_para", "s_para", "phi", "a_eq", "a_ferro", "phi_ferro")},
           "beta_stab": _phys("Eqs. (6.13), (6.22)", "linear stability of chi = 1/q (closed form)"),
           "beta_stab_ising": _phys("Sec. 4.2; Eq. (6.13)", "beta_Ising = -beta_Potts/2 at q = 2"),
           "beta_s": fo, "beta_c": fo,
           **{k: _stat("pd_status", "Sec. 4.3.2") for k in ("phi_pd_planted", "phi_pd_random", "b0_pd_planted")},
           **{k: _stat("bp_status", "Sec. 5.2", "bethe_labels_bp") for k in ("phi_planted_bp", "phi_random_bp")}}
    meta = ("graph", "q", "c_tilde", "phase", "fixed_points", "transition_order", "transition_status",
            "transition_status_note", "pd_status", "pd_status_note", "pd_phase", "N", "seeds", "phases_bp", "bp_status",
            "bp_status_note", "bethe_labels_bp", "phase_note")
    return entry("potts_bethe", ModelSpec("potts_bethe", "potts", "none", constraint="sparse random graph"), KZ,
                 {"cavity": steps}, refs, {"cavity": "BP / cavity fixed points (physics-level); finite-N BP and "
                                           "population dynamics statistical"},
                 "the q-state Potts model on random d-regular / ER graphs: paramagnetic and ferromagnetic BP fixed "
                 "points, beta_stab, the q >= 3 first-order spinodal and free-entropy crossing", default_route="cavity",
                 finality=fin, meta=meta, parameters={"cavity": ("graph", "d", "c", "q", "beta", "pop", "sweeps", "N",
                                                                 "seeds", "iters")})


# ---- coloring --------------------------------------------------------------------------------------------
def _mean_degree(p):
    c = p.get("c", 3)
    try:
        v = float(PB.beta_value(c)) if not isinstance(c, bool) else math.nan
    except Refusal:
        v = math.nan
    if not (math.isfinite(v) and v > 0):
        raise Refusal("precondition:c", f"c = {c!r}: the ER mean degree must be a finite number > 0")
    return v


def _col_para(p):
    q, c, beta = p.get("q", 3), _mean_degree(p), p.get("beta", "inf")
    if not PB.beta_value(beta) > 0:
        raise Refusal("precondition:beta", "coloring is the antiferromagnet: beta > 0 (or inf); beta < 0 is potts_bethe")
    pa = PB.potts_paramagnetic(q, c, beta)
    return Out({"phi_para": pa.phi, "e_para": pa.e, "s_para": pa.s, "s_max": math.log(q), "e_at_s_max": c / (2 * q),
                "q": q}, f"paramagnetic (= annealed) Phi = {pa.phi:.12g}, s(e*) = {pa.s:.12g}")


def _col_thr(p):
    q, beta = p.get("q", 3), p.get("beta", "inf")
    return Out({"c_ann": PB.coloring_annealed_threshold(q, beta), "c_ks": PB.kesten_stigum(q, beta),
                "c_ann_inf": PB.coloring_annealed_threshold(q), "c_ks_inf": PB.kesten_stigum(q)},
               "Eq. (6.17) and Eq. (6.24) (ER: c~ = c)")


def _col_bp(p):
    q, c, beta, n = p.get("q", 3), _mean_degree(p), p.get("beta", "inf"), p["N"]
    seeds = _checks.seeds(p.get("seeds", _SEEDS))
    runs = [PG.bp_convergence(q, c, beta, n=n, seed=s, iters=p.get("iters", 2000), damping=p.get("damping"))
            for s in seeds]
    frac = sum(r.converged for r in runs) / len(runs)
    return Out({"bp_converged_fraction": frac, "bp_max_deviation": [r.max_deviation for r in runs], "N": n,
                "seeds": list(seeds), "bp_damping": runs[0].damping, "bp_status": runs[0].status,
                "bp_status_note": runs[0].note},
               f"damped BP (damping {runs[0].damping:.4g}) on {len(seeds)} sampled ER graphs (N = {n}): converged "
               f"fraction {frac} ({runs[0].status})", statistical=("bp_converged_fraction",))


def coloring():
    steps = [K7.precheck_step(KZ),
             fstep("coloring-paramagnetic", f"{KZ}, Sec. 6.4", _col_para),
             fstep("coloring-thresholds", f"{KZ}, Sec. 6.6, Eqs. (6.17), (6.24)", _col_thr),
             fstep("coloring-bp-graphs", f"{KZ}, Sec. 6.6 (BP on large random graphs)", _col_bp,
                   when=lambda p: p.get("N") is not None and p.get("planted", False) is not True,
                   skip="no N given, or planted=true (BP from the planted start on planted graphs runs in "
                        "coloring-planted-graphs instead of this random-start step)")] + K7.steps(_mean_degree, KZ)
    refs = [RefCheck("factor", "coloring_c_ks_q3", lambda v: v["c_ks"], params={"q": 3}, route="cavity"),
            RefCheck("factor", "coloring_c_ann_q3", lambda v: v["c_ann"], params={"q": 3}, route="cavity")] + K7.refs()
    ann = _phys("Sec. 6.6, Eq. (6.17)", "closed form of the exact first moment (an upper bound on the quenched entropy "
                "by Jensen); equal to the paramagnetic Bethe free entropy")
    fin = {"phi_para": ann, "e_para": ann, "s_para": ann, "c_ann": ann, "c_ann_inf": ann,
           "s_max": _phys("Sec. 6.4", "maximum of the paramagnetic s(e): ln q at e = c/(2q)"),
           "e_at_s_max": _phys("Sec. 6.4", "c/(2q): a random coloring violates each edge w.p. 1/q"),
           "c_ks": _phys("Sec. 6.6, Eqs. (6.18)-(6.24)", "linear (variance) stability of chi = 1/q: physics-level"),
           "c_ks_inf": _phys("Eq. (6.24)", "(q - 1)^2 at beta = oo"),
           "bp_converged_fraction": _stat("bp_status", "Sec. 6.6"), **K7.finality()}
    return entry("coloring", ModelSpec("coloring", "potts", "none", constraint="antiferromagnetic, ER graph"), KZ,
                 {"cavity": steps}, refs, {"cavity": "paramagnetic BP / annealed closed forms (physics-level); "
                                           "finite-N BP statistical"},
                 "graph q-coloring (the Potts antiferromagnet) on ER graphs: paramagnetic entropy, annealed threshold "
                 "c_ann, Kesten-Stigum c_KS; with ansatz=1rsb the clusters (frozen SP c_col, 1RSB c_d / c_c, rigidity), "
                 "with planted=true the planted coloring (SBM link, BP from the planted start, quiet planting)", default_route="cavity", finality=fin,
                 meta=("q", "bp_max_deviation", "N", "seeds", "bp_damping", "bp_status", "bp_status_note") + K7.META,
                 parameters={"cavity": ("q", "c", "beta", "N", "seeds", "iters", "damping") + K7.PARAMS})


def factor_entries():
    return [potts_bethe(), coloring()]


__all__ = ["coloring", "factor_entries", "potts_bethe"]
