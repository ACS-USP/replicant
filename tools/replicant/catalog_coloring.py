"""K7 steps of the `coloring` entry (route cavity; KZ Chs. 13-14): ansatz=1rsb (frozen SP, 1RSB cavity, rigidity)
and planted=true (the planted coloring, its SBM link, BP from the planted start, quiet planting).

Sources: KZ (catalog_factor.KZ); ZK = Zdeborova & Krzakala, PRE 76, 031131 (2007), arXiv:0704.1269 (fetched).
Conventions: q colors, c = ER mean degree (the entry's graph); the regular-graph values (d, integer or real) are
labelled "_regular".  Run parameters added to `coloring` (cavity):
  ansatz      "rs" (default: K3's steps only) | "1rsb": closed forms (rigidity c_r(m = 1) on ER and regular graphs,
              the notes' regular SP: d_sp, d_uncol, the real root d_col), and with
  m, pop      the ER 1RSB population of populations at Parisi m (pop = N1 = N2, default 300), over `seeds`
  pop alone   the colour-symmetric frozen-SP population at (q, c) over `seeds` (pop members, default 10000)
  thresholds  "colorability" (ER c_col: the frozen-SP root), "clustering" (ER c_d, c_c by the 1RSB cavity),
              "regular" (integer c_d, c_c of d-regular graphs by the m = 1 population); over `seeds`
  planted     true: the SBM affinities (c_in = 0 at beta = oo, Eq. (13.3)) and c_l of Eq. (13.9); with N, BP from
              the planted start on planted graphs (overlap, Phi_planted - Phi_para, the quiet-planting verdict)
              in place of K3's random-start step
Values from populations and sampled graphs are statistical over named seeds (never final); closed forms are
physics-level floats.
"""
import math

from .amp._stats import student_band
from .catalog import UNCLAIMED, Out, RefCheck, Stability, fstep
from .coloring import planted as PL
from .coloring import rsb as R
from .coloring import rsb_pop as RP
from .core import Refusal
from .csp import _checks

KZ = ("Krzakala & Zdeborova, Statistical Physics Methods in Optimization and Machine Learning, lecture notes "
      "(June 9 2024)")
ZK = "Zdeborova & Krzakala, PRE 76, 031131 (2007), arXiv:0704.1269"
SEEDS = (0, 1, 2, 3)
PARAMS = ("ansatz", "planted", "thresholds", "m", "pop", "sweeps")
_THRESHOLDS = ("colorability", "clustering", "regular")


def precheck(p):
    a = p.get("ansatz", "rs")
    if a not in ("rs", "1rsb"):
        raise Refusal("unsupported:ansatz", f"ansatz = {a!r}: 'rs' | '1rsb'")
    if p.get("planted", False) not in (True, False):
        raise Refusal("precondition:planted", f"planted = {p.get('planted')!r}: true | false")
    t = p.get("thresholds")
    if t is not None and t not in _THRESHOLDS:
        raise Refusal("unsupported:thresholds", f"thresholds = {t!r}: one of {_THRESHOLDS}")
    if a == "rs" and any(p.get(k) is not None for k in ("thresholds", "m")):
        raise Refusal("precondition:ansatz", "thresholds and m belong to ansatz = '1rsb'")
    if a == "rs" and p.get("pop") is not None:
        raise Refusal("precondition:ansatz", "pop belongs to ansatz = '1rsb' (the coloring entry has no RS population)")
    return Out({}, f"ansatz {a}, planted {p.get('planted', False)}, thresholds {t}")


def _band(vals, errs, key, xs):
    b = student_band(xs)
    vals[key], vals[f"{key}_halfwidth99"], errs[key] = b.mean, b.half_width, b.stderr


def closed(p):
    q = p.get("q", 3)
    col = R.colorability_threshold(q, "regular")
    cor = R.rigidity_corrections(q)
    return Out({"c_r_m1": R.rigidity_threshold(q, "er").c_r, "c_r_m1_regular": R.rigidity_threshold(q, "regular").c_r,
                "rigidity_leading": cor["leading"], "rigidity_zk_eq44": cor["zk_eq44"], "d_sp_regular": col.d_sp,
                "d_uncol_regular": col.d_uncol, "d_col_regular": col.value},
               f"rigidity (Sec. 13.2.3) and the regular SP (Sec. 14.3): d_sp = {col.d_sp}, d_uncol = {col.d_uncol}")


def population(p, c):
    q, seeds = p.get("q", 3), _checks.seeds(p.get("seeds", SEEDS))
    vals, errs, bad = {"seeds": list(seeds)}, {}, []
    sw = p.get("sweeps", 150)
    if p.get("m") is not None:
        n = p.get("pop", 300)
        runs = [RP.coloring_1rsb(q, c, p["m"], graph="er", pop=(n, n), sweeps=sw, measure=max(2, sw // 3), seed=s)
                for s in seeds]
        bad += [f"1rsb seed {s}: {r.note}" for s, r in zip(seeds, runs) if r.status != "converged"]
        for k in ("psi", "phi", "sigma", "q1"):
            _band(vals, errs, f"{k}_1rsb", [getattr(r, k) for r in runs])
        vals["psi_annealed"] = runs[0].psi_annealed
    else:
        runs = [RP.frozen_sp(q, c, graph="er", pop=p.get("pop", 10000), sweeps=sw, measure=max(2, sw // 3), seed=s)
                for s in seeds]
        bad += [f"sp seed {s}: {r.note}" for s, r in zip(seeds, runs) if r.status != "converged"]
        for k in ("sigma", "eta"):
            _band(vals, errs, f"{k}_sp", [getattr(r, k) for r in runs])
    if len(seeds) < 2:
        bad.append("a band needs at least two seeds")
    vals.update(pop_status="undetermined" if bad else "converged", pop_status_note="; ".join(bad))
    return Out(vals, f"population over seeds {list(seeds)}" + (f" (UNDETERMINED: {'; '.join(bad)})" if bad else ""),
               errors=errs, statistical=tuple(errs))


def thresholds(p):
    q, seeds, t = p.get("q", 3), _checks.seeds(p.get("seeds", SEEDS)), p["thresholds"]
    vals, errs = {"seeds": list(seeds)}, {}
    if t == "colorability":
        col = R.colorability_threshold(q, "er", seeds=seeds, pop=p.get("pop", 10000), sweeps=p.get("sweeps", 150))
        vals.update(c_col_sp=col.value, c_col_sp_halfwidth99=col.half_width, c_col_status=col.status,
                    c_col_status_note=col.note)
        errs["c_col_sp"] = col.stderr
        return Out(vals, f"frozen-SP root c_col = {col.value:.5g} +- {col.half_width:.2g} ({col.status})", errors=errs,
                   statistical=("c_col_sp",))
    cd, cc = R.cluster_thresholds(q, "er" if t == "clustering" else "regular", seeds=seeds,
                                  **({"pop": p["pop"]} if p.get("pop") is not None else {}))
    sfx = "_1rsb" if t == "clustering" else "_regular"
    for key, th in (("c_d", cd), ("c_c", cc)):
        vals.update({f"{key}{sfx}_status": th.status, f"{key}{sfx}_status_note": th.note, f"{key}{sfx}_kind": th.kind})
        if th.status == "determined":
            vals[key + sfx] = th.value
            if t == "clustering":
                errs[key + sfx] = student_band(th.estimates).stderr
    return Out(vals, f"{t}: c_d {cd.value} ({cd.status}), c_c {cc.value} ({cc.status})", errors=errs,
               statistical=tuple(errs))


def planted(p, c):
    q, beta = p.get("q", 3), p.get("beta", "inf")
    c_in, c_out = PL.sbm_affinities(q, c, beta)
    return Out({"c_in": c_in, "c_out": c_out, "c_l_planted": PL.bp_threshold(q, beta)},
               f"SBM link (Eq. (13.3)): c_in = {c_in:.6g}, c_out = {c_out:.6g}; c_l (13.9) = {PL.bp_threshold(q, beta):.6g}")


def planted_graphs(p, c):
    q, n, seeds = p.get("q", 3), p["N"], _checks.seeds(p.get("seeds", SEEDS))
    if p.get("beta", "inf") not in ("inf", "oo", "+inf", math.inf):
        raise Refusal("unsupported:beta", "BP on planted graphs is implemented for proper colorings (beta = inf)")
    qp = PL.quiet_planting(q, c, n=n, seeds=seeds, iters=p.get("iters", 3000))
    vals, errs = {"N": n, "seeds": list(seeds), "quiet": qp.verdict, "quiet_status": qp.status,
                  "quiet_status_note": qp.note}, {}
    _band(vals, errs, "overlap_planted_bp", list(qp.overlaps))
    vals["delta_phi_planted_bp"], vals["delta_phi_planted_bp_halfwidth99"] = qp.delta, qp.half_width
    return Out(vals, f"BP from the planted start, N = {n}: overlaps {qp.overlaps}, quiet planting: {qp.verdict}",
               errors=errs, statistical=("overlap_planted_bp",))


def steps(mean_degree, cite):
    one = lambda p: p.get("ansatz", "rs") == "1rsb"
    return [fstep("coloring-k7-parameters", f"{cite}, Chs. 13-14", precheck,
                  when=lambda p: any(p.get(k) is not None for k in PARAMS), skip="K3's parameter set"),
            fstep("coloring-1rsb-closed-forms", f"{cite}, Secs. 13.2.3, 14.3; {ZK}, Eq. (44)", closed, when=one,
                  skip="ansatz rs"),
            fstep("coloring-1rsb-population", f"{cite}, Secs. 14.2-14.3 (population dynamics: statistical)",
                  lambda p: population(p, mean_degree(p)), when=lambda p: one(p) and (p.get("pop") is not None or
                                                                                    p.get("m") is not None)
                  and p.get("thresholds") is None, skip="no m or pop given (or thresholds asked)"),
            fstep("coloring-1rsb-thresholds", f"{cite}, Secs. 13.2.2, 14.1, 14.3; {ZK}, Tables I-II", thresholds,
                  when=lambda p: one(p) and p.get("thresholds") is not None, skip="no thresholds asked"),
            fstep("coloring-planted", f"{cite}, Sec. 13.1, Eqs. (13.3), (13.9)", lambda p: planted(p, mean_degree(p)),
                  when=lambda p: p.get("planted", False) is True, skip="planted not asked"),
            fstep("coloring-planted-graphs", f"{cite}, Secs. 13.2.2-13.2.3 (sampled planted graphs: statistical)",
                  lambda p: planted_graphs(p, mean_degree(p)),
                  when=lambda p: p.get("planted", False) is True and p.get("N") is not None, skip="no planted N")]


def refs():
    R1 = {"q": 3, "ansatz": "1rsb"}
    out = [RefCheck("factor", "coloring_c_ann_q5", lambda v: v["c_ann"], params={"q": 5}, route="cavity")]
    out += [RefCheck("factor", f"coloring_c_r_m1_q{q}", lambda v: v["c_r_m1"], params={**R1, "q": q}, route="cavity")
            for q in range(3, 11)]
    for q in (3, 4, 5):
        out += [RefCheck("factor", f"coloring_regular_d_sp_q{q}", lambda v: v["d_sp_regular"], params={**R1, "q": q},
                         route="cavity"),
                RefCheck("factor", f"coloring_regular_d_uncol_q{q}", lambda v: v["d_uncol_regular"],
                         params={**R1, "q": q}, route="cavity"),
                RefCheck("factor", f"coloring_c_col_q{q}", lambda v: v["c_col_sp"], statistical=True, slow=True,
                         params={**R1, "q": q, "thresholds": "colorability"}, route="cavity",
                         error=lambda e: e["c_col_sp"], runs=len(SEEDS))]
    cl = {**R1, "thresholds": "clustering"}
    out += [RefCheck("factor", f"coloring_{k}_q3", (lambda key: lambda v: v[key])(f"{k}_1rsb"), statistical=True,
                     slow=True, params=cl, route="cavity", error=(lambda key: lambda e: e[key])(f"{k}_1rsb"),
                     runs=len(SEEDS)) for k in ("c_d", "c_c")]
    for q, keys in ((3, ("c",)), (4, ("d", "c")), (5, ("d", "c"))):
        out += [RefCheck("factor", f"coloring_regular_d_{k}_q{q}", (lambda key: lambda v: v[key])(f"c_{k}_regular"),
                         slow=True, params={**R1, "q": q, "thresholds": "regular"}, route="cavity") for k in keys]
    return out


def _phys(where, note):
    return Stability("not-computed", "rs", f"{KZ}, {where}", note)


def _stat(status_key, where):
    def f(v):
        und = v.get(status_key) not in (None, "converged", "determined")
        return Stability("not-computed", "finite-N", f"{KZ}, {where}", "statistical over named seeds: NOT final"
                         + (f"; UNDETERMINED: {v.get(status_key + '_note', '')}" if und else ""))
    return f


def finality():
    rig = _phys("Sec. 13.2.3", "the freezing recursion on the tree, its fold (physics-level; ER form derived here)")
    sp = _phys("Sec. 14.3", "the notes' colour-symmetric SP on d-regular graphs (conjecture-level, physics)")
    pop = _stat("pop_status", "Secs. 14.2-14.3 (population dynamics)")
    fin = {"c_r_m1": rig, "c_r_m1_regular": rig, "d_sp_regular": sp, "d_uncol_regular": sp, "d_col_regular": sp,
           "rigidity_leading": _phys("Sec. 13.2.3", "the notes' leading large-q term q ln q (asymptotic form)"),
           "rigidity_zk_eq44": _phys("Sec. 13.2.4", f"{ZK}, Eq. (44): q (ln q + ln ln q + 1) (asymptotic form)"),
           "c_in": _phys("Eq. (13.3)", "the SBM affinities of the planted coloring (exact map)"),
           "c_out": _phys("Eq. (13.3)", "the SBM affinities of the planted coloring (exact map)"),
           "c_l_planted": _phys("Eq. (13.9)", "linear stability of the paramagnetic BP fixed point (physics-level)"),
           "psi_annealed": _phys("Sec. 6.6", "ln q + (c/2) ln(1 - 1/q): Psi(m = 1) of the 1RSB solution"),
           "c_col_sp": _stat("c_col_status", "Sec. 14.3 (frozen-SP population root)"),
           "overlap_planted_bp": _stat("quiet_status", "Sec. 13.2.2 (BP on sampled planted graphs)"),
           "delta_phi_planted_bp": _stat("quiet_status", "Sec. 13.2.2 (Bethe on sampled loopy graphs: approximation)")}
    for k in ("psi", "phi", "sigma", "q1"):
        fin[f"{k}_1rsb"] = pop
    for k in ("sigma", "eta"):
        fin[f"{k}_sp"] = pop
    for k in ("c_d", "c_c"):
        fin[f"{k}_1rsb"] = _stat(f"{k}_1rsb_status", "Secs. 13.2.2, 14.1 (1RSB population of populations)")
        fin[f"{k}_regular"] = _stat(f"{k}_regular_status", "Sec. 14.2 (m = 1 population, integer degrees)")
    return fin


META = ("ansatz", "planted", "seeds", "pop_status", "pop_status_note", "c_col_status", "c_col_status_note",
        "quiet", "quiet_status", "quiet_status_note") + tuple(
    f"{k}{s}_{x}" for k in ("c_d", "c_c") for s in ("_1rsb", "_regular") for x in ("status", "status_note", "kind")) \
    + tuple(f"{k}_halfwidth99" for k in ("psi_1rsb", "phi_1rsb", "sigma_1rsb", "q1_1rsb", "sigma_sp", "eta_sp",
                                         "c_col_sp", "overlap_planted_bp", "delta_phi_planted_bp"))

UNCLAIMED.update({("factor", f"coloring_{k}_q4"): (
    "ZK Table II, q = 4 ER; the K4 planted-measure route (sbm entry, q = 4, c_in = 0, seeds 0-3) brackets c_d in "
    "[8.350, 8.367] on seed 0 but labels the pair undetermined: c_c = 8.46 lies 0.1 above c_d and Delta Phi is already "
    "positive at the first grid point (8.367), so its zero is not resolved; the K7 1RSB ER first-order path is not "
    "run at reference size.  Banked for the table, checked by no route in this build") for k in ("c_d", "c_c")})

__all__ = ["META", "PARAMS", "finality", "refs", "steps"]
