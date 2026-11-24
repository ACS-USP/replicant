"""The q-group path of the `sbm` catalog entry (K4, KZ Ch. 10), imported by catalog_spiked_sbm.sbm.

Convention (inference/sbm_q.py): q groups of equal expected size 1/q, edge probability c_in/N inside a group and
c_out/N across (the q = 2 parameters a, b of the original entry are c_in, c_out); mean degree
c = (c_in + (q - 1) c_out)/q; epsilon = c_out/c_in.  The path is chosen by `q`; without `q` the entry is the q = 2
entry of E6b, unchanged, and the q-path parameters are refused.  Run parameters (with q):
  c_in, c_out      the Kesten-Stigum margin c lambda_2^2 = (c_in - c_out)^2/(q^2 c) (c_in < c_out accepted)
  c                the assortative KS point at mean degree c: c_in_ks, c_out_ks, inverse_eps_ks_q
  c_in (no c_out)  the line c_in fixed: c_l (KS point), c_d and c_c from planted-measure population dynamics
                   over named seeds (pop, sweeps; default 3000 x 400 per probe, seeds 0-3): statistical
  c_in, c_out, N   BP on sampled graphs over named seeds (default 0-3, iters 600, damping 1/2) from init=planted |
                   random (default random): overlap, Nishimori overlap, Bethe Delta Phi (an approximation)
  learn=true       with c_in, c_out, N: EM (BP-learning) of n_a, c_ab from a spectral start, over named seeds
c_d, c_c are reported only when the planted start, the random start and the free-entropy crossing agree
(inference/sbm_q_pd.py); otherwise they are absent and `transitions_status` is undetermined with the reason.
"""
import numpy as np

from .amp._stats import student_band
from .catalog import Out, RefCheck, Stability, fstep
from .core import Refusal
from .csp import _checks
from .inference import sbm_learn as EM
from .inference import sbm_q as SQ
from .inference import sbm_q_pd as PD

KZ = ("Krzakala & Zdeborova, Statistical Physics Methods in Optimization and Machine Learning, lecture notes "
      "(June 9 2024), Ch. 10")
DKMZ = ("Decelle, Krzakala, Moore & Zdeborova, Phys. Rev. E 84, 066106 (2011), arXiv:1109.3041, Eqs. (26)-(35), "
        "(44), Fig. 3")
Q_PARAMS = ("q", "c_in", "c_out", "init", "learn", "pop", "sweeps", "damping")
_SEEDS = (0, 1, 2, 3)


def stray_q(p):
    """Without q: the q-path parameters are refused (the q = 2 entry does not read them)."""
    bad = [k for k in Q_PARAMS if k != "q" and p.get(k) is not None]
    if bad:
        raise Refusal("precondition:known-parameter", f"{', '.join(bad)}: q-group parameters need q (the entry "
                      "without q is the two-group model in a, b)")


def _check(p):
    bad = [k for k in ("a", "b") if p.get(k) is not None]
    if bad:
        raise Refusal("precondition:known-parameter", f"{', '.join(bad)}: with q the affinities are c_in, c_out "
                      "(a, b are the q = 2 parameters of the entry without q)")
    q = SQ.group_count(p["q"])
    if p.get("c") is not None and p.get("c_in") is not None:
        raise Refusal("precondition:state-point", "give c (the KS point) or c_in [, c_out], not both")
    if p.get("c") is None and p.get("c_in") is None:
        raise Refusal("precondition:state-point", "with q the sbm entry needs c_in (and c_out) or c")
    if p.get("N") is not None and p.get("c_out") is None:
        raise Refusal("precondition:state-point", "finite-N BP and learning need c_in, c_out and N")
    line = p.get("c_out") is None and p.get("c") is None
    if p.get("N") is None:
        bad = [k for k in ("iters", "init", "learn", "damping") + (() if line else ("seeds",)) if p.get(k) is not None]
    else:
        bad = [k for k in ("pop", "sweeps") if p.get(k) is not None]
    if not line:
        bad += [k for k in ("pop", "sweeps") if p.get(k) is not None and k not in bad]
    if bad:
        raise Refusal("precondition:known-parameter", f"{', '.join(bad)}: not read at this state point (seeds, pop, "
                      "sweeps: the c_in line; seeds, iters, init, damping, learn: finite N)")
    init, learn = p.get("init", "random"), p.get("learn", False)
    if init not in ("planted", "random"):
        raise Refusal("precondition:known-parameter", f"init = {init!r}: accepted values are planted and random")
    if not (isinstance(learn, bool) or learn in ("true", "false")):
        raise Refusal("precondition:known-parameter", f"learn = {learn!r}: accepted values are true and false")
    unread = [k for k in ("init", "damping") if learn in (True, "true") and p.get(k) is not None]
    if unread:
        raise Refusal("precondition:known-parameter", f"{', '.join(unread)}: not read with learn=true (EM starts BP "
                      "from random messages at damping 1/2, inference/sbm_learn.py)")
    return q


def ks_q(p):
    q = _check(p)
    if p.get("c_out") is None:
        return Out({"q": q}, "no c_out given")
    k = SQ.sbm_q_kesten_stigum(q, p["c_in"], p["c_out"])
    eps = k.c_out / k.c_in if k.c_in else None
    return Out({"q": q, "c_in": k.c_in, "c_out": k.c_out, "mean_degree": k.mean_degree, "lambda2": k.lambda2,
                "ks_margin_q": k.margin, "ks_detectable_q": k.detectable, "eps": eps},
               f"c lambda_2^2 = (c_in - c_out)^2/(q^2 c) = {float(k.margin):.6g}: "
               f"{'above' if k.detectable else 'not above'} the Kesten-Stigum line")


def line_q(p):
    L = SQ.sbm_q_ks_line(SQ.group_count(p["q"]), p["c"])
    return Out({"c": L.c, "c_in_ks": L.c_in, "c_out_ks": L.c_out, "inverse_eps_ks_q": L.ratio},
               "bisection of c lambda_2^2 - 1 along c_in + (q - 1) c_out = q c")


def transitions(p):
    q, seeds = SQ.group_count(p["q"]), _checks.seeds(p.get("seeds", _SEEDS))
    kw = {"pop": p.get("pop", 3000), "sweeps": p.get("sweeps", 400)}
    runs = [PD.sbm_q_transitions(q, float(p["c_in"]), seed=s, **kw) for s in seeds]
    c_l = runs[0].c_l
    vals = {"c_l": c_l, "seeds": list(seeds), "pop": kw["pop"], "sweeps": kw["sweeps"]}
    bad = [f"seed {s}: {r.note}" for s, r in zip(seeds, runs) if r.status != "determined"]
    if bad or len(seeds) < 2:
        why = "; ".join(bad) or "a band needs at least two seeds"
        return Out({**vals, "transitions_status": "undetermined", "transitions_note": why},
                   "c_d, c_c undetermined: " + why, notes=(f"UNDETERMINED: {why}",))
    errs, stat = {}, []
    for key in ("c_d", "c_c"):
        b = student_band([getattr(r, key) for r in runs])
        vals.update({key: b.mean, f"{key}_halfwidth99": b.half_width})
        errs[key] = b.stderr
        stat.append(key)
    note = (f"planted / random population dynamics over seeds {list(seeds)} ({kw['pop']} members, {kw['sweeps']} "
            f"sweeps per c_d probe): c_d = {vals['c_d']:.4g} +- {vals['c_d_halfwidth99']:.2g}, c_c = "
            f"{vals['c_c']:.4g} +- {vals['c_c_halfwidth99']:.2g} (99% Student-t), c_l = {c_l:.6g}")
    return Out({**vals, "transitions_status": "determined", "transitions_note": runs[0].note}, note, errors=errs,
               statistical=tuple(stat), notes=(note,))


def _band(vals, errs, key, xs):
    b = student_band(xs)
    vals.update({key: b.mean, f"{key}_halfwidth99": b.half_width})
    errs[key] = b.stderr


def finite(p):
    q, seeds = SQ.group_count(p["q"]), _checks.seeds(p.get("seeds", _SEEDS))
    if p.get("learn", False) in (True, "true"):
        return learn(p, q, seeds)
    init, iters, damp = p.get("init", "random"), p.get("iters", 600), p.get("damping", 0.5)
    runs = [SQ.sbm_q_bp(SQ.sbm_q_sample(p["N"], q, p["c_in"], p["c_out"], s), q, p["c_in"], p["c_out"],
                        init=init, iters=iters, seed=s, damping=damp) for s in seeds]
    ok = [r for r in runs if r.converged]
    vals = {"N": p["N"], "seeds": list(seeds), "iters": iters, "init": init, "damping": damp,
            "q_bp_converged_fraction": len(ok) / len(runs),
            "q_bp_phi_label": runs[0].phi_label if ok else "undetermined"}
    if len(ok) < max(2, len(runs)):
        why = "; ".join(f"seed {s}: {r.note}" for s, r in zip(seeds, runs) if not r.converged) or "one seed"
        return Out({**vals, "q_bp_status": "undetermined", "q_bp_status_note": why},
                   f"BP values undetermined: {why}", notes=(f"UNDETERMINED: {why}",))
    errs = {}
    _band(vals, errs, "q_bp_overlap", [r.overlap for r in runs])
    _band(vals, errs, "q_bp_overlap_nishimori", [r.overlap_nishimori for r in runs])
    _band(vals, errs, "q_bp_delta_phi", [r.phi - r.phi_para for r in runs])
    vals["q_bp_status"] = "converged"
    b = vals["q_bp_overlap"]
    return Out(vals, f"BP ({init} start) over seeds {list(seeds)}: overlap {b:.4g} +- "
               f"{vals['q_bp_overlap_halfwidth99']:.2g}; Delta Phi {vals['q_bp_delta_phi']:.4g} (Bethe on loopy "
               "graphs: an approximation)", errors=errs, statistical=tuple(errs))


def learn(p, q, seeds):
    iters = p.get("iters", 60)
    runs = [EM.sbm_em(SQ.sbm_q_sample(p["N"], q, p["c_in"], p["c_out"], s), q, iters=iters, seed=s) for s in seeds]
    vals = {"N": p["N"], "seeds": list(seeds), "iters": iters, "learn": True}
    bad = [f"seed {s}: {r.status}: {r.note}" for s, r in zip(seeds, runs) if r.status != "ok"]
    if bad or len(seeds) < 2:
        why = "; ".join(bad) or "a band needs at least two seeds"
        return Out({**vals, "em_status": "undetermined", "em_status_note": why}, f"EM undetermined: {why}",
                   notes=(f"UNDETERMINED: {why}",))
    errs = {}
    off = ~np.eye(q, dtype=bool)
    _band(vals, errs, "em_c_in", [float(np.mean(np.diag(r.affinity))) for r in runs])
    _band(vals, errs, "em_c_out", [float(np.mean(r.affinity[off])) for r in runs])
    _band(vals, errs, "em_n_max", [float(np.max(r.sizes)) for r in runs])
    vals["em_status"] = "ok"
    return Out(vals, f"EM over seeds {list(seeds)}: c_in {vals['em_c_in']:.4g} +- {vals['em_c_in_halfwidth99']:.2g}, "
               f"c_out {vals['em_c_out']:.4g} +- {vals['em_c_out_halfwidth99']:.2g}", errors=errs,
               statistical=tuple(errs))


def _qwhen(*keys, lack=()):
    return lambda p: p.get("q") is not None and all(p.get(k) is not None for k in keys) and \
        all(p.get(k) is None for k in lack)


def steps():
    return [fstep("sbm-q-kesten-stigum", f"{KZ}, Eqs. (10.33)-(10.37); {DKMZ}", ks_q, when=_qwhen(), skip="no q"),
            fstep("sbm-q-ks-line", f"{KZ}, caption of Fig. 10.4.1; {DKMZ}", line_q, when=_qwhen("c"), skip="no q, c"),
            fstep("sbm-q-transitions", f"{KZ}, Sec. 10.4.3; {DKMZ} (population dynamics: statistical)", transitions,
                  when=_qwhen("c_in", lack=("c_out", "N")), skip="needs q, c_in without c_out"),
            fstep("sbm-q-finite-n", f"{KZ}, Eqs. (10.15)-(10.28); {DKMZ} (sampled graphs: statistical)", finite,
                  when=_qwhen("c_in", "c_out", "N"), skip="needs q, c_in, c_out and N")]


def _phys(note):
    return Stability("not-computed", "rs", f"{KZ}; {DKMZ}", note)


def _stat(status_key, note):
    def f(v):
        und = v.get(status_key) not in (None, "converged", "determined", "ok")
        return Stability("not-computed", "finite-N", f"{KZ}; {DKMZ}", "statistical over named seeds: NOT final; "
                         + note + (f"; UNDETERMINED: {v.get(status_key + '_note', '')}" if und else ""))
    return f


def finality():
    ks = _phys("the instability of the factorized BP fixed point (KZ Eq. (10.36)); for q >= 3 no detection theorem "
               "was fetched (the cited proofs are for q = 2): physics-level, NOT final")
    bp = _stat("q_bp_status", "Bethe free entropy on sampled loopy graphs: an approximation")
    return {"ks_margin_q": ks, "lambda2": ks, "c_in_ks": ks, "c_out_ks": ks, "inverse_eps_ks_q": ks, "c_l": ks,
            "c_d": _stat("transitions_status", "planted-measure population dynamics (N -> oo, Poisson degrees), "
                         "finite population and sweeps"),
            "c_c": _stat("transitions_status", "zero of the Bethe Delta Phi of the planted branch (population "
                         "dynamics), finite population and sweeps"),
            "q_bp_overlap": bp, "q_bp_overlap_nishimori": bp, "q_bp_delta_phi": bp,
            "em_c_in": _stat("em_status", "EM (BP-learning) from a spectral start"),
            "em_c_out": _stat("em_status", "EM (BP-learning) from a spectral start"),
            "em_n_max": _stat("em_status", "EM (BP-learning) from a spectral start")}


META = ("q", "c_in", "c_out", "eps", "ks_detectable_q", "pop", "sweeps", "transitions_status", "transitions_note",
        "c_d_halfwidth99", "c_c_halfwidth99", "init", "damping", "q_bp_converged_fraction", "q_bp_phi_label",
        "q_bp_status", "q_bp_status_note", "q_bp_overlap_halfwidth99", "q_bp_overlap_nishimori_halfwidth99",
        "q_bp_delta_phi_halfwidth99", "learn", "em_status", "em_status_note", "em_c_in_halfwidth99",
        "em_c_out_halfwidth99", "em_n_max_halfwidth99")

def refs():
    line = {"q": 5, "c_in": 0}
    stat = lambda key: RefCheck("sbm", f"{key}_q5_cin0", lambda v: v[key], statistical=True, params=line,
                                route="cavity", slow=True, error=lambda e: e[key], runs=len(_SEEDS))
    return [RefCheck("sbm", "inverse_eps_c_q4_c16", lambda v: v["inverse_eps_ks_q"], params={"q": 4, "c": 16},
                     route="cavity"),
            RefCheck("sbm", "c_l_q5_cin0", lambda v: v["c_l"], params=line, route="cavity", slow=True),
            stat("c_d"), stat("c_c")] + [
        # K7: the planted q-coloring is this model at c_in = 0 (KZ Sec. 13.1): ZK's Table II c_d, c_c at q = 5
        # (factor.json; the q = 4 keys were removed in K7's fix round, no route resolves them).  Two keys per quantity
        # on purpose: KZ's print (sbm.json c_d_q5_cin0, c_c_q5_cin0) and ZK's (factor.json), each with its own tol
        RefCheck("factor", f"coloring_{key}_q{q}", (lambda k: lambda v: v[k])(key), statistical=True,
                 params={"q": q, "c_in": 0}, route="cavity", slow=True, error=(lambda k: lambda e: e[k])(key),
                 runs=len(_SEEDS)) for key in ("c_d", "c_c") for q in (5,)]


__all__ = ["META", "Q_PARAMS", "finality", "refs", "steps", "stray_q"]
