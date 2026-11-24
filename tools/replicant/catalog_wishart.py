"""Catalog entry `spiked_wishart` (route `amp`), K5: the rank-one spiked Wishart (bipartite) model.

Y = (lambda/n) u v^T + W, n x m, W_ij ~ N(0, 1/n), alpha = m/n, unit-variance priors (inference/wishart.py states the
conventions: the notes' (12.87) lambda_notes = lambda^2; LKZ's Delta = 1/lambda^2; BBP's gamma^2 = alpha,
l_1 = 1 + lambda^2).  Route `amp`: the Bayes two-overlap state evolution, the BBP edge, finite-N rectangular Bayes AMP
and SVD.  Run parameters:
  alpha                   lambda_c = alpha^(-1/4) (BBP / LKZ (198)), lambda_c_se (the SE's linear instability from
                          numerical slopes), lambda_c_notes = lambda_c^2, delta_c_lkz = sqrt(alpha), sigma_edge
  alpha, snr              SE from the informative start: m_u_se, m_v_se, mmse_u_se, mmse_v_se; sigma_top (BBP)
  alpha, snr, N           over named seeds (default 0-7): AMP from the informative start E[v | sqrt(init_snr) v + z]
                          (init_snr default 0.3, iters default 20): m_u_amp, m_v_amp; the SE at the same t from the
                          same start (m_u_se_t, m_v_se_t); SVD: sigma_top_svd (99% Student-t bands, statistical)
  prior_u, prior_v        gaussian | rademacher (default gaussian; any unit-variance bayes/channels.py prior name)
"""
import numpy as np

from .amp._stats import student_band
from .catalog import Out, RefCheck, Stability, entry, fstep, has
from .core import Refusal
from .csp import _checks
from .inference import wishart as WS
from .spec import ModelSpec

BBP = ("Baik, Ben Arous & Peche, Ann. Probab. 33, 1643 (2005), arXiv:math/0403022, Eqs. (4)-(5), Thm 1.1 (36), (39) "
       "(complex samples; the real case is their Sec. 1.3 conjecture)")
LKZ = ("Lesieur, Krzakala & Zdeborova, J. Stat. Mech. (2017) 073403, arXiv:1701.00858, Eqs. (195)-(198), (248)")
KZ = ("Krzakala & Zdeborova, Statistical Physics Methods in Optimization and Machine Learning, lecture notes "
      "(June 9 2024), Sec. 12.3.3, Eqs. (12.87)-(12.100) (U* and V* swapped back in (12.97)-(12.100))")
CAVEAT = ("AMP route: the two-overlap Bayes state evolution and the BBP edge are N -> oo predictions; BBP is a "
          "theorem for complex Gaussian samples (the real case is conjectured there), the SE's Bayes optimality is "
          "the replica prediction of LKZ (no theorem for the bipartite model was fetched): physics-level; finite-N "
          "AMP and SVD values are statistical (mean +- stderr over named seeds, 99% Student-t), never certified")
_SEEDS = tuple(range(8))
PARAMS = ("alpha", "snr", "prior_u", "prior_v", "N", "seeds", "iters", "init_snr")


def _check(p):
    if p.get("alpha") is None:
        raise Refusal("precondition:state-point", "spiked_wishart needs alpha (= m/n)")
    if p.get("N") is not None and p.get("snr") is None:
        raise Refusal("precondition:state-point", "finite-N AMP needs snr (and N)")
    if p.get("N") is None:
        bad = [k for k in ("seeds", "iters", "init_snr") if p.get(k) is not None]
        if bad:
            raise Refusal("precondition:known-parameter", f"{', '.join(bad)}: finite-N parameters need N (and snr)")


def _pri(p):
    return p.get("prior_u", "gaussian"), p.get("prior_v", "gaussian")


def _threshold(p):
    _check(p)
    a = WS._alpha(p["alpha"])
    pu, pv = _pri(p)
    lam = WS.bbp_threshold(a)
    return Out({"alpha": a, "prior_u": WS.prior(pu).name, "prior_v": WS.prior(pv).name, "lambda_c": lam,
                "lambda_c_se": WS.wishart_se_threshold(a, pu, pv), "lambda_c_notes": WS.notes_from_snr(lam),
                "delta_c_lkz": WS.lkz_delta_from_snr(lam), "sigma_edge": 1 + a ** 0.5},
               "lambda_c = alpha^(-1/4); lambda_c_se from alpha lambda^4 m_U'(0) m_V'(0) = 1")


def _se(p):
    se = WS.wishart_se(p["snr"], p["alpha"], *_pri(p))
    vals = {"snr": se.snr, "se_converged": se.converged, "sigma_top": WS.bbp_top_singular(se.snr, se.alpha)}
    if se.converged:
        vals.update({"m_u_se": se.m_u, "m_v_se": se.m_v, "mmse_u_se": 1 - se.m_u, "mmse_v_se": 1 - se.m_v})
    return Out(vals, se.note, notes=() if se.converged else (f"UNDETERMINED: {se.note}",))


def _band(vals, errs, key, xs):
    b = student_band(xs)
    vals[key], vals[f"{key}_halfwidth99"], errs[key] = b.mean, b.half_width, b.stderr


def _finite(p):
    """AMP and SVD over the seeds.  A run whose iterates go non-finite makes m_u_amp, m_v_amp undetermined (absent,
    amp_status `undetermined`); finite runs that have not converged in t keep their bands (they are AMP at that t, the
    SE's statement) but amp_status is `not-converged` and the finality says so."""
    seeds, iters = _checks.seeds(p.get("seeds", _SEEDS)), p.get("iters", 20)
    s0 = p.get("init_snr", "0.3")
    pu, pv = _pri(p)
    mu, mv, tops, fin, conv = [], [], [], [], []
    for s in seeds:
        Y, u, v = WS.wishart_matrix(p["N"], p["alpha"], p["snr"], pu, pv, s)
        r = WS.wishart_amp(Y, pu, pv, snr=p["snr"], iters=iters, v_init=WS.informative_start(v, pv, s0, s))
        fin.append(r.finite)
        conv.append(r.converged)
        if r.finite:
            mu.append(float(r.u_hats[-1] @ u) / u.size)
            mv.append(float(r.v_hats[-1] @ v) / v.size)
        tops.append(float(np.linalg.svd(Y, compute_uv=False)[0]))
    se = WS.wishart_se(p["snr"], p["alpha"], pu, pv, iters=iters, init_snr=s0)
    n = len(seeds)
    status = "undetermined" if not all(fin) else ("converged" if all(conv) else "not-converged")
    vals, errs = {"N": p["N"], "seeds": list(seeds), "iters": iters, "init_snr": s0, "amp_status": status,
                  "amp_finite_fraction": sum(fin) / n, "amp_converged_fraction": sum(conv) / n,
                  "m_u_se_t": se.trajectory[-1][0], "m_v_se_t": se.trajectory[-1][1]}, {}
    bands = (("sigma_top_svd", tops),) if status == "undetermined" else (("m_u_amp", mu), ("m_v_amp", mv),
                                                                        ("sigma_top_svd", tops))
    for key, xs in bands:
        _band(vals, errs, key, xs)
    amp = ("AMP UNDETERMINED: a run went non-finite" if status == "undetermined" else
           f"AMP m_u {vals['m_u_amp']:.5g} +- {vals['m_u_amp_halfwidth99']:.2g} (SE {vals['m_u_se_t']:.5g}; {status})")
    return Out(vals, f"over seeds {list(seeds)} at t = {iters}: {amp}; top singular value {vals['sigma_top_svd']:.5g} "
               f"+- {vals['sigma_top_svd_halfwidth99']:.2g}", errors=errs, statistical=tuple(errs),
               notes=(f"UNDETERMINED: AMP finite in {sum(fin)}/{n} runs",) if status == "undetermined" else ())


def _amp_label(v):
    note = {"converged": "", "not-converged": f"; NOT converged in t in {1 - v.get('amp_converged_fraction', 0):.0%} "
            "of runs (the values are AMP at the given t: compare m_*_se_t)"}.get(v.get("amp_status"), "")
    return Stability("not-computed", "amp-finite-N", KZ, "statistical: mean +- stderr over named seeds (99% "
                     "Student-t half-width in <key>_halfwidth99), never certified: NOT final" + note)


def spiked_wishart():
    steps = [fstep("wishart-threshold", f"{BBP}; {LKZ}", _threshold),
             fstep("wishart-state-evolution", f"{KZ}; {LKZ}", _se, when=has("snr"), skip="no snr given"),
             fstep("wishart-finite-n", f"{KZ} (rectangular AMP); {BBP} (SVD) (finite N: statistical)", _finite,
                   when=has("snr", "N"), skip="needs snr and N")]
    refs = [RefCheck("wishart", "lambda_c_alpha4", lambda v: v["lambda_c_se"],
                     params={"alpha": 4, "prior_v": "rademacher"}, route="amp")]
    thr = Stability("not-computed", "amp-se", f"{BBP}; {LKZ}",
                     "BBP prove l_c = 1 + 1/gamma for complex samples and conjecture it for real ones (Sec. 1.3); "
                     "LKZ's Delta_c is the SE instability; lambda_c_se reproduces it numerically: NOT final")
    phys = Stability("not-computed", "amp-se", f"{KZ}; {LKZ}",
                     "the Bayes two-overlap SE (replica prediction for the bipartite model; no theorem fetched): "
                     "physics-level, NOT final")
    stat = Stability("not-computed", "amp-finite-N", KZ, "statistical: mean +- stderr over named seeds (99% "
                     "Student-t half-width in <key>_halfwidth99), never certified: NOT final")
    fin = {**{k: thr for k in ("lambda_c", "lambda_c_se", "lambda_c_notes", "delta_c_lkz", "sigma_edge",
                               "sigma_top")},
           **{k: phys for k in ("m_u_se", "m_v_se", "mmse_u_se", "mmse_v_se", "m_u_se_t", "m_v_se_t")},
           "m_u_amp": _amp_label, "m_v_amp": _amp_label, "sigma_top_svd": stat}
    meta = ("alpha", "prior_u", "prior_v", "snr", "se_converged", "N", "seeds", "iters", "init_snr", "amp_status",
            "amp_finite_fraction", "amp_converged_fraction",
            "m_u_amp_halfwidth99", "m_v_amp_halfwidth99", "sigma_top_svd_halfwidth99")
    spec = ModelSpec("spiked_wishart", "real", "gaussian", constraint="rank-one bipartite spike, Bayes-optimal",
                     prior="gaussian")
    return entry("spiked_wishart", spec, KZ, {"amp": steps}, refs, {"amp": CAVEAT},
                 "rank-one spiked Wishart: two-overlap Bayes SE, BBP edge lambda_c = alpha^(-1/4), rectangular AMP",
                 default_route="amp", finality=fin, meta=meta, parameters={"amp": PARAMS})


__all__ = ["spiked_wishart"]
