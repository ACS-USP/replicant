"""Catalog route `amp` (E3b) for the `sk` and `teacher_student` entries: amp/ as a route.

  sk_amp_route() -> AmpRoute     state evolution of SK AMP; finite-N AMP when N is given.  With
                                 ansatz=1rsb: approximate survey propagation (ASP) at the given m,
                                 its SE fixed point (asp-state-evolution) and, when N is given, finite-N
                                 ASP (asp-finite-n, amp/asp_finite.py: bands q0_amp, q1_amp against the
                                 SE at the same t, q0_se_t, q1_se_t; seeds 0-5, iters 20, start
                                 (0.5, 0.95) by default)
  ts_amp_route() -> AmpRoute     state evolution of Bayes-optimal GAMP (gaussian prior);
                                 finite-N GAMP when N is given
An AmpRoute carries the Steps, the honest caveat, the declared run parameters, the
finality declaration of every value key and the meta keys; catalog_entries.py adds it
to its entry.  The route calls amp/ only (no family, replica or cavity code).

Finality.  A state-evolution value is `exact-by-literature` only where the cited paper
proves SE for the setting: SK AMP with the Lipschitz tanh on a Gaussian (GOE-type) J,
Bayati & Montanari 2011 Thm 4 (and Bolthausen 2014 for this iteration).  GAMP's SE for
the perceptron (a GLM) is stated as Claim 1 by Barbier et al. 2019 (from Rangan 2011), so
its values are `not-computed`.  Finite-N values are statistical (mean +- stderr over the
named seeds; 99% Student-t half-width in the meta key <key>_halfwidth99), `not-computed`,
never final.  Defaults (echoed in the values): seeds 0-5, law gaussian; iters 100 (SK),
30 (GAMP).
"""
from dataclasses import dataclass

import mpmath as mp

from .catalog import Out, Stability, fstep, has
from .core import Refusal
from .csp import _checks

BM11 = ("Bayati & Montanari, IEEE Trans. Inf. Theory 57, 764 (2011), arXiv:1001.3448, Thm 4; "
        "Bolthausen, Commun. Math. Phys. 325, 333 (2014), arXiv:1201.2891")
BKMMZ = ("Barbier, Krzakala, Macris, Miolane & Zdeborova, PNAS 116, 5451 (2019), "
         "arXiv:1708.03395, Claim 1 and eq. (10); Rangan, IEEE ISIT 2011, arXiv:1010.5141")
SEEDS = (0, 1, 2, 3, 4, 5)
SK_CAVEAT = (
    "AMP route: state evolution (SE) is a deterministic N -> oo prediction for SK AMP "
    "(amp/sk.py), a theorem for Gaussian J at each fixed t (Bayati-Montanari 2011 Thm 4; "
    "Bolthausen 2014), not a statement about the Gibbs measure; finite-N AMP values are "
    "statistical (mean +- stderr over named seeds, 99% Student-t half-width), never "
    "certified; +-J matrices: numerical evidence only")
TS_CAVEAT = (
    "AMP route: Bayes-optimal GAMP (amp/gamp.py) with an i.i.d. N(0, 1) prior and teacher "
    "(the spherical model's overlap equation is the same); its state evolution is stated as "
    "a claim for GLMs (Barbier et al. 2019, Claim 1, from Rangan 2011), not a theorem here; "
    "Bayes-optimal only on the Nishimori line; finite-N values are statistical (mean +- "
    "stderr over named seeds, 99% Student-t half-width), never certified")


@dataclass(frozen=True)
class AmpRoute:
    steps: list
    caveat: str
    params: tuple
    finality: dict
    meta: tuple


def _need(p, key, model, finite):
    """The state point is required; the finite-N knobs (`finite`) are read only with N, so
    one given without N is refused rather than echoed and ignored."""
    if p.get(key) is None:
        raise Refusal("precondition:state-point", f"the {model} amp route needs {key}")
    stray = [k for k in finite if k in p] if p.get("N") is None else []
    if stray:
        raise Refusal("precondition:known-parameter",
                      f"{', '.join(stray)}: finite-N parameters need N (the {model} amp route "
                      "without N runs state evolution only, which reads "
                      f"{key}{', h' if model == 'sk' else ', learning'})")


def _stat(key, note):
    return Stability("not-computed", "amp-finite-N", BM11 if key == "q_amp" else BKMMZ,
                     "statistical: finite-N mean +- stderr over the named seeds (99% Student-t "
                     f"half-width in {key}_halfwidth99), never certified; " + note)


def _band_values(key, band):
    return {key: band.mean, f"{key}_halfwidth99": band.half_width}


# ---- SK ------------------------------------------------------------------------------
def _sk_se(p):
    from .amp import sk_state_evolution
    from .amp.sk import LAWS
    _need(p, "beta", "sk", ("seeds", "iters", "law"))
    law = p.get("law", "gaussian")
    if law not in LAWS:
        raise Refusal("precondition:known-law", f"law={law!r}; known: {', '.join(LAWS)}")
    se = sk_state_evolution(p["beta"], p.get("h", 0))
    return Out({"q_se": se.q, "law": law, "se_converged": se.converged,
                "se_fixed_points": [mp.nstr(f.q, 12) for f in se.fixed_points],
                "se_amp_converges": se.amp_converges}, "SE fixed point from Q_0 = 1: " + se.note)


def _sk_finite(p):
    from .amp import random_init_q, sk_amp_ensemble, sk_state_evolution
    seeds, iters, law = _checks.seeds(p.get("seeds", SEEDS)), p.get("iters", 100), p.get("law",
                                                                                      "gaussian")
    beta, h = p["beta"], p.get("h", 0)
    ens = sk_amp_ensemble(p["N"], float(mp.mpf(beta)), float(mp.mpf(h)), seeds=seeds,
                          iters=iters, law=law)
    se_t = sk_state_evolution(beta, h, q0=random_init_q(beta, h), iters=iters).trajectory[-1]
    vals = {**_band_values("q_amp", ens.band), "N": p["N"], "seeds": list(seeds),
            "iters": iters, "q_se_t": se_t, "q_amp_runs": ens.values,
            "amp_converged": ens.converged}
    return Out(vals, f"AMP over seeds {list(seeds)}: q = {ens.band.mean:.6g} +- "
               f"{ens.band.half_width:.3g} (99% Student-t, df = {ens.band.df}); SE at t = "
               f"{iters} from the same initial law: {se_t:.6g}",
               errors={"q_amp": ens.band.stderr}, statistical=("q_amp",), notes=(ens.note,))


_PM_ONE = ("+-J matrix: SE is not proved for it here (Bayati-Lelarge-Montanari 2015 cover "
           "polynomial nonlinearities); the value is the Gaussian-J SE, numerical evidence only "
           "for this law")


def _sk_q_se(v):
    if not v.get("se_converged"):
        return Stability("not-computed", "amp-se", BM11, "the SE trajectory did not converge")
    if v.get("law") != "gaussian":
        return Stability("not-computed", "amp-se", BM11, _PM_ONE)
    return Stability("exact-by-literature", "amp-se", BM11,
                     "what is proved: for Gaussian (GOE-type) J and the Lipschitz tanh, lim_N of "
                     "AMP's Q_t = <m^t, m^t> equals the SE Q_t at each fixed t (Thm 4 is stated "
                     "with diagonal variance 2/N; amp/sk.py uses a zero diagonal); q_se = lim_t "
                     "of SE from Q_0 = 1 (N -> oo first), the same limit as from AMP's random "
                     "start (Q_0 > 0) when one fixed point attracts; it is not a statement about "
                     "the Gibbs measure (at h = 0, beta > 1 the RS saddle it matches is "
                     "AT-unstable)")


def _sk_q_se_t(v):
    if v.get("law") != "gaussian":
        return Stability("not-computed", "amp-se", BM11, _PM_ONE)
    return Stability("exact-by-literature", "amp-se", BM11,
                     "SE Q_t at the finite t of the AMP runs, from the same initial law (proved "
                     "for each fixed t as N -> oo, Gaussian J)")


def _sk_ansatz(p):
    """ansatz rs (default: AMP / TAP, its SE) or 1rsb (ASP, its SE: amp/asp.py, K5); m only with 1rsb."""
    a = p.get("ansatz", "rs")
    if a not in ("rs", "1rsb"):
        raise Refusal("precondition:known-ansatz", f"sk amp: ansatz = {a!r}: rs (AMP) or 1rsb (approximate survey "
                      "propagation)")
    if a == "rs" and p.get("m") is not None:
        raise Refusal("precondition:known-parameter", "m (the Parisi parameter) applies to ansatz=1rsb only")
    if a == "rs" and p.get("start") is not None:
        raise Refusal("precondition:known-parameter", "start (the ASP initial pair) applies to ansatz=1rsb only")
    if a == "1rsb":
        if p.get("law") is not None:
            raise Refusal("precondition:known-parameter", "law: ansatz=1rsb is Gaussian-J only (the ASP finite-N run "
                          "samples Gaussian matrices)")
        stray = [k for k in ("seeds", "iters", "start") if p.get(k) is not None] if p.get("N") is None else []
        if stray:
            raise Refusal("precondition:known-parameter", f"{', '.join(stray)}: finite-N parameters need N (the sk amp "
                          "route with ansatz=1rsb and no N runs the ASP state evolution only, which reads beta, m, h)")
        if p.get("beta") is None or p.get("m") is None:
            raise Refusal("precondition:state-point", "the sk amp route with ansatz=1rsb needs beta and m")
    return Out({}, f"ansatz {a}")


def _sk_asp(p):
    from .amp.asp import asp_fixed_point
    r = asp_fixed_point(p["beta"], p["m"], p.get("h", 0))
    vals = {"m": r.m, "asp_converged": r.converged}
    if r.converged:
        vals.update({"q0_asp": r.q0, "q1_asp": r.q1})
    return Out(vals, r.note, notes=() if r.converged else (f"UNDETERMINED: {r.note}",))


_ASP_CITE = ("Krzakala & Zdeborova, lecture notes (June 9 2024), Sec. 12.2.5, Eqs. (12.63)-(12.65); Antenucci, "
             "Krzakala, Urbani & Zdeborova, arXiv:1807.01296, Sec. IV B")


def _sk_asp_finite(p):
    """Finite-N ASP (amp/asp_finite.py) against the ASP SE at the same t (amp/asp.py).  asp_trajectory runs first, so
    a malformed start (the CLI hands decimal strings) is refused before any matrix is sampled."""
    from .amp.asp import asp_trajectory
    from .amp.asp_finite import asp_amp_ensemble
    seeds, iters = _checks.seeds(p.get("seeds", SEEDS)), p.get("iters", 20)
    beta, m, h = float(mp.mpf(p["beta"])), float(mp.mpf(p["m"])), float(mp.mpf(p.get("h", 0)))
    tr = asp_trajectory(beta, m, h, start=p.get("start"), iters=iters)
    start = tr[0]
    ens = asp_amp_ensemble(p["N"], beta, m, h, seeds=seeds, iters=iters, start=start)
    (q0_t, q1_t), b0, b1 = tr[iters], ens.bands["q0"], ens.bands["q1"]
    frac = sum(bool(c) for c in ens.converged) / len(ens.converged)
    vals = {**_band_values("q0_amp", b0), **_band_values("q1_amp", b1), "q0_se_t": q0_t, "q1_se_t": q1_t,
            "N": p["N"], "seeds": list(seeds), "iters": iters, "start": list(start), "asp_converged_fraction": frac}
    return Out(vals, f"ASP over seeds {list(seeds)}: q0 = {b0.mean:.6g} +- {b0.half_width:.3g}, q1 = {b1.mean:.6g} +- "
               f"{b1.half_width:.3g} (99% Student-t, df = {b0.df}); ASP SE at t = {iters} from start {list(start)}: "
               f"({q0_t:.6g}, {q1_t:.6g})",
               errors={"q0_amp": b0.stderr, "q1_amp": b1.stderr}, statistical=("q0_amp", "q1_amp"),
               notes=(ens.note,) if frac == 1 else (ens.note, f"asp_converged_fraction = {frac:.3g}: some seeds did not "
                                                              "reach the tolerance within iters (labelled, not refused)"))


def sk_amp_route():
    rs = lambda p: p.get("ansatz", "rs") == "rs"
    steps = [fstep("amp-ansatz", "AMP (rs) or approximate survey propagation (1rsb)", _sk_ansatz),
             fstep("amp-state-evolution", BM11, _sk_se, when=rs, skip="ansatz=1rsb (ASP)"),
             fstep("amp-finite-n", BM11 + " (finite-N iterates: statistical)", _sk_finite,
                   when=lambda p: rs(p) and has("N")(p), skip="no N given (finite-N AMP runs only with N)"),
             fstep("asp-state-evolution", _ASP_CITE, _sk_asp, when=lambda p: not rs(p), skip="ansatz rs"),
             fstep("asp-finite-n", _ASP_CITE + "; " + BM11 + " (finite-N iterates: statistical)", _sk_asp_finite,
                   when=lambda p: not rs(p) and has("N")(p),
                   skip="no N given (finite-N ASP runs only with N), or ansatz rs")]
    asp = Stability("not-computed", "1rsb-asp-se", _ASP_CITE,
                    "fixed point of ASP's state evolution at the given Parisi m: the 1RSB q0, q1 equations (the m "
                    "equation is not imposed); 1RSB is not the exact SK solution: physics-level, NOT final")
    se_t = Stability("exact-by-literature", "asp-se", _ASP_CITE + "; " + BM11,
                     "ASP state evolution at the finite t of the AMP runs, from the same start: SE at a fixed t for a "
                     "Lipschitz AMP with deterministic q0, q1 (Bayati-Montanari 2011, Gaussian J).  The finite-N run "
                     "uses the empirical q0^t, q1^t as plug-ins, not literally the theorem's deterministic-q "
                     "algorithm; BM covers the deterministic-q f_t (Lipschitz)")
    fin = {"q_se": _sk_q_se, "q_se_t": _sk_q_se_t,
           "q_amp": _stat("q_amp", "AMP on sampled matrices (amp/sk.py)"), "q0_asp": asp, "q1_asp": asp,
           "q0_se_t": se_t, "q1_se_t": se_t,
           "q0_amp": _stat("q0_amp", "ASP on sampled matrices (amp/asp_finite.py)"),
           "q1_amp": _stat("q1_amp", "ASP on sampled matrices (amp/asp_finite.py)")}
    meta = ("se_converged", "se_fixed_points", "se_amp_converges", "N", "seeds", "iters",
            "law", "q_amp_halfwidth99", "q_amp_runs", "amp_converged", "m", "asp_converged", "start",
            "q0_amp_halfwidth99", "q1_amp_halfwidth99", "asp_converged_fraction")
    return AmpRoute(steps, SK_CAVEAT, ("beta", "h", "N", "seeds", "iters", "law", "ansatz", "m", "start"), fin, meta)


# ---- teacher-student -------------------------------------------------------------------
def _learning(p):
    learning = p.get("learning", "bayes")
    if learning not in ("bayes", "gibbs"):
        raise Refusal("precondition:known-learning",
                      f"learning = {learning!r}: use 'bayes' (posterior mean, GAMP's estimate) "
                      "or 'gibbs' (a posterior sample: overlap q by the Nishimori identity)")
    return learning


def _ts_se(p):
    from .amp import perceptron_state_evolution
    _need(p, "alpha", "teacher_student", ("seeds", "iters"))
    learning = _learning(p)
    se = perceptron_state_evolution(p["alpha"], "gaussian")
    eps = se.eps_bayes if learning == "bayes" else se.eps_gibbs
    return Out({"q_se": se.overlap, "mse_se": se.mse, "eps_se": eps, "learning": learning,
                "prior": "gaussian", "se_converged": se.converged},
               f"GAMP SE from q = 0 (gaussian prior): {se.note}")


def _ts_finite(p):
    from .amp import gamp_ensemble, perceptron_state_evolution
    seeds, iters = _checks.seeds(p.get("seeds", SEEDS)), p.get("iters", 30)
    ens = gamp_ensemble(p["N"], float(mp.mpf(p["alpha"])), "gaussian", seeds=seeds,
                        iters=iters)
    se_t = perceptron_state_evolution(p["alpha"], "gaussian", iters=iters).trajectory[-1]
    vals = {**_band_values("q_gamp", ens.band), **_band_values("eps_gamp", ens.eps_band),
            "N": p["N"], "seeds": list(seeds), "iters": iters, "q_se_t": se_t,
            "gamp_converged": ens.converged}
    return Out(vals, f"GAMP over seeds {list(seeds)}: q = {ens.band.mean:.6g} +- "
               f"{ens.band.half_width:.3g} (99% Student-t, df = {ens.band.df}); SE at t = "
               f"{iters}: {se_t:.6g}; eps_gamp is the plug-in (Bayes) error of the GAMP "
               "estimate whatever `learning`",
               errors={"q_gamp": ens.band.stderr, "eps_gamp": ens.eps_band.stderr},
               statistical=("q_gamp", "eps_gamp"), notes=(ens.note,))


def ts_amp_route():
    claim = Stability("not-computed", "gamp-se", BKMMZ,
                      "GAMP state evolution for GLMs is stated as Claim 1 by Barbier et al. "
                      "(from Rangan 2011, steps of the proof missing; Javanmard-Montanari 2013 "
                      "state SE for general AMP covering GAMP, not counted as a proof there): "
                      "no theorem is applied by this route.  The replica route's eps is "
                      "exact-by-literature (Barbier et al. Thm 2) and agrees with eps_se "
                      "(tests/test_gamp.py); the agreement checks an identity: with the "
                      "Gaussian prior the SE equation is Gyorgyi's spherical Gibbs equation")
    steps = [fstep("gamp-state-evolution", BKMMZ, _ts_se),
             fstep("gamp-finite-n", BKMMZ + " (finite-N iterates: statistical)", _ts_finite,
                   when=has("N"), skip="no N given (finite-N GAMP runs only with N)")]
    fin = {"q_se": claim, "mse_se": claim, "eps_se": claim, "q_se_t": claim,
           "q_gamp": _stat("q_gamp", "GAMP on sampled instances (amp/gamp.py)"),
           "eps_gamp": _stat("eps_gamp", "plug-in error of the GAMP estimate")}
    meta = ("learning", "prior", "se_converged", "N", "seeds", "iters", "q_gamp_halfwidth99",
            "eps_gamp_halfwidth99", "gamp_converged")
    return AmpRoute(steps, TS_CAVEAT, ("alpha", "learning", "N", "seeds", "iters"), fin, meta)


__all__ = ["AmpRoute", "sk_amp_route", "ts_amp_route"]
