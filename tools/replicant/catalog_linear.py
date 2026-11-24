"""Catalog entries `lasso` (routes `amp`, `replica`) and `compressed_sensing` (route `amp`), K6 (KZ notes Ch. 12, 16,
18).

Convention (the notes'; linear/lasso.py): A is M x N with i.i.d. N(0, 1/N) entries, alpha = M/N, y = A x0 +
sqrt(Delta) xi (`sigma2` = Delta), x0 i.i.d. Gauss-Bernoulli (1 - rho) delta_0 + rho N(0, 1), LASSO
(1/2)|y - A x|^2 + lam |x|_1.  Bayati-Montanari's delta = alpha, sigma_BM^2 = Delta/alpha, lam_BM = lam/alpha.

`lasso` run parameters (alpha, rho, lam, sigma2 required):
  replica   the state-evolution / RS fixed point solved directly (BM's calibration lam(kappa) inverted by root
            finding): risk, tau2, threshold, kappa, b (the fraction of non-zeros)
  amp       KZ Alg. 4's state evolution iterated from its start: risk_se, threshold_se, b_se, se_converged; with N
            (seeds default 0-3, iters default 2000): AMP and coordinate descent on sampled instances: risk_exact,
            risk_amp, threshold_amp (99% Student-t bands), amp_lasso_gap (max |x_amp - x_cd|), kkt_max,
            amp_converged_fraction
`compressed_sensing` (noiseless, Gauss-Bernoulli; route amp), rho required:
            alpha_l1 (DMM Eq. (5)), alpha_l1_se (soft-threshold SE local stability), alpha_bayes (the Bayes SE
            spinodal, only when the two SE starts confirm it; bayes_status otherwise), alpha_it = rho; with alpha
            and N (seeds default 0-9): l1_success_fraction (LP) and gamp_mse (Bayes GAMP, 99% band), statistical
"""
from .amp._stats import student_band
from .bayes.channels import Prior
from .catalog import Out, RefCheck, Stability, entry, fstep, has
from .core import Refusal
from .csp import _checks
from .linear import cs as CS
from .linear import gamp as G
from .linear import lasso as L
from .spec import ModelSpec

KZ = ("Krzakala & Zdeborova, Statistical Physics Methods in Optimization and Machine Learning, lecture notes (June 9 "
      "2024)")
BM = ("Bayati & Montanari, IEEE Trans. Inf. Theory 58, 1997 (2012), arXiv:1008.2581, Thm 1.5, Cor. 1.6, Eqs. (1.7), "
      "(1.15)")
DMM = "Donoho, Maleki & Montanari, PNAS 106, 18914 (2009), arXiv:0907.3574, Eq. (5)"
KMSSZ = ("Krzakala, Mezard, Sausset, Sun & Zdeborova, PRX 2, 021005 (2012), arXiv:1109.4424, Fig. 3 caption and main "
         "text")
HMRT = "Hastie, Montanari, Rosset & Tibshirani, Ann. Statist. 50, 949 (2022), arXiv:1903.08560, Prop. 2"
CAVEAT_R = ("replica route: the LASSO's state-evolution fixed point (equal to the RS saddle point of the replica "
            "derivations BM cite, KWT09 / RFG09), solved directly; BM prove its risk for Gaussian A, Delta > 0, "
            "lam > 0 (exact-by-literature there); lam = 0, alpha > 1 is least squares (HMRT Prop. 2)")
CAVEAT_A = ("amp route: the state evolution of KZ Alg. 4 iterated (BM Thm 1.1 / 1.8: AMP's SE and its convergence to "
            "the LASSO, Gaussian A); finite-N AMP and coordinate-descent values are statistical (named seeds, 99% "
            "Student-t), never certified")
CAVEAT_CS = ("amp route: alpha_l1 is DMM's formula (their Finding 1 equates it with the Donoho-Tanner l1 transition, "
             "not fetched), alpha_bayes the Bayes SE spinodal confirmed by two SE starts, alpha_it = rho KMSSZ's "
             "replica boundary: physics-level; finite-N LP and GAMP values are statistical, never final")
_SEEDS = tuple(range(4))
LASSO_PARAMS = ("alpha", "rho", "lam", "sigma2")
CS_PARAMS = ("rho", "alpha", "N", "seeds")


def _need(p, keys, name):
    miss = [k for k in keys if p.get(k) is None]
    if miss:
        raise Refusal("precondition:state-point", f"{name} needs {', '.join(miss)}")


def _lasso_state(p):
    _need(p, LASSO_PARAMS, "lasso")
    if p.get("N") is None and any(p.get(k) is not None for k in ("seeds", "iters")):
        raise Refusal("precondition:known-parameter", "seeds and iters are finite-N parameters: need N")
    return Out({k: p[k] for k in LASSO_PARAMS}, "Gauss-Bernoulli signal, Gaussian A (variance 1/N), AWGN Delta")


def _args(p):
    return p["alpha"], p["rho"], p["lam"], p["sigma2"]


def _fixed_point(p):
    se = L.lasso_se(*_args(p))
    return Out({"risk": se.risk, "tau2": se.tau2, "threshold": se.threshold, "kappa": se.kappa, "b": se.b},
               se.note)


def _se_iterate(p):
    se = L.lasso_se_iterate(*_args(p))
    vals = {"se_converged": se.converged}
    if se.converged:
        vals.update({"risk_se": se.risk, "threshold_se": se.threshold, "b_se": se.b})
    return Out(vals, se.note, notes=() if se.converged else (f"UNDETERMINED: {se.note}",))


def _band(vals, errs, key, xs):
    b = student_band(xs)
    vals[key], vals[f"{key}_halfwidth99"], errs[key] = b.mean, b.half_width, b.stderr


def _lasso_finite(p):
    seeds, iters = _checks.seeds(p.get("seeds", _SEEDS)), L._iters(p.get("iters", 2000))
    a, rho, lam, s2 = _args(p)
    ch, pr = G.Channel("awgn", Delta=s2), Prior("gauss_bernoulli", rho=str(rho))
    rex, ramp, th, gaps, kkts, conv, exact_ok = [], [], [], [], [], [], []
    for s in seeds:
        A, y, x0 = G.linear_instance(p["N"], a, ch, pr, s)
        amp, ex = L.lasso_amp(A, y, lam, iters=iters), L.lasso_exact(A, y, lam)
        rex.append(float((ex.x - x0) @ (ex.x - x0)) / x0.size)
        kkts.append(ex.kkt_residual)
        exact_ok.append(ex.converged)
        conv.append(amp.converged)
        if amp.converged:
            ramp.append(float((amp.x - x0) @ (amp.x - x0)) / x0.size)
            th.append(amp.threshold)
            gaps.append(float(abs(amp.x - ex.x).max()))
    vals, errs = {"N": p["N"], "seeds": list(seeds), "iters": iters, "kkt_max": max(kkts),
                  "amp_converged_fraction": sum(conv) / len(seeds)}, {}
    notes = ()
    if all(exact_ok):
        _band(vals, errs, "risk_exact", rex)
    else:
        notes += (f"UNDETERMINED: coordinate descent reached its KKT tolerance in {sum(exact_ok)}/{len(seeds)} runs",)
    if len(ramp) >= 2 and all(exact_ok):
        _band(vals, errs, "risk_amp", ramp)
        _band(vals, errs, "threshold_amp", th)
        vals["amp_lasso_gap"] = max(gaps)
    elif len(ramp) < 2:
        notes += (f"UNDETERMINED: AMP converged in {len(ramp)}/{len(seeds)} runs",)
    head = (f"LASSO risk {vals['risk_exact']:.5g} +- {vals['risk_exact_halfwidth99']:.2g}" if "risk_exact" in vals
            else "LASSO risk undetermined")
    return Out(vals, f"N = {p['N']}, seeds {list(seeds)}: {head} (coordinate descent, KKT <= {max(kkts):.2g})",
               errors=errs, statistical=tuple(errs), notes=notes)


def _cs_state(p):
    _need(p, ("rho",), "compressed_sensing")
    if (p.get("N") is None) != (p.get("alpha") is None) or (p.get("N") is None and p.get("seeds") is not None):
        raise Refusal("precondition:known-parameter", "alpha, N (and seeds) go together: the finite-N runs")
    return Out({"rho": p["rho"]}, "noiseless y = A x0, Gauss-Bernoulli x0 (unit-variance non-zeros)")


def _cs_boundary(p):
    b = CS.cs_phase_boundary(p["rho"])
    vals = {"alpha_l1": b.alpha_l1, "alpha_l1_se": b.alpha_l1_se, "alpha_it": b.alpha_it,
            "bayes_status": b.bayes_status}
    if b.bayes_status == "spinodal":
        vals["alpha_bayes"] = b.alpha_bayes
    return Out(vals, b.note, notes=(b.note,) if b.bayes_status == "undetermined" else ())


def _cs_finite(p):
    seeds = _checks.seeds(p.get("seeds", tuple(range(10))))
    a, rho, N = p["alpha"], p["rho"], p["N"]
    frac = CS.l1_success_fraction(N, a, rho, seeds)
    ch, pr = G.Channel("awgn", Delta=0), Prior("gauss_bernoulli", rho=str(rho))
    mses, finite = [], []
    for s in seeds:
        A, y, x0 = G.linear_instance(N, a, ch, pr, s)
        r = G.gamp(A, y, ch, pr, iters=200, seed=s, teacher=x0)
        mses.append(r.mse_trace[-1])
        finite.append(r.finite)
    vals = {"alpha": a, "N": N, "seeds": list(seeds), "l1_success_fraction": frac.fraction}
    errs = {"l1_success_fraction": (frac.fraction * (1 - frac.fraction) / frac.trials) ** 0.5}   # binomial stderr
    if all(finite):
        _band(vals, errs, "gamp_mse", mses)
        gm = f"Bayes GAMP MSE {vals['gamp_mse']:.3g} (200 steps)"
    else:
        gm = f"Bayes GAMP UNDETERMINED: non-finite in {len(seeds) - sum(finite)}/{len(seeds)} runs"
    return Out(vals, f"LP success {frac.successes}/{frac.trials}; {gm}", errors=errs,
               statistical=tuple(errs), notes=() if all(finite) else (gm,))


def _bm_label(v):
    if v.get("lam") is not None and float(v["lam"]) == 0:
        return Stability("exact-by-literature", "rs", HMRT, "lam = 0, alpha > 1: least squares; risk Delta/(alpha - 1) "
                         "is HMRT's ridgeless gamma < 1 limit at gamma = 1/alpha")
    return Stability("exact-by-literature", "rs", BM, "the LASSO risk and the calibrated threshold for Gaussian A with "
                     "i.i.d. N(0, 1/M) entries in BM's scaling (delta = alpha, sigma_BM^2 = Delta/alpha, lam_BM = "
                     "lam/alpha), Delta > 0, lam > 0, P(X0 != 0) > 0; the n, N -> oo almost-sure limit")


def _sparsity_label(v):
    return Stability("not-computed", "rs", BM, "the SE's P(|X0 + tau Z| > theta): the predicted fraction of non-zeros "
                     "of x_hat; BM Thm 1.5 covers pseudo-Lipschitz test functions and the indicator 1{x_i != 0} is "
                     "not one, so no fetched theorem covers it: NOT final")


def _stat_label(v):
    return Stability("not-computed", "finite-N", KZ, "statistical: mean +- stderr over named seeds (99% Student-t "
                     "half-width in <key>_halfwidth99), never certified: NOT final")


def lasso():
    state = fstep("lasso-state-point", f"{KZ}, Sec. 12.7.3, Eq. (12.146)", _lasso_state)
    steps = {"replica": [state, fstep("lasso-fixed-point", f"{BM}; {KZ}, Eqs. (18.13)-(18.14)", _fixed_point)],
             "amp": [state, fstep("lasso-state-evolution", f"{KZ}, Alg. 4, (16.90)-(16.95); {BM}, Eq. (1.7)",
                                  _se_iterate),
                     fstep("lasso-finite-n", f"{KZ}, Alg. 4; coordinate descent (finite N: statistical)",
                           _lasso_finite, when=has("N"), skip="no N given")]}
    refs = [RefCheck("ridge", "ridgeless_g0p5_s1_r1", lambda v: v["risk"],
                     params={"alpha": "2", "rho": "0.2", "lam": "0", "sigma2": "1"}, route="replica"),
            RefCheck("ridge", "ridgeless_g0p5_s1_r1", lambda v: v["risk_se"],
                     params={"alpha": "2", "rho": "0.2", "lam": "0", "sigma2": "1"}, route="amp")]
    fin = {**{k: _bm_label for k in ("risk", "tau2", "threshold", "kappa", "risk_se", "threshold_se")},
           "b": _sparsity_label, "b_se": _sparsity_label,
           **{k: _stat_label for k in ("risk_exact", "risk_amp", "threshold_amp", "amp_lasso_gap")}}
    meta = LASSO_PARAMS + ("se_converged", "N", "seeds", "iters", "kkt_max", "amp_converged_fraction",
                           "risk_exact_halfwidth99", "risk_amp_halfwidth99", "threshold_amp_halfwidth99")
    spec = ModelSpec("lasso", "real", "gaussian", constraint="l1-penalised least squares, Gauss-Bernoulli signal")
    return entry("lasso", spec, BM, steps, refs, {"replica": CAVEAT_R, "amp": CAVEAT_A},
                 "the LASSO on Gaussian designs: risk and calibrated threshold (BM), KZ Alg. 4 and an exact solver",
                 default_route="replica", finality=fin, meta=meta,
                 parameters={"replica": LASSO_PARAMS, "amp": LASSO_PARAMS + ("N", "seeds", "iters")})


def compressed_sensing():
    steps = [fstep("cs-state-point", f"{KZ}, Sec. 16.6.2, Fig. 16.6.1", _cs_state),
             fstep("cs-phase-boundaries", f"{DMM}; {KMSSZ}; linear/cs.py", _cs_boundary),
             fstep("cs-finite-n", "basis pursuit by LP (scipy HiGHS) and Bayes GAMP (finite N: statistical)",
                   _cs_finite, when=has("alpha", "N"), skip="needs alpha and N")]
    refs = [RefCheck("linear", "cs_alpha_bp_rho0p4", lambda v: v["alpha_bayes"], params={"rho": "0.4"}, route="amp")]
    l1 = Stability("not-computed", "amp-se", DMM, "DMM Eq. (5) (the soft-threshold SE boundary, here also from the "
                   "SE's "
                   "local stability); its identity with the l1 phase transition is DMM's Finding 1 (numerical) and "
                   "Donoho-Tanner's theorem (not fetched): NOT final")
    bp = Stability("not-computed", "amp-se", KMSSZ, "the Bayes SE spinodal, confirmed by the uninformative and "
                   "informative SE starts (physics-level; no theorem fetched): NOT final")
    it = Stability("not-computed", "rs", KMSSZ, "the replica (information-theoretic) boundary alpha = rho: NOT final")
    fin = {"alpha_l1": l1, "alpha_l1_se": l1, "alpha_bayes": bp, "alpha_it": it, "gamp_mse": _stat_label,
           "l1_success_fraction": _stat_label}
    meta = ("rho", "bayes_status", "alpha", "N", "seeds", "gamp_mse_halfwidth99")
    spec = ModelSpec("compressed_sensing", "real", "gaussian", constraint="noiseless linear measurements, "
                     "Gauss-Bernoulli signal")
    return entry("compressed_sensing", spec, KMSSZ, {"amp": steps}, refs, {"amp": CAVEAT_CS},
                 "noiseless compressed sensing (Gauss-Bernoulli): l1 (DMM), Bayes spinodal and IT boundaries",
                 default_route="amp", finality=fin, meta=meta, parameters={"amp": CS_PARAMS})


def linear_entries():
    return [lasso(), compressed_sensing()]


__all__ = ["compressed_sensing", "lasso", "linear_entries"]
