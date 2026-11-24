"""The diluted-Rademacher path of the `spiked_wigner` entry (K5, KZ Ex. 8.2), imported by catalog_spiked_sbm.

Chosen by prior=diluted_rademacher (inference/spiked_diluted.py states the convention: the unit-variance prior x in
{0, +-1/sqrt(rho)}, so LKZ's and AFUZ's Delta / rho^2 is 1 / lambda^2).  Run parameters:
  (no rho)       the tricritical point (LKZ Fig. 7): rho_tri, snr_tri, delta_tri_over_rho2, delta_tri
  rho            lambda_c (instability of q = 0, = LKZ's Delta_c = rho^2) and, when the transition is first order, the
                 spinodals and the IT threshold: snr_dyn, snr_alg, snr_it and delta_*_over_rho2 = 1 / snr_*^2
  rho, snr       every SE fixed point (se_fixed_points, se_stable), the uninformative and informative starts, the phase
                 (easy | hard | impossible | undetermined) and, unless undetermined, the Bayes gamma_se, q_se,
                 overlap_se, mmse_se, matrix_mmse_se.  On this path these keys are the BAYES (posterior-sample) values
                 at the fixed point of lower RS potential; in the hard phase AMP does not reach them (AMP's limit is
                 se_uninformative_gamma).  On the gaussian / rademacher path overlap_se and mmse_se are Bayes AMP's
                 performance (MV Thm 2); here they are the RS fixed point's sqrt(q) and 1 - q, covered by no theorem
                 fetched here (LM Thm 1 covers q and the matrix MMSE 1 - q^2), so they are not-computed
Finite-N AMP (N, seeds, iters) is not implemented for this prior and is refused.
"""
from .catalog import Out, RefCheck, Stability, fstep
from .core import Refusal
from .inference import spiked_diluted as SD

LKZ = ("Lesieur, Krzakala & Zdeborova, J. Stat. Mech. (2017) 073403, arXiv:1701.00858, Sec. IV C, Eqs. (33), "
       "(193), (213)-(214), Figs. 3, 6-7")
AFUZ = "Antenucci, Franz, Urbani & Zdeborova, PRX 9, 011020 (2019), arXiv:1805.05857, Fig. 1 and p. 5"
KZ = ("Krzakala & Zdeborova, Statistical Physics Methods in Optimization and Machine Learning, lecture notes "
      "(June 9 2024), Ex. 8.2")
LM = ("Lelarge & Miolane, Probab. Theory Relat. Fields 173, 859 (2019), arXiv:1611.03888, Thm 1 (bounded priors)")
PARAMS = ("rho",)
DILUTED = SD.PRIOR


def is_diluted(p):
    return p.get("prior") == DILUTED


def stray(p):
    """Without the diluted prior, rho is refused; with it, the finite-N knobs are."""
    if not is_diluted(p):
        if p.get("rho") is not None:
            raise Refusal("precondition:known-parameter", f"rho applies to prior={DILUTED} only")
        return
    bad = [k for k in ("N", "seeds", "iters") if p.get(k) is not None]
    if bad:
        raise Refusal("unsupported:diluted-finite-n", f"{', '.join(bad)}: finite-N AMP is not implemented for the "
                      f"{DILUTED} prior (its SE, fixed points and thresholds are)")
    if p.get("snr") is not None and p.get("rho") is None:
        raise Refusal("precondition:rho", f"snr with prior={DILUTED} needs rho")


def _tri(p):
    t = SD.diluted_tricritical()
    return Out({"rho_tri": t.rho, "snr_tri": t.snr, "delta_tri_over_rho2": 1 / t.snr**2,
                "delta_tri": t.rho**2 / t.snr**2},
               f"rho_tri = {t.rho:.8g}: the interior maximum of gamma m' - m (the sign of D') vanishes")


def _thresholds(p):
    rho = SD._rho(p["rho"])
    vals = {"rho": rho, "lambda_c": 1 / SD.diluted_overlap_slope(rho, 0.0) ** 0.5}
    sp = SD.diluted_spinodals(rho)
    if sp is None:
        return Out({**vals, "first_order": False}, f"rho = {rho}: D(gamma) is monotone, no first-order transition")
    it = SD.diluted_it_threshold(rho)
    for name, lam in (("dyn", sp.snr_dyn), ("alg", sp.snr_alg), ("it", it)):
        vals[f"snr_{name}"], vals[f"delta_{name}_over_rho2"] = lam, 1 / lam**2
    return Out({**vals, "first_order": True}, f"rho = {rho}: Delta/rho^2 dyn {1 / sp.snr_dyn**2:.6g}, IT "
               f"{1 / it**2:.6g}, alg {1 / sp.snr_alg**2:.6g} (hard phase between alg and IT)")


def _se(p):
    se = SD.diluted_state_evolution(p["snr"], p["rho"])
    vals = {"snr": se.snr, "se_fixed_points": [f.gamma for f in se.fixed_points],
            "se_stable": [f.stable for f in se.fixed_points], "se_phase": se.phase, "se_metastable": se.metastable,
            "se_uninformative_gamma": se.uninformative.gamma, "se_informative_gamma": se.informative.gamma}
    if se.phase == "undetermined":
        return Out({**vals, "se_status": "undetermined", "se_status_note": se.note}, se.note,
                   notes=(f"UNDETERMINED: {se.note}",))
    return Out({**vals, "se_status": "determined", "gamma_se": se.gamma, "q_se": se.q, "overlap_se": se.overlap,
                "mmse_se": se.mmse, "matrix_mmse_se": se.matrix_mmse, "se_bayes": se.bayes}, se.note)


def _when(rho, snr=None):
    return lambda p: is_diluted(p) and (p.get("rho") is not None) == rho and (snr is None or
                                                                            (p.get("snr") is not None) == snr)


def steps():
    return [fstep("diluted-tricritical", f"{LKZ} (Fig. 7 caption)", _tri, when=_when(False),
                  skip="not the diluted prior, or rho given"),
            fstep("diluted-thresholds", f"{LKZ}; {AFUZ}; {KZ}", _thresholds, when=_when(True),
                  skip="not the diluted prior, or no rho"),
            fstep("diluted-state-evolution", f"{LKZ}, Sec. IV C 1 (both starts, free-entropy decision); {LM}", _se,
                  when=_when(True, True), skip="needs the diluted prior, rho and snr")]


def _phys(note):
    return Stability("not-computed", "rs", f"{LKZ}; {AFUZ}", note + ": physics-level SE quantity, NOT final")


def _bayes(v):
    if v.get("se_status") == "determined" and v.get("se_phase") in ("easy", "hard", "impossible"):
        return Stability("exact-by-literature", "rs", LM,
                         "LM Thm 1 (priors of bounded support): the limit of the MMSE is set by the maximiser of the "
                         "RS free entropy; this run found that maximiser (the lower RS potential, DAM Eq. (9)) among "
                         "the converged uninformative and informative SE limits; not at the IT point itself")
    return Stability("not-computed", "rs", LM, "UNDETERMINED: the two SE starts did not both converge (or tie)")


def finality():
    sp = _phys("spinodal of the SE fixed points (a local extremum of D = m/gamma)")
    it = Stability("not-computed", "rs", f"{LM}; {LKZ}",
                   "where the RS potential of the two stable SE branches ties (LM Thm 1 makes the RS formula exact; "
                   "this number locates its kink numerically): NOT final")
    tri = _phys("the rho where the spinodals merge")
    return {"lambda_c": _phys("instability of the q = 0 SE fixed point (LKZ Eq. (193), Delta_c = rho^2); for this "
                              "prior it is not the IT threshold"),
            **{f"snr_{k}": sp for k in ("dyn", "alg")}, **{f"delta_{k}_over_rho2": sp for k in ("dyn", "alg")},
            "snr_it": it, "delta_it_over_rho2": it,
            **{k: tri for k in ("rho_tri", "snr_tri", "delta_tri_over_rho2", "delta_tri")},
            **{k: _bayes for k in ("gamma_se", "q_se", "matrix_mmse_se")},
            **{k: Stability("not-computed", "rs", f"{LM}; {LKZ}",
                            "the RS fixed point's vector quantity (sqrt(q), 1 - q) at the Bayes fixed point: LM Thm 1 "
                            "covers the free entropy and the matrix MMSE, not a cosine or a vector MSE (the posterior "
                            "mean of a sign-symmetric prior is 0); in the hard phase it is not AMP's performance: NOT "
                            "final") for k in ("overlap_se", "mmse_se")}}


META = ("rho", "first_order", "se_fixed_points", "se_stable", "se_phase", "se_metastable", "se_uninformative_gamma",
        "se_informative_gamma", "se_status", "se_status_note")


def refs():
    rho = {"prior": DILUTED, "rho": "0.08"}
    out = [RefCheck("spiked", f"diluted_rho0p08_delta_{k}_over_rho2", lambda v, k=k: v[f"delta_{k}_over_rho2"],
                    params=rho, route="amp") for k in ("dyn", "it", "alg")]
    tri = {"prior": DILUTED}
    out += [RefCheck("spiked", key, lambda v, x=x: v[x], params=tri, route="amp", slow=True)
            for key, x in (("diluted_rho_tri", "rho_tri"), ("diluted_delta_tri_over_rho2", "delta_tri_over_rho2"),
                           ("diluted_delta_tri", "delta_tri"))]
    return out


__all__ = ["DILUTED", "META", "PARAMS", "finality", "is_diluted", "refs", "steps", "stray"]
