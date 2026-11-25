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
  rho, snr, N    finite-N Bayes AMP over named seeds (default 0-7, iters 30): q_amp, overlap_amp, mmse_amp,
                 matrix_mmse_amp and gamma_amp_t (the empirical gamma at the last t), each with <key>_halfwidth99, and
                 gamma_se_t, the SE at the same t from the same start.  init=spectral (default, lambda > 1): gamma_se_t
                 is the SE fixed point reached from the uninformative start (MV Thm 2's t -> oo limit).  init=planted
                 with gamma0 (any lambda > 0): x^0 = gamma0 x + sqrt(gamma0) g, gamma_se_t =
                 diluted_se_trajectory(snr, rho, gamma0, iters)[iters].  The run uses the empirical plug-in gamma_t
                 (inference/spiked.py _gamma_hat), not literally the theorem's SE-driven algorithm.
"""
from .catalog import Out, RefCheck, Stability, fstep
from .csp import _checks
from .core import Refusal
from .inference import spiked as S
from .inference import spiked_diluted as SD

LKZ = ("Lesieur, Krzakala & Zdeborova, J. Stat. Mech. (2017) 073403, arXiv:1701.00858, Sec. IV C, Eqs. (33), "
       "(193), (213)-(214), Figs. 3, 6-7")
AFUZ = "Antenucci, Franz, Urbani & Zdeborova, PRX 9, 011020 (2019), arXiv:1805.05857, Fig. 1 and p. 5"
KZ = ("Krzakala & Zdeborova, Statistical Physics Methods in Optimization and Machine Learning, lecture notes "
      "(June 9 2024), Ex. 8.2")
LM = ("Lelarge & Miolane, Probab. Theory Relat. Fields 173, 859 (2019), arXiv:1611.03888, Thm 1 (bounded priors)")
MV = ("Montanari & Venkataramanan, Ann. Statist. 49, 321 (2021), arXiv:1711.01682, Thm 2, "
      "Eqs. (2.24), (2.28)-(2.29), Prop. 2.2, Cor. 2.3")
BM = ("Bayati & Montanari, IEEE Trans. Inf. Theory 57, 764 (2011), arXiv:1001.3448, Thm 4; "
      "Bolthausen, Commun. Math. Phys. 325, 333 (2014), arXiv:1201.2891")
PARAMS = ("rho",)
FINITE = ("seeds", "iters", "init", "gamma0")
_SEEDS = tuple(range(8))
_AMP = {"q": "q_amp", "overlap": "overlap_amp", "mse": "mmse_amp", "matrix_mse": "matrix_mmse_amp",
        "gamma_t": "gamma_amp_t"}
DILUTED = SD.PRIOR


def is_diluted(p):
    return p.get("prior") == DILUTED


def stray(p):
    """Without the diluted prior, rho, init and gamma0 are refused; with it, N needs snr and rho, the finite-N knobs
    need N, and gamma0 needs init=planted."""
    if not is_diluted(p):
        bad = [k for k in ("rho", "init", "gamma0") if p.get(k) is not None]
        if bad:
            raise Refusal("precondition:known-parameter", f"{', '.join(bad)}: apply to prior={DILUTED} only")
        return
    if p.get("N") is None:
        bad = [k for k in FINITE if p.get(k) is not None]
        if bad:
            raise Refusal("precondition:known-parameter", f"{', '.join(bad)}: finite-N parameters need N (and snr, "
                          "rho)")
    elif p.get("snr") is None:
        raise Refusal("precondition:state-point", "finite-N AMP needs snr (and N, rho)")
    if p.get("snr") is not None and p.get("rho") is None:
        raise Refusal("precondition:rho", f"snr with prior={DILUTED} needs rho")
    if p.get("gamma0") is not None and p.get("init") != "planted":
        raise Refusal("precondition:known-parameter", "gamma0 applies to init=planted only (the spectral start is "
                      "x^0 = sqrt(N lambda^2 (lambda^2 - 1)) phi_1)")


def _tri(p):
    t = SD.diluted_tricritical()
    return Out({"rho_tri": t.rho, "snr_tri": t.snr, "delta_tri_over_rho2": 1 / t.snr**2,
                "delta_tri": t.rho**2 / t.snr**2},
               f"rho_tri = {t.rho:.8g}: the interior maximum of gamma m' - m (the sign of D') vanishes")


def _thresholds(p):
    rho = SD._rho(p["rho"])
    vals = {"rho": rho, "lambda_c": 1 / SD.diluted_overlap_slope(rho, 0.0) ** 0.5}
    kind = SD.diluted_first_order(rho)
    if kind == "undetermined":
        return Out({**vals, "first_order_status": "undetermined", "first_order_status_note": "within numerical "
                    "resolution of rho_tri (|max s| < 1e-10): the sign of D' is not resolved"},
                   f"rho = {rho}: first-order or continuous not resolved (at the tricritical point)")
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


def _finite(p):
    seeds, iters, init = _checks.seeds(p.get("seeds", _SEEDS)), p.get("iters", 30), p.get("init", "spectral")
    ens = S.spiked_amp_ensemble(p["N"], p["snr"], DILUTED, seeds=seeds, iters=iters, rho=p["rho"], init=init,
                                gamma0=p.get("gamma0"))
    if init == "planted":
        gse = SD.diluted_se_trajectory(p["snr"], p["rho"], ens.gamma0, iters)[iters]
        src = f"diluted_se_trajectory from gamma0 = {ens.gamma0:.6g} at t = {iters}"
    else:
        gse = SD.diluted_state_evolution(p["snr"], p["rho"]).uninformative.gamma
        src = "the SE fixed point reached from the uninformative start (MV Thm 2's t -> oo limit)"
    vals = {"rho": SD._rho(p["rho"]), "N": p["N"], "seeds": list(seeds), "iters": iters, "init": init,
            "gamma0": ens.gamma0, "amp_converged_fraction": sum(ens.converged) / len(seeds), "gamma_se_t": gse}
    for k, name in _AMP.items():
        vals.update({name: ens.bands[k].mean, f"{name}_halfwidth99": ens.bands[k].half_width})
    b = ens.bands["gamma_t"]
    return Out(vals, f"Bayes AMP ({init} start) over seeds {list(seeds)}: gamma_t = {b.mean:.6g} +- "
               f"{b.half_width:.3g} (99% Student-t, df = {b.df}); gamma_se_t = {gse:.6g} ({src})",
               errors={name: ens.bands[k].stderr for k, name in _AMP.items()},
               statistical=tuple(_AMP.values()), notes=(ens.note,))


def _when(rho, snr=None):
    return lambda p: is_diluted(p) and (p.get("rho") is not None) == rho and (snr is None or
                                                                            (p.get("snr") is not None) == snr)


def steps():
    return [fstep("diluted-tricritical", f"{LKZ} (Fig. 7 caption)", _tri, when=_when(False),
                  skip="not the diluted prior, or rho given"),
            fstep("diluted-thresholds", f"{LKZ}; {AFUZ}; {KZ}", _thresholds, when=_when(True),
                  skip="not the diluted prior, or no rho"),
            fstep("diluted-state-evolution", f"{LKZ}, Sec. IV C 1 (both starts, free-entropy decision); {LM}", _se,
                  when=_when(True, True), skip="needs the diluted prior, rho and snr"),
            fstep("diluted-amp-finite-n", f"{MV}; {BM} (finite-N iterates: statistical)", _finite,
                  when=lambda p: is_diluted(p) and p.get("N") is not None, skip="needs the diluted prior and N")]


def _phys(note):
    return Stability("not-computed", "rs", f"{LKZ}; {AFUZ}", note + ": physics-level SE quantity, NOT final")


def _bayes(v):
    if v.get("se_status") == "determined" and v.get("se_phase") in ("easy", "hard", "impossible"):
        return Stability("exact-by-literature", "rs", LM,
                         "LM Thm 1 (priors of bounded support): the limit of the MMSE is set by the maximiser of the "
                         "RS free entropy; this run found that maximiser (the lower RS potential, DAM Eq. (9)) among "
                         "the converged uninformative and informative SE limits; not at the IT point itself")
    return Stability("not-computed", "rs", LM, "UNDETERMINED: the two SE starts did not both converge (or tie)")


_PLUG = ("the finite-N run uses the empirical plug-in gamma_t (|x^t|^2 / N = gamma_t^2 + gamma_t, inference/spiked.py "
         "_gamma_hat), not literally the theorem's algorithm, whose gamma_t comes from SE")


def _stat():
    return Stability("not-computed", "amp-finite-N", f"{MV}; {BM}",
                     "statistical: finite-N mean +- stderr over the named seeds (99% Student-t half-width in "
                     f"<key>_halfwidth99), never certified; {_PLUG}: NOT final")


def _se_t(v):
    if v.get("init") == "planted":
        return Stability("exact-by-literature", "amp-se", BM,
                         "SE at the fixed t from gamma_0 = gamma0: the N -> oo limit of the iterate's empirical "
                         "statistics at fixed t, for a Lipschitz denoiser (the diluted prior has bounded support) and "
                         f"a start x^0 = gamma0 x + sqrt(gamma0) g independent of the noise W given x; {_PLUG}")
    if v.get("snr", 0) > 1 and v.get("se_status") == "determined" and v.get("se_phase") == "easy":
        return Stability("exact-by-literature", "amp-se", MV,
                         "MV Thm 2, Eqs. (2.24), (2.28)-(2.29): the N -> oo then t -> oo limit of Bayes AMP with "
                         "spectral start (lambda > 1; F Lipschitz: bounded support) is the SE fixed point reached from "
                         "the uninformative start; here both SE starts reach the same fixed point (phase easy), so it "
                         "is also the Bayes one; gamma_se_t is that t -> oo limit, the run stops at a finite t; "
                         f"{_PLUG}")
    return Stability("not-computed", "amp-se", MV, "MV Thm 2 needs lambda > 1, and here the SE has more than one "
                     "stable fixed point (or was undetermined): the spectral start's limit is the uninformative "
                     "one, not shown to be reached at this t: NOT final")


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
                            "final") for k in ("overlap_se", "mmse_se")},
            **{k: _stat() for k in _AMP.values()}, "gamma_se_t": _se_t}


META = ("rho", "first_order", "first_order_status", "first_order_status_note", "se_fixed_points", "se_stable", "se_phase", "se_metastable", "se_uninformative_gamma",
        "se_informative_gamma", "se_status", "se_status_note", "init", "gamma0", "gamma_amp_t_halfwidth99")


def refs():
    rho = {"prior": DILUTED, "rho": "0.08"}
    out = [RefCheck("spiked", f"diluted_rho0p08_delta_{k}_over_rho2", lambda v, k=k: v[f"delta_{k}_over_rho2"],
                    params=rho, route="amp") for k in ("dyn", "it", "alg")]
    tri = {"prior": DILUTED}
    out += [RefCheck("spiked", key, lambda v, x=x: v[x], params=tri, route="amp", slow=True)
            for key, x in (("diluted_rho_tri", "rho_tri"), ("diluted_delta_tri_over_rho2", "delta_tri_over_rho2"),
                           ("diluted_delta_tri", "delta_tri"))]
    return out


__all__ = ["DILUTED", "FINITE", "META", "PARAMS", "finality", "is_diluted", "refs", "steps", "stray"]
