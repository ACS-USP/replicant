"""Catalog route "rsb" of the spherical perceptron entry: the capacity at T = 0 beyond RS for kappa < 0.

  python3 -m replicant run perceptron --route rsb --param kappa=-1                  1RSB (default)
  python3 -m replicant run perceptron --route rsb --param kappa=-1 --param ansatz=krsb --param levels=2
  python3 -m replicant run perceptron --route rsb --param kappa=-1 --param ansatz=frsb --param levels=3

`1rsb` (families/perceptron_rsb.py, audited mpmath root, ~3 s) is the default; `krsb` (K = levels, 2..5)
and `frsb` (the staircase K = 1..levels extrapolated geometrically, levels 3..5) are the float64 grid
code of families/perceptron_krsb.py (tens of seconds to minutes: slow tier).  kappa >= 0: 1RSB = RS.
Finality: a 1RSB / K-step value is labelled by its COMPUTED replicon 1 - c (c the contact index): `unstable`
for c > 1 (every kappa < 0 checked), `stable` only for a computed non-negative replicon; the full-RSB
estimate is `not-computed` (an extrapolation, never final).
"""
from dataclasses import dataclass

import mpmath as mp

from .catalog import Out, RefCheck, Stability, fstep, from_flag
from .core import Refusal
from .families.perceptron import gardner_capacity

RSB_CITE = ("Franz, Parisi, Sevelev, Urbani & Zamponi, SciPost Phys. 2, 019 (2017), arXiv:1702.06919, "
            "Eq. (59), App. B (zero-temperature K-step RSB, jamming scaling); Franz & Parisi, "
            "J. Phys. A 49, 145001 (2016), arXiv:1501.03397")
RSB_CAVEAT = ("physics-level: replica n -> 0 continuation and the 1RSB / K-step RSB ansatz at T = 0 (jamming "
              "scaling m = mu (1 - q1)); K-step values are variational upper bounds on the next level, the "
              "full-RSB value is a staircase extrapolation; numbers carry quadrature error only")
ANSATZE = ("1rsb", "krsb", "frsb")
_KAPPAS = {"m2p7": -2.7, "m2p3": -2.3, "m2p0": -2.0, "m1p7": -1.7, "m1p5": -1.5, "m1p3": -1.3,
           "m1p0": -1.0, "m0p7": -0.7, "m0p5": -0.5, "m0p3": -0.3}


def _levels(p, default):
    L = p.get("levels")
    if L is None:
        return default
    try:
        ok = float(L) == int(float(L)) and 1 <= int(float(L)) <= 5
    except (TypeError, ValueError, OverflowError):
        ok = False
    if not ok:
        raise Refusal("precondition:levels", f"levels = {L!r}: an integer 1..5")
    return int(float(L))


def _capacity(p):
    a, kappa = p.get("ansatz", "1rsb"), p.get("kappa", 0)
    if a not in ANSATZE:
        raise Refusal("precondition:known-ansatz", f"perceptron rsb route: ansatz = {a!r}: use "
                      f"{', '.join(ANSATZE)}")
    levels = _levels(p, {"1rsb": 1, "krsb": 2, "frsb": 3}[a])
    if a == "1rsb" and levels != 1:
        raise Refusal("precondition:levels", "ansatz 1rsb has one level; use ansatz=krsb for K >= 2")
    if a == "krsb" and levels < 2:
        raise Refusal("precondition:levels", "ansatz krsb needs levels >= 2 (1 is ansatz=1rsb)")
    if a == "frsb" and levels < 3:
        raise Refusal("precondition:levels", "ansatz frsb extrapolates a staircase of >= 3 levels")
    if a == "1rsb":
        c = gardner_capacity("spherical", kappa, "1rsb")
        out = {"alpha_c": c.alpha_c, "capacity_replicon": c.replicon, "capacity_stable": c.stable,
               "capacity_ansatz": c.ansatz}
        if c.point is not None:
            out.update(rsb_q0=c.point.q0, rsb_mu=c.point.mu, rsb_contact_index=c.point.contact_index)
        return Out(out, c.note)
    from .families.perceptron_krsb import frsb_estimate, krsb_jamming
    if a == "krsb":
        j = krsb_jamming(float(kappa), levels)
        return Out({"alpha_c": j.alpha_c, "capacity_replicon": j.replicon,
                    "capacity_stable": bool(j.replicon >= 0), "capacity_ansatz": "krsb",
                    "rsb_q0": j.qs[0], "rsb_mu": j.mus[0], "rsb_contact_index": j.contact_index},
                   f"{levels}-step RSB jamming point (float64 grid; BFGS minimum of the upper-bound "
                   f"functional); contact index c = {j.contact_index:.6g} (1 = marginal full RSB)",
                   errors={"alpha_c": j.grid_error})
    alpha_inf, band, ladder = frsb_estimate(float(kappa), levels)
    return Out({"alpha_c": alpha_inf, "capacity_replicon": None, "capacity_stable": None,
                "capacity_ansatz": "frsb", "frsb_band": band, "rsb_contact_index": ladder[-1].contact_index,
                "rsb_ladder": tuple(x.alpha_c for x in ladder)},
               f"full-RSB estimate from the staircase K = 1..{levels}: geometric (Aitken) extrapolation; "
               f"{levels}-RSB value {ladder[-1].alpha_c:.8g}, last difference {band:.3g}",
               errors={"alpha_c": band})


def capacity_finality(v):
    """alpha_c and the 1RSB / K-step parameters: labelled by the computed replicon (never stable unless it is)."""
    a = v["capacity_ansatz"]
    if a == "frsb":
        return Stability("not-computed", "frsb", RSB_CITE,
                         "extrapolation of the K-step staircase to K -> infinity (band = last difference); "
                         "the continuum Parisi solution is not computed here")
    if "rsb_contact_index" not in v:                      # kappa >= 0: 1RSB reduces to RS
        return from_flag(v["capacity_stable"], a, RSB_CITE, "kappa >= 0: 1RSB reduces to RS (replicon of RS)")
    return from_flag(v["capacity_stable"], a, RSB_CITE,
                     "computed contact index c = " + mp.nstr(mp.mpf(v["rsb_contact_index"]), 6)
                     + " (replicon 1 - c; c > 1: unstable toward further RSB, the capacity is lower at "
                     "the next level)")


@dataclass(frozen=True)
class RsbRoute:
    steps: list
    caveat: str
    params: tuple
    finality: dict
    meta: tuple
    refs: list


def perceptron_rsb_route():
    steps = [fstep("gardner-capacity-rsb", RSB_CITE, _capacity)]
    # all slow: the audited mpmath root costs ~3 s per kappa (the default-tier guard of the number is
    # tests/test_perceptron_rsb.py::test_one_rsb_capacity_kappa_minus_one_float_stage_matches_the_bank)
    refs = [RefCheck("perceptron", "spherical_1rsb_capacity_kappa_m1p0", lambda v: v["alpha_c"],
                     params={"kappa": -1}, route="rsb", slow=True),
            RefCheck("perceptron", "spherical_1rsb_q0_kappa_m1p0", lambda v: v["rsb_q0"],
                     params={"kappa": -1}, route="rsb", slow=True)]
    refs += [RefCheck("perceptron", f"spherical_1rsb_capacity_kappa_{t}", lambda v: v["alpha_c"],
                      params={"kappa": k}, route="rsb", slow=True)
             for t, k in _KAPPAS.items() if t != "m1p0"]
    refs += [RefCheck("perceptron", "spherical_1rsb_q0_kappa_m1p5", lambda v: v["rsb_q0"],
                      params={"kappa": -1.5}, route="rsb", slow=True)]
    refs += [RefCheck("perceptron", f"spherical_2rsb_capacity_kappa_{t}", lambda v: v["alpha_c"],
                      params={"kappa": _KAPPAS[t], "ansatz": "krsb", "levels": 2}, route="rsb", slow=True)
             for t in ("m2p0", "m1p5", "m1p0", "m0p5")]
    fin = {k: capacity_finality for k in ("rsb_q0", "rsb_mu", "rsb_contact_index", "frsb_band", "rsb_ladder")}
    return RsbRoute(steps, RSB_CAVEAT, ("kappa", "ansatz", "levels"), fin, ("capacity_ansatz",), refs)
