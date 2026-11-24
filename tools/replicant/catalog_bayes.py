"""Catalog entries of KZ Ch. 7 (K2): `scalar_denoising` (route exact) and `sparse_denoising` (route replica); and the
`rem` entry's condensation step (KZ Ch. 15), wired in catalog_entries.rem.

Source: F. Krzakala & L. Zdeborova, "Statistical Physics Methods in Optimization and Machine Learning" (lecture
notes, version of June 9 2024), "KZ"; Guo, Shamai & Verdu, IEEE Trans. Inf. Theory 51, 1261 (2005),
arXiv:cs/0412108 (read), "GSV".  Conventions (bayes/channels.py, bayes/sparse_vector.py): KZ's y_i = x* +
sqrt(Delta) z_i (Eq. (7.4)) is the channel y = sqrt(snr) x + z at snr = N/Delta (one observation: snr = 1/Delta);
the sparse vector is y = x* + sqrt(Delta/N) z with x* one of d = 2^N basis vectors (Eq. (7.51)).
Ball values are reported as their midpoint (40 digits) with an upper bound of |value - x| in `errors`.  The exact
route's rigor depends on the prior (scalar_rigor): "certified" for gaussian and +-1 (every value a ball); for the
Gauss-Bernoulli and sparse priors the label names the binary64 quadrature of the MMSE and I.
Run parameters:
  scalar_denoising (exact)    prior (gaussian (default) | rademacher | gauss_bernoulli | sparse), rho, var, a (the
                              prior's parameters), Delta (noise variance, rational > 0, default 1), data (a KZ data
                              set "7.6" | "7.8" | "7.11", or a list of numbers: the posterior of x given them)
  sparse_denoising (replica)  Delta (rational > 0; without it Delta_c only), d (= 2^N: sigma_c^2 and, with Delta,
                              the exact finite-d (1/N) E log Z)
"""
import mpmath as mp

from .bayes import channels as CH
from .bayes import sparse_vector as SV
from .catalog import Out, RefCheck, Stability, entry, fstep, has
from .core import Refusal
from .families import rem as REM
from .lattice.ball import rational
from .spec import ModelSpec

KZ = ("Krzakala & Zdeborova, Statistical Physics Methods in Optimization and Machine Learning, lecture notes "
      "(June 9 2024)")
GSV = "Guo, Shamai & Verdu, IEEE Trans. Inf. Theory 51, 1261 (2005), arXiv:cs/0412108"


def _put(**balls):
    vals, errs = {}, {}
    for k, b in balls.items():
        vals[k], errs[k] = b.value_and_error(40)
    return vals, errs


def _lit(cite, note=""):
    return Stability("exact-by-literature", "exact", cite, note)


# ---- scalar_denoising ----------------------------------------------------------------------------------------
def _prior(p):
    return CH.Prior(p.get("prior", "gaussian"), **{k: p.get(k) for k in ("rho", "var", "a") if p.get(k) is not None})


def _delta(p):
    return rational(p.get("Delta", 1), "Delta")


def _channel(p):
    P, d = _prior(p), _delta(p)
    return Out({"prior": P.name, "prior_params": repr(P), "Delta": str(d), "snr": str(1 / d)},
               f"{P!r}; y = x + sqrt(Delta) z is y' = sqrt(snr) x + z at snr = 1/Delta = {1 / d}")


def _information(p):
    P, s = _prior(p), 1 / _delta(p)
    if P.name in ("gaussian", "rademacher"):
        v, e = _put(mmse=CH.mmse_ball(P, s), mutual_information=CH.information_ball(P, s))
        return Out(v, f"certified (arb): mmse = {v['mmse'][:14]}, I = {v['mutual_information'][:14]} nats", errors=e)
    return Out({"mmse": CH.mmse(P, s), "mutual_information": CH.mutual_information(P, s)},
               "binary64 trapezoid quadrature over y (not certified)")


def _data(p):
    d = p["data"]
    if isinstance(d, str):
        if d not in CH.DATASETS:
            raise Refusal("precondition:data", f"data = {d!r}: a KZ data set is one of {', '.join(CH.DATASETS)}, "
                          "or give a list of numbers")
        return CH.DATASETS[d]
    return d


def _posterior(p):
    P = _prior(p)
    post = CH.posterior_ball(P, _data(p), _delta(p))
    balls = {"posterior_mean": post["mean"], "posterior_var": post["var"]}
    if any(mu > 0 and v == 0 for _, mu, v in P.components):
        balls["p_plus"] = post["p_plus"]
    if any(mu == 0 and v == 0 for _, mu, v in P.components):
        balls["p_nonzero"] = post["p_nonzero"]
    v, e = _put(**balls)
    return Out({**v, "N": post["N"]}, f"posterior of x given {post['N']} observations (sufficient statistic, arb)",
               errors=e)


def scalar_denoising():
    steps = [fstep("scalar-channel", f"{KZ}, Sec. 7.2.1, Eq. (7.4); {GSV}, Eq. (3)", _channel),
             fstep("scalar-mmse-information", f"{GSV}, Theorem 1, Eqs. (11), (13), (17), (18); {KZ}, Sec. 7.2.2-"
                   "7.2.3, Eqs. (7.18), (7.39)", _information),
             fstep("scalar-posterior", f"{KZ}, Sec. 7.2.1, Eqs. (7.5), (7.7), (7.9), (7.10)", _posterior,
                   when=has("data"), skip="no data given")]
    R = lambda key, get, prior, data: RefCheck("bayes", key, get, params={"prior": prior, "data": data, "Delta": 1},
                                               route="exact")
    refs = [R("gauss_posterior_mean_eq7_8", lambda v: v["posterior_mean"], "gaussian", "7.8"),
            R("gauss_posterior_var_eq7_8", lambda v: v["posterior_var"], "gaussian", "7.8"),
            R("gb_p_nonzero_eq7_11", lambda v: v["p_nonzero"], "gauss_bernoulli", "7.11")]
    gsv = _lit(f"{GSV}, Eqs. (11), (13) (gaussian), (17), (18) (+-1)", "certified ball (arb; the +-1 integrals by "
               "acb.integral with a rigorous tail, bounds/gauss.py)")
    quad = Stability("not-computed", "exact", f"{KZ}, Eqs. (7.18), (7.39)", "binary64 trapezoid quadrature, not "
                     "certified (no closed form for this prior)")
    by_prior = lambda v: gsv if v["prior"] in ("gaussian", "rademacher") else quad
    bayes = _lit(f"{KZ}, Eqs. (7.5), (7.7), (7.9), (7.10) (Bayes' formula)", "certified ball (arb, the decimal data "
                 "exact)")
    fin = {"mmse": by_prior, "mutual_information": by_prior,
           **{k: bayes for k in ("posterior_mean", "posterior_var", "p_plus", "p_nonzero")}}
    return entry("scalar_denoising", ModelSpec("scalar_denoising", "real", "gaussian",
                                               constraint="scalar Gaussian channel, Bayes-optimal"),
                 f"{KZ}, Ch. 7; {GSV}", {"exact": steps}, refs,
                 {"exact": "closed forms and certified integrals (gaussian, +-1 priors); binary64 quadrature for the "
                           "Gauss-Bernoulli and sparse priors"},
                 "Bayes-optimal scalar denoising: posterior, MMSE, mutual information for four priors",
                 default_route="exact", finality=fin, meta=("prior", "prior_params", "Delta", "snr", "N"),
                 parameters={"exact": ("prior", "rho", "var", "a", "Delta", "data")}, rigor={"exact": scalar_rigor})


def scalar_rigor(values):
    """The Result's rigor: "certified" when every value is an arb enclosure (gaussian, +-1 priors); otherwise the
    posteriors are certified and the MMSE / I are binary64 quadrature, and the label says so."""
    if values.get("prior") in ("gaussian", "rademacher"):
        return "certified"
    return QUADRATURE_RIGOR


QUADRATURE_RIGOR = "certified posteriors; binary64 quadrature (mmse, mutual_information: no closed form)"


# ---- sparse_denoising ----------------------------------------------------------------------------------------
def _threshold(p):
    v = {"Delta_c": SV.delta_c()}
    if p.get("d") is not None:
        v["sigma_c2"] = SV.sigma_c2(p["d"])
    return Out(v, "Delta_c = 1/(2 log 2)" + (f"; sigma_c^2 = 1/(2 log d) at d = {p['d']}" if "sigma_c2" in v else ""))


def _theorem(p):
    D = p["Delta"]
    f = SV.sparse_vector_free_entropy(D)
    m = SV.sparse_vector_mmse(D)
    return Out({"f": f, "q": 1 - m, "mmse": m}, f"Theorem 15 at Delta = {D}: f = {str(f)[:14]}, MMSE = {m}")


def _rem_route(p):
    beta, s = SV.rem_parameters(p["Delta"])
    return Out({"f_rem": SV.free_entropy_via_rem(p["Delta"]), "beta_rem": beta,
                "Delta_c_rem": SV.delta_c_via_rem()},
               f"null terms as a REM at beta = sqrt(2/Delta) = {str(beta)[:12]}, N_scale = 1/2")


def _finite(p):
    d = p["d"]
    return Out({"f_N": SV.sparse_vector_free_entropy(p["Delta"], d=d)},
               f"exact (1/N) E log Z at d = {d} (binary64 quadrature)")


def sparse_denoising():
    steps = [fstep("sparse-threshold", f"{KZ}, Sec. 7.3, Theorem 15 and p. 127", _threshold),
             fstep("sparse-theorem15", f"{KZ}, Theorem 15, Eqs. (7.57)-(7.58); I-MMSE Eq. (7.56); App. 7.B, Eqs. "
                   "(7.92)-(7.93)", _theorem, when=has("Delta"), skip="no Delta given"),
             fstep("sparse-rem-route", f"{KZ}, Eq. (7.54); Derrida, Phys. Rev. B 24, 2613 (1981) (families/rem.py)",
                   _rem_route, when=has("Delta"), skip="no Delta given"),
             fstep("sparse-finite-d", f"{KZ}, Eqs. (7.54)-(7.55)", _finite, when=has("Delta", "d"),
                   skip="needs Delta and d")]
    refs = [RefCheck("bayes", "sparse_delta_c", lambda v: v["Delta_c"]),
            RefCheck("bayes", "sparse_f_delta_half", lambda v: v["f"], params={"Delta": "1/2"})]
    thm = _lit(f"{KZ}, Theorem 15 (proved in the notes, Lemmas 8-9) and Theorem 14 (I-MMSE)")
    rem = _lit(f"Derrida, Phys. Rev. B 24, 2613 (1981) (the REM free energy), mapped by {KZ}, Eq. (7.54)",
               "an independent route to f and Delta_c")
    fin = {"Delta_c": thm, "sigma_c2": _lit(f"{KZ}, Sec. 7.3, p. 127"), "f": thm, "q": thm, "mmse": thm,
           "f_rem": rem, "Delta_c_rem": rem, "beta_rem": rem,
           "f_N": Stability("not-computed", "exact", f"{KZ}, Eq. (7.54)", "exact finite-d value by binary64 "
                            "quadrature (not certified, never final)")}
    return entry("sparse_denoising", ModelSpec("sparse_denoising", "real", "gaussian",
                                               constraint="one-hot vector of dimension 2^N in Gaussian noise"),
                 f"{KZ}, Sec. 7.3, App. 7.A-7.B", {"replica": steps}, refs,
                 {"replica": "Theorem 15 (rigorous in the notes) beside the replica saddles of App. 7.B; the REM "
                             "route independent"},
                 "denoising a sparse (one-hot) vector: the REM-like transition at Delta_c = 1/(2 log 2)",
                 finality=fin, parameters={"replica": ("Delta", "d")})


# ---- the rem entry's condensation step (K2; wired in catalog_entries.rem) -----------------------------------------
def _condensation_flag(p):
    """`condensation`: true | false (default false); anything else is refused, never read as false."""
    v = p.get("condensation", False)
    if not (isinstance(v, bool) or v in ("true", "false")):
        raise Refusal("precondition:known-parameter", f"condensation = {v!r}: accepted values are true and false")
    return v in (True, "true")


def rem_condensation(scale):
    """(step, refs, finality) of KZ Eqs. (15.39), (15.66), (15.41)-(15.42) for the rem entry: E[Y], E[Y^2], P(q = 1),
    P(q = 0).  Replica values (physics-level), so behind `condensation=true`: the earlier parameter sets keep their
    all-final results.  `scale(p)` is the entry's N_scale."""
    def compute(p):
        with mp.workdps(30):
            s, b = scale(p), mp.mpf(p["beta"])
            ov = REM.rem_overlap_distribution(b, s)
            return Out({"participation_ratio": REM.rem_participation_ratio(b, s), "participation_ratio_sq":
                        REM.rem_y2(b, s), "overlap_p1": ov[1], "overlap_p0": ov[0]},
                       "condensation: m = min(beta_c/beta, 1), E[Y] = 1 - m")
    step = fstep("rem-condensation", f"{KZ}, Secs. 15.2.4-15.2.5, Eqs. (15.39)-(15.42); Ex. 15.2, Eq. (15.66)",
                 compute, when=lambda p: _condensation_flag(p) and p.get("beta") is not None,
                 skip="condensation not requested (or no beta)")
    R = lambda key, get: RefCheck("rem", key, get, params={"beta": 3, "condensation": True})
    refs = [R("participation_ratio_kz1539_beta3", lambda v: v["participation_ratio"]),
            R("participation_ratio_sq_kz1566_beta3", lambda v: v["participation_ratio_sq"])]
    st = Stability("not-computed", "1rsb", f"{KZ}, Eqs. (15.39)-(15.42), (15.66)", "replica 1RSB (physics-level); "
                   "checked against exact REM sampling and the exact finite-N moments (tests/test_rem_condensation.py)")
    return step, refs, {k: st for k in ("participation_ratio", "participation_ratio_sq", "overlap_p1", "overlap_p0")}


def bayes_entries():
    return [scalar_denoising(), sparse_denoising()]


__all__ = ["bayes_entries", "rem_condensation", "scalar_denoising", "sparse_denoising"]
