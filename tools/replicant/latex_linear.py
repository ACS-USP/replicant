"""Displayed equations of the K6 entries `lasso` (amp, replica) and `compressed_sensing` (amp).  Same rule as
latex_entries.py: each sympy block carries a tex.Check against linear/ at three points (tests/test_latex.py); the
algorithms and theorem statements are cited strings.
"""
import sympy as sp

from .tex import Check, cited, sym, text

AL, D, K, R = sp.symbols("alpha Delta kappa rho", positive=True)
_KZ = ("Krzakala & Zdeborova, Statistical Physics Methods in Optimization and Machine Learning, lecture notes "
       "(June 9 2024)")
_BM = "Bayati & Montanari, IEEE Trans. Inf. Theory 58, 1997 (2012), arXiv:1008.2581"
_DMM = "Donoho, Maleki & Montanari, PNAS 106, 18914 (2009), arXiv:0907.3574"
_KMSSZ = "Krzakala, Mezard, Sausset, Sun & Zdeborova, PRX 2, 021005 (2012), arXiv:1109.4424"
_CONV = ("A is M x N with i.i.d. N(0, 1/N) entries, alpha = M/N, y = A x0 + sqrt(Delta) xi, x0 i.i.d. (1 - rho) "
         "delta_0 + rho N(0, 1); LASSO (1/2)|y - A x|^2 + lambda |x|_1.  Bayati-Montanari's delta = alpha, "
         "sigma_BM^2 = Delta/alpha, lambda_BM = lambda/alpha")


def _f(x):
    return float(sp.Rational(x))


def lasso(route):
    from .linear import lasso as L
    blocks = [
        text("AMP route" if route == "amp" else "Replica route", _CONV, source=_KZ),
        cited("AMP for the LASSO (KZ Alg. 4, (16.90)-(16.95); (16.86) read with R < -lambda Sigma)",
              r"\Sigma^t = \frac{1 + V^t}{\alpha},\quad R_i = a_i + \frac{1}{\alpha}\sum_\mu F_{\mu i}(y_\mu - "
              r"\omega_\mu),\quad a_i = \mathrm{ST}(R_i; \lambda\Sigma^t),\quad v_i = \Sigma^t\,"
              r"\mathbb{1}\{|R_i| > \lambda\Sigma^t\}", _KZ),
        cited("State evolution (KZ (18.13)-(18.14); BM Eq. (1.7))",
              r"\tau_t^2 = \frac{E^t + \Delta}{\alpha},\quad E^{t+1} = \mathbb{E}\big[(\mathrm{ST}(X_0 + \tau_t Z; "
              r"\theta_t) - X_0)^2\big],\quad b_t = \mathbb{P}(|X_0 + \tau_t Z| > \theta_t)", f"{_KZ}; {_BM}"),
        cited("Calibration (derived: at a fixed point Sigma = 1/(alpha - b) and theta = lambda Sigma; BM Eq. (1.15) "
              "mapped), checked at three state points", r"\lambda = \theta\,(\alpha - b)", f"{_BM}; {_KZ}",
              check=Check((sp.Symbol("lam"), AL, R, D), lambda lam, a, r, d: _f(lam),
                          (("1/20", "1/2", "1/10", "1/100"), ("2/5", "4/5", "3/10", "1/5"), ("1", "3/2", "1/2", "1/2")),
                          display=lambda lam, a, r, d: (lambda s: s.threshold * (_f(a) - s.b))(
                              L.lasso_se(_f(a), _f(r), _f(lam), _f(d))), tol="1e-11")),
        sym("Least squares (lambda = 0, alpha > 1): the risk Delta/(alpha - 1), the ridgeless HMRT limit at "
            "gamma = 1/alpha", sp.Eq(sp.Symbol("E"), D / (AL - 1)), source=_KZ,
            check=Check((AL, D), lambda a, d: L.lasso_se(_f(a), "0.2", 0, _f(d)).risk,
                        (("2", "1"), ("3/2", "1/2"), ("4", "1/4")), "rhs", tol="1e-12")),
        cited("Theorem (BM Thm 1.5, Cor. 1.6): for Gaussian A, Delta > 0, lambda > 0, the LASSO risk is the SE fixed "
              "point's", r"\lim_{N\to\infty}\frac{1}{N}\|\hat x - x_0\|^2 = \mathbb{E}\big[(\mathrm{ST}(X_0 + \tau_* Z;"
              r"\theta_*) - X_0)^2\big] = \alpha\tau_*^2 - \Delta", _BM),
    ]
    if route == "amp":
        blocks.append(text("Finite-N checks", "KZ Alg. 4 and coordinate descent (KKT-audited) on sampled instances "
                           "over named seeds: the AMP fixed point is the LASSO minimiser; 99% Student-t bands of the "
                           "risk and threshold (statistical)"))
    return blocks


def compressed_sensing(route):
    from .linear import cs as CS
    S = (1 + K**2) * (1 - sp.erf(K / sp.sqrt(2))) / 2 - K * sp.exp(-K**2 / 2) / sp.sqrt(2 * sp.pi)
    return [
        text("AMP route", "noiseless y = A x0, A with i.i.d. N(0, 1/N) entries, alpha = M/N, x0 Gauss-Bernoulli with "
             "non-zero fraction rho (DMM's rho is rho/alpha, delta = alpha)", source=_KZ),
        sym("Slope of the soft-threshold SE at E = 0, threshold kappa tau (checked against the SE's own mse/tau^2, "
            "extrapolated to tau = 0); alpha_l1 = min over kappa",
            sp.Eq(sp.Symbol("s"), R * (1 + K**2) + 2 * (1 - R) * S), source=_DMM,
            check=Check((R, K), lambda r, k: CS._slope(_f(r), _f(k)), (("2/5", "1/2"), ("1/10", "2"), ("1/5", "6/5")),
                        "rhs", tol="1e-6")),
        cited("The l1 boundary (DMM Eq. (5), kappa_chi = 2): alpha_l1 solves alpha rho_SE(alpha) = rho",
              r"\rho_{SE}(\delta) = \max_{z\ge 0}\frac{1 - \frac{2}{\delta}\left[(1+z^2)\Phi(-z) - z\phi(z)\right]}"
              r"{1 + z^2 - 2\left[(1+z^2)\Phi(-z) - z\phi(z)\right]}", _DMM),
        cited("The Bayes spinodal (derived in linear/cs.py from E' = mmse(alpha/E)), confirmed by two SE starts",
              r"\alpha_{BP} = \max_{s>0}\, s\,\mathrm{mmse}(s)", _KMSSZ),
        text("Information-theoretic boundary", "alpha_it = rho (KMSSZ's replica free entropy has its global maximum at "
             "D = 0 iff alpha > rho0): physics-level", source=_KMSSZ),
        text("Finite-N checks", "basis pursuit by linear programming (success fraction over named seeds) and Bayes "
             "GAMP (MSE band), statistical"),
    ]


__all__ = ["compressed_sensing", "lasso"]
