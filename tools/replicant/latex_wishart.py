"""Displayed equations of the `spiked_wishart` catalog entry (amp route, K5).  Same rule as latex_entries.py: each
sympy block carries a tex.Check against inference/wishart.py at three points (tests/test_latex.py); the algorithm and
the theorem statements are cited strings.
"""
import sympy as sp

from .tex import Check, cited, sym, text

LAM, AL, MV = sp.symbols("lambda alpha m_v", positive=True)
_KZ = ("Krzakala & Zdeborova, Statistical Physics Methods in Optimization and Machine Learning, lecture notes "
       "(June 9 2024), Sec. 12.3.3")
_LKZ = "Lesieur, Krzakala & Zdeborova, J. Stat. Mech. (2017) 073403, arXiv:1701.00858"
_BBP = "Baik, Ben Arous & Peche, Ann. Probab. 33, 1643 (2005), arXiv:math/0403022"


def spiked_wishart(route):
    from .inference import wishart as WS
    pts = (("4",), ("1",), ("1/4",))
    return [
        text("AMP route", "Y = (lambda/n) u v^T + W, n x m, W_ij ~ N(0, 1/n), alpha = m/n, u and v i.i.d. from "
             "unit-variance priors; the notes' lambda (Eq. (12.87)) is lambda^2 here, LKZ's Delta is 1/lambda^2, "
             "and with v Gaussian the columns of sqrt(n) Y are BBP's samples (gamma^2 = alpha, l_1 = 1 + lambda^2)",
             source=_KZ),
        sym("Threshold (BBP Thm 1.1 (36): l_c = 1 + 1/gamma; LKZ Eq. (198): Delta_c = sqrt(alpha))",
            sp.Eq(sp.Symbol("lambda_c"), AL ** sp.Rational(-1, 4)), source=f"{_BBP}; {_LKZ}",
            check=Check((AL,), lambda a: WS.wishart_se_threshold(float(sp.Rational(a)), "gaussian", "gaussian"),
                        pts, "rhs", tol="1e-9")),
        sym("Top singular value above the edge (BBP Thm 1.1 (39), sigma_1^2 = alpha lambda_1(S))",
            sp.Eq(sp.Symbol("sigma_1"), sp.sqrt((1 + LAM**2) * (AL * LAM**2 + 1)) / LAM), source=_BBP,
            check=Check((LAM, AL), lambda l, a: WS.bbp_top_singular(float(l), float(sp.Rational(a))),
                        (("1.9", "1/4"), ("1", "4"), ("2", "1")), "rhs")),
        sym("Gaussian u: the u-channel of the SE (LKZ Eq. (248), Delta = 1/lambda^2)",
            sp.Eq(sp.Symbol("m_u"), AL * LAM**2 * MV / (1 + AL * LAM**2 * MV)), source=_LKZ,
            check=Check((LAM, AL, MV), lambda l, a, m: WS._m(WS.prior("gaussian"),
                                                             float(a) * float(l)**2 * float(m)),
                        (("1.5", "0.5", "0.7"), ("2", "1", "0.3"), ("1.2", "2", "0.9")), "rhs", tol="1e-10")),
        cited("Bayes state evolution (the notes' (12.97)-(12.100) with U* and V* swapped back; LKZ (195)-(196))",
              r"m_u^t = \mathbb{E}\big[U\,\eta_U(\sqrt{\alpha\lambda^2 m_v^t}\,U + Z)\big],\quad "
              r"m_v^{t+1} = \mathbb{E}\big[V\,\eta_V(\sqrt{\lambda^2 m_u^t}\,V + Z)\big]", f"{_KZ}; {_LKZ}"),
        cited("Rectangular AMP (the notes' (12.88)-(12.89) in this scaling)",
              r"h_u^t = \lambda Y\hat v^t - \lambda^2\alpha\langle f_v'\rangle\,\hat u^{t-1},\quad "
              r"h_v^{t+1} = \lambda Y^\top\hat u^t - \lambda^2\langle f_u'\rangle\,\hat v^t", _KZ),
        text("Finite-N checks", "rectangular Bayes AMP from an informative start independent of W, and the top "
             "singular value by SVD, over named seeds with 99% Student-t bands (statistical)"),
    ]


__all__ = ["spiked_wishart"]
