"""Displayed equations of the K1 catalog entries `curie_weiss`, `rfim` and `random_matrix` (KZ Ch. 1-3).  Same rule
as latex_entries.py: each sympy block carries a tex.Check against meanfield/ or rmt/ at three points
(tests/test_latex.py); the derivations' intermediate steps are cited.
"""
import mpmath as mp
import sympy as sp

from .tex import Check, cited, sym, text

_KZ = "Krzakala & Zdeborova, lecture notes (June 9 2024)"
B, H, M, X, AL, D, J = sp.symbols("beta h m x alpha Delta J", real=True)


def _cw():
    from .meanfield import curie_weiss as CW
    return CW


def _potential():
    Hm = -((1 + M) / 2 * sp.log((1 + M) / 2) + (1 - M) / 2 * sp.log((1 - M) / 2))
    return Hm + B * M**2 / 2 + B * H * M


def _root(beta, h, branch=None):
    """(beta, h, m*) with m* the certified root of the given / selected branch (a point for the checks)."""
    sol = _cw().cw_solution(beta, h)
    fp = sol.selected or sol.fixed_points[-1]
    if branch is not None:
        fp = next(f for f in sol.fixed_points if f.branch == branch)
    return beta, h, str(fp.m.mid())


def _root_hat(beta, h):
    """(beta, h, m*, mhat*) at the stationary point of g: mhat* = -atanh m* (Sec. 1.5)."""
    b, hh, m = _root(beta, h)
    with mp.workdps(40):
        return b, hh, m, str(-mp.atanh(mp.mpf(m)))


def curie_weiss(route):
    CW = _cw()
    model = text("Model", "N spins S_i = +-1, H_N = -(1/2N) sum_ij S_i S_j - h sum_i S_i (i = j included), J = 1, "
                 "Sbar = (1/N) sum_i S_i; Phi = lim (1/N) ln Z_N, f = -Phi/beta", source=f"{_KZ}, Eq. (1.1)")
    if route == "exact":
        return [model,
                cited("The binomial count (Sec. 1.1), summed exactly in ball arithmetic",
                      r"Z_N = \sum_{k=0}^{N} \binom{N}{k} e^{\beta N (m_k^2/2 + h m_k)},\quad m_k = \frac{2k - N}{N}",
                      f"{_KZ}, Sec. 1.1, Eq. (1.2)"),
                cited("Lemma 2 (audited at every run) and the finite-N rate (Lemma 1)",
                      r"\phi(m^*) - \frac{\ln N(N+1)}{N} \le \Phi_N \le \phi(m^*) + \frac{\ln(N+1)}{N},\quad "
                      r"I_N(m) = -\frac{1}{N} \ln P_N(\bar S = m)", f"{_KZ}, Eqs. (1.4), (1.6)"),
                text("Finite-size term (derived here)", "N (Phi_N - Phi) -> -(1/2) ln(1 - beta (1 - m*^2)) + "
                     "ln(number of global maxima), by the Laplace method on the binomial sum")]
    if route == "cavity":
        phit = -B * M**2 / 2 + sp.log(2 * sp.cosh(B * (M + H)))
        return [model,
                cited("Adding a spin (Eqs. (1.18)-(1.21))",
                      r"\frac{Z_{N+1}}{Z_N} = \left\langle e^{-\beta \bar S^2/2}\, 2\cosh\beta(\bar S + h)\right\rangle_N",
                      f"{_KZ}, Eq. (1.21)"),
                sym("Cavity potential (Eq. (1.23)), maximised by m = tanh beta (m + h); its value at the root is the "
                    "cavity route's Phi (Theorem 6, App. 1.C)",
                    sp.Eq(sp.Symbol(r"\tilde\phi"), phit), source=f"{_KZ}, Eq. (1.23)",
                    check=Check((B, H, M), lambda b, h, m: CW.cw_free_entropy_cavity(b, h)[1],
                                tuple(lambda a=a: _root(*a) for a in (("1.5", "0.05"), ("0.5", "0.2"), ("2", "-0.1"))),
                                "rhs"))]
    g = B * M**2 / 2 + B * H * M + sp.Symbol(r"\hat m") * M + sp.log(2) + sp.log(sp.cosh(sp.Symbol(r"\hat m")))
    s = sp.sqrt(1 - 1 / B)
    return [model,
            sym("Theorem 1: Phi = max over m of the potential, at a root of beta (m + h) = atanh m (certified)",
                sp.Eq(sp.Symbol(r"\phi"), _potential()), source=f"{_KZ}, Theorem 1, Eqs. (1.3), (1.7)",
                check=Check((B, H, M), lambda b, h, m: CW.cw_free_entropy(b, h).mid(),
                            tuple(lambda a=a: _root(*a) for a in (("1.5", "0.05"), ("0.5", "0.2"), ("2", "-0.1"))),
                            "rhs")),
            sym("Field-theoretic toolbox: Phi = extr over (m, mhat) of g (Sec. 1.5)", sp.Eq(sp.Symbol("g"), g),
                source=f"{_KZ}, Sec. 1.5",
                check=Check((B, H, M, sp.Symbol(r"\hat m")), lambda b, h, m, k: CW.cw_free_entropy_field(b, h)[1],
                            tuple(lambda a=a: _root_hat(*a) for a in (("1.5", "0.05"), ("0.5", "0.2"), ("2", "-0.1"))),
                            "rhs")),
            sym("Spinodal field, beta > 1 (App. 1.B; KZ p. 33 prints sqrt((1/beta)(1 - 1/beta)) in place of "
                "sqrt(1 - 1/beta): an erratum, contradicted by Fig. 1.B.4)", sp.Eq(sp.Symbol("h_{sp}"), s - sp.atanh(s) / B),
                source=f"{_KZ}, App. 1.B (corrected)",
                check=Check((B,), lambda b: CW.cw_spinodal(b).mid(), (("1.5",), ("2",), ("5",)), "rhs")),
            cited("Rate functions: the true rate (Eq. (1.8)) and its convex envelope (Legendre, Gartner-Ellis)",
                  r"I^*(m) = \Phi - \phi(m),\quad I(m) = \sup_k \left[k m - \lambda(k)\right],\ "
                  r"\lambda(k) = \Phi(\beta, h + k/\beta) - \Phi(\beta, h)", f"{_KZ}, Eq. (1.8), Secs. 1.2.2-1.2.3")]


def rfim(route):
    from .meanfield import rfim as RF
    return [text("Model", "H = -(J N/2) Sbar^2 - sum_i h_i s_i, h_i i.i.d. N(0, Delta) (Delta the variance) or +-h0",
                 source=f"{_KZ}, Ch. 2, p. 37"),
            cited("Replica-symmetric free entropy and its saddle (Eqs. (2.1)-(2.2))",
                  r"\Phi = \max_m \left[-\frac{\beta J m^2}{2} + \mathbb{E}_h \ln 2\cosh\beta(h + J m)\right],\quad "
                  r"m = \mathbb{E}_h \tanh\beta(h + J m)", f"{_KZ}, Sec. 2.2, Eq. (2.1); Ex. 2.2"),
            cited("Zero temperature (Eq. (2.3))", r"e_{\min} = \min_m \left[\frac{J m^2}{2} - \mathbb{E}_h |h + J m|\right]",
                  f"{_KZ}, Eq. (2.3)"),
            sym("Gaussian fields above Delta_c (m = 0): e_min = -E|h| (derived here from Eq. (2.3))",
                sp.Eq(sp.Symbol(r"e_{\min}"), -sp.sqrt(2 * D / sp.pi)), source=f"{_KZ}, Eq. (2.3)",
                check=Check((D,), lambda d: RF.rfim_ground_energy(1, RF.field_law(Delta=d)).e,
                            (("0.81",), ("1",), ("2",)), "rhs", tol="1e-9")),
            sym("T = 0 ordering threshold of the Gaussian RFIM, from the zero-temperature saddle equation (derived "
                "here, not printed by KZ)", sp.Eq(sp.Symbol(r"\Delta_c"), 2 * J**2 / sp.pi), source="derived here",
                check=Check((J,), lambda j: RF.rfim_delta_c(j), (("1",), ("0.5",), ("2",)), "rhs")),
            text("Exact checks", "ground states by sorting the fields (k up spins on the k largest fields), the free "
                 "entropy by a Hubbard-Stratonovich quadrature, and Guerra's lower / upper bounds on each sample")]


def random_matrix(route):
    from .rmt import stieltjes as RM
    head = text("Ensembles", "wigner: A = (G + G^T)/sqrt(2N); wishart: Sigma = X X^T / N, X of size M x N, alpha = M/N "
                "(gamma of learning/ridge.py); S(z) = (1/N) sum 1/(lambda_i - z), nu = Im S(x + i0)/pi",
                source=f"{_KZ}, Secs. 3.1-3.2, Ex. 3.1")
    lp = (1 + sp.sqrt(AL))**2
    lm = (1 - sp.sqrt(AL))**2
    mpd = sp.sqrt((lp - X) * (X - lm)) / (2 * sp.pi * AL * X)
    if route == "cavity":
        return [head,
                cited("Cavity recursion (Sec. 3.3) and its Wishart analogue (derived here), iterated as Moebius maps",
                      r"S_{N+1}(z) = -\frac{1}{z + S_N(z)},\quad S \leftarrow \frac{1}{-z + 1/(1 + \alpha S)}",
                      f"{_KZ}, Sec. 3.3"),
                sym("Semicircle from the iterated recursion at z = x + i 1e-30",
                    sp.Eq(sp.Symbol(r"\nu"), sp.sqrt(4 - X**2) / (2 * sp.pi)), source=f"{_KZ}, Sec. 3.2.3",
                    check=Check((X,), lambda x: RM.density("wigner", x, "cavity"), (("0",), ("1",), ("1.5",)), "rhs")),
                sym("Marchenko-Pastur density (alpha = 1/2) from the Wishart recursion",
                    sp.Eq(sp.Symbol(r"\nu"), mpd.subs(AL, sp.Rational(1, 2))), source=f"{_KZ}, Ex. 3.1",
                    check=Check((X,), lambda x: RM.density("wishart", x, "cavity", alpha="1/2"),
                                (("1",), ("0.5",), ("2",)), "rhs"))]
    return [head,
            cited("RS saddle of the replicated determinant (Sec. 3.2.2)",
                  r"\hat q = \frac{1}{q},\ q = -\hat q - \lambda\ \Rightarrow\ S(z) = \frac{-z + \sqrt{z^2 - 4}}{2}",
                  f"{_KZ}, Sec. 3.2.2"),
            sym("Semicircle (Sec. 3.2.3)", sp.Eq(sp.Symbol(r"\nu"), sp.sqrt(4 - X**2) / (2 * sp.pi)),
                source=f"{_KZ}, Sec. 3.2.3",
                check=Check((X,), lambda x: RM.density("wigner", x).mid(), (("0",), ("1",), ("1.5",)), "rhs")),
            sym("Marchenko-Pastur density (Ex. 3.1), the continuous part", sp.Eq(sp.Symbol(r"\nu"), mpd),
                source=f"{_KZ}, Ex. 3.1",
                check=Check((X, AL), lambda x, a: RM.density("wishart", x, alpha=a).mid(),
                            (("1", "0.5"), ("0.5", "0.25"), ("3", "2")), "rhs")),
            sym("Upper edge of the Marchenko-Pastur law (Ex. 3.1)", sp.Eq(sp.Symbol(r"\lambda_+"), lp),
                source=f"{_KZ}, Ex. 3.1",
                check=Check((AL,), lambda a: RM.edges("wishart", alpha=a)[1].mid(), (("0.25",), ("0.5",), ("2",)), "rhs"))]


__all__ = ["curie_weiss", "random_matrix", "rfim"]
