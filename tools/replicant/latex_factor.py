"""Displayed equations of the K3 catalog entries `potts_bethe`, `coloring` and the `rfim` route cavity (KZ Ch. 4-6).
Same rule as latex_entries.py: each sympy block carries a tex.Check against coloring/ or catalog_rfim_sparse.py at
three points (tests/test_latex.py); the BP and Bethe formulas of Ch. 4-5 are cited strings.
"""
import sympy as sp

from .tex import Check, cited, sym, text

_KZ = "Krzakala & Zdeborova, lecture notes (June 9 2024)"
Q, C, B, D, A, CT, J = sp.symbols("q c beta d a ctilde J", real=True)
TH = 1 - sp.exp(-B)


def _pb():
    from .coloring import potts_bp as PB
    return PB


def _root(q, d, b):
    """(q, d, beta, a*) at the stable ferromagnetic root of Eq. (6.11) (a point for the checks)."""
    PB = _pb()
    f = max((f for f in PB.potts_ferro_fixed_points(int(q), int(d), b) if f.stable), key=lambda f: f.a)
    return q, d, b, repr(f.a)


def _convention():
    return text("Convention", "P(s) ~ prod over edges of exp(-beta delta(s_i, s_j)): beta > 0 antiferromagnetic "
                "(coloring, beta = oo counts proper colorings), beta < 0 ferromagnetic; beta_Potts = -2 beta_Ising "
                "(s s' = 2 delta - 1); c the mean degree, d the degree of a d-regular graph, c~ the mean excess degree",
                source=f"{_KZ}, Ch. 6, Sec. 6.5.1")


def _bp_bethe():
    return [cited("BP on a factor graph (KZ Sec. 5.2.2)",
                  r"\chi^{j\to a}_{s} \propto g_j(s) \prod_{b \in \partial j \setminus a} \psi^{b \to j}_s,\quad "
                  r"\psi^{a\to i}_{s_i} \propto \sum_{s_{\partial a \setminus i}} f_a(s_{\partial a}) "
                  r"\prod_{j \in \partial a \setminus i} \chi^{j \to a}_{s_j}", f"{_KZ}, Sec. 5.2.2"),
            cited("Bethe free entropy (Eq. (5.1)): exact on trees, an approximation on loopy graphs",
                  r"N \Phi = \sum_i \ln Z^i + \sum_a \ln Z^a - \sum_{(ia)} \ln Z^{ia}", f"{_KZ}, Eq. (5.1)")]


def _para():
    PB = _pb()
    return sym("Paramagnetic fixed point chi = 1/q (Sec. 6.4): the Bethe (= annealed) free entropy",
               sp.Eq(sp.Symbol(r"\Phi_{\rm para}"), sp.log(Q) + C / 2 * sp.log(1 - TH / Q)), source=f"{_KZ}, Sec. 6.4",
               check=Check((Q, C, B), lambda q, c, b: PB.potts_paramagnetic(int(q), c, b).phi,
                           (("3", "5", "-1/2"), ("4", "6", "1"), ("2", "3", "-7/10")), "rhs"))


def potts_bethe(route):
    PB = _pb()
    A_, B_ = 1 - TH * A, 1 - TH * (1 - A) / (Q - 1)
    rhs = A_**(D - 1) / (A_**(D - 1) + (Q - 1) * B_**(D - 1))
    phi = sp.log((Q - 1) * B_**D + A_**D) - D / 2 * sp.log(1 - TH * ((1 - A)**2 / (Q - 1) + A**2))
    pts = tuple(lambda p=p: _root(*p) for p in (("3", "5", "-7/10"), ("2", "5", "-7/10"), ("6", "5", "-1")))
    return [_convention(), *_bp_bethe(), _para(),
            sym("Ferromagnetic ansatz chi_1 = a on d-regular graphs: the fixed-point equation (Eq. (6.11))",
                sp.Eq(A, rhs), source=f"{_KZ}, Eqs. (6.9)-(6.11)",
                check=Check((Q, D, B, A), lambda q, d, b, a: float(a), pts, "rhs")),
            sym("Bethe free entropy of the ansatz (Eq. (6.12)), evaluated at its stable roots",
                sp.Eq(sp.Symbol(r"\Phi"), phi), source=f"{_KZ}, Eq. (6.12)",
                check=Check((Q, D, B, A), lambda q, d, b, a: max(
                    (f for f in PB.potts_ferro_fixed_points(int(q), int(d), b) if f.stable), key=lambda f: f.a).phi,
                    pts, "rhs")),
            sym("Stability of chi = 1/q (Eqs. (6.13), (6.22); c~ = d - 1 on d-regular graphs)",
                sp.Eq(sp.Symbol(r"\beta_{\rm stab}"), -sp.log(1 + Q / (CT - 1))), source=f"{_KZ}, Eqs. (6.13), (6.22)",
                check=Check((Q, CT), lambda q, ct: PB.potts_beta_stab(int(q), float(ct)),
                            (("2", "3"), ("3", "4"), ("10", "5")), "rhs")),
            text("First-order structure (q >= 3)", "beta_s: the saddle node RHS(a) = a, RHS'(a) = 1 at a > 1/q; beta_c: "
                 "Phi(a_ferro) = Phi(1/q); reported only when the planted (a = 1) and the random (a = 1/q + 0.01) "
                 "iterations of Eq. (6.11) both converge at beta_c; beta_stab < beta_c < beta_s",
                 source=f"{_KZ}, Sec. 6.5.2")]


def coloring(route):
    PB = _pb()
    return [_convention(), *_bp_bethe(), _para(),
            sym("Annealed threshold (Eq. (6.17)): the paramagnetic / annealed entropy vanishes",
                sp.Eq(sp.Symbol(r"c_{\rm ann}"), -2 * sp.log(Q) / sp.log(1 - TH / Q)), source=f"{_KZ}, Eq. (6.17)",
                check=Check((Q, B), lambda q, b: PB.coloring_annealed_threshold(int(q), b),
                            (("3", "2"), ("4", "1/2"), ("5", "3")), "rhs")),
            sym("Kesten-Stigum (Eq. (6.24)): the variance of BP perturbations grows when c~ lambda_max^2 > 1",
                sp.Eq(sp.Symbol(r"\tilde c_{\rm KS}"), (Q - TH)**2 / TH**2), source=f"{_KZ}, Eqs. (6.19)-(6.24)",
                check=Check((Q, B), lambda q, b: PB.kesten_stigum(int(q), b), (("3", "2"), ("4", "1/2"), ("5", "3")),
                            "rhs")),
            text("Limits", "beta = oo: c_ann = 2 ln q / (-ln(1 - 1/q)) and c~_KS = (q - 1)^2; for q = 3 on ER graphs "
                 "(c~ = c) c_KS = 4 < c_ann = 5.42 (printed 5.4)", source=f"{_KZ}, Sec. 6.6")]


def rfim_cavity():
    from . import catalog_rfim_sparse as RS
    return [text("Model", "H = -J sum over edges s_i s_j - sum_i h_i s_i on a c-regular or Erdos-Renyi graph, "
                 "h_i ~ N(0, Delta); the fully connected entry is the c -> oo limit with coupling J/c",
                 source=f"{_KZ}, Sec. 4.3"),
            cited("Cavity recursion (Eq. (4.15)), solved by population dynamics",
                  r"Q^{\rm cav}(m) = \int dh_0\, \mathcal N(h_0; 0, \Delta) \prod_{i=1}^{c-1} \int dm_i\, "
                  r"Q^{\rm cav}(m_i)\, \delta\!\left(m - \tanh\!\left(\beta h_0 + \sum_i "
                  r"{\rm atanh}(m_i \tanh \beta J)\right)\right)", f"{_KZ}, Eqs. (4.14)-(4.15)"),
            *_bp_bethe(),
            sym("Zero-field ferromagnetic point on c-regular graphs (Ex. 4.2.1: (c - 1) tanh beta J = 1)",
                sp.Eq(sp.Symbol(r"\beta_c"), sp.atanh(1 / (C - 1)) / J), source=f"{_KZ}, Sec. 4.2; Ex. 4.2.1",
                check=Check((C, J), lambda c, j: RS._state({"c": int(c), "J": j}).values["beta_c_zero_field"],
                            (("4", "1"), ("3", "1/2"), ("6", "2")), "rhs"))]


__all__ = ["coloring", "potts_bethe", "rfim_cavity"]
