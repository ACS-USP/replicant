"""Displayed equations of the K2 catalog entries `scalar_denoising` and `sparse_denoising` (KZ Ch. 7).  Same rule as
latex_entries.py: each sympy block carries a tex.Check against bayes/ at three points (tests/test_latex.py).
"""
import mpmath as mp
import sympy as sp

from .gauss import GaussE
from .tex import Check, cited, sym, text

_KZ = "Krzakala & Zdeborova, lecture notes (June 9 2024)"
_GSV = "Guo, Shamai & Verdu, IEEE Trans. Inf. Theory 51, 1261 (2005)"
S, D, Z = sp.symbols("snr Delta z", positive=True)
_PTS = (("0.5",), ("1",), ("3.7",))


def _ch():
    from .bayes import channels as CH
    return CH


def scalar_denoising(route):
    CH = _ch()
    g, r = CH.Prior("gaussian"), CH.Prior("rademacher")
    return [text("Channel", "y_i = x* + sqrt(Delta) z_i, i = 1..N (z_i ~ N(0, 1)); the sum is sufficient and is the channel "
                 "y = sqrt(snr) x + z at snr = N/Delta", source=f"{_KZ}, Eq. (7.4); {_GSV}, Eq. (3)"),
            cited("Posterior for N observations: a mixture prior w_k N(mu_k, v_k) (atoms: v_k = 0) stays a mixture "
                  "(Eqs. (7.7), (7.9), (7.10) are its three cases)",
                  r"P(x \mid y) \propto \sum_k w_k\, \mathcal{N}\!\left(\bar y; \mu_k, v_k + \tfrac{\Delta}{N}\right)"
                  r"\mathcal{N}\!\left(x; \mu_k + \frac{v_k (\bar y - \mu_k)}{v_k + \Delta/N}, "
                  r"\frac{v_k \Delta/N}{v_k + \Delta/N}\right)", f"{_KZ}, Eqs. (7.5)-(7.10)"),
            sym("Gaussian prior: MMSE (posterior variance, Eq. (7.18))", sp.Eq(sp.Symbol("mmse"), 1 / (1 + S)),
                source=f"{_GSV}, Eq. (13)",
                check=Check((S,), lambda s: CH.mmse_ball(g, s).mid(), _PTS, "rhs")),
            sym("Gaussian prior: mutual information (nats)", sp.Eq(sp.Symbol("I"), sp.log(1 + S) / 2),
                source=f"{_GSV}, Eq. (11)",
                check=Check((S,), lambda s: CH.information_ball(g, s).mid(), _PTS, "rhs")),
            sym("+-1 prior: MMSE, E tanh = E tanh^2 by Nishimori (KZ Theorem 13)",
                sp.Eq(sp.Symbol("mmse"), 1 - GaussE(sp.tanh(S + sp.sqrt(S) * Z)**2, Z)), source=f"{_GSV}, Eq. (17)",
                check=Check((S,), lambda s: CH.mmse_ball(r, s).mid(), _PTS, "rhs")),
            sym("+-1 prior: mutual information (nats)",
                sp.Eq(sp.Symbol("I"), S - GaussE(sp.log(sp.cosh(S + sp.sqrt(S) * Z)), Z)), source=f"{_GSV}, Eq. (18)",
                check=Check((S,), lambda s: CH.information_ball(r, s).mid(), _PTS, "rhs")),
            cited("I-MMSE (Theorem 1 of GSV; KZ Theorem 14 with snr = 1/Delta), checked as a validator",
                  r"\frac{d I}{d\,\mathrm{snr}} = \frac{1}{2}\,\mathrm{mmse}(\mathrm{snr})", f"{_GSV}, Eq. (15)")]


def sparse_denoising(route):
    from .bayes import sparse_vector as SV
    beta = sp.sqrt(2 / D)
    null = -sp.log(2) - 1 / (2 * D) + beta * sp.sqrt(sp.log(2))      # the frozen REM, beta > beta_c
    return [text("Model", "x* one of the d = 2^N basis vectors, y = x* + sqrt(Delta/N) z; Phi = lim (1/N) E log Z with Z "
                 "the likelihood ratio of Eq. (7.54)", source=f"{_KZ}, Eqs. (7.51)-(7.55)"),
            sym("Theorem 15, Delta below Delta_c (above it f = 0)", sp.Eq(sp.Symbol("f"), 1 / (2 * D) - sp.log(2)),
                source=f"{_KZ}, Theorem 15, Eq. (7.57)",
                check=Check((D,), lambda d: SV.sparse_vector_free_entropy(d), (("0.3",), ("0.5",), ("0.7",)), "rhs")),
            sym("The threshold", sp.Eq(sp.Symbol(r"\Delta_c"), 1 / (2 * sp.log(2))), source=f"{_KZ}, Sec. 7.3, p. 126",
                check=Check((), lambda: SV.delta_c(), ((),), "rhs")),
            sym("REM route: the d - 1 null terms are a REM at beta = sqrt(2/Delta), N_scale = 1/2; frozen below "
                "Delta_c, the null part is negative (derived here), so the planted term wins",
                sp.Eq(sp.Symbol(r"\phi_{null}"), null), source=f"{_KZ}, Eq. (7.54); Derrida, PRB 24, 2613 (1981)",
                check=Check((D,), _null, (("0.3",), ("0.5",), ("0.7",)), "rhs")),
            cited("Replica-symmetric saddles (App. 7.B): all replicas on the true index, or spread",
                  r"\lim \tfrac{1}{N} \log \mathbb{E} Z^n \in \left\{ n\left(\tfrac{1}{2\Delta} - \log 2\right),\ 0 "
                  r"\right\},\quad f = \max", f"{_KZ}, Eqs. (7.92)-(7.93)")]


def _null(d):
    from .bayes import sparse_vector as SV
    from .families.rem import rem_free_energy
    beta, s = SV.rem_parameters(d)
    with mp.workdps(30):
        return -mp.log(2) - 1 / (2 * SV._delta(d)) - beta * rem_free_energy(beta, N_scale=s)


__all__ = ["scalar_denoising", "sparse_denoising"]
