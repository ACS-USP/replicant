"""The `rfim` entry's route `cavity` (K3): the random field Ising model on random graphs (KZ Sec. 4.3, Ex. 4.2) by
population dynamics over a factor law (factor/population.py), checked by BP on sampled graphs (factor/bp.py).

Convention (KZ Ch. 4): H = -J sum_(ij) s_i s_j - sum_i h_i s_i on a random graph, beta the Ising beta, h_i i.i.d.
from the field law of meanfield/rfim.py (gaussian N(0, Delta), Delta the variance, or bimodal +-h0).  graph:
regular (every vertex has c neighbours, excess c - 1) or poisson (Erdos-Renyi, mean degree c).  The fully
connected entry (route replica, H = -(J N/2) Sbar^2 - sum h_i s_i) is the c -> oo limit with the edge coupling
J/c.  State 0 is s = +1; the planted start is every message on s = +1.
Run parameters (route cavity): graph (regular | poisson, default regular), c (default 4), J (default 1), law,
Delta, h0 (as the replica route), beta (required for the population), pop (default 2000), sweeps (default 20),
seeds (default 0-3), N (BP on one sampled graph per seed), iters.
Values: beta_c_zero_field, the Delta = 0 ferromagnetic point (KZ Ex. 4.2.1: (c - 1) tanh beta J = 1 regular,
c tanh beta J = 1 Erdos-Renyi; closed form); m_cavity, phi_cavity (population, statistical; distinct keys from
the replica route's m, phi); m_bp, phi_bp (BP, statistical).
"""
import math

import numpy as np

from .amp._stats import student_band
from .catalog import Out, Stability, fstep, has
from .core import Refusal
from .csp import _checks
from .factor import bp as BP
from .factor import graph as FG
from .factor import population as POP
from .meanfield import rfim as RF

KZ = ("Krzakala & Zdeborova, Statistical Physics Methods in Optimization and Machine Learning, lecture notes "
      "(June 9 2024)")
PARAMS = ("graph", "c", "J", "law", "Delta", "h0", "beta", "pop", "sweeps", "seeds", "N", "iters")


def _law(p):
    kind = p.get("law", "gaussian")
    return RF.field_law(kind, p.get("Delta", "1/2" if kind == "gaussian" else None), p.get("h0"))


def _graph(p):
    g, c = p.get("graph", "regular"), p.get("c", 4)
    if g == "regular":
        if isinstance(c, bool) or not isinstance(c, int) or c < 2:
            raise Refusal("precondition:c", f"c = {c!r}: a regular graph needs an int degree >= 2")
        return g, c, ("regular", c)
    if g == "poisson":
        cf = float(RF._pos(c, "c"))
        return g, cf, ("poisson", cf)
    raise Refusal("unsupported:graph", f"graph = {g!r}: regular | poisson")


def _fields(law):
    if law.kind == "gaussian":
        return lambda rng, n: math.sqrt(law.scale) * rng.standard_normal(n)
    return lambda rng, n: law.scale * rng.choice(np.array([-1.0, 1.0]), n)


def _state(p):
    g, c, _ = _graph(p)
    j = float(RF._pos(p.get("J", 1), "J"))
    branching = c - 1 if g == "regular" else c
    v = {"graph": g, "c": c, "J": j, "law": _law(p).kind, "law_scale": _law(p).scale}
    if branching > 1:
        v["beta_c_zero_field"] = math.atanh(1 / branching) / j
    return Out(v, f"{g} graph, c = {c}, J = {j}; Delta = 0 ferromagnetic point beta_c = atanh(1/{branching})/J"
               if branching > 1 else f"{g} graph, c = {c}: branching <= 1, no zero-field transition")


def _pd(p):
    g, c, dl = _graph(p)
    law, j, beta = _law(p), float(RF._pos(p.get("J", 1), "J")), float(RF._pos(p["beta"], "beta"))
    r = POP.population_dynamics(POP.rfim_law(beta, j, _fields(law)), dl, pop=p.get("pop", 2000),
                                sweeps=p.get("sweeps", 20), seeds=_checks.seeds(p.get("seeds", (0, 1, 2, 3))),
                                init="planted")
    m = float(r.marginal[0] - r.marginal[1])
    return Out({"m_cavity": m, "phi_cavity": r.phi, "pd_status": r.status, "pd_status_note": r.note, "seeds": list(r.seeds)},
               f"population dynamics ({p.get('pop', 2000)} x {p.get('sweeps', 20)} sweeps, planted s = +1): m = "
               f"{m:.6g}, Phi = {r.phi:.8g} ({r.status})", errors={"m_cavity": 2 * float(r.marginal_stderr[0]),
                                                                    "phi_cavity": r.phi_stderr},
               statistical=("m_cavity", "phi_cavity"))


def rfim_graph(n, c, graph, beta, j, law, seed, iters=3000):
    """(m, phi, converged) of BP on one sampled graph with fields from `law`, from the planted (+1) start."""
    rng = np.random.default_rng(seed)
    e = FG.random_regular_edges(n, c, rng) if graph == "regular" else FG.erdos_renyi_edges(n, c, rng)
    s = np.array([1.0, -1.0])
    h = _fields(law)(rng, n)
    g = FG.pairwise(n, 2, e, np.exp(beta * j * np.outer(s, s)), np.exp(beta * np.outer(h, s)))
    st = BP.bp(g, iters=iters, init=("planted", np.zeros(n, dtype=int)))
    b = BP.marginals(g, st)
    return float(np.mean(b[:, 0] - b[:, 1])), BP.bethe_free_entropy(g, st).phi, st.converged


def _bp(p):
    g, c, _ = _graph(p)
    law, j, beta, n = _law(p), float(RF._pos(p.get("J", 1), "J")), float(RF._pos(p["beta"], "beta")), p["N"]
    seeds = _checks.seeds(p.get("seeds", (0, 1, 2, 3)))
    runs = [rfim_graph(n, c, g, beta, j, law, s, p.get("iters", 3000)) for s in seeds]
    bad = [s for s, r in zip(seeds, runs) if not r[2]]
    bm, bp = student_band([r[0] for r in runs]), student_band([r[1] for r in runs])
    return Out({"m_bp": bm.mean, "phi_bp": bp.mean, "N": n, "bp_status": "undetermined" if bad else "converged",
                "bp_status_note": f"BP not converged on seeds {bad}" if bad else ""},
               f"BP on {len(seeds)} sampled graphs (N = {n}): m = {bm.mean:.6g}, Phi = {bp.mean:.8g}",
               errors={"m_bp": bm.stderr, "phi_bp": bp.stderr}, statistical=("m_bp", "phi_bp"))


def cavity_steps():
    return [fstep("rfim-graph-state", f"{KZ}, Sec. 4.3; Ex. 4.2.1", _state),
            fstep("rfim-population", f"{KZ}, Sec. 4.3.2, Eq. (4.15); Ex. 4.2.2", _pd, when=has("beta"),
                  skip="no beta given"),
            fstep("rfim-bp-graphs", f"{KZ}, Sec. 4.2.1 (BP), Sec. 4.3", _bp, when=has("beta", "N"),
                  skip="no beta or no N given")]


def _stat(key, where):
    def f(v):
        und = v.get(key) not in (None, "converged")
        return Stability("not-computed", "finite-N", f"{KZ}, {where}", "statistical over named seeds: NOT final"
                         + (f"; UNDETERMINED: {v.get(key + '_note', '')}" if und else ""))
    return f


FINALITY = {"beta_c_zero_field": Stability("not-computed", "rs", f"{KZ}, Sec. 4.2; Ex. 4.2.1",
                                           "zero-field linear stability of the cavity recursion (physics-level)"),
            "m_cavity": _stat("pd_status", "Sec. 4.3.2"), "phi_cavity": _stat("pd_status", "Sec. 4.3.2"),
            "m_bp": _stat("bp_status", "Sec. 4.2.1"), "phi_bp": _stat("bp_status", "Sec. 4.2.1")}
META = ("graph", "c", "pd_status", "pd_status_note", "bp_status", "bp_status_note")

__all__ = ["FINALITY", "META", "PARAMS", "cavity_steps", "rfim_graph"]
