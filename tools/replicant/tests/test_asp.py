"""K5: approximate survey propagation - the Parisi 1RSB equations of SK as a state evolution (KZ Sec. 12.2.5, Eqs.
(12.63)-(12.65); Antenucci, Krzakala, Urbani & Zdeborova, 'Approximate survey propagation for statistical inference',
arXiv:1807.01296, Sec. IV B; Antenucci, Franz, Urbani & Zdeborova, PRX 9, 011020 (2019), arXiv:1805.05857).

At fixed Parisi parameter m the ASP state evolution maps (q0, q1) to (E_U <tanh>^2, E_U <tanh^2>), <g> the cosh^m
reweighted V-average at Xi = beta (h + sqrt(q0) U + sqrt(q1 - q0) V): its fixed points are the q0, q1 stationarity
conditions of the 1RSB free energy, which families/sk.py `sk_1rsb` solves together with the m equation by Newton on a
Gauss-Hermite grid.  amp/asp.py iterates the map on a trapezoid grid and imports nothing from families/.
"""
import ast
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.amp import asp as A
from replicant.core import Refusal
from replicant.families.sk import sk_1rsb

PKG = Path(__file__).resolve().parents[1]


def _planted(key, value, fn):
    old = A._HOOKS[key]
    A._HOOKS[key] = value
    try:
        return fn()
    finally:
        A._HOOKS[key] = old
        assert A._HOOKS[key] is old


def test_asp_reproduces_sk_1rsb():
    """beta = 2 and 3, h = 0, m from sk_1rsb: the ASP fixed point equals sk_1rsb's (q0, q1).  Tolerances: sk.py states
    its grid error (~1e-11 at beta = 2; (q0, q1, m) to ~1e-6 at beta = 3, from its 80- vs 160-node comparison); the
    trapezoid of asp.py is converged to ~1e-12 (its fixed point is one of the h/2 map too, checked below), so the
    bands are 1e-9 and 3e-6.  Control: the plain (m = 1, un-reweighted) V-average, i.e. dropping cosh^m, misses q1
    by > 1e-2."""
    for beta, tol in ((2, 1e-9), (3, 3e-6)):
        s = sk_1rsb(beta)
        r = A.asp_fixed_point(beta, float(s.m))
        assert r.converged, r
        assert abs(r.q0 - float(s.q0)) < tol and abs(r.q1 - float(s.q1)) < tol, (beta, r, s)
        fine = A.asp_fixed_point(beta, float(s.m), grid=2, start=(r.q0, r.q1))
        assert abs(fine.q0 - r.q0) < 1e-12 and abs(fine.q1 - r.q1) < 1e-12, (fine, r)
    s = sk_1rsb(2)
    bad = _planted("reweight", False, lambda: A.asp_fixed_point(2, float(s.m), max_iter=60))
    assert abs(bad.q1 - float(s.q1)) > 1e-2, bad


def test_asp_q0_zero_branch_and_refusals():
    """At h = 0, q0 = 0 is an exact fixed point of the q0 map (the reweighted <tanh> is odd); started there ASP stays
    on it and q1 solves q1 = <tanh^2> alone.  Bad inputs are named refusals."""
    r = A.asp_fixed_point(2, 0.3, start=(0.0, 0.6))
    assert r.converged and r.q0 == 0 and 0.5 < r.q1 < 1, r
    for bad, reason in (((2, 0), "precondition:parisi-m"), ((2, 1.5), "precondition:parisi-m"),
                        ((-1, 0.3), "precondition:beta")):
        try:
            A.asp_fixed_point(*bad)
        except Refusal as e:
            assert e.reason == reason, (bad, e.reason)
        else:
            raise AssertionError(bad)


def test_asp_imports_nothing_from_families():
    """The check is independent: amp/asp.py reaches neither families/ nor sk_1rsb (AST audit)."""
    tree = ast.parse((PKG / "amp" / "asp.py").read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mods.add(("." * node.level) + (node.module or "") + "".join(f":{a.name}" for a in node.names))
        elif isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
    assert not any("famil" in m or "sk_1rsb" in m or m.endswith(".sk") or ":sk" in m for m in mods), mods
