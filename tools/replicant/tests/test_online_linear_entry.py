"""O2 entry: the online_linear catalog entry, its refusals, its undetermined labels and the GUIDE rows
(split from test_online_linear.py, which keeps the ODE, offline and simulation tests).
"""
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.catalog import catalog, check_ref, run_cached
from replicant.compose import compose
from replicant.core import Refusal
from replicant.online import linear as L
from replicant.online import linear_offline as OFF


def _refuses(fn, reason):
    try:
        fn()
    except Refusal as r:
        assert r.reason == reason, (r.reason, r.detail)
        return r
    raise AssertionError(f"expected Refusal {reason}")


@contextmanager
def replaced(module, name, fn):
    """module.<name> = fn for the block; the original object is restored (identity asserted)."""
    saved = getattr(module, name)
    try:
        setattr(module, name, fn)
        yield
    finally:
        setattr(module, name, saved)
        assert getattr(module, name) is saved


def _undetermined(reason):
    def fn(*a, **k):
        raise Refusal(reason, "planted")
    return fn


# ---- refusals ----------------------------------------------------------------------------------
def test_refusals():
    _refuses(lambda: L.linear_flow("hebb"), "unsupported:rule")
    _refuses(lambda: L.linear_flow("standard", eta=0), "precondition:eta")
    _refuses(lambda: L.linear_flow("standard", eta="fast"), "precondition:eta")
    _refuses(lambda: L.linear_flow("optimal", eta=0.5), "precondition:optimal-eta")
    _refuses(lambda: L.linear_flow("optimal", eta="greedy"), "precondition:optimal-eta")
    _refuses(lambda: L.linear_flow("standard", constraint="ball"), "precondition:constraint")
    _refuses(lambda: L.linear_flow("standard", M=0), "precondition:teacher-norm")
    _refuses(lambda: L.linear_flow("standard", averages="mc"), "precondition:averages")
    f = L.linear_flow("standard", constraint="sphere")
    _refuses(lambda: L.integrate_linear(f, 1.0), "precondition:state-point")
    _refuses(lambda: L.integrate_linear(f, 1.0, (0.0, 0.0)), "precondition:initial-overlap")
    _refuses(lambda: L.integrate_linear(f, 1.0, (2.0, 1.0)), "precondition:initial-overlap")
    _refuses(lambda: L.integrate_linear(f, -1.0, (0.0, 1.0)), "precondition:alpha")
    _refuses(lambda: L.integrate_linear(L.linear_flow("optimal"), 1.0, (0.0, 1.0)),
             "precondition:initial-overlap")
    _refuses(lambda: L.integrate_linear(L.linear_flow("standard", eta="greedy"), 1.0, (0.0, 0.0)),
             "precondition:initial-overlap")
    _refuses(lambda: OFF.offline(0.5, "adaline"), "unsupported:algorithm")
    _refuses(lambda: OFF.offline(0, "standard"), "precondition:alpha")
    _refuses(lambda: OFF.offline(0.5, "standard", M=1, Q=0.25), "precondition:offline-window")
    _refuses(lambda: OFF.offline(0.5, "standard", constraint="ball"), "precondition:constraint")


def test_catalog_entry_runs_and_refuses():
    e = catalog()["online_linear"]
    v = run_cached(e, "online", {})[0]                           # every default (generic tables)
    assert v["rule"] == "optimal" and v["constraint"] == "sphere"
    v = e.run(route="online", alpha="0.5")
    assert abs(v["rho"] - math.sqrt(1 - math.exp(-0.5))) < 1e-5
    assert abs(v["rho_offline_standard"] - 0.5) < 1e-12 and abs(v["rho_offline_optimal_ridge"]
                                                                - math.sqrt(0.5)) < 1e-12
    assert not e.finality["rho"].final and not e.finality["rho_offline_optimal"].final
    _refuses(lambda: compose(e.spec, "online").run(rul="optimal"), "precondition:known-parameter")
    _refuses(lambda: e.run(route="online", rule="standard", eta="greedy", constraint="none",
                           R0=0, Q0=0, alpha=1), "precondition:initial-overlap")
    _refuses(lambda: e.run(route="online", rule="hebb"), "unsupported:rule")
    _refuses(lambda: e.run(route="online", constraint="ball"), "precondition:constraint")
    _refuses(lambda: e.run(route="online", rule=lambda b, h, Q, R: b - h), "unsupported:rule")
    for ref in e.refs[:2]:                     # test_catalog checks all 13 in the default tier
        out = check_ref(e, ref)
        assert out.ok, out


def test_guide_is_true():
    """The GUIDE's online_linear rows: every refusal reason listed is raised by the code, and the
    worked example's numbers are what the CLI prints."""
    import io
    import json

    from replicant.cli import main
    root = Path(__file__).resolve().parent.parent
    guide = (root / "GUIDE.md").read_text()
    assert "| `online_linear`" in guide and "`online_linear` (online): `rule`" in guide
    code = "".join(p.read_text() for p in [*(root / "online").glob("linear*.py"),
                                           root / "online" / "simulate_linear.py",
                                           root / "catalog_online_linear.py"])
    for reason in ("unsupported:rule", "unsupported:algorithm", "precondition:eta",
                   "precondition:optimal-eta", "precondition:constraint", "precondition:teacher-norm",
                   "precondition:averages", "precondition:state-point", "precondition:initial-overlap",
                   "precondition:alpha", "precondition:asymptote", "precondition:offline-window",
                   "undetermined:asymptote", "undetermined:crossing", "no-solution:rho-zero",
                   "precondition:method", "undetermined:descent"):
        assert f'"{reason}"' in code, reason
    out = io.StringIO()
    assert main(["run", "online_linear", "--route", "online", "--param", "alpha=0.5"], out,
                io.StringIO()) == 0
    v = json.loads(out.getvalue())["values"]
    assert "alpha_cross" not in v and "rho_inf" not in v
    v.update(run_cached(catalog()["online_linear"], "online", {})[0])     # the run without alpha
    for key, fmt in (("rho", ".6f"), ("gamma_temperature", ".6f"), ("alpha_cross", ".6f"),
                     ("rho_offline_optimal_ridge", ".6f"), ("tail_constant", ".7f")):
        assert f"`{key}` {v[key]:{fmt}}" in guide, (key, v[key])


def test_undetermined_asymptote_and_crossing_are_labelled_not_refused():
    """A diagnostic that cannot decide (the ODE not converged at alpha = 60, or not exactly one sign
    change of rho_online - rho_offline on the grid) labels asymptote_status / crossing_status and omits
    the value; the run is not refused and the other values are kept.  No real input reaches either
    branch, so both are planted; the library functions still refuse by name."""
    e = catalog()["online_linear"]
    with replaced(L, "asymptote", _undetermined("undetermined:asymptote")):
        res = compose(e.spec, "online").run()
    v = res.values
    assert v["asymptote_status"].startswith("undetermined:asymptote: planted")
    assert not {"rho_inf", "eg_inf", "tail_constant"} & set(v) and "alpha_cross" in v
    assert any("asymptote UNDETERMINED" in n for n in res.notes) and res.final is False
    with replaced(OFF, "crossing_alpha", _undetermined("undetermined:crossing")):
        res = compose(e.spec, "online").run()
    v = res.values
    assert v["crossing_status"].startswith("undetermined:crossing: planted")
    assert "alpha_cross" not in v and "rho_inf" in v and "asymptote_status" not in v
    with replaced(L, "asymptote", _undetermined("precondition:asymptote")):    # other reasons still refuse
        _refuses(lambda: compose(e.spec, "online").run(), "precondition:asymptote")
    v = compose(e.spec, "online").run().values                                 # restored: computed again
    assert "rho_inf" in v and "alpha_cross" in v
