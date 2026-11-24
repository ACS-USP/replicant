"""Catalog: entries, routes, and the banked numbers each entry reproduces.

Conventions: every catalog entry runs its level-(b) family pipeline (or its cavity
pipeline) through core.run_pipeline; a RefCheck names a banked key in
reference/<file>.json, the run parameters at which the entry reproduces it, and how the
compared number is read off the run's values.  Published numbers are never retyped
here: the comparison is refdata.check against the bank.  Statistical (population-
dynamics) keys are compared in a 99% Student-t band (cavity/_stats.t_quantile) on
hypot(stderr, banked tol), in the slow tier only.
"""
import json
import os
import sys
from pathlib import Path

import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.catalog import UNCLAIMED, RefCheck, catalog, check_ref
from replicant.testing import slow

BANK = Path(__file__).resolve().parent.parent / "reference"

ROUTES = {"rem": {"replica"}, "sk": {"replica", "cavity", "amp"}, "pspin": {"replica"},
          "ising_pspin": {"replica"}, "xorsat": {"cavity"}, "ksat": {"cavity"}, "ldpc_bec": {"cavity"},
          "spiked_wigner": {"amp"}, "sbm": {"cavity"}, "ridge": {"replica"}, "series_ising2d": {"series"},
          "ising2d": {"exact"}, "ising1d": {"exact"}, "bethe_ising": {"cavity"},
          "ising2d_anisotropic": {"exact"}, "ising_triangular": {"exact"}, "ising_honeycomb": {"exact"},
          "sk_bounds": {"bound"}, "online_perceptron": {"online"}, "online_linear": {"online"}, "online_drift": {"online"},
          "online_tcm": {"online"}, "online_scm": {"online"}, "perceptron": {"replica", "cavity", "rsb"}, "teacher_student": {"replica", "amp"},
          "hopfield": {"replica", "cavity"}, "viana_bray": {"cavity"},
          "percolation_bethe": {"cavity", "sample"}, "percolation_graph": {"cavity", "sample"},
          "percolation_lattice": {"exact", "sample"}, "percolation_kcore": {"cavity", "sample"},
          "percolation_bootstrap": {"cavity", "sample"}, "cluster_mc": {"sample"},
          "wang_landau_mc": {"sample"}, "histogram_reweighting_mc": {"sample"},
          "population_annealing_mc": {"sample"}, "dense_memory": {"replica"}, "dense_memory_ising": {"replica"},
          "curie_weiss": {"replica", "cavity", "exact"}, "rfim": {"replica", "cavity"},
          "random_matrix": {"replica", "cavity"}, "scalar_denoising": {"exact"}, "sparse_denoising": {"replica"},
          "potts_bethe": {"cavity"}, "coloring": {"cavity"}, "spiked_wishart": {"amp"}}


def _bank_keys():
    return {(p.stem, k) for p in BANK.glob("*.json") for k in json.loads(p.read_text())}


def test_catalog_lists_the_v1_entries_with_their_routes():
    cat = catalog()
    assert {n: e.routes for n, e in cat.items()} == ROUTES
    for name, e in cat.items():
        assert e.name == name and e.cite and e.spec.family and e.refs, name
        for route in e.routes:
            steps = e.route_steps[route]
            assert steps and all(s.cite for s in steps), (name, route)


def test_catalog_entry_reproduces_its_banked_numbers():
    """Default tier (ruling R5): every non-slow, non-statistical key of every entry."""
    before = mp.mp.dps
    for name, e in catalog().items():
        fast = [r for r in e.refs if not (r.slow or r.statistical)]
        assert fast, f"{name}: no default-tier banked check"
        for ref in fast:
            out = check_ref(e, ref)
            assert out.ok, (name, ref.key, out)
    assert mp.mp.dps == before


def test_check_ref_can_fail():
    """Positive control: the comparison is not vacuous (wrong quantity, wrong point)."""
    sk = catalog()["sk"]
    assert not check_ref(sk, RefCheck("sk", "tc", lambda v: v["rs_ground_energy"])).ok
    rem = catalog()["rem"]
    wrong = RefCheck("rem", "ground_energy_derrida", lambda v: v["f"], params={"beta": 1})
    assert not check_ref(rem, wrong).ok          # paramagnet at beta = 1: not the ground energy


def test_statistical_detail_names_a_systematic_only_when_there_is_one():
    """Final review m-11: a statistical RefCheck without a systematic prints no '+ systematic 0' (compare on
    synthetic values: the banked value with a stderr; P2's FSS RefChecks keep their systematic text)."""
    from replicant.catalog import compare
    from replicant.catalog_percolation_fss import BAND_P
    from replicant.refdata import load
    e = catalog()["percolation_graph"]
    ref = next(r for r in e.refs if r.statistical)
    want = float(load(ref.file)[ref.key].value)
    out = compare(e, ref, {"P_inf_sampled": want}, {"P_inf_sampled": 1e-3})
    assert out.ok and out.statistical and "systematic" not in out.detail, out.detail
    lat = catalog()["percolation_lattice"]
    pc = next(r for r in lat.refs if r.statistical)
    v = {"p_c_crossing": float(load(pc.file)[pc.key].value), "fss_bands": {BAND_P: (0, 0, 0, 0.002, 0)}}
    assert "+ systematic 0.002" in compare(lat, pc, v, {"p_c_crossing": 1e-3}).detail


@slow
def test_catalog_entry_reproduces_its_slow_and_statistical_numbers():
    for name, e in catalog().items():
        for ref in e.refs:
            if ref.slow or ref.statistical:
                out = check_ref(e, ref)
                assert out.ok, (name, ref.key, out)


def test_every_ref_key_exists_in_the_bank():
    keys = _bank_keys()
    for name, e in catalog().items():
        for ref in e.refs:
            assert (ref.file, ref.key) in keys, (name, ref.file, ref.key)
            assert ref.route in e.routes, (name, ref.key, ref.route)
            if ref.statistical:
                assert ref.slow and ref.error is not None, (name, ref.key)


# the 38 names catalog() listed, in order, before K1 moved the registry into catalog_index.py (HEAD 9db72fa)
_PRE_K1 = ("rem", "sk", "pspin", "ising_pspin", "perceptron", "teacher_student", "hopfield", "viana_bray", "xorsat",
           "ksat", "ldpc_bec", "spiked_wigner", "sbm", "ridge", "series_ising2d", "ising2d", "ising1d", "bethe_ising",
           "ising2d_anisotropic", "ising_triangular", "ising_honeycomb", "sk_bounds", "online_perceptron",
           "online_linear", "online_drift", "online_tcm", "online_scm", "percolation_bethe", "percolation_graph",
           "percolation_lattice", "percolation_kcore", "percolation_bootstrap", "cluster_mc", "wang_landau_mc",
           "histogram_reweighting_mc", "population_annealing_mc", "dense_memory", "dense_memory_ising")


def test_registry_move_is_identity():
    """K1 moved the import-and-concatenate block of catalog() into catalog_index.entries(): the 38 entries keep
    their order (later tasks append after them), and catalog() is exactly the registry, keyed by name."""
    from replicant import catalog_index
    names = list(catalog())
    assert tuple(names[:len(_PRE_K1)]) == _PRE_K1, names
    reg = catalog_index.entries()
    assert [e.name for e in reg] == names and len(set(names)) == len(names)
    assert all(catalog()[e.name].name == e.name for e in reg)


def test_unclaimed_bank_keys_are_listed():
    """Guard against bank rot: every banked key is claimed by an entry or allowlisted."""
    keys = _bank_keys()
    claimed = {(r.file, r.key) for e in catalog().values() for r in e.refs}
    unclaimed = set(UNCLAIMED)
    assert keys == claimed | unclaimed, (keys - claimed - unclaimed, (claimed | unclaimed) - keys)
    assert not claimed & unclaimed, claimed & unclaimed
    assert all(isinstance(v, str) and len(v) > 20 for v in UNCLAIMED.values())
