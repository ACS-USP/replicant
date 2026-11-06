# Automated Tests

`replicant` has its own battery runner, `tools/replicant/selftest.py`. It discovers the
`tests/test_*.py` functions without pytest and exits 0 only if every test passes.

## Tiers

| Tier | Command | Size | Time |
|---|---|---|---|
| Default | `python3 tools/replicant/selftest.py` | 748 tests | about 30 s (the 662 before the margin law and pinned directions measured 30.2 s cold) |
| Full | `python3 tools/replicant/selftest.py --full` | adds 231 `@slow` tests (979 in all) | about 35 min (measured 34 min 39 s with 859 tests; not re-measured since) |
| Repository | `python3 run_selftests.py replicant` | the default tier, through the repo runner | — |

`@slow` (`testing.py`) is a plain function attribute that only `selftest.py` honours. **pytest
ignores it and would run every test, the slow ones included.**

The slow tier contains:
- full RSB (the K = 3 RSB test alone takes about 7 minutes);
- population dynamics at reference sizes;
- the K-SAT and LDPC crossings;
- the Monte Carlo and enumeration held-out routes;
- the online simulations at N = 1000;
- the lattice, percolation and Monte Carlo checks at reference sizes.

## Test map

| Area | Files |
|---|---|
| Core, spec, composer, CLI | `test_core.py`, `test_compose.py`, `test_compose_rsb.py`, `test_cli.py`, `test_catalog.py`, `test_refdata.py` |
| Level (a) | `test_derive.py`, `test_derive_ising.py`, `test_derive_rsb.py`, `test_solve*.py`, `test_replicon_derived.py`, `test_gauss*.py` |
| Families | `test_sk.py`, `test_rem_pspin.py`, `test_perceptron*.py`, `test_hopfield.py`, `test_parisi.py`, `test_gardner_transition.py`, `test_margin_law.py`, `test_pinned.py`, `test_dense_memory.py` |
| Cavity, CSP, inference | `test_cavity_*.py`, `test_xorsat.py`, `test_ksat.py`, `test_ldpc.py`, `test_spiked.py`, `test_sbm.py` |
| AMP | `test_amp_sk.py`, `test_amp_route.py`, `test_gamp.py` |
| Lattices, bounds, series | `test_lattice*.py`, `test_guerra.py`, `test_series.py` |
| Online learning | `test_online_*.py` |
| Percolation | `test_percolation*.py` |
| Monte Carlo | `test_mc_*.py`, `test_pt.py`, `test_enumeration.py` |
| Held-out | `test_heldout*.py` (helpers `heldout_common.py`, `heldout_mc.py`) |
| LaTeX | `test_latex.py` |

## Conventions

- **Held-out tests** check composed models (not catalog entries) by routes that share no code with them: Monte Carlo, parallel tempering and exact enumeration (`numerics/`).
- **Planted controls**:
  - every check that could pass vacuously has a planted error it must reject;
  - a hook is replaced, the check is asserted to fail, and the hook is restored with an identity assert.
- **Power**: statistical tests state their power against the planted error and are sized for power of at least 0.9.
- **Estimator-property tests** may use undetermined runs, but they are named and documented as such, and no run-path value may come from them.
- **python-flint**: run the battery from the project venv. Without it, the certified lattice, bounds and percolation tests and the Monte Carlo exact targets fail loudly (`unsupported:python-flint`), by design.

## Related pages

- [Reference Store and Provenance](Reference-Store-and-Provenance)
- [Certified and Statistical Computation](Certified-and-Statistical-Computation)
- [Troubleshooting](Troubleshooting)
