# Automated Tests

`replicant` has its own battery runner, `tools/replicant/selftest.py`. It discovers the
`tests/test_*.py` functions without pytest and exits 0 only if every test passes.

## Tiers

| Tier | Command | Size | Time |
|---|---|---|---|
| Default | `python3 tools/replicant/selftest.py` | 874 tests | about 70 s (measured 69.70 s at 55fd208, load averages 4.93 at the start and 3.64 at the end; the 850 before the finite-N checks: 62.92 s, load 3.4-4.5; K8's 841 tests: 60.57 s / 60.56 s on clean exports, load 3.5-4.3) |
| Full | `python3 tools/replicant/selftest.py --full` | adds 273 `@slow` tests (1147 in all) | about 48 min (last timed at 1a37a65, 1120 tests: 1120 passed, 48 min 04 s, under load: load averages about 5-6; not re-timed since) |
| Repository | `python3 run_selftests.py replicant` | the default tier, through the repo runner | — |

`@slow` (`testing.py`) is a plain function attribute that only `selftest.py` honours. **pytest
ignores it and would run every test, the slow ones included.**

The slow tier contains:
- full RSB (the K = 3 RSB test alone takes about 7 minutes);
- population dynamics at reference sizes;
- the K-SAT and LDPC crossings;
- the Monte Carlo and enumeration held-out routes;
- the online simulations at N = 1000;
- the lattice, percolation and Monte Carlo checks at reference sizes;
- the KZ series at reference sizes (the q = 5 SBM c_d, c_c and the coloring tables by population dynamics, the planted-BP threshold at N = 5000);
- every entry's slow and statistical RefChecks (`test_catalog.py`: one `@slow` test per entry, `coloring` split by `thresholds`; slowest 85 s, coloring colorability; about 7.5 minutes in all).

## Test map

| Area | Files |
|---|---|
| Core, spec, composer, CLI | `test_core.py`, `test_compose.py`, `test_compose_rsb.py`, `test_cli.py`, `test_catalog.py`, `test_refdata.py` |
| Level (a) | `test_derive.py`, `test_derive_ising.py`, `test_derive_rsb.py`, `test_solve*.py`, `test_replicon_derived.py`, `test_gauss*.py` |
| Families | `test_sk.py`, `test_rem_pspin.py`, `test_perceptron*.py`, `test_hopfield.py`, `test_parisi.py`, `test_gardner_transition.py`, `test_margin_law.py`, `test_pinned.py`, `test_dense_memory.py` |
| Cavity, CSP, inference | `test_cavity_*.py`, `test_xorsat.py`, `test_ksat.py`, `test_ldpc.py`, `test_spiked.py`, `test_sbm.py` |
| KZ notes (K1-K7) | `test_curie_weiss.py`, `test_rfim.py`, `test_random_matrix.py`; `test_bayes_scalar.py`, `test_bayes_identities.py`, `test_rem_condensation.py`; `test_factor_bp.py`, `test_potts_coloring.py`, `test_rfim_sparse.py`; `test_sbm_q.py`, `test_sbm_learn.py`, `test_nonbacktracking.py`; `test_spiked_diluted.py`, `test_wishart.py`, `test_asp.py`, `test_spiked_diluted_amp.py`, `test_asp_finite.py` (finite-N checks of the diluted AMP and ASP); `test_linear_gamp.py`, `test_lasso.py`, `test_cs.py`, `test_ridge_covariates.py`; `test_coloring_rsb.py`, `test_coloring_planted.py` |
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
