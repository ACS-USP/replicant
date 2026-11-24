# replicant

Published replica-method and cavity-method calculations of disordered systems as typed,
recorded steps. Each step names its preconditions and its citation; a missing precondition
is a named refusal (`precondition:<name>`, `missing-step:<hook>`, ...), never a silent
fallback; every run returns a trace, a per-value finality label (`stable`, `unstable`,
`not-computed`, `exact-by-literature`) and a comparison with banked, cited reference values.
Results are physics-level (the n -> 0 continuation and the RS/RSB ansaetze are not proved; the
online-learning ODEs assume the N -> oo limit), except what is certified in ball arithmetic (the lattice
entries `ising2d`, `ising1d`, `bethe_ising`, `ising2d_anisotropic`, `ising_triangular`, `ising_honeycomb`,
the bound entry `sk_bounds`, and the `cavity` / `exact` routes of the percolation entries);
population-dynamics, AMP, finite-N simulation, the percolation `sample` route and the Monte Carlo entries
are statistical. Full documentation: [GUIDE.md](GUIDE.md).

Requirements: Python 3 with `sympy`, `mpmath`, `numpy`, `scipy` and `python-flint` (the certified
entries go through `tools/baller`). Run the battery from the project venv (INSTALL.md: `.venv`, which
has python-flint): without python-flint the lattice and bounds tests fail loudly
(`unsupported:python-flint`), by design.

## Quick start

From the repository root:

```
python3 tools/replicant/selftest.py                        # default battery, about 30 s
cd tools && python3 -m replicant list                      # the 48 catalog entries
python3 -m replicant run perceptron --param alpha=1.5 --out /tmp/result.json --latex /tmp/p.tex
```

`run` also takes a spec file for a model near a catalogued one
(`python3 -m replicant run spec.json --param beta=1.5 --param h=0`; schema in the GUIDE),
`--route rsb` for `perceptron` at T = 0 with kappa < 0 (1RSB, K-step RSB, full-RSB estimate: `python3 -m replicant run
perceptron --route rsb --param kappa=-1`; results in [RESULTS_negative_perceptron_rsb.md](RESULTS_negative_perceptron_rsb.md)), `--route amp` for `sk`, `teacher_student`, `spiked_wigner`, `spiked_wishart`, `lasso` and `compressed_sensing` (state evolution, and finite-N AMP / GAMP with
`--param N=...`), `--route cavity` for the entries that have a cavity route (required for `viana_bray`, `xorsat`,
`ksat`, `ldpc_bec`, `sbm` and `bethe_ising`, which are cavity-only: `python3 -m replicant run viana_bray --route
cavity`, `python3 -m replicant run xorsat --route cavity`; the percolation entries `percolation_bethe`,
`percolation_graph`, `percolation_kcore` and `percolation_bootstrap` also default to `cavity`), `--route exact` for
`ising2d`, `ising1d`, `ising2d_anisotropic`, `ising_triangular`, `ising_honeycomb` and `percolation_lattice`, `--route
series` for `series_ising2d`, `--route bound` for `sk_bounds`, `--route online` for the online-learning entries
(`online_perceptron`, `online_linear`, `online_drift`, `online_tcm`, `online_scm`: `python3 -m replicant run online_scm
--route online`), `--route sample` for the sampled percolation routes and the Monte Carlo entries (`cluster_mc`,
`wang_landau_mc`, `histogram_reweighting_mc`, `population_annealing_mc`: `python3 -m replicant run cluster_mc --route
sample`), and `--pdf` (with `--latex`) to compile the derivation.

## Layout

- `core.py`, `spec.py`, `testing.py`: steps, state, trace, `Refusal`, `ModelSpec`, the `@slow` marker.
- `refdata.py`, `reference/*.json`: the reference store, banked cited values (schema in
  `reference/_schema.md`).
- `derive*.py`, `gauss.py`, `ansatz.py`, `agree.py`, `solve.py`, `solve_derived.py`: level (a),
  the symbolic derivation (replicate, average, order parameters, RS ansatz, n -> 0,
  stationarity), the derived replicon (`derive_replicon*.py`: SK, perceptrons, Hopfield), the
  solver for the derived equations, and the 1RSB functional (`derive_rsb*.py`, `gauss_rsb.py`,
  `solve_derived_rsb.py`, `_rsb_num.py`; composer route `compose_rsb.py`, `--param ansatz=1rsb`).
- `families/`: level (b), the published pipelines (SK, REM, spherical and Ising p-spin,
  perceptron (and `margin_law.py`: the same with a margin law kappa_l per pattern, `derive_margin.py` its level-(a)
  derivation, `numerics/margin_perceptron.py` its finite-N least-norm check), teacher-student, Hopfield, Parisi full RSB); `families/gardner.py`: the 1RSB replicon
  (stability toward further RSB, the Gardner transition) for the two p-spin entries.
  `families/perceptron_rsb.py`, `perceptron_krsb.py`: the negative-margin spherical perceptron at T = 0, where RS is only an
  upper bound: the 1RSB jamming (SAT-UNSAT) capacity from the zero-temperature scaling of the 1RSB entropy (audited
  mpmath roots, contact-index replicon), K-step RSB and the full-RSB staircase estimate (float64 grids, slow tier);
  catalog route `rsb` of `perceptron` (`catalog_perceptron_rsb.py`, `latex_perceptron_rsb.py`).
- `cavity/`: dense cavity, sparse BP / population dynamics, sparse 1RSB.
- `amp/`: AMP, state evolution and the TAP free energy for SK, GAMP and its state evolution for the
  teacher-student perceptron (finite N, statistical; independent of the replica and cavity code);
  catalog route `amp` (`catalog_amp.py`, `latex_amp.py`) for `sk` and `teacher_student`.
- `csp/`: random K-XORSAT (E5a): leaf-removal / zero-temperature cavity closed forms (`alpha_d`,
  `alpha_s`, 2-core, complexity) through a degree-law hook, and independent finite-N checks (exact GF(2)
  elimination, leaf removal on sampled hypergraphs); catalog entry `xorsat` (`catalog_csp.py`,
  `latex_csp.py`). Random K-SAT (E5b): `csp/ksat.py` (instances, exhaustive counts, the first-moment
  bound, BP) and `csp/sp.py` (survey propagation on instances, the SP population dynamics and the
  root of its complexity); catalog entry `ksat`.
- `inference/`: regular (l, r) LDPC codes on the binary erasure channel (E6a): density evolution,
  the BP threshold, the MAP threshold (Maxwell construction: trial-entropy root and area theorem)
  through a degree-polynomial hook, and independent finite-N checks (configuration-model graphs, the
  peeling decoder); catalog entry `ldpc_bec` (`catalog_inference.py`, `latex_inference.py`).
  E6b: the rank-one spiked Wigner model (`inference/spiked.py`: Bayes-optimal state evolution, the
  threshold, Bayes AMP with spectral start) and the two-group stochastic block model
  (`inference/sbm.py`: Kesten-Stigum line, sampler, BP with the external field); catalog entries
  `spiked_wigner` (amp) and `sbm` (cavity) (`catalog_spiked_sbm.py`, `latex_spiked_sbm.py`). K4 (the notes'
  Ch. 10): the q-group path of `sbm` (`catalog_sbm_q.py`; `inference/sbm_q.py` BP with planted / random starts and
  the Bethe free entropy, `sbm_q_pd.py` population dynamics and the first-order thresholds, `sbm_learn.py` EM,
  `nonbacktracking.py` the non-backtracking spectrum). K5 (the notes' Ch. 8, 9, 11, 12): `spiked_wigner` with
  `prior=diluted_rademacher` (`inference/spiked_diluted.py`: every SE fixed point, the spinodals, the IT threshold,
  the hard phase, the tricritical point; `catalog_spiked_diluted.py`), the entry `spiked_wishart` (amp)
  (`inference/wishart.py`: two-overlap SE, rectangular AMP, the BBP edge; `catalog_wishart.py`, `latex_wishart.py`),
  the dense spin-glass game as the Rademacher spike (`inference/spin_glass_game.py`), and approximate survey
  propagation for `sk` (`amp/asp.py`, `--route amp --param ansatz=1rsb --param m=...`).  K6 (the notes' Ch. 12, 16,
  18): `linear/` (GAMP with a Channel x Prior interface and its Bayes SE, the LASSO by AMP with the derived
  calibration and an exact coordinate-descent solver, the compressed-sensing boundaries), the entries `lasso`
  (replica, amp) and `compressed_sensing` (amp) (`catalog_linear.py`, `latex_linear.py`), and `ridge`'s `covariance`.
- `learning/` (E7): ridge and ridgeless least squares on isotropic Gaussian design (`ridge.py`: the
  Marchenko-Pastur closed forms of Hastie-Montanari-Rosset-Tibshirani, the optimal penalty;
  `ridge_rs.py`: the RS saddle point of the linear student, a cited closed form solved here;
  `ridge_sim.py`: SVD fits over named seeds, extrapolated in 1/n); catalog entry `ridge` (replica)
  (`catalog_learning.py`, `latex_learning.py`).
- `dense_memory/` (D1): dense associative memories with P = e^(alpha N) patterns and exponential interaction (Lucibello &
  Mezard 2024; Demircigil et al. 2017), at zero temperature: the pattern ensembles' zeta and rate function (`ensembles.py`:
  Gaussian, spherical, and binary derived here), the REM noise by four routes (`rem_noise.py`), the retrieval thresholds, basins,
  all-patterns bound and Ramsauer bound (`thresholds.py`), the Ising entry's fixed-point exponent and single-flip thresholds
  (`ising.py`), independent finite-N checks (`sim.py`: exact-law sampler, the paper's gradient descent, single-flip dynamics,
  Gray-code enumeration) and the InfoNCE dictionary (`infonce.py`, `MAPPING_INFONCE.md`); `derive_dense_memory.py` (the
  label-replica derivation, reusing `families/rem.py`); catalog entries `dense_memory` and `dense_memory_ising`
  (`catalog_dense_memory.py`, `latex_dense_memory.py`).
- `series/` (E8): the exact high-temperature series of the 2D Ising free energy from the Onsager
  integral (`ising2d.py`, with the independent L x L torus route), holonomic recognition through
  `tools/annihilator` (`recognize.py`), exact local exponents (`local.py`), the elliptic operator
  identity (`elliptic.py`), the banked susceptibility series (`susceptibility.py`); catalog entry
  `series_ising2d` (route `series`) (`catalog_series.py`, `latex_series.py`).
- `lattice/` (E9): solvable lattice models certified in ball arithmetic (python-flint arb through
  `tools/baller`: `ball.py`): Onsager's free energy and Yang's magnetization (`onsager.py`), Kaufman's
  finite torus and its integer polynomial recovered from balls (`kaufman.py`), the exact integer
  transfer matrix and the 1D ring (`transfer.py`), every BP fixed point of the Bethe-lattice
  ferromagnet (`bethe.py`); catalog entries `ising2d`, `ising1d` (route `exact`) and `bethe_ising`
  (route `cavity`), rigor "certified" (`catalog_lattice.py`, `latex_lattice.py`).
- `bounds/` (E10): Guerra's rigorous upper bounds on the SK pressure, certified in ball arithmetic
  (python-flint arb through `tools/baller`): certified Gaussian expectations (`gauss.py`), the K-step
  broken-replica bound by nested `acb.integral` (`krsb.py`), the RS bound with all its stationary points
  and its certified minimum (`guerra.py`), the chain check of the package's own RS / 1RSB / K-RSB
  optimisers (`chain.py`); catalog entry `sk_bounds` (route `bound`), rigor "certified"
  (`catalog_bounds.py`, `latex_bounds.py`).
- `numerics/` (E11): Monte Carlo and exact enumeration as independent checks of the held-out models
  (not a catalog route): Gray-code enumeration of +-J SK spectra and Ising-perceptron solution counts
  (`enumeration.py`), parallel tempering with an equilibration diagnostic (`pt.py`), finite-size
  extrapolation (`finite_size.py`), exact finite-N moments and the quenched-annealed closure (`moments.py`);
  used by `tests/test_heldout_mc.py` through `tests/heldout_mc.py`.
- `online/` (O1): online learning, the N -> oo order-parameter ODEs in alpha = examples / N
  (`fields.py`: Gaussian averages over the local fields, closed forms and an independent polar
  quadrature; `odes.py`: the perceptron flow for Hebb, perceptron, AdaTron, the optimal rule or any
  F(h, b, Q), with a learning rate or schedule and output noise, its integrator and asymptotes;
  `optimal.py`: the variationally optimal modulation of Kinouchi-Caticha and its noise-robustness
  boundaries; `simulate.py`: finite-N online learning over named seeds, sharing no code with the ODEs);
  catalog entry `online_perceptron` (route `online`) (`catalog_online.py`, `latex_online.py`).
  O2: `linear.py` (the linear perceptron with quadratic error: standard and optimal rules, sphere or
  unconstrained, greedy eta, closed forms), `linear_offline.py` (zero-temperature offline solutions and the
  ridge route), `simulate_linear.py` (finite-N online and offline); catalog entry `online_linear`
  (`catalog_online_linear.py`, `latex_online_linear.py`).
  O3: `drift.py` (the drift term, random or deterministic, composed onto the O1 flow; integration with
  jumps; stationary states; small-drift laws), `drift_rules.py` (VKC's annealed Hebb, step, symmetric
  and optimal modulations, their averages, closed-form stationary states and KC93's I(lambda)),
  `simulate_drift.py` (finite-N with the teacher drifting); catalog entry `online_drift`
  (`catalog_online_drift.py`, `latex_online_drift.py`).
  O4: `tcm.py` (the tree committee machine: the other hidden units' internal-representation average,
  the committee error, the branch field averages for Hebb, the local and CC95's nonlocal optimal rules, and
  the CKC96 noise mapping), `tcm_flow.py` (its ODEs, integration, closed forms and asymptotic constants),
  `simulate_tcm.py` (finite-N TCM learning); catalog entry `online_tcm` (`catalog_online_tcm.py`,
  `latex_online_tcm.py`).
  O5: `scm.py` (the soft committee machine: Saad-Solla ODEs, the closed-form erf averages I2, I3, I4 and an
  independent Gauss-Hermite route), `scm_flow.py` (integration, BRW / SS initial conditions, symmetric fixed
  points, the asymptotic modes, critical learning rates), `plateau.py` (plateau detection by BRW's definition,
  `undetermined` reports, the rtol check), `simulate_scm.py` (finite-N online gradient descent); catalog entry
  `online_scm` (`catalog_online_scm.py`, `latex_online_scm.py`).
  O6: `optimal_multilayer.py` (VC97's variationally optimal modulation in the soft committee machine: Eq. (3),
  its gradients and Hessian, the closed optimal flow, the teacher-output posterior W, the K = M = 2 plateau),
  `optimal_ml_flow.py` (integration through det H = 0, VC97's initial overlaps), `simulate_optimal_ml.py`
  (finite-N runs of the algorithm); `rule=optimal` of `online_scm` (`catalog_online_scm_opt.py`,
  `latex_online_scm_opt.py`).
- `lattice/` (I1-I3, beside E9's files): exact 1D observables and finite rings (`observables1d.py`), Onsager's 2D
  energy and specific heat (`onsager_thermo.py`), the torus u, c by Kaufman derivatives and integer moments
  (`torus_thermo.py`), the 2D correlation length from Onsager's strip spectrum (`correlation2d.py`) with the direct
  strip matrix (`strip_tm.py`), the critical diagonal correlations (`critical2d.py`), the anisotropic square lattice
  (`anisotropic.py`), the triangular and honeycomb lattices (`tri_honey.py`) and their exact tori (`geom_tori.py`),
  the exact coupling K = beta J (`coupling.py`); entries `ising2d_anisotropic`, `ising_triangular`,
  `ising_honeycomb` (`catalog_lattice_obs.py`, `catalog_lattice_xi.py`, `catalog_lattice_aniso.py`,
  `catalog_lattice_geom.py`, `catalog_lattice_common.py`, `latex_lattice_obs.py`, `latex_lattice_xi.py`,
  `latex_lattice_geom.py`).
- `percolation/` (P1-P3): the shared union-find (`unionfind.py`), degree laws and generating functions (`laws.py`),
  the certified cavity fixed points (`cavity.py`), cluster-size distributions (`clusters.py`), independent closed
  forms (`closed_forms.py`), finite-graph samplers (`sampling.py`) and the ER critical window (`window.py`); 2D
  lattices: exact thresholds and exponents (`exact2d.py`), Cardy's formula (`cardy.py`), exact crossing polynomials by
  enumeration (`enumerate2d.py`), Newman-Ziff sweeps (`lattice2d.py`) and finite-size scaling (`fss2d.py`); k-cores
  and bootstrap percolation: the certified core equation (`kcore.py`, `kcore_audit.py`), the birth point
  (`kcore_threshold.py`), DGM's form in mpmath (`kcore_dgm.py`), exact peeling and spreading (`kcore_sample.py`);
  entries `percolation_bethe`, `percolation_graph`, `percolation_lattice`, `percolation_kcore`,
  `percolation_bootstrap` (`catalog_percolation*.py`, `latex_percolation*.py`).
- `montecarlo/` (M1-M3): Fortuin-Kasteleyn bond probabilities and critical points (`fk.py`), Swendsen-Wang and Wolff
  on the shared union-find (`cluster.py`) and a csgraph SW for large L (`cluster_fast.py`), checkerboard Metropolis
  (`metropolis.py`), exact small-torus distributions and the stationarity / ergodicity chi^2 (`exact_small.py`,
  `balance.py`), FK at q = 1 against P2's crossing polynomial (`fk_percolation.py`), runs over named seeds (`run.py`),
  the shared statistics module (`stats.py`: tau_int, binning, jackknife, Student-t bands, the equilibration
  diagnostic, chi^2, z fits); the exact g(E) (`dos_exact.py`), the flat-histogram walk (`flat_walk.py`), Wang-Landau
  (`wang_landau.py`), multicanonical reweighting (`multicanonical.py`), Ferrenberg-Swendsen (`reweight.py`);
  population annealing (`population.py`) with an SK Metropolis sweep (`sk_metropolis.py`); entries `cluster_mc`,
  `wang_landau_mc`, `histogram_reweighting_mc`, `population_annealing_mc` (`catalog_mc*.py`, `latex_mc*.py`).
- pinned directions (the joint pair problem of random documents): `derive_pinned.py` (the level-(a) hook:
  `margin kappa pinned directions` for the spherical perceptron, two fixed unit vectors of overlap c; the entropic
  term with the pinned components fixed), `pinned.py` (the capacity boundary as the q -> 1 limit of the derived
  functional, RS state points audited against the derived equations, the untied refusal, the composer run path),
  `pinned_modes.py` (the stability sectors: the replicon, which cannot mix with the pinned overlap, the standard
  block that does, the longitudinal mode), `pinned_check.py` (integer-n Hessian, an independent check),
  `pinned_report.py` (the table of `docs/replicant/pinned_joint_replicon.md`).
- `catalog*.py`, `compose.py` (+ `compose_result.py`: `Result` and the finality rule), `cli.py`,
  `__main__.py`: the 46-entry catalog (registered in `catalog_index.py`; `catalog_ts.py` holds the
  teacher-student entry, `catalog_pspin.py` the two p-spin entries; LaTeX in `latex_pspin.py`), the
  composer, the command line.
- `meanfield/`, `rmt/` (K1, Krzakala & Zdeborova's lecture notes, Ch. 1-3): `curie_weiss.py` (Theorem 1 in arb,
  the cavity and field-theoretic toolboxes as separate code, the spinodal), `cw_exact.py` (Z_N by the binomial
  sum in arb, the rate functions), `rfim.py` (the fully connected RFIM: RS, exact sorted ground states,
  Guerra's bounds), `rmt/stieltjes.py` (Wigner and Wishart spectra by replica and cavity routes); catalog
  entries `curie_weiss`, `rfim`, `random_matrix` (`catalog_meanfield.py`, `latex_meanfield.py`).
- `bayes/` (K2, the notes' Ch. 7): `channels.py` (the scalar Gaussian channel y = sqrt(snr) x + z, priors as
  point-mass / Gaussian mixtures, the vectorised posterior-mean denoiser, MAP, MMSE and mutual information, certified
  for the Gaussian and +-1 priors), `identities.py` (Nishimori, Stein and I-MMSE validators returning a CheckResult),
  `sparse_vector.py` (one-hot vector denoising, Theorem 15 and its REM route); catalog entries `scalar_denoising`,
  `sparse_denoising` (`catalog_bayes.py`, `latex_bayes.py`); `families/rem.py` gains the participation ratio, its
  second moment and the overlap law (Ch. 15).
- `factor/`, `coloring/` (K3, the notes' Ch. 4-6): `factor/graph.py` (factor graphs with q-state variables, samplers of
  trees, d-regular and Erdos-Renyi graphs), `factor/bp.py` (sum-product BP, marginals, the Bethe free entropy labelled
  exact on trees and an approximation on loops), `factor/population.py` (population dynamics over a factor law),
  `factor/enumerate.py` (exact Z by enumeration); `coloring/potts_bp.py` (paramagnetic and ferromagnetic fixed points,
  beta_stab, the q >= 3 first-order spinodal and crossing, c_ann, c_KS), `coloring/potts_graph.py` (BP on sampled
  graphs from the planted and a random start), `coloring/first_moment.py` (exact E Z over G(N, M)); catalog entries
  `potts_bethe`, `coloring` (`catalog_factor.py`, `latex_factor.py`) and the `rfim` route `cavity`
  (`catalog_rfim_sparse.py`).
- `coloring/planted.py`, `rsb.py`, `rsb_pop.py`, `sample.py` (K7, the notes' Ch. 13-14): the planted coloring and its
  SBM link, the frozen survey propagation (colorability), the 1RSB cavity by population dynamics (clustering,
  condensation), the rigidity recursion, exhaustive counts and BP on sampled graphs; the `coloring` entry gains
  `ansatz=1rsb` and `planted=true` (`catalog_coloring.py`).
- `latex*.py`, `tex*.py`, `derive_tex.py`: the LaTeX derivation writer.
- `tests/`, `selftest.py`: the battery.

## Test tiers

- Default: `python3 tools/replicant/selftest.py` (781 tests after K2, 803 after K4, 821 after K5, 837 after K6; the 662 before the margin law and the pinned
  directions took about 30 s, measured 30.2 s cold on a clean export after the final fix wave). Also `python3 run_selftests.py replicant` from the repository root, with the venv activated (it runs
  the `python3` on PATH; without python-flint the certified lattice, bounds and percolation tests and the Monte
  Carlo exact targets fail, by design).
- Full: `python3 tools/replicant/selftest.py --full` adds the 231 `@slow` tests (full RSB,
  population dynamics at reference sizes, the K-SAT and LDPC crossings, Monte Carlo and enumeration
  held-out routes, solver-based checks, the online simulations at N = 1000, the lattice, percolation and
  Monte Carlo checks at reference sizes); about 35 minutes on a laptop (measured 34 min 39 s with 859 tests before the final fix wave; the K = 3 RSB test alone is about 7 minutes; not re-measured after the margin-law, pinned-direction, negative-margin RSB and dense-memory additions).
- The held-out tests are `tests/test_heldout*.py` (helpers `tests/heldout_common.py`, `tests/heldout_mc.py`).
- `@slow` is a plain function attribute honoured by `selftest.py` only. pytest ignores the
  marker and would run every test, slow ones included.
