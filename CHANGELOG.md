# Changelog

All notable changes to BootLoops are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) (1.0.x = fixes and tool additions that change nothing existing, 1.x.0 = changes to existing tools).

## Unreleased

### Fixed
- `replicant` (spectral first step): the spectral start of `spiked_wigner`'s Bayes AMP (`inference/spiked.py`, every
  prior) takes its first Onsager term with f_{-1} = x^0/lambda (x^0 with MV's (2.20) scaling), the convention of the
  proof of MV Theorem 1 (App. A.2, E{L f(X~_{-1}; -1)} = 1/lambda); it was 0 (Sec. 6's choice for the general
  algorithm), and the run missed MV's SE at small t (rademacher, lambda = 1.5, N = 2000, t = 1: -23 half-widths).
  The planted start keeps f_{-1} = 0; `reaction=False` drops this term too. Finite-N spectral values change at small
  t (none banked; at t = 30 the same to the printed digits). Tests: fixed-t spectral bands at t = 1, 2 hold SE(t) and
  exclude the neighbours (rademacher lambda = 1.5, seeds 0-39; diluted rho = 0.08, lambda = 1.2, slow), the f_{-1} = 0
  control asserted to miss by more than 5 half-widths (private hook `_spectral_first_step`); the q literals are
  recaptured (`test_existing_priors_spectral_literals`; the old ones reproduced with the hook), the 58953a7 golden
  test checks the finite-N keys under the hook and every other key unchanged. Default tier 882 tests, 79.48 s (load
  averages 3.94 at the start, 5.21 at the end); slow tier 274 tests, full tier not re-timed (last timed at 1a37a65).
- `replicant` (final-review fix wave of the finite-N checks): the diluted `spiked_wigner` spectral start's
  `gamma_se_t` is the SE at the run's t from gamma_0 = lambda^2 - 1 (x^0 = sqrt(N lambda^2 (lambda^2 - 1)) phi_1 has
  |x^0|^2/N = lambda^2 (lambda^2 - 1); checked: gamma_trace[0] is lambda^2 - 1 to rounding and the band of <x^0, x>/N
  holds it), labelled exact-by-literature for every lambda > 1 (MV Thm 2 at fixed t, with the plug-in note); it was
  the t -> oo fixed point, labelled exact whenever the phase was easy, which is wrong in the metastable easy window
  (rho = 0.09, lambda = 1.0085: 0.0220 reported, 0.0184 at t = 30). `diluted_se_trajectory` sets only quadrature
  noise in (-1e-15, 0) to 0; a more negative step is refused `precondition:snr` (it was clipped). The `catalog_amp`
  docstring names the `asp-finite-n` step. Tests: the no-Onsager diluted control must miss by more than 3
  half-widths; off-by-one checks at t = 1, 2 (diluted planted) and t = 1 (ASP q0) with SE(t - 1), SE(t + 1) outside
  the band; a default-tier two-branch variant (N = 2000, margin 3, pilots 4.87-7.67); the spectral gamma_0 check; the
  metastable probe point; the spectral band excludes 0. Planted / gaussian / rademacher values unchanged. Default tier
  880 tests, 75.07 s (load averages 6.52 at the start, 6.45 at the end); slow tier 273 tests, full tier not re-timed
  (last timed at 1a37a65).
- `replicant` (parked items of the Krzakala-Zdeborova series): `coloring` ER clustering at q >= 4 is refused in the
  entry's first step, so a run with `N` no longer pays for K3's BP on graphs before the refusal
  (`coloring/rsb.refuse_er_first_order`, shared with `cluster_thresholds`); the `rfim` note reads `d =` on a regular
  graph; the `sbm` q = 2 Kesten-Stigum step is skipped, not run empty, when `q` is given; `transitions_note` lists
  every seed. Tests: the ER refusal test stubs both engines with a private exception and covers the catalog with
  `N`; `test_lasso_amp_is_the_lasso_at_small_n` requires all four named seeds to converge. No value changed.
  Default tier 850 passed (63.7 s); full tier 1120 passed, 0 failed, 48 min 04 s under load (load averages about
  5 to 6).
- `replicant` (final-review fix wave of the Krzakala-Zdeborova series K1-K8): `coloring` ER clustering /
  condensation at q >= 4 (first-order, ZK 2007) is refused `unsupported:er-first-order` on the input, before any
  population runs (it used to fire after about 60 s of populations, seed-dependently); at q = 3 a nontrivial probe
  labels c_d, c_c undetermined (kind first-order). `random_matrix`'s cavity edges are located by certified root
  isolation (arb bisection) of trace^2 - 4 det on the recursion's Moebius matrix instead of the closed form typed by
  hand (the `semicircle_edge` / `mp_edge_*` routes now say so; same values; planted hook `cavity_scale`).
  `curie_weiss`: the cavity route's findroot fallback is audited (mnewton at the triple root of beta = 1, h = 0,
  residual checked, named refusals) and the Lemma 2 audit text corrected (it runs at beta = 1, h = 0).
  `rfim` cavity takes the regular degree as `d` (as `potts_bethe` / `coloring`); `c` keeps its old meaning on regular
  graphs with identical values, both together or `d` on poisson are refused by name. `ridge` with an identity
  spectrum (`covariance=1:1`) carries the isotropic labels. `spiked_wigner` diluted: the spinodals within ~1e-7 of
  rho_tri (missed by the grid) are bracketed around the maximum of s, and at rho_tri the first-order flag is
  labelled undetermined (`first_order_status`). `compressed_sensing`'s LP success fraction carries the exact 99%
  Clopper-Pearson interval (its error was 0 at fraction 0 or 1). `sk` amp ASP refuses a malformed `start`
  `precondition:start`. The CLI JSON spec names the prior that ran (`catalog_prior` keeps the matched default).
  Entry descriptions of `sbm`, `spiked_wigner`, `ridge`, `sk`, `rem` (and the `sbm` spec constraint) match the code.
  Bank text (sources unchanged): the edge routes; `coloring_c_ann_q3` O(1/N) as tested; `coloring_c_ks_q3` no longer
  claims an untested Eq. (6.23) route; `coloring_c_ann_q5` 2 ln 5 / ln(5/4) = 14.4251 (was 14.427; value 14.4 +-
  0.05 unchanged) and no first-moment sign-change claim; the two keys per q = 5 c_d / c_c (notes and ZK prints)
  explained. Tests: the aggregate slow catalog RefCheck (about 7.5 min) split into one `@slow` test per entry
  (coloring by `thresholds`; slowest 85 s); positive controls for the graph samplers, the Potts ER population and
  the RFIM c -> oo limit; the tautological `test_rfim.py` assert replaced by the band check; `test_lasso.py` checks
  named seeds 0-3 instead of a survey-picked seed; the layering audits share `testing.imports` (the
  `test_potts_coloring` copy dropped imported names); `sk_1rsb`'s h = 0 selection checked as the best Guerra 1RSB
  bound (the q0 = 0 point is a saddle 1.919e-4 higher in -beta f), and the GUIDE / As-built no longer call it open.
  Default tier 850 tests, 62.92 s in the working tree under load (load averages 3.4-4.5); slow tier 270 tests.
- `replicant`: `online_linear` (asymptote, crossing) and `online_drift` (small-drift law) no longer refuse a run
  whose diagnostic cannot decide; they label `asymptote_status`, `crossing_status`, `small_drift_status`
  undetermined and omit the value, as `online_tcm` and `online_scm` already did (the library functions still
  refuse by name). `sk` replica at a beta with an ansatz other than rs / 1rsb (e.g. `2rsb`), which ran no saddle
  and returned no values, is refused `precondition:known-ansatz`. `families/sk.py` evaluates the h = q0 = 0
  1RSB candidate on the half line (Gauss-Legendre), removing the Gauss-Hermite kink error (3.8e-6 at beta = 3,
  now ~1e-13 against an independent mpmath quadrature); no selected result changes. Tests: four new, and
  `test_online_linear.py` / `test_lattice_geom.py` split (`test_online_linear_entry.py`,
  `test_lattice_geom_entries.py`) under the ~400-line convention; default tier 752 tests.

### Added
- `replicant` (finite-N checks of K5): finite-N Bayes AMP for `spiked_wigner` with `prior=diluted_rademacher` (parameters `N`, `seeds`, `iters`, `init` = `spectral` | `planted`, `gamma0`; values
  `q_amp`, `overlap_amp`, `mmse_amp`, `matrix_mmse_amp`, `gamma_amp_t` with `<key>_halfwidth99`, and `gamma_se_t`) and
  finite-N approximate survey propagation for the `sk` amp route with `ansatz=1rsb` (`amp/asp_finite.py`; `N`, `seeds`, `iters`, `start`; `q0_amp`, `q1_amp` against `q0_se_t`, `q1_se_t`). New library
  functions: `spiked.diluted_prior`, `spiked_amp(init="planted", gamma0, x0, rho)`, `spiked_amp_ensemble(rho, init,
  gamma0)` with a `gamma_t` band, `spiked_diluted.diluted_se_trajectory`, `asp.asp_trajectory`, `asp_finite.asp_amp`,
  `asp_amp_ensemble`. The SE value at the same fixed t is exact-by-literature with a note that the run uses empirical
  plug-ins; the finite-N bands are statistical, NOT final. The two `unsupported:` refusals that stated
  these gaps are removed (the cases now run); new `precondition:gamma0`, `precondition:x0`, and
  `precondition:known-parameter` for `init` / `gamma0` / `rho` off the diluted prior, `law` with 1rsb and finite-N
  parameters without `N`. Measured checks (pilot seeds disjoint from the test seeds): planted gamma0 = 1 at rho =
  0.08, lambda = 0.99, N = 3000, t = 4 inside the SE band with its no-Onsager control outside; two branches at rho =
  0.05, lambda = 0.95, N = 8000, separated by more than 10 summed half-widths; ASP at beta = 2, N = 2000, t = 10 inside
  both bands with the no-Onsager and m = 0 controls outside. Existing gaussian / rademacher runs: values unchanged
  (the same q at the same seeds; the ensemble's note text changed). Default
  tier 874 tests, 69.70 s (load averages 4.93 at the start, 3.64 at the end); slow tier 273 tests, full tier not
  re-timed (last timed at 1a37a65: 1120 passed, 48 min 04 s).
- `replicant` (pinned directions): the perceptron family with fixed external directions at level (a), the joint
  pair problem of random documents (maximal margin of a pair {i, j} of unit vectors of overlap c against n - 2
  iid Gaussian documents). Constraint `margin kappa pinned directions` (the composer no longer refuses it) with the
  extra RS order parameter M (overlap with the pinned vectors), the entropic term (1/2)[ln(1-q) + (q - M^T G^-1 M)/(1-q)]
  derived from the Gram determinant, and the tied/untied cases (untied refused `no-solution:untied-slack` by
  derivation). `pinned.capacity` (the q -> 1 limit: reproduces the paper's pair margin m_RS to 1e-12 and the
  support-vector law k = (n-2) Phi(-h*)), the RS saddle below capacity audited against the derived equations, and the
  stability sectors: the replicon of the joint problem (it cannot mix with M, by Schur's lemma, checked on the
  integer-n Hessian) is 1 - k/d at capacity, +0.32 .. +0.44 at the paper's five states, against -0.52 .. -2.22 for the
  fixed-a slice taken from the catalog perceptron; the standard-irrep block (anomalous Q mode mixed with the
  antisymmetric M mode) and the longitudinal M-mode also have the stable sign there. Spherical variables, two tied
  vectors; physics-level. Report: `docs/replicant/pinned_joint_replicon.md`.
- `replicant`: published replica-method and cavity-method calculations of disordered systems as
  typed, recorded steps (each with its citation and named preconditions; a missing one is a named
  refusal, never a silent fallback), with per-value finality labels (stable, unstable,
  not-computed, exact-by-literature) and checks against banked, cited reference values. v1
  catalog: random energy model, Sherrington-Kirkpatrick (RS, AT stability, 1RSB, full RSB in the
  slow tier), spherical p-spin, spherical perceptron (Gardner capacity), teacher-student learning
  curves, Hopfield (AGS), sparse +-J spin glass (Viana-Bray / Bethe lattice, RS and 1RSB). Routes:
  replica (symbolic level (a) derivation and level (b) family pipelines, with agreement tests) and
  cavity (dense, sparse BP / population dynamics, sparse 1RSB). A composer adapts a catalogued
  derivation to a nearby model and reports what it reused, adapted or could not do; a LaTeX writer
  renders any run as a derivation document (`python3 -m replicant list|run ... --latex out.tex
  [--pdf]`). Held-out evidence: the composed +-J SK model and the composed Ising perceptron are
  checked by independent routes (enumeration, bank targets, an annealed bound) with positive
  controls; its limits are stated in the GUIDE (the Ising perceptron 4/pi RS endpoint is not
  checked, small-N enumeration is coarse, the +-J agreement tests the adapted step and not
  universality). Results are physics-level, not proofs; population-dynamics results are
  statistical. Battery: `python3 tools/replicant/selftest.py` (about 16 s; `--full` about 6 min).
- `replicant` (E1): the replicon at level (a) for Gaussian-integral energies. The derived RS
  solution of the spherical and Ising perceptron and of Hopfield now carries a computed replicon
  (Price covariance derivative, RS split, n -> 0, eigenvector projection; the spherical volume by an
  explicit-n projection; for Hopfield also the m-mode from the Hessian of the derived functional),
  so `solve_derived_rs` returns `stable` True/False for them and composed
  Ising-perceptron values get computed finality; `derived_replicon(spec, solution)`; the replicon
  derivation appears in the LaTeX documents. Checked against the families' replicons (perceptron,
  Hopfield) at 12 points to 1e-8 and at the Ising AT root; planted hook errors move it.
- `replicant` (E2): 1RSB at level (a). The 1RSB functional is derived by rules (block overlap sums,
  two-level Hubbard-Stratonovich, the 1RSB split and spectrum, nested expectations, frozen q1 = 1 for
  counting measures, a T = 0 limit for SK-type energies) and solved with an independent-rule audit
  (`solve_derived_1rsb`); composer route `ansatz=1rsb` (values never final). Checked against the SK
  family 1RSB (T = 0 and finite T, 1e-8) and the banked Krauth-Mezard frozen capacity of the Ising
  perceptron; each rule against brute force at integer n.
- `replicant` (E3a): AMP, state evolution and the TAP free energy for SK (`replicant.amp`), a finite-N
  route that imports none of the replica or cavity code: Bolthausen's iteration with the Onsager term
  for Gaussian and +-J matrices (`sk_amp`, `sk_amp_ensemble` with 99% Student-t bands over named
  seeds), the scalar and two-time state evolution (`sk_state_evolution`, all fixed points listed, the
  predicted AMP convergence from psi'(q)), the TAP free energy (averaged, Plefka and naive forms) and
  TAP fixed points. Checked: the SE fixed point equals the RS q of `families/sk.py` and of the derived
  RS equations (1e-10); psi'(q) equals 1 - lambda_AT (1e-5); at N = 1500 the AMP q lies in the band
  around the SE value for both coupling laws while naive mean field does not; the TAP functional is
  stationary at the AMP fixed point and its value lies in the band around the RS -beta f. Statistical,
  not certified; validity statements (SE theorem for Gaussian J, convergence only above the AT line,
  many TAP solutions at low temperature) are in the GUIDE. No catalog route yet.
- `replicant` (E3b): GAMP for the Bayes-optimal teacher-student perceptron (gaussian or +-1 prior,
  noiseless sign channel) and its state evolution (`amp/gamp.py`, `amp/gamp_se.py`), the RS potential
  and the binary-prior thresholds alpha_IT 1.244684 / alpha_AMP 1.492632 (banked targets from Barbier
  et al. 2019 and a secondary source, which disagree by 0.004 on alpha_IT); catalog route `amp` for
  `sk` (SE value exact-by-literature by Bayati-Montanari Thm 4; finite-N AMP statistical) and
  `teacher_student` (GAMP SE values not-computed: a claim, not a theorem); a new independent held-out
  check, the composed +-J SK q against AMP on sampled +-J matrices (slow).
- `replicant` (E4): the Gardner transition. The 1RSB replicon toward further RSB
  (`families/gardner.py`, the cited Crisanti-Leuzzi / Montanari-Ricci-Tersenghi formula, checked
  against an explicit integer-n replica functional) gives computed finality for the p-spin entries;
  new Ising p-spin family and catalog entry `ising_pspin` (T_s, T_d, T_G = 0.240260 at p = 3 against
  Rizzo 2013; state-point values stable only for T_G < T < T_s); the spherical pure p-spin has no
  Gardner transition (refused with the computed scan) and its 1RSB values are now computed stable;
  the frozen Ising-perceptron 1RSB stability is refused as not computed.
- `replicant` (E5): constraint satisfaction. Random K-XORSAT (`csp/xorsat.py`, entry `xorsat`:
  leaf-removal / zero-temperature cavity thresholds, 2-core, complexity; exact GF(2) elimination and
  leaf removal on sampled instances) and random K-SAT (`csp/ksat.py`, `csp/sp.py`, entry `ksat`:
  first-moment bound, BP, survey propagation on instances and as population dynamics, the root of the
  SP complexity 4.2623 +- 0.0108 against Mertens-Mezard-Zecchina's 4.26675; exhaustive counts for
  N <= 22). Only the proved values are final; the K-SAT thresholds are cavity conjectures. The
  3-coloring stretch was not done.
- `replicant` (E6): inference and codes. Regular LDPC codes on the erasure channel (`inference/ldpc.py`,
  entry `ldpc_bec`: density evolution, BP and MAP thresholds by tangency, trial entropy and the area
  theorem; the peeling decoder on configuration-model graphs); the rank-one spiked Wigner model
  (`inference/spiked.py`, entry `spiked_wigner`, route amp: Bayes-optimal state evolution, the
  threshold, Bayes AMP) and the two-group stochastic block model (`inference/sbm.py`, entry `sbm`: the
  Kesten-Stigum line, BP on sampled graphs).
- `replicant` (E7): ridge and ridgeless regression on isotropic Gaussian design (`learning/`, entry
  `ridge`): the Marchenko-Pastur closed forms of Hastie et al. 2022 and the optimal penalty, the RS
  saddle point of Loureiro et al. 2022 (a cited closed form, not a level-(a) derivation), SVD
  simulations over named seeds extrapolated in 1/n.
- `replicant` (E8): the exact high-temperature series of the 2D Ising free energy from the Onsager
  integral and its holonomic operator through `tools/annihilator` (`series/`, entry `series_ising2d`,
  route series): theta-order 3, degree 8, inhomogeneous, with the elliptic right factor proved
  exactly; the critical point 3 - 2 sqrt 2 and alpha = 0 from exact local exponents; an independent
  exact torus route; the susceptibility has no operator within (6, 40) (a bound, not a proof).
- `replicant` (E9): solvable lattice models certified in ball arithmetic through `tools/baller`
  (`lattice/`, entries `ising2d`, `ising1d` route exact, `bethe_ising` route cavity; `rigor`
  "certified" with radii): Onsager's free energy, Yang's magnetization, Kaufman's torus against the
  exact integer transfer matrix, the 1D ring, every BP fixed point of the Bethe ferromagnet. Needs
  python-flint (the project venv).
- `replicant` (E10): Guerra's RS / 1RSB / 2RSB upper bounds on the SK pressure, certified (`bounds/`,
  entry `sk_bounds`, route bound), with every stationary point of the RS bound certified and a
  consistency check of the package's own RS / 1RSB / K-RSB optimisers against the bound chain.
- `replicant` (E11): Monte Carlo and exact enumeration as independent checks of the held-out models
  (`numerics/`: Gray-code enumeration of +-J SK spectra up to N = 26 and of Ising-perceptron counts up
  to N = 30, numpy parallel tempering with an equilibration diagnostic, N^-2/3 finite-size
  extrapolation, exact finite-N moments). The held-out tests now exclude RS for the SK ground energy
  (-0.7610 +- 0.0124 contains Parisi), tell quenched from annealed for the +-J E ln Z, and exclude the
  annealed entropy and capacity of the Ising perceptron; what they do not resolve (1RSB against
  Parisi, RS against 1RSB at beta = 2) is printed, not asserted.

- `replicant` (online learning, O1-O6): the N -> oo order-parameter ODEs of online learning in alpha = examples / N
  (`online/`, new route `online`), each checked against finite-N simulation over named seeds that shares no code with
  the ODE, closed forms and an independent quadrature, with planted ODE terms that must fall outside the band. Entries
  `online_perceptron` (Hebb, perceptron, AdaTron and the variationally optimal rule of Kinouchi-Caticha, output
  noise: Biehl-Riegler 1994, Copelli et al. 1997), `online_linear` (online versus offline, Kinouchi-Caticha 1995),
  `online_drift` (random and deterministic drift, Vicente-Kinouchi-Caticha 1998, Kinouchi-Caticha 1993),
  `online_tcm` (tree committee machine and its noisy-perceptron equivalence, Copelli-Caticha 1995), `online_scm`
  (soft committee machines, plateaus and their escape, critical learning rates: Saad-Solla read through their 1996 /
  1997 restatements, Biehl-Schwarze 1995, Biehl-Riegler-Woehler 1996; `rule=optimal`: Vicente-Caticha 1997's
  functional optimisation). Every ODE value is `not-computed` (no concentration theorem fetched); "optimal" names its
  class; a plateau not left inside the window is `undetermined`. Not fetched: Kinouchi-Caticha 1992 (0.88 read from
  two secondary prints), Saad-Solla's PRL / PRE, Biehl-Schwarze 1992 / 1993; Copelli-Kinouchi-Caticha 1996 abstract
  only. Default tier +4.0 s (ruling: <= 4 s).
- `replicant` (exact Ising, I1-I3): certified 1D magnetization, susceptibility, connected correlation and
  correlation length checked against exact finite rings with the N -> oo bound; Onsager's 2D energy and specific heat
  (the log amplitude (2/pi) ln^2(1 + sqrt 2)) against Kaufman-torus derivatives and the E8 series; the 2D correlation
  length above and below T_c from Onsager's strip spectrum (below T_c `not-computed`: the two-particle gap's
  identification with xi is derived here), nu / eta consistency checks, the critical nearest-neighbour and diagonal
  correlations (Perk-Au-Yang, Deift-Its-Krasovsky); new entries `ising2d_anisotropic`, `ising_triangular` and
  `ising_honeycomb` (route `exact`, certified: Athanasopoulos-Ueltschi's Kac-Ward integral, Baxter's maps, Pierre et
  al.; the triangular antiferromagnet's ground-state entropy), with star-triangle and four-sector duality identities on
  exact tori and a high-temperature torus identity. `ising2d` and `ising1d` keep their earlier values. Default tier
  +1.6 s (ruling: <= 1 s; the task reports' own direct accounting gave about 0.9 s, see Changed).
- `replicant` (percolation, P1-P3): new package `percolation/` with a shared union-find and entries
  `percolation_bethe`, `percolation_graph` (certified cavity fixed points of the generating-function equations, P_inf,
  mean cluster size, Molloy-Reed, cluster-size distributions; route `sample`: finite graphs over named seeds, the ER
  critical window in the slow tier), `percolation_lattice` (square, triangular, honeycomb; certified thresholds 1/2,
  2 sin(pi/18), 1 - 2 sin(pi/18), the Smirnov-Werner exponents, Cardy's formula by two routes, exact crossing
  polynomials; Newman-Ziff sweeps and finite-size scaling validated on the solved lattices), `percolation_kcore` and
  `percolation_bootstrap` (certified k-core and bootstrap fixed points and birth points of Dorogovtsev-Goltsev-Mendes
  and Janson, against csp/xorsat.py's 2-core and Balogh-Pittel's closed forms; exact peeling and spreading). The
  sample route never refuses a correct sampler. Default tier +1.1 s (ruling: <= 1 s).
- `replicant` (advanced Monte Carlo, M1-M3): new package `montecarlo/` and entries `cluster_mc` (Swendsen-Wang,
  Wolff and a Metropolis baseline for the Ising / q-state Potts torus; stationarity chi^2 of stated power; z fits
  printed against the literature), `wang_landau_mc` (Wang-Landau with the 1/t refinement against the exact torus
  g(E), multicanonical reweighting), `histogram_reweighting_mc` (Ferrenberg-Swendsen single and multiple histogram as
  printed by Kumar et al. 1992, with a stated overlap window and refusals outside it) and `population_annealing_mc`
  (Hukushima-Iba as described by Machta and Wang-Machta-Katzgraber; free energy against Kaufman and exact enumeration,
  family diagnostics, the finite-R bias and its corrections), all route `sample`, statistical, unequilibrated runs
  labelled `undetermined`; a shared statistics module (tau_int, binning, jackknife, Student-t bands, equilibration
  diagnostic, chi^2). Default tier +0.5 s (ruling: <= 1 s).
- `replicant` (margin law): the `perceptron` margin kappa can be a LAW, kappa_l drawn iid per pattern and averaged
  inside the replica integrals (`kappa_law` = constant | gaussian (`kappa`, `kappa_s`) | mixture (`kappa_weights`,
  `kappa_values`); other laws and stray parameters are refused `precondition:supported-margin-law`). Level (b)
  (`families/margin_law.py`, spherical): RS saddle, entropy, replicon and the q -> 1 capacity alpha_c = 1/E_kappa D(kappa)
  with the merged Gaussian x = (m - sqrt(q + s^2) u)/sqrt(1-q), and the unmerged 2-D audit; the catalog `perceptron`
  entry takes the parameters. Level (a) (`derive_margin.py`, spherical and Ising, RS): constraint `"margin kappa law"`,
  the formal `LnGaussProdLaw`, derived equations, solve and replicon through the generic steps (additive registries
  `derive.RS_RULES`, `derive.REPLICON_RULES`); the 1RSB routes refuse it (`not-implemented:1rsb:LnGaussProdLaw`).
  Banked internal check `reference/perceptron.json` `margin_law_collapse`: the Gaussian offset (constraints a g_l + b y_l
  <= h, a^2 + b^2 = 1) has alpha_c(law)/(1 - a^2) = the catalog capacity at kappa~ = -h, to 1e-15 for eight (a, h)
  pairs (an identity at q -> 1; the entropy and replicon below capacity do not collapse). Independent evidence: degenerate
  laws equal the catalog (capacity 1e-15, replicon 1e-8, observed 1e-17); level (a) equals level (b) (1e-22; the
  stationarity equations proved equal symbolically); finite-N least-norm sampling (`numerics/margin_perceptron.py`,
  Lawson-Hanson least-distance programming, the package's finite-size band) brackets the RS capacity; planted controls
  (wrong merged variance, kappa replaced by its mean, kappa averaged per replica) are each caught. Honest limits in the
  GUIDE (section MARGIN LAW): spherical only at level (b), physics-level RS, non-convex laws carry the catalog's kappa < 0
  caveat and the sampling there is the ball threshold. Default tier 685 tests (23 new, about 21 s CPU measured on a
  loaded machine, not re-measured quiet); full tier 896.
- `replicant` (negative-margin perceptron): the capacity of the spherical perceptron with kappa < 0 beyond RS, where the
  RS value 1/D(kappa) is only an upper bound. `families/perceptron_rsb.py`: the 1RSB jamming (SAT-UNSAT) point from the
  zero-temperature scaling (1 - q1 = delta -> 0, m = mu delta) of the 1RSB entropy of Franz, Parisi, Sevelev, Urbani &
  Zamponi (arXiv:1702.06919, Eq. (59)): alpha_c is the minimum of A(mu, q0) = -ent/en, solved with float64 bracketing,
  an mpmath quasi-Newton polish and an audit (stationarity residuals, Hessian, two quadratures), with a contact-index
  replicon (FPSUZ Eq. (62) at leading order: 1RSB is unstable for every kappa < 0, labelled `unstable`, never final);
  `gardner_capacity("spherical", kappa, ansatz="1rsb")` no longer raises `not-implemented:spherical-rsb`
  (kappa >= 0 returns the RS value, kappa < 0 the 1RSB capacity). `families/perceptron_krsb.py`: K-step RSB (K <= 5, float64
  grids, slow tier), its contact index and a geometric extrapolation to full RSB (`not-computed`). Catalog route `rsb`
  of `perceptron` (`catalog_perceptron_rsb.py`, parameters `kappa`, `ansatz` 1rsb|krsb|frsb, `levels`; displayed equations in
  `latex_perceptron_rsb.py`). Banked from Stojnic (arXiv:2312.16531 Tables 2, 5, 6; arXiv:1306.3980 Table 2) and
  Franz-Parisi (arXiv:1501.03397): the 1RSB capacity agrees with his second level of lifting at ten margins
  (kappa = -2.7 ... -0.3) to 6.2e-4 relative and the 2RSB with his third level at four. At the effective margins of the
  random-documents appendix (kappa = -0.55, -0.85, -1.12, -1.36, -1.66) the full-RSB / RS capacity ratio is 0.981, 0.948,
  0.905, 0.860, 0.796 (RESULTS_negative_perceptron_rsb.md). Checks: the scaling limit against Eq. (59) evaluated as
  written, the integer-n replica functional against Gaussian orthant probabilities, planted controls (wrong block
  structure, dropped m, wrong scaling). The arXiv number 1606.04380 sometimes quoted for FPSUZ is a mathematics paper:
  the paper is arXiv:1702.06919. Default tier 676 tests (+14, about 1.8 s by their own timing, over the 1 s ruling: the audited mpmath roots and the scaling-limit
  and replicon checks are slow-tier), full tier 898 tests (+16 slow: K = 2 about 30 s per kappa, K = 3 about 7 minutes).
- `replicant` (D1): dense associative memories with exponentially many patterns (`dense_memory/`, entries `dense_memory` and
  `dense_memory_ising`, route replica, 38 entries in all). `dense_memory` reproduces Lucibello & Mezard (PRL 132, 077301, arXiv:2304.14964)
  for Gaussian and spherical patterns: the Random Energy Model noise of the softmax denominator by a seven-step label-replica
  derivation (`derive_dense_memory.py`, reusing `families/rem.py`) and by three further routes agreeing to 1e-30, the REM freezing line
  and Parisi parameter, the single-pattern threshold alpha_1(lam) and its inverse lambda_1(alpha), the all-patterns bound (Gaussian
  log(2)/4 for lam >= 0.70091, spherical alpha_c = alpha_1), one-step basins, the small-load and scaled-dot-product regimes and the
  Ramsauer bound as restated there; 17 banked values from the printed formulas and numbers. `dense_memory_ising` takes Demircigil et
  al.'s rigorous fixed-point exponent I(1 - 2 rho)/2 (log(2)/2, lam = 1) and derives the binary REM noise and the single-flip
  thresholds (labelled not-computed). Independent checks: the exact finite-N overlap law (agreeing with `rem_quenched_finite_n` at
  alpha = log 2), the paper's gradient descent, the single-flip dynamics, Gray-code enumeration; planted controls fail. Polynomial
  interactions are refused `unsupported:interaction`; a finite-temperature free energy of the memory is not in the paper and is not
  claimed. `dense_memory/MAPPING_INFONCE.md` maps the model to InfoNCE (lam = 1/(tau d)) and states what it does not say about the
  critical n(d, tau) and the per-query threshold c(d) tau.
- `replicant` (K1-K7): the methods of Krzakala & Zdeborova's lecture notes (*Statistical Physics Methods in Optimization
  and Machine Learning*, June 9 2024), 10 new entries (48 in all) and generic layers. K1 (Ch. 1-3, `meanfield/`, `rmt/`):
  `curie_weiss` (Theorem 1 in arb with every root, the cavity and field-theoretic toolboxes as disjoint code, Z_N by
  the binomial sum, the rate functions, the corrected spinodal), `rfim` (RS, exact sorted ground states, Guerra's
  bounds on each sample), `random_matrix` (Wigner and Wishart Stieltjes transforms by replica and cavity, against
  eigvalsh). K2 (Ch. 7, 15, `bayes/`): one scalar channel with a prior registry, vectorised denoisers, MMSE and mutual
  information (certified for the gaussian and +-1 priors), the Nishimori / Stein / I-MMSE validators; `scalar_denoising`,
  `sparse_denoising` (Theorem 15 beside a REM route); `rem` `condensation=true` (participation ratio, its second
  moment, the overlap law against exact finite-N quadrature and sampling). K3 (Ch. 4-6, `factor/`, `coloring/`):
  factor-graph BP with the Bethe free entropy labelled exact on trees and an approximation on loops, population
  dynamics over a factor law, enumeration; `potts_bethe`, `coloring`, and the route `cavity` of `rfim`. K4 (Ch. 10):
  `sbm` with `q` groups (BP with the non-edge field, planted / random starts, the Bethe free entropy, the q = 5
  first-order c_d, c_c by population dynamics against DKMZ's 12.84, 13.23, EM with a BIC group-count rule, the
  non-backtracking spectrum). K5 (Ch. 8, 9, 11, 12): `spiked_wigner` with `prior=diluted_rademacher` (every SE fixed
  point, the spinodals, the IT threshold, the hard phase, the tricritical point, against AFUZ and LKZ),
  `spiked_wishart` (two-overlap SE, rectangular AMP, the BBP edge), ASP as `sk`'s amp `ansatz=1rsb`, the dense
  spin-glass game as the Rademacher spike. K6 (Ch. 12, 16, 18, `linear/`): GAMP with a channel x prior interface and
  its Bayes SE (reproducing `amp/gamp.py`), `lasso` (KZ Alg. 4 with the derived calibration lam = theta (alpha - b),
  against coordinate descent and Bayati-Montanari), `compressed_sensing` (l1, Bayes-spinodal and IT boundaries), and
  `ridge` `covariance`. K7 (Ch. 13-14): `coloring` `ansatz=1rsb` (the notes' regular frozen-SP closed form and the ER
  survey population against ZK 2007 Tables I-II, the 1RSB cavity by Parisi-reweighted population dynamics, the
  rigidity recursion) and `planted=true` (the planted coloring as the SBM with c_in = 0, BP from the planted start,
  quiet planting), exhaustive small-N counts against the first moment and Bethe. Every check has a planted control;
  statistical values carry 99% Student-t bands over named seeds; a run whose diagnostic fails is labelled
  `undetermined`, never refused. The notes' misprints (the p. 33 spinodal, p. 18's g(m), Fig. 2.2.1's axis, (11.26),
  (12.62), (12.97)-(12.100), (12.163)-(12.164) = (18.17)-(18.18), (12.173), (16.86), the p. 208 table's "c_col" = the
  annealed thresholds and others) are not banked as printed; the GUIDE's KNOWN HAZARDS lists each with the corrected
  form and the confirming route. Primaries fetched: GSV, DKMZ, KMMNSZZ, LKZ, BBP, AFUZ (arXiv:1805.05857, not the
  plan's 1802.01535), ASP (arXiv:1807.01296), DMM, BM, KMSSZ, Rangan, HMRT, ZK; Wigner, Marchenko-Pastur and
  Krzakala-Zdeborova 2009 were not, and nothing is banked from them. Existing entries keep their numbers (golden
  values from a pre-K5 export; every old parameter set unchanged).

### Changed
- `replicant` (documentation, K8): GUIDE (the K1 / K7 route summary, the catalog header with per-task counts, the
  refusal rows, API paragraphs for `meanfield/`, `rmt/`, `bayes/`, `factor/`, `coloring/`, worked examples for every
  K1-K7 entry run literally, the errata of the notes and of the primaries, the KZ-series hazards and what is not
  implemented), README, `tools/README.md`, REFERENCES (each source of the series: read, fetched with its arXiv id, or
  not fetched), the spec's as-built section and the wiki bundle (`docs/wiki/`); `compose.py`'s module docstring
  (a rigor may be a callable of the values). No number or behaviour changed. Time standard, re-timed on clean
  `git archive` exports of bcc55fc, cold, on a LOADED machine (the earlier ~30 s and 34 min 39 s were idle-machine
  figures and are not comparable): default tier 841 tests, 60.57 s and 60.56 s (load averages 3.5-4.3); full tier
  (`--full`) 1099 tests, 0 failed, 47 min 37 s (load averages 3.1 to 8.4 during the run). Series growth by the
  per-task same-session baselines: about 6 s for K1-K7 (baseline 752 tests, 54.7-56.6 s at load about 5).
  `selftest.py`'s summary line and its test (`tests/test_cli.py`) state the new full-tier time.
- `replicant` (documentation and cleanup, O7): GUIDE (catalog, refusal table, routes, finality notes, hazards,
  every worked example re-run literally on a clean export), README, REFERENCES (every source of the four plans
  classified primary / secondary / abstract only) and the online spec's and three plans' as-built sections updated;
  `catalog_percolation_kcore.py` split (the `percolation_bootstrap` entry moved to `catalog_percolation_bootstrap.py`,
  outputs unchanged); no number or behaviour changed. Time standard, measured cold on clean `git archive` exports:
  default tier 655 tests, 28.3-28.9 s (425 tests, 20.3-21.1 s at the last push, 9f97a28: +7.8 s for the four plans,
  of which about 0.6 s is the REM level-(a) derivation committed between them); full tier (`--full`) 859 tests,
  34 min 39 s. Per-plan default-tier growth by default-selftest differences at the plan boundaries (the rulings'
  method; three cold runs each, run-to-run noise about +-0.5 s): online +4.0 s, Ising +1.6 s, percolation +1.1 s,
  Monte Carlo +0.5 s; the new test functions alone (in-suite) cost 2.7, 0.7, 0.5 and 0.7 s, the rest is the growth of
  the shared catalog-RefCheck and LaTeX tests with each new entry. Cold isolated, each plan's own test modules take
  3.6, 1.3, 1.3 and 1.3 s (including about 0.45 s of imports each).
- `replicant`: finality labels that became computed: the derived RS stability of the perceptrons and
  Hopfield (E1), the p-spin 1RSB values (E4). `tests/test_heldout.py` split by model
  (`test_heldout_sk.py`, `test_heldout_perceptron.py`, `test_heldout_mc.py`, helpers in
  `heldout_common.py`); no assertion changed. Default battery `python3 tools/replicant/selftest.py`
  425 tests, about 22 s (measured 21.4 s); `--full` 557 tests, about 18 minutes (measured 18 min 11 s). GUIDE, README, REFERENCES and the extension spec's
  as-built notes updated.
- `replicant` (margin law): `_derived_num.safe` now undoes sympy's own rewrite erfc(-y) -> 2 - erfc(y) where it has
  already happened (numeric mixture atoms <= 0 make the argument a numeric multiple of the integration variable); exact,
  no existing result changes. `tests/test_compose.py::test_reference_checks_available_through_result` expects the
  new banked key `margin_law_collapse` beside `spherical_capacity_kappa0`. `cli._SYMBOLS` gains `kappa_s`.

### Fixed
- `replicant` (CLI regression from E2): `--param ansatz` was taken by the level-(a) 1RSB composer for
  every spec, so the v1 `run sk --param ansatz=1rsb --param beta=2` no longer reached the family 1RSB
  pipeline and `run viana_bray --route cavity ... --param ansatz=1rsb` was refused. For a catalog
  entry `ansatz` is again the entry's own parameter; the composer switch is `ansatz` for specs that
  are not catalog entries and `--param level_a_ansatz=1rsb` for any spec.
- `replicant`: `run sk|teacher_student --route amp --param N=... --param seeds=1` ended in
  `ERROR: TypeError` (exit 1), and an empty seed list fell back to the default; both are now refused
  `precondition:named-seeds`. `lattice.ball.certify` no longer treats an exact ball (radius 0 at two
  precisions) as a radius stall. The LDPC MAP-threshold cite names MMU Example 6 only for (3, 6); the
  spherical p-spin `no-transition:gardner` detail states its scan and resolution first.
- `replicant` (final review of the online, exact-Ising, percolation and Monte Carlo plans): `ising1d` crashed
  (`ValueError`) on a NaN ring ball (N = 3, r = 1; beta = 50): a non-finite ball is now refused
  `no-solution:enclosure`, the ring check raises its precision until its balls are finite and omits the ring
  correlation length where the ring's reflection symmetry makes it undefined (2 r + 1 = N). The percolation FSS
  crashed (`IndexError`) or printed `p_c_crossing` NaN with stderr 0 when too few seeds or sweeps left the
  crossing unresolved: such values are omitted and labelled `undetermined:crossing` (also `:collapse`, `:width`).
  `histogram_reweighting_mc`'s fixed default beta / betas were refused at L >= 32 by its own window: for L > 8 they
  now move towards beta_c by 8/L (L <= 8 unchanged), and an implicit default outside the window is labelled
  undetermined, not refused. The online asymptote and finite-N rho-sign diagnostics label instead of discarding the
  run (`asymptote_status`, `sim_status`); online runs that compute no value say so. The Monte Carlo LaTeX documents
  no longer describe their route as union-find graph sampling. `population_annealing_mc` refuses parameters its
  model does not read and sweeps = 0; one s_max cap for both percolation routes; no "+ systematic 0" in RefCheck
  details. No banked number changed. Default tier 662 tests, 30.2 s cold on a clean export (three runs, 30.17-30.21 s;
  +1.4 s, mostly the FSS and online regression tests); full tier 868 tests.

### Known issues
- `replicant`: `families/sk.py` `_rsb_phi_e` (v1, float64 Gauss-Hermite) is inaccurate in the q0 = 0
  / large q1 - q0 sector (2.8e-9, 2.8e-7, 3.8e-6 at beta = 2, 2.5, 3 against the certified balls of
  `bounds/`); no v1 result is affected (that candidate loses the selection by 1.9e-4 or more). Left
  unchanged; recorded in the GUIDE hazards.

## 1.0.1 (2026-10-09)

### Added
- `cutbasis`: geometric master-basis selector on the maximal cut (arXiv:2511.15381
  Algorithms 1-3) writing Kira `preferred_masters` lists in three forms, with a certified
  reducer comparison (`evaluate`) and a self-test; worked example: the two-loop Møller
  double box with a massive loop.

### Fixed
- `run_selftests.py`: a crash in a data-gated package was reported as
  "REFUSED (by design)" and left the run green. A data-gated battery now
  signals a refusal only by exit code 4 (`DATA_GATED_REFUSAL_RC`) with a
  printed reason, kept in `selftest_results.json`; any other nonzero exit,
  an exit 4 without a reason, or one whose last line is an uncaught
  exception, is a FAIL. `ffcapital`, `frobenius-boundary` (was exit 3) and
  `galois` follow the contract; `galois` resolves its reference data before
  any check runs. Documented in `tools/README.md` (Verification classes)
  and `INSTALL.md`. Reported by @Aspect5 (BootLoops-ai/bootloops#3).
- `run_selftests.py`: a timeout in the Julia `Pkg.instantiate()` pre-step
  aborted the whole run and `selftest_results.json` was never written. The
  timeout now fails that one package (with the cause and the remedy in the
  record) and the run continues, exiting 1. Fix by @hzshen88
  (BootLoops-ai/bootloops#2), reported in BootLoops-ai/feedback#1.
- `nestor`: the farm's orphan guard compared each worker's parent pid with
  the pool parent's pid, which is wrong under the `forkserver` start method
  (Python 3.14's default on Linux): every worker exited at its first chunk
  and the farm self-test (P5) never finished. The guard now uses the parent
  pid seen from inside the worker and also checks that the pool parent is
  alive, so it behaves the same under `fork`, `forkserver` and `spawn`.
  Reported by @drdavidglew-afk (BootLoops-ai/feedback#2).
