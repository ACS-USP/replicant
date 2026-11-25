# Finite-N Checks (diluted spiked prior, ASP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `spiked_wigner` with `prior=diluted_rademacher` (spectral and planted starts) and `sk` amp with
`ansatz=1rsb` (ASP) finite-N AMP checks against their state evolution at a fixed t, and remove the two
`unsupported:*-finite-n` refusals.

**Architecture:** The diluted prior becomes one more `Prior` in `inference/spiked.py`, and `spiked_amp` gains
a planted start. The existing loop, ensemble and bands are reused. A new `amp/asp_finite.py` runs ASP on
`amp/sk.py`'s matrices, independent of `amp/asp.py`. The SE side gains trajectory functions in
`inference/spiked_diluted.py` and `amp/asp.py`. The catalog steps are wired in `catalog_spiked_diluted.py`
and `catalog_amp.py`.

**Tech Stack:** Python 3, numpy, scipy, mpmath. Tests are plain `test_*` functions run by
`python -m replicant.selftest` (`--full` adds `@slow`).

**Spec:** `docs/superpowers/specs/2026-11-24-finite-n-checks-design.md`. Read it first; its **[call]** items
are binding.

## Global Constraints

- Work on `main`; commit after each task with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; never commit `.superpowers`; **no push**.
- Python: `/Users/renatovicente/Projects/bootloops/.venv/bin/python3`. Run tests from `tools/` (`cd tools && python -m replicant.selftest`, or one file: `python -m replicant.tests.test_<name>` if the file has a `__main__` runner, else `python -c "from replicant.tests import test_x as t; t.test_y()"`).
- Independence:
  - `amp/asp_finite.py` imports nothing from `amp/asp.py` or `families/`.
  - The new diluted functions in `inference/spiked.py` (F, dF, the sampler and the planted start) reference nothing from `bayes/` or `inference/spiked_diluted.py`.
- Each check is pinned by an AST audit test.
- Bands are 99% Student-t over named seeds (`amp/_stats.student_band`). A branch separation must exceed 10x the summed half-widths, and that margin is fixed before the test runs. No tolerance is tuned on the test's own seeds.
- Planted controls must be able to fail, and each one is asserted to fail.
- A non-converged finite-N run is labelled (`*_converged_fraction`), never refused.
- Refusal names:
  - `unsupported:diluted-finite-n` and `unsupported:asp-finite-n` are removed everywhere, code and docs;
  - new: `precondition:gamma0` (gamma0 not a finite number > 0) and `precondition:known-init` (already exists in `spiked_amp`);
  - `gamma0` without `init=planted` is refused `precondition:known-parameter`;
  - `law` with `ansatz=1rsb` is refused `precondition:known-parameter`.
- Labelling rule (spec **[call]**):
  - SE values at a fixed t (`gamma_se_t`, `q0_se_t`, `q1_se_t`) are `exact-by-literature`, with a note that the finite-N run uses empirical plug-ins (`_gamma_hat`, empirical q0/q1), not literally the theorem's algorithm;
  - finite-N bands are `not-computed` (`amp-finite-N`), statistical, NOT final;
  - for the spectral start, `gamma_se_t` uses MV Thm 2's conditions, as `catalog_spiked_sbm._sw_amp` does.
- Strong start `gamma0 = 1` literally (not LKZ's m_0 = 1).
- State points:
  - hard phase in (IT, alg); negative control below dyn and away from it (e.g. lambda = 0.7 at rho = 0.05); not above alg;
  - weak gamma0 below the SE's unstable middle fixed point;
  - all state-point numbers come from the package's own SE and are recorded in the test docstring.
- The default tier stays within about 10 s of its current 63.7 s; heavier points are `@slow` (`replicant.testing.slow`).

## Review Focus

1. **`rho` near 1 or very small (e.g. 1e-3).** The denoiser must not overflow: cosh(a y) with a = 1/sqrt(rho) large, and e^{gamma a^2/2}. A reasonable person expects finite F, dF for |y| up to 50 a. Task 1 pins it.
2. **The CLI path with `start=0.5,0.95` for ASP.** The values arrive as decimal strings in a tuple and must be converted, not refused. A malformed pair, or q0 > q1, is `precondition:start`. Task 5 pins it.
3. **`init=planted` with `snr` <= 1 on the catalog.** It must run, since the spectral refusal applies to the spectral start only, and `init=spectral` with snr <= 1 must still refuse `precondition:snr-above-one`. Task 3 pins it.
4. **ASP with `h = 0` and a start with `q0 = 0`.** The SE stays at q0 = 0 exactly (the `asp_fixed_point` branch). The finite-N run from x^0 = 0 gives f = 0 identically at h = 0; its q0 band must agree with 0, not divide by zero. Task 4 pins it.
5. **The existing gaussian / rademacher finite-N runs are unchanged.** Same values at the same seeds. Task 2 pins it with a before/after equality on one seed.

---

### Task 1: Diluted prior in `inference/spiked.py`

**Files:**
- Modify: `tools/replicant/inference/spiked.py` (Prior registry, `_prior`)
- Test: `tools/replicant/tests/test_spiked_diluted_amp.py` (new)

**Interfaces:**
- Produces:
  - `spiked.diluted_prior(rho: float) -> Prior` (cached per rho; name `"diluted_rademacher"`, nodes `[-a, 0, a]`, weights `[rho/2, 1 - rho, rho/2]`, a = 1/sqrt(rho));
  - `spiked._prior(name, rho=None) -> Prior`, which returns `diluted_prior(rho)` for `"diluted_rademacher"` and refuses `precondition:rho` when that name has no rho or rho is outside (0, 1].
- Existing `_prior(name)` callers are unchanged.

- [ ] **Step 1: Write the failing tests**
  - `test_diluted_denoiser_matches_channels`: for rho in (0.05, 0.08, 0.5, 1.0), gamma in (0.1, 1, 5) and y on a grid of 41 points in [-10a, 10a]:
    - `|P.F(y, g) - E[x|y]| < 1e-12`, with E[x|y] computed in the test by mpmath from the three-atom posterior (the test may import `bayes.channels` as a second witness);
    - `dF` against a central difference (h = 1e-6), relative error < 1e-6.
  - `test_diluted_denoiser_stable_at_extremes`: rho = 1e-3, gamma = 10, |y| up to 50a: F and dF are finite, |F| <= a, and dF >= 0.
  - `test_diluted_sampler_moments`: 200000 samples from `default_rng(0)` have mean within 0.01, E x^2 within 0.01 of 1, and the fraction of zeros within 0.005 of 1 - rho, at rho = 0.08.
  - `test_diluted_functions_import_nothing_from_se`: an AST check over the function bodies of `diluted_prior` and the helpers it defines. No `Name` or `Attribute` may resolve to `channels`, `spiked_diluted` or `bayes`, and no `Import`/`ImportFrom` of them may appear inside those bodies.
  - `test_prior_refusals`: `_prior("diluted_rademacher")` -> `precondition:rho`; rho = 0, 1.5 or "x" -> `precondition:rho`.
- [ ] **Step 2: Run them; they fail** (`diluted_prior` not defined).
- [ ] **Step 3: Implement**
  - F(y; g) = a rho sinh(a y) / (rho cosh(a y) + (1 - rho) e^{g a^2/2}), evaluated in log-space: with u = |a y|, divide through by e^{u}.
  - dF from the posterior variance: dF = Var[x|y] = E[x^2|y] - F^2, where E[x^2|y] = a^2 rho cosh(a y) / (same denominator).
- [ ] **Step 4: Run them; they pass.** Also run `test_spiked.py` and `test_spiked_diluted.py`, which must stay green.
- [ ] **Step 5: Commit** `replicant: diluted_rademacher Prior for finite-N spiked AMP (closed-form denoiser, independent of bayes/channels)`.

### Task 2: Planted start and the diluted SE trajectory

**Files:**
- Modify: `tools/replicant/inference/spiked.py` (`spiked_amp`, `spiked_amp_ensemble`, `AMPEnsemble`)
- Modify: `tools/replicant/inference/spiked_diluted.py` (new `diluted_se_trajectory`)
- Test: `tools/replicant/tests/test_spiked_diluted_amp.py`

**Interfaces:**
- Consumes: `_prior(name, rho)` from Task 1.
- Produces:
  - `spiked_matrix(N, snr, prior, seed, rho=None)`: x drawn from `_prior(prior, rho)`; same rngs as today.
  - `spiked_amp(Y, prior, *, snr, iters, seed, init="spectral", reaction=True, tol=1e-12, rho=None, gamma0=None, x0=None)`:
    - `init="planted"` needs `gamma0` and `x0`. It starts from x^0 = gamma0 x0 + sqrt(gamma0) g, with g ~ N(0, 1) from `default_rng([seed, 4])`. It does not call eigsh, and runs at any lambda > 0.
    - `init="spectral"` keeps `precondition:snr-above-one`.
    - `gamma0` must be finite and > 0, else `precondition:gamma0`.
    - With the spectral start, `gamma0` or `x0` -> `precondition:known-parameter`.
  - `spiked_amp_ensemble(N, snr, prior, *, seeds, iters, reaction=True, rho=None, init="spectral", gamma0=None)`: `AMPEnsemble` gains `gamma_t: list` (each run's `gamma_trace[-1]`), `bands["gamma_t"]`, `init` and `gamma0`. The note names the start and says gamma_t is the empirical plug-in.
  - `spiked_diluted.diluted_se_trajectory(snr, rho, gamma0, iters) -> list[float]`: gamma_0 = gamma0 and gamma_{t+1} = lambda^2 `diluted_overlap(rho, gamma_t)`, for t = 0..iters (length iters + 1).

- [ ] **Step 1: Write the failing tests**
  - `test_diluted_se_trajectory_reaches_both_branches`: at the chosen rho and lambda in (IT, alg), the weak gamma0 trajectory tends to `se.uninformative.gamma` and the gamma0 = 1 trajectory to `se.informative.gamma`, within 1e-8 at t = 2000. Here `se = diluted_state_evolution`.
  - `test_planted_amp_tracks_se_trajectory`: one state point in (IT, alg) and gamma0 = 1, at the default-tier N, seeds and t chosen by the builder. The SE value gamma_t (from `diluted_se_trajectory`) lies inside the `gamma_t` band, and the SE overlap at t, `diluted_overlap(rho, gamma_t)` (= E[x F(x^t; gamma_t)]), lies inside the `q` band from `amp_metrics`. The test may import `spiked_diluted`; only the finite-N code may not.
  - `test_planted_amp_no_onsager_fails`: the same point with `reaction=False`: the SE value at t lies outside the `gamma_t` band.
  - `test_existing_priors_unchanged`: `spiked_amp_ensemble(400, 1.5, "rademacher", seeds=(0,), iters=10)` gives the same `q` list before and after. Capture the before value by running it at the start of the task and paste it as a literal with 15 significant digits.
  - `test_planted_refusals`: planted without gamma0 -> `precondition:gamma0`; gamma0 = 0, -1 or nan -> `precondition:gamma0`; spectral with gamma0 -> `precondition:known-parameter`; spectral with snr = 0.9 -> `precondition:snr-above-one`.
- [ ] **Step 2: Run them; they fail.**
- [ ] **Step 3: Implement** as in Interfaces. The planted run's `gammas[0]` is `_gamma_hat(x^0)`.
- [ ] **Step 4: Run them; they pass.** The existing spiked tests stay green.
- [ ] **Step 5: Commit** `replicant: planted start for spiked AMP and the diluted SE trajectory`.

### Task 3: Catalog wiring for the diluted prior, plus the two-branch tests

**Files:**
- Modify: `tools/replicant/catalog_spiked_diluted.py` (`stray`, `steps`, `finality`, `META`, docstring)
- Modify: `tools/replicant/catalog_spiked_sbm.py` (`parameters["amp"]` += `"init", "gamma0"`)
- Test: `tools/replicant/tests/test_spiked_diluted_amp.py`; update any test asserting `unsupported:diluted-finite-n` (grep `tests/`)

**Interfaces:**
- Consumes: `spiked_amp_ensemble(..., rho, init, gamma0)` and `diluted_se_trajectory` from Task 2.
- Produces:
  - **Step.** `fstep("diluted-amp-finite-n", <MV + BM cite>, _finite, when=diluted and N given)`, after `diluted-state-evolution`.
  - **Values.** `q_amp`, `overlap_amp`, `mmse_amp`, `matrix_mmse_amp`, `gamma_amp_t`, each with `<key>_halfwidth99`; `gamma_se_t`; `init`, `gamma0`, `N`, `seeds`, `iters` and `amp_converged_fraction`.
  - **Defaults.** `seeds` `tuple(range(8))`, `iters` 30, `init` `"spectral"`.
  - **`gamma_se_t` source.**
    - planted: `diluted_se_trajectory(...)[iters]`;
    - spectral: `diluted_state_evolution(snr, rho).uninformative.gamma`.
  - **Refusals in `stray`.**
    - `unsupported:diluted-finite-n` is removed.
    - N needs snr and rho: `precondition:state-point` / `precondition:rho`.
    - Without N, the finite-N parameters (`seeds`, `iters`, `init`, `gamma0`) are refused `precondition:known-parameter`.
    - `gamma0` without planted is refused `precondition:known-parameter`.
- **Finality.**
  - The amp keys get the `not-computed` statistical label.
  - `gamma_se_t` gets `exact-by-literature` (planted) or the MV Thm 2 conditions (spectral, snr > 1 and `se_phase == "easy"`), else `not-computed`. Use `v.get("init")` to pick.
  - The META keys are added.

- [ ] **Step 1: Write the failing tests**
  - `test_diluted_catalog_planted_runs_below_one`: `catalog()["spiked_wigner"].execute("amp", prior="diluted_rademacher", rho=<r>, snr=<lam < 1 in (IT, alg)>, N=<n>, init="planted", gamma0=1, seeds=(0, 1, 2), iters=<t>)` returns `gamma_amp_t`, `gamma_se_t` and `init == "planted"`, with `gamma_se_t` final and `gamma_amp_t` not final (via `compose(...).run` `finality`).
  - `test_diluted_catalog_refusals`: `N` without snr; `gamma0` without planted; `init` without N; spectral with snr 0.95 -> `precondition:snr-above-one`. And `grep -rn "unsupported:diluted-finite-n" tools/replicant` returns nothing (asserted in the test by reading the package files).
  - `@slow test_diluted_two_branches_in_the_hard_phase`:
    - at the chosen point, the weak gamma0 and gamma0 = 1 runs end with `gamma_amp_t` bands whose means differ by more than 10x the summed half-widths;
    - each band contains its own `gamma_se_t`.
  - `@slow test_diluted_negative_control_below_dyn`:
    - at lambda = 0.7 (or the builder's recorded point below dyn), the two starts' `gamma_se_t` differ by less than the summed half-widths;
    - each band contains its own `gamma_se_t`.
  - `test_diluted_spectral_easy_phase`: lambda = 1.5 at rho = 0.08, spectral: `gamma_se_t` inside the band.
- [ ] **Step 2: Run them; they fail.**
- [ ] **Step 3: Implement** as in Interfaces; update the module docstring (drop "Finite-N AMP ... is refused").
- [ ] **Step 4: Run** the new file and `test_spiked_diluted.py`, `test_spiked.py` and `test_amp_route.py`; then the default selftest, which must be green.
- [ ] **Step 5: Commit** `replicant: finite-N AMP on the diluted spiked prior (spectral and planted starts); unsupported:diluted-finite-n removed`.

### Task 4: `amp/asp_finite.py` and the ASP SE trajectory

**Files:**
- Create: `tools/replicant/amp/asp_finite.py`
- Modify: `tools/replicant/amp/asp.py` (new `asp_trajectory`, `__all__`)
- Modify: `tools/replicant/amp/__init__.py` (export the new names, if the package re-exports)
- Test: `tools/replicant/tests/test_asp_finite.py` (new)

**Interfaces:**
- Produces:
  - **SE side.** `asp.asp_trajectory(beta, m, h=0, *, start=None, iters, grid=1) -> list[tuple[float, float]]`:
    - (q0_t, q1_t) for t = 0..iters, t = 0 being the start (default (0.5, 0.95));
    - each step is `_map`;
    - argument checks as in `asp_fixed_point`;
    - the q0 = 0, h = 0 branch held at 0 as there.
  - **Finite-N side.** `asp_finite.asp_amp(J, beta, m, *, h=0, iters, seed, start=None, reaction=True) -> ASPAMPResult` (dataclass: `q0_trace`, `q1_trace` (length iters + 1), `converged`, `increments`, `beta`, `m`, `h`, `seed`, `label="statistical"`). The algorithm is the spec's:
    - x^0 = sqrt(q0_start) g with g from `default_rng([seed, 2])`, q1^0 = q1_start, f_{-1} = 0;
    - f_t, f_t' by Gauss–Hermite in V under the cosh^m weight (log-space);
    - b_t = mean f_t';
    - x^{t+1} = J f_t - b_t f_{t-1};
    - q0^{t+1} = mean f_t^2 and q1^{t+1} = mean <tanh^2 Xi>.
  - **Ensemble.** `asp_finite.asp_amp_ensemble(N, beta, m, h=0, *, seeds, iters, start=None, reaction=True) -> ASPAMPEnsemble`:
    - J from `amp.sk.sk_matrix(N, seed)`;
    - fields `q0`, `q1` (last t per seed), `bands` {"q0", "q1"}, `converged`, `seeds`, `N`, `iters`, `note`.
  - **Hook.** `asp_finite._HOOKS = {"reweight": True}`. False sets m = 0 in both f_t and f_t'.

- [ ] **Step 1: Write the failing tests**
  - `test_asp_trajectory_ends_at_fixed_point`: `asp_trajectory(2, m, iters=3000)[-1]` equals `asp_fixed_point(2, m)` (q0, q1) within 1e-12, with m from `sk_1rsb(2)` as `test_asp.py` does.
  - `test_asp_denoiser_derivative`: f' against a central difference (h = 1e-6), relative error < 1e-7, at beta 2 and 3, the sk_1rsb m, q1 - q0 in (0.1, 0.5), x in [-3, 3].
  - `test_asp_quadrature_order`: the chosen Gauss–Hermite order and twice it agree on f, f' and <tanh^2> to 1e-10 at the same points.
  - `test_asp_finite_tracks_se`: beta = 2, m = sk_1rsb's, h = 0, the builder's N, seeds and t: the SE (q0_t, q1_t) lie inside the q0 and q1 bands.
  - `@slow test_asp_finite_tracks_se_more_points`: beta = 3 at h = 0, and beta = 2 at h = 0.3 with that h's sk_1rsb m.
  - `test_asp_finite_controls_fail`:
    - `reaction=False`: the SE q0_t lies outside the band;
    - `_HOOKS["reweight"] = False`: the SE q1_t (at sk_1rsb's m) lies outside the band. Measure the gap first; if it is below 10x the half-width, use the point with larger m recorded per the spec and say so in the docstring.
  - `test_asp_finite_q0_zero_start`: h = 0 and start (0, 0.9): every q0 trace is 0, the q0 band is (0, 0), and nothing is NaN.
  - `test_asp_finite_imports_nothing_from_asp`: AST check that `asp_finite.py` has no import of `.asp`, `replicant.amp.asp` or `families`.
- [ ] **Step 2: Run them; they fail.**
- [ ] **Step 3: Implement**, with a module docstring stating:
  - the model;
  - the plug-in caveat;
  - independence (Gauss–Hermite vs asp.py's trapezoid; no import of asp.py);
  - the hooks.
- [ ] **Step 4: Run** the new file and `test_asp.py`; they pass.
- [ ] **Step 5: Commit** `replicant: finite-N approximate survey propagation (amp/asp_finite.py) and the ASP SE trajectory`.

### Task 5: Catalog wiring for `sk` amp with `ansatz=1rsb`

**Files:**
- Modify: `tools/replicant/catalog_amp.py` (`_sk_ansatz`, new `_sk_asp_finite`, `sk_amp_route` steps, `fin`, `meta`, declared params += `"start"`)
- Test: `tools/replicant/tests/test_asp_finite.py`; update any test asserting `unsupported:asp-finite-n`

**Interfaces:**
- Consumes: `asp_amp_ensemble` and `asp_trajectory` from Task 4.
- Produces:
  - **Step.** `fstep("asp-finite-n", _ASP_CITE + " (finite-N iterates: statistical)", _sk_asp_finite, when=1rsb and N given)`, after `asp-state-evolution`.
  - **Values.** `q0_amp`, `q1_amp`, each with `_halfwidth99`; `q0_se_t`, `q1_se_t`; `N`, `seeds` (default `tuple(range(6))`), `iters` (default 20), `start` and `asp_converged_fraction`.
  - **Refusals.**
    - `law` with 1rsb -> `precondition:known-parameter`.
    - Without N, the parameters `seeds`, `iters` and `start` -> `precondition:known-parameter` (the existing "need N" pattern).
    - `start` malformed -> `precondition:start` (from `asp_trajectory`).
- **Finality.**
  - `q0_se_t`, `q1_se_t`: `exact-by-literature` (SE at a fixed t, with the plug-in note).
  - `q0_amp`, `q1_amp`: `_stat`.

- [ ] **Step 1: Write the failing tests**
  - `test_sk_asp_catalog_finite`: `compose(spec, "amp").run(beta=2, m=<sk_1rsb m as a decimal string>, ansatz="1rsb", N=<n>, seeds=(0, 1, 2), iters=20)` reports `q0_se_t`, `q1_se_t` (final) and `q0_amp`, `q1_amp` (not final), with `q0_asp` still present.
  - `test_sk_asp_catalog_start_from_cli_strings`: `start=("0.5", "0.95")` runs; `start=("0.9", "0.5")` -> `precondition:start`.
  - `test_sk_asp_catalog_refusals`: `law="gaussian"` with 1rsb -> `precondition:known-parameter`; `iters` without N -> `precondition:known-parameter`. And `unsupported:asp-finite-n` appears nowhere in `tools/replicant` (file scan).
- [ ] **Step 2: Run them; they fail.**
- [ ] **Step 3: Implement** as in Interfaces.
- [ ] **Step 4: Run** the new tests, `test_amp_route.py` and `test_asp.py`, then the default selftest; all pass.
- [ ] **Step 5: Commit** `replicant: finite-N ASP on the sk amp route (ansatz=1rsb); unsupported:asp-finite-n removed`.

### Task 6: Docs and timing

**Files:**
- Modify: `tools/replicant/GUIDE.md`, which needs these changes:
  - INPUTS for `spiked_wigner` (diluted) and `sk` amp;
  - the refusal table: the two `unsupported:*` names removed, `precondition:gamma0` added;
  - the K5 "Not implemented" sentence, which keeps only the Wishart spectral start and BBP;
  - the INVOKE signatures of the Task 1, 2 and 4 functions;
  - a worked example for each new run.
- Modify: `tools/replicant/README.md` and `selftest.py`'s summary line, with the test counts and times.
- Modify: `docs/wiki/Catalog-and-Routes.md`, `docs/wiki/Automated-Tests.md` and `docs/wiki/Home.md`, with the counts and the new parameters line in "Existing entries gained parameters".
- Modify: `CHANGELOG.md`, with an Unreleased/Added entry.

- [ ] **Step 1:** `grep -rn "diluted-finite-n\|asp-finite-n" tools docs` returns nothing.
- [ ] **Step 2:** Run each GUIDE worked example literally (`cd tools && python3 -m replicant run ...`) and paste the printed numbers into the GUIDE.
- [ ] **Step 3:** Time the default tier (`time python -m replicant.selftest`) and the full tier (`--full`, in the background) and record both, with the load averages at start and end, as in commit baf3994.
- [ ] **Step 4: Commit** `replicant: finite-N checks docs (GUIDE, README, wiki, CHANGELOG) and re-timing`.
