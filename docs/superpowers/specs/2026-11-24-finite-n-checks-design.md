# Finite-N checks for the diluted spiked prior and for ASP: design

Status: approach approved in conversation (2026-11-24); this document records it.
Marks: **[RV]** = decided by Renato; **[call]** = my call, open to override.

## Intent

**[RV]** Every AMP route in `replicant` has a finite-N simulation that checks its state evolution (SE) from
outside, except two from the KZ series (K5):

- `spiked_wigner` with `prior=diluted_rademacher`. The SE, its fixed points, the spinodals, the IT threshold
  and the tricritical point exist (`inference/spiked_diluted.py`). Finite-N parameters are refused
  `unsupported:diluted-finite-n` (`catalog_spiked_diluted.py:43`).
- `sk` on the amp route with `ansatz=1rsb`: approximate survey propagation (ASP). The SE fixed point
  (q0, q1) exists (`amp/asp.py`). Finite-N parameters are refused `unsupported:asp-finite-n`
  (`catalog_amp.py:148`).

The series closes that asymmetry. It is the house rule (every value has an independent route) applied to the
two routes that lack one. No new printed value is banked. The finite-N runs are statistical checks of SE
values that the package already computes.

The ASP case matters in particular. ASP was validated against `sk_1rsb`'s (q0, q1), and at h = 0 the q0 > 0
branch is the best Guerra bound. A finite-N ASP gives a check of that branch that does not go through any
free-energy functional.

## Goals

1. **Diluted prior, easy phase.** Bayes AMP with the spectral start (lambda > 1, as today for the gaussian and
   rademacher priors), compared with the SE fixed point.
2. **Diluted prior, hard phase [RV: option (b)].** Bayes AMP with a planted start that is correlated with
   x0 at a known SNR gamma0 and independent of W. Inside a first-order window it shows both branches the SE
   predicts at a fixed t:
   - a weak start (small gamma0) stays on the uninformative branch;
   - a strong start (gamma0 = 1 literally) reaches the informative branch. **[call]** This is not the LKZ
     informative start m_0 = 1, which in this convention is gamma0 -> oo (x^0 proportional to x0). The builder
     check measured that gamma0 = 1 lies above the informative fixed point at rho = 0.05 (gamma_high = 0.575,
     0.728, 0.839 at lambda = 0.90, 0.95, 0.99), so the map's monotonicity carries it to that branch.
   The spectral start cannot do this: the window lies mostly below lambda = 1, and just above 1 a finite-N
   spectral overlap is noise.
3. **Finite-N ASP.** On an SK matrix, the ASP iteration's empirical (q0^t, q1^t) at a fixed t is compared
   with the ASP SE trajectory from the same initial law. This follows the existing `sk` rs pattern (`q_se_t`
   in `catalog_amp._sk_finite`).
4. **Planted controls** that make each check able to fail (below).

## Non-goals

- Banking any new reference value (the finite-N outputs are statistical and NOT final, like `q_amp`).
- Finite-N ASP for the ±J law (`law=pm_one`), the spectral start for the Wishart AMP, and the other GUIDE
  not-implemented items (LDPC l = 2, XORSAT K = 2, SP stability, the Ising pinned perceptron, T = 0 1RSB for
  Gaussian-integral energies). Each is a later series.
- Claiming that AMP's t -> oo limit at finite N is the equilibrium. The theorems are for a fixed t as
  N -> oo. Below the AT line ASP's iterates need not converge, so the target is the SE at a fixed t, not a
  fixed point.

## Anti-goals

- A tolerance tuned on the test's own seeds. Bands are 99% Student-t over named seeds; the separation between
  branches must exceed them by a margin fixed before the run.
- A finite-N check that imports the code it checks. The diluted denoiser must not import `bayes/channels.py`
  or `inference/spiked_diluted.py`. `amp/asp_finite.py` must not import `amp/asp.py`. An AST audit test pins
  both, as `asp.py`'s audit against `families/` does today.
- A runtime refusal where a label will do: a non-converged run is labelled, never refused.

## Approach [RV: approach 1, extend what exists]

Considered:
1. **Extend the existing modules (chosen).** The diluted prior becomes one more `Prior` in
   `inference/spiked.py`, and `spiked_amp` gains a planted start, so the loop, matrix, ensemble and bands
   are reused. Finite-N ASP goes in a new `amp/asp_finite.py`, which reuses `amp/sk.py`'s matrix generator.
   - Costs: `spiked.py` grows; the `precondition:snr-above-one` refusal becomes specific to the spectral
     start.
2. **Standalone modules** for both. Full isolation, but this duplicates the AMP loop and the ensemble, which
   are the code the existing planted controls (`reaction=False`) validate. The copies could drift apart
   without anyone noticing.
3. **Generalise `sk_amp` to take any denoiser** and run ASP through it. Rejected: it touches a banked route
   to save about 40 lines.

## Design

### Diluted prior in `inference/spiked.py`

- **The prior.** It is the unit-variance prior x in {0, ±a}, a = 1/sqrt(rho), with P(0) = 1 - rho and
  P(±a) = rho/2, the same convention as `inference/spiked_diluted.py`. Its denoiser on
  y = gamma x + sqrt(gamma) G is written in closed form in numpy:
  F(y; gamma) = a rho sinh(a y) / (rho cosh(a y) + (1 - rho) e^{gamma a^2 / 2}),
  with dF its y-derivative, computed in a numerically stable form (log-space for large |a y|).
  - Name in `PRIORS`: `diluted_rademacher` (the catalog's name).
  - It takes `rho`. **[call]** Pass it as `prior=("diluted_rademacher", rho)`, or through a
    `spiked_prior(name, rho=None)` factory: the builder picks whichever keeps the existing `_prior(name)`
    callers unchanged.
- **The planted start.** `spiked_amp(..., init="planted", gamma0=...)` starts from
  x^0 = gamma0 x0 + sqrt(gamma0) g, with g ~ N(0, 1) iid from `default_rng([seed, 4])`, independent of W. It
  needs the truth x0 as an argument (`x0=`), and `spiked_amp_ensemble` passes it.
  - SE from this start begins at gamma_0 = gamma0 exactly (the scalar channel), so the comparison target is
    the SE trajectory gamma_t from gamma0.
  - The SE side of that comparison is a trajectory function next to `diluted_state_evolution`:
    `diluted_se_trajectory(snr, rho, gamma0, iters)` in `inference/spiked_diluted.py`. The SE module may gain
    code; the finite-N module may not import it.
- **The refusals.** `precondition:snr-above-one` stays for `init="spectral"` only. A planted start runs at
  any lambda > 0. `gamma0` must be finite and > 0, else `precondition:gamma0`; `gamma0` without
  `init=planted` is `precondition:known-parameter`.

### Catalog: `spiked_wigner` with the diluted prior

- **Accepted parameters.** `N`, `seeds` (default as the plain path, 0-7), `iters` (default 30), `init`
  (`spectral` default | `planted`) and `gamma0` (required with `planted`).
- **Removed refusal.** `unsupported:diluted-finite-n` is removed. `snr` and `rho` are required with `N`, as
  `precondition:state-point` / `precondition:rho` already require.
- **The new step.** It is `diluted-amp-finite-n` in `catalog_spiked_diluted.steps()`, after the SE step, with
  `when=` N given.
- **Its values.** The bands of `q_amp`, `overlap_amp`, `mmse_amp` and `matrix_mmse_amp`
  (`<key>_halfwidth99`), `gamma_amp_t` (the band of the empirical gamma at the last t), `gamma_se_t` (SE at
  the same t from the same start: gamma0 for planted, the fixed point for spectral), `init`, `gamma0`, `N`,
  `seeds`, `iters` and `amp_converged_fraction`.
- **Stability.**
  - The amp keys: `not-computed` (`amp-finite-N`, statistical, NOT final), as `_stat` does today.
  - `gamma_se_t`: `exact-by-literature` for the planted start, the SE at fixed t (Bayati–Montanari for a
    Lipschitz denoiser with an initial condition independent of W; the diluted prior has bounded support).
  - **[call] One labelling rule for both new routes, the `sk` rs precedent.** The SE value at a fixed t is a
    statement about the theorem's algorithm, whose parameters (gamma_t here, q0^t and q1^t for ASP) are
    deterministic. The finite-N runs use consistent empirical plug-ins (`_gamma_hat` here, as
    `spiked_amp` already does; the empirical Q_t in `amp/sk.py`'s Onsager term, whose `q_se_t` is
    `exact-by-literature`). So the SE-at-t keys are `exact-by-literature`, the finite-N bands are statistical
    and NOT final, and every note says the run uses the plug-in, not literally the theorem's algorithm.
  - **[call]** For the spectral start, `gamma_se_t` is labelled with MV Thm 2's conditions, as `_sw_amp` does.

### Finite-N ASP: `amp/asp_finite.py`

- **The iteration.** It is the symmetric AMP x^{t+1} = J f_t(x^t) - b_t f_{t-1}(x^{t-1}), with the 1RSB
  denoiser f_t(x) = <tanh Xi>, Xi = beta (x + sqrt(q1^t - q0^t) V + h), where the average over V is weighted
  by cosh^m Xi.
  - Onsager: b_t = <f_t'(x^t)>, with f' = beta (<sech^2 Xi> + m (<tanh^2 Xi> - <tanh Xi>^2)) under the same
    cosh^m average. This closed form is to be verified against a finite difference in a test.
  - The V average is computed by Gauss–Hermite (`asp.py` uses a trapezoid; the two quadratures are
    independent).
  - q0^t and q1^t are the empirical plug-ins: q0^t = <f_{t-1}^2> and q1^{t+1} = mean_i <tanh^2 Xi_i>.
- **The start.** x^0 = sqrt(q0_start) g, with g iid from `default_rng([seed, 2])`, and q1^0 = q1_start.
  The default is (0.5, 0.95), the same default as `asp_fixed_point`.
- **The SE side.** A trajectory function in `amp/asp.py`, `asp_trajectory(beta, m, h, start, iters)`, returns
  (q0_t, q1_t) for t = 0..iters, reusing `_map`.
- **What is proved.** The finite-N denoiser uses the empirical q1^t, while the SE theorem is stated for a
  deterministic f_t (with deterministic q's this f_t is Lipschitz, so Bayati–Montanari covers it). The
  labelling rule is the one stated for the diluted prior: `q0_se_t`, `q1_se_t` are `exact-by-literature` as SE
  values at a fixed t, the finite-N bands are statistical and NOT final, and the note names the plug-in. The
  ASP fixed point (`q0_asp`, `q1_asp`) keeps its current label.
- **The matrix.** J comes from `amp/sk.py`'s `sk_matrix(N, seed)` (Gaussian law only).
- **API.** `asp_amp(J, beta, m, *, h=0, iters, seed, start=None, reaction=True) -> ASPAMPResult` and
  `asp_amp_ensemble(N, beta, m, h=0, *, seeds, iters, reaction=True)`, with 99% Student-t bands of q0 and q1
  at the last t (`amp/_stats.student_band`).

### Catalog: `sk` amp with `ansatz=1rsb`

- **Accepted parameters.** `N`, `seeds` (default 0-5), `iters` (**[call]** default 20: a fixed t, not a
  convergence run) and `start` (optional, the pair).
- **Removed refusal.** `unsupported:asp-finite-n` is removed. `law` with 1rsb is refused
  `precondition:known-parameter` (Gaussian only, a non-goal).
- **The new step.** `asp-finite-n`, with `when=` ansatz 1rsb and N given.
- **Its values.** `q0_amp`, `q1_amp` (bands), `q0_se_t`, `q1_se_t`, `N`, `seeds`, `iters` and
  `asp_converged_fraction` (labelled, never required).

### Planted controls (each must make its check fail)

- **Diluted, no Onsager term.** `reaction=False` at a planted start: the empirical gamma_t leaves the SE
  trajectory's band.
- **Diluted, the two branches.**
  - Inside the window: the weak and the strong start end on different branches at t, separated by more than
    the summed half-widths, and each agrees with its own SE trajectory.
  - Outside the window: both starts end on the same branch. This is the negative control. **[call]** It sits
    below dyn, far enough from it that the strong start decays to the uninformative branch within t (e.g.
    lambda = 0.7 at rho = 0.05). "Agree" means each start against its own SE trajectory, as inside the
    window, and the two SE values at t closer than the band. Above alg is not used: there the weak start
    escapes slowly near the former spinodal, so a fixed t is not a fair test.
- **ASP without the reweighting.** **[call]** The hook (`asp_finite._HOOKS`, tests only) sets m = 0 in both
  f_t and the Onsager term, so the control is a valid AMP (the RS average) and fails only because m is wrong,
  not for the same reason as the no-Onsager control. Its q1 at t must disagree with the ASP SE at the
  sk_1rsb m beyond the band. sk_1rsb's m is small (0.282 at beta = 2, 0.257 at beta = 3), so the builder
  first measures the m = 0 gap at the chosen t; if it is below 10x the half-width, the control uses a state
  point with a larger m, chosen and recorded before the test is written.
- **ASP, no Onsager term.** It fails the same comparison.
- **Independence audit.** An AST check that `asp_finite.py` imports nothing from `amp/asp.py` or
  `families/`. **[call]** For `spiked.py` the audit is scoped to the new functions (the diluted prior's F,
  dF, sampler and the planted start): their bodies reference nothing from `bayes/` or
  `inference/spiked_diluted.py`. A whole-module check would fail on today's file, because
  `spiked_state_evolution` (`inference/spiked.py:195`) imports `spiked_diluted` to dispatch the diluted SE;
  that dispatch is SE code and stays.

### State points [call: the builder fixes them from the package's own SE before writing any test]

- **The hard-phase window.** It comes from the existing SE. The two stable branches coexist on (dyn, alg);
  the two-branch test uses the hard phase proper, (IT, alg), not the metastable part (dyn, IT). At rho = 0.05
  that is lambda in (0.8645, 1.0027) (dyn 0.8494), wider than at rho = 0.08 (0.9856, 1.0097; dyn 0.9802). The
  SE's unstable middle fixed point (0.134, 0.079, 0.039 at lambda = 0.90, 0.95, 0.99) bounds the weak gamma0
  from above. The builder picks rho and lambda
  inside a window where the two SE branches at the chosen t differ by at least 10x the expected 99%
  half-width at the chosen N, and records the SE numbers in the test docstring.
- **The easy point.** A lambda well above alg (e.g. 1.5) for the spectral start.
- **ASP.** beta = 2 and beta = 3 with m from `sk_1rsb` (the points `test_asp.py` already uses), h = 0
  (q0 > 0 branch) and one h > 0 point.
- N and the seed counts are chosen so the default tier stays near its current time. The heavy points go in
  the slow tier (`@slow`).

### Docs

- **GUIDE:**
  - the INPUTS lines for `spiked_wigner` and `sk` amp;
  - the refusal table, with the two `unsupported:*` names removed and `precondition:gamma0` added;
  - the "Not implemented" sentence of K5, shortened;
  - the INVOKE signatures.
- **README, the wiki pages and CHANGELOG:**
  - README: the test counts;
  - the wiki pages Catalog-and-Routes, Automated-Tests and Home;
  - CHANGELOG: an Unreleased/Added entry.
- The selftest summary line is re-timed at the end, as after K8.

## Left to the builder

- Internal names and the exact numerically stable form of F and dF.
- The Gauss–Hermite order: chosen so the quadrature error is below 1e-10 at the test points, checked against
  a higher order in a test.
- How `rho` reaches `Prior`.
- N, the seed counts and t at each test point, within the rules above.
