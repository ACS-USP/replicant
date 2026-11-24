# The Krzakala–Zdeborová lecture notes in `replicant`: design

Status: approved in conversation (2026-11-23); this document records it.
Marks: **[RV]** = decided by Renato; **[call]** = my call, open to override.

## Intent

**[RV]** Add to `replicant` every method presented in F. Krzakala and L. Zdeborová, *Statistical Physics
Methods in Optimization and Machine Learning: An Introduction to Replica, Cavity & Message-Passing
techniques* (lecture notes, version of June 9, 2024, 287 pp.; the user's copy, `Notes.pdf`).

The notes present 121 methods and computations across 18 chapters. replicant already has 21, has 41 in part
and lacks 59 (gap maps of 2026-11-23, kept in the build workspace, not committed). The gaps are mostly the
generic machinery behind entries replicant already has for one model: BP on an arbitrary factor graph with
its Bethe free entropy, q-state models (Potts, coloring, the SBM with q > 2), the scalar Bayes toolbox that
AMP denoisers are built from, rectangular AMP and GAMP with a general channel, and the convex problems
(LASSO, compressed sensing).

Success means:
- every method of the notes is in replicant as a recorded derivation and a run, or is listed in the GUIDE
  as not implemented with the reason;
- each result is checked by a route that did not produce it: exact enumeration, finite-N BP / AMP /
  convex solves over named seeds, Monte Carlo, `eigvalsh`, or an existing entry;
- the generic layers (factor-graph BP, q-state BP, scalar denoisers, GAMP channels) are reused across
  entries, and a nearby model the notes do not treat runs through the same code and says what it reused.

## Sources [RV for the notes; call for the primaries]

The notes are the guide, not the bank. They print few numbers: most results are closed forms or
figures. The build banks from:
1. the notes themselves, cited by equation, section or figure, for the numbers they print. Examples are
   the Bethe–Peierls inverse temperatures for c = 2d, d = 2..5 (Sec. 4.2), the SBM thresholds c_d = 12.84,
   c_c = 13.23, c_l = 16 at q = 5, c_in = 0 (Ch. 10), and the colorability table (Ch. 14);
2. the primary papers the notes rest on, fetched during the build. Pointers to confirm (venues, years and
   equation numbers are checked when fetched):
   - Decelle, Krzakala, Moore & Zdeborová, PRE 84, 066106 (2011), already cited by `sbm`;
   - Krzakala et al., PNAS 110, 20935 (2013), the non-backtracking spectrum;
   - Zdeborová & Krzakala, PRE 76, 031131 (2007), coloring;
   - Krzakala, Montanari, Ricci-Tersenghi, Semerjian & Zdeborová, PNAS 104, 10318 (2007);
   - Krzakala & Zdeborová, PRL 102, 238701 (2009), quiet planting;
   - Lesieur, Krzakala & Zdeborová, J. Stat. Mech. (2017) 073403, low-rank matrix estimation;
   - Baik, Ben Arous & Péché, Ann. Probab. 33, 1643 (2005), the BBP transition;
   - Antenucci, Franz, Urbani & Zdeborová, PRX 9, 011020 (2019), approximate survey propagation;
   - Donoho, Maleki & Montanari, PNAS 106, 18914 (2009), AMP for the LASSO;
   - Bayati & Montanari, IEEE Trans. Inf. Theory 58, 1997 (2012), the LASSO risk;
   - Krzakala, Mézard, Sausset, Sun & Zdeborová, PRX 2, 021005 (2012), compressed sensing;
   - Guo, Shamai & Verdú, IEEE Trans. Inf. Theory 51, 1261 (2005), I-MMSE;
   - Marchenko & Pastur (1967) and Wigner (1958), or a textbook that states them.
   A paper that cannot be fetched is reported, and its numbers are not banked.

**Errata [RV: not banked as printed].** The gap maps found misprints in the notes, among them Eq. (11.26)
against (11.25), Eqs. (12.62), (12.163)–(12.164), (12.173) and (18.17)–(18.18). Two were confirmed against
values already banked (`reference/ridge.json`). Each one is recorded in the bank's `cite` field or the
GUIDE with the corrected form and the route that confirms it. Formulas damaged by the text extraction
(13.9, 15.29, 16.25, 16.70–16.72, 17.1, 18.38–18.39, the Ch. 14 table) are reread from the PDF pages.

**Conventions.** Each entry states its own and maps the notes into it:
- the notes' spiked-model λ is λ_LM = λ_replicant²;
- the notes' α in the regression chapters is 1/γ of `learning/ridge.py`;
- the notes' Potts β is twice the Ising β and has the opposite sign for the ferromagnet.

## Shape [call]

Seven extensions (K1–K7) and one documentation task (K8):

| | Extension | Notes | New code | Entries |
|---|---|---|---|---|
| K1 | Mean-field foundations | Ch. 1–3 | `meanfield/` (Curie–Weiss, fully connected RFIM), `rmt/` (Stieltjes transforms) | `curie_weiss`, `rfim`, `random_matrix` |
| K2 | Bayes-optimal denoising and the REM extras | Ch. 7, 15 | `bayes/` (scalar channels, priors, denoisers, Nishimori / Stein / I-MMSE validators, sparse-vector denoising) | `scalar_denoising`, `sparse_denoising`; `rem` gains the participation ratio and the overlap distribution |
| K3 | Generic factor-graph BP | Ch. 4–6 | `factor/` (factor graphs, BP, Bethe free entropy, population dynamics over a factor law), `coloring/` (Potts / coloring BP) | `rfim` gains `cavity`; `potts_bethe`, `coloring` (`cavity`) |
| K4 | SBM with q groups | Ch. 10 | `inference/sbm_q.py`, `inference/nonbacktracking.py`, EM parameter learning | `sbm` gains `q`, `planted`, `learn` |
| K5 | Low rank and AMP | Ch. 8, 9, 11, 12 | `inference/wishart.py` (rectangular AMP and its two-overlap state evolution, BBP), diluted-Rademacher prior, `amp/asp.py` | `spiked_wishart` (`amp`); `spiked_wigner` gains `prior=diluted_rademacher`; `sk` route `amp` gains `ansatz=1rsb` (ASP) |
| K6 | Linear estimation and convex problems | Ch. 12, 16, 18 | `linear/` (GAMP with a channel × prior interface, compressed-sensing state evolution, LASSO AMP, proximal gradient, an exact LASSO solver) | `lasso`, `compressed_sensing` (`amp`, `replica`); `ridge` gains a covariance spectrum |
| K7 | Coloring with clusters | Ch. 13–14 | `coloring/` 1RSB cavity, frozen survey propagation, planted coloring, rigidity | `coloring` gains `ansatz=1rsb`, `planted` |
| K8 | Documentation and final review | — | GUIDE, README, wiki, deck, full-tier re-timing | — |

- **No new route names.** The existing eight cover everything: `replica`, `cavity`, `amp`, `exact` (Curie–Weiss at finite N, scalar channels, random-matrix densities), and so on. A new route is a ruling, not a default.
- **Generic layers first.** K3's `factor/` is the BP that K4 and K7 specialise. K2's denoisers are the ones K5 and K6 plug into AMP. The existing single-model code (`cavity/sparse.py`, `inference/sbm.py`, `amp/`) is left as it is. The new layers reproduce its numbers as a cross-check: q = 2 SBM against `sbm`, and the q = 2 Potts against `bethe_ising`.
- **Validators.** The Nishimori, Stein and I-MMSE identities (K2) are checks that run on results of the right class (`spiked_wigner`, `teacher_student`, `sbm`, the new denoising entries), each with a planted violation. This is item 7 of the generalisation notes, done for the identities these notes present.

## Checks (each must be able to fail)

- **Exact small systems.** Curie–Weiss Z_N by its binomial sum. RFIM ground states by exact sorting at finite N. BP and the Bethe free entropy against enumeration on trees and on small loopy graphs (where Bethe is an approximation and is labelled so). Exhaustive q-coloring counts at small N.
- **Finite-N algorithms over named seeds.** BP on sampled graphs (Potts, coloring, SBM q). AMP and GAMP at N ~ 10³ against their state evolution. LASSO by an exact convex solver (coordinate descent) against AMP's fixed point. Non-backtracking spectra by sparse eigensolvers. Spiked Wishart by SVD against BBP.
- **Existing entries as independent routes.** The REM for sparse denoising. `bethe_ising` for the q = 2 Potts. `sbm` for the q = 2 case. `sk` 1RSB (`families/sk.py`) for ASP's fixed point. The ridge closed forms for the LASSO at λ → 0 where they coincide, and for ridge with an isotropic covariance.
- **Planted errors** go through module hooks restored with identity asserts, as in every earlier series. Statistical tests state their power against the plant (≥ 0.9).

## Labels

- Replica, cavity, BP and AMP state-evolution values are physics-level unless a fetched theorem proves them (for example the Bayes-optimal spiked models, the LASSO risk of Bayati–Montanari, the scalar channels). Those are `exact-by-literature`, with the cite.
- The Bethe free entropy on a loopy finite graph is an approximation; on a tree it is exact. Say which.
- Scalar Bayes integrals and random-matrix densities evaluated in balls are `certified` where the evaluation is a true enclosure.
- BP / AMP / Monte Carlo at finite N are statistical and never final. A run whose diagnostic fails (BP not converged, two fixed points from planted and random starts with no free-entropy decision) is labelled `undetermined` for the affected values.

## Goals and non-goals

Goals:
- Every method of the notes implemented, or listed as not implemented with the reason.
- The generic layers reused across at least two entries each.
- Default battery growth ≲ 7 s for K1–K7, about 1 s per extension; heavy runs in `@slow`.

Non-goals [call]:
- The rigorous proofs in the notes (Guerra interpolation as a proof, the AMP-based proofs of Ch. 12.4–12.5, Appendix 1.C). They are cited as the theorems behind `exact-by-literature` labels. Guerra's bound is computed where it gives a number (RFIM), as `sk_bounds` already does for SK.
- The probability appendix (Ch. 19).
- Deep networks and kernels, which the notes do not cover.

Anti-goals:
- A number banked from the notes where the notes are wrong (the errata above).
- A Bethe free entropy on a loopy graph called exact.
- A first-order transition point stated where the planted and random BP runs did not both converge.

## Execution [call]

The same discipline as the earlier series:
- a plan with one task per extension, executed subagent-driven, with a review per task and a final whole-branch review;
- work directly on `main`, the user's standing choice, with no push until the user asks.

Order: K1, K2, K3, K4, K5, K6, K7, K8. K1 and K2 are cheap and give the bases. K3 is the BP that K4 and K7 need. K5 and K6 reuse K2's denoisers. K8 documents everything and re-times the full tier.

## As built

Deviations of the implementation from the text above, taken from the build ledger
(`.superpowers/sdd/2026-11-23-kz-notes/progress.md`, not committed), the task reports and the code.
Commits 9db72fa (this design and its plan) .. f6df7f8 (K7 fix round), then the documentation task K8. Every task had
one review and one fix round; every re-review was clean.

- **Shape.** The seven extensions as designed, with these file-level deviations: `catalog_index.py` holds the
  registry that was inside `catalog()` (K1, behaviour unchanged, tested); `meanfield/cw_exact.py` (K1),
  `coloring/rsb_pop.py` and `catalog_coloring.py` (K7) were split out to keep files under 400 lines; the K3 `rfim`
  cavity route lives in `catalog_rfim_sparse.py` and K4's q path in `catalog_sbm_q.py`; `rem`'s condensation values
  (K2) sit behind a new parameter `condensation=true` (default false), because adding them unconditionally made old
  `rem` runs non-final. K4's `sbm` gained `q`, `c_in`, `c_out`, `init` (planted | random), `learn` and the population
  parameters (the design's `planted` is `init=planted`). Interfaces that differ from the plan: K7's thresholds take
  `(q, graph)`, not `(q, c)`; `sbm_q_transitions` needs a seed and returns a `Transitions` object;
  `sparse_vector_free_entropy(Delta, d=None)` has `d` optional (None: the limit).
- **Routes.** No new route name. `curie_weiss` uses `replica`, `cavity` and `exact`; `scalar_denoising` `exact`;
  `random_matrix` `replica` and `cavity`.
- **Generic layers.** `factor/` (K3) is reused by `coloring/` in K3 and K7 (BP on sampled graphs, Bethe counts) and its
  relaxation diagnostic is shared with K7's 1RSB populations (`factor.population.relaxation`, a review ruling). K4's
  q-group BP is specialised (`inference/sbm_q.py`: the non-edge field needed a hook `factor/bp.py` does not have;
  ruling: allowed, with the q = 2 check against `sbm.py` independent and AST-audited). K7's 1RSB population is its
  own engine (the Z^m reweighting; ruling). `bayes/channels.py` (K2) is the one place for priors and denoisers:
  K5's diluted prior was first duplicated in `inference/spiked_diluted.py` and moved into channels in the fix round
  (the unit-variance `diluted(rho)`, amplitude exact as `Sqrt(1/rho)`); K6's GAMP uses its denoisers, and
  `channels.sample` is shared by K5's Wishart and K6's GAMP instances.
- **Cross-checks against existing entries.** As designed: q = 2 Potts against `bethe_ising`, q = 2 SBM against `sbm`,
  ASP against `sk_1rsb`, the REM for sparse denoising, the ridge closed forms for the LASSO at lam = 0 and for ridge
  with an identity covariance (all 14 `ridge.json` keys). Golden values recorded from a pre-K5 export
  (`tests/golden_58953a7.json`) pin the old `spiked_wigner` and `sk` amp outputs. The Nishimori / Stein / I-MMSE
  validators run on `spiked_wigner` and `teacher_student` results (not on `sbm`).
- **Labels.** As designed, plus: an entry's rigor may be a callable of the values (K2's `scalar_denoising`:
  certified for the gaussian and +-1 priors, a quadrature label otherwise; `compose.py`). Labelled `not-computed`
  where no fetched theorem covers the value: the `random_matrix` densities and edges (Wigner and Marchenko-Pastur
  not fetched), the q >= 3 KS points, the Wishart thresholds (BBP proves the complex case), the diluted-prior
  `overlap_se` / `mmse_se`, the LASSO non-zero fraction `b` (BM's theorem needs pseudo-Lipschitz functions).
  `undetermined` (never refused): Potts first-order points without both starts converged, K4's first-order pair and
  EM decisions on non-stationary runs, the diluted SE without both starts, the ER q = 3 coloring c_c (only an upper
  bound) and the regular q = 3 c_d, c_c (ZK's critical "5+"). One data-dependent refusal: an ER first-order
  clustering transition is refused `unsupported:er-first-order` after the runs show it (the ER 1RSB path is
  continuous-only).
- **Sources.** Fetched and read: GSV, DKMZ, KMMNSZZ, LKZ, BBP, AFUZ, ASP, DMM, BM, KMSSZ, Rangan, HMRT, ZK (REFERENCES.md,
  "Krzakala-Zdeborova lecture notes"). Not fetched, nothing banked from them: Wigner (1958), Marchenko-Pastur (1967),
  Krzakala-Zdeborova PRL 102 (2009) (quiet planting from the notes only), Mulet et al. (2002); KMRSZ (PNAS 104) was not
  reread. Pointer corrections: AFUZ is arXiv:1805.05857 (the plan's 1802.01535 is an unrelated paper) and is the hard-
  phase paper, not ASP; ASP is Antenucci, Krzakala, Urbani & Zdeborova, arXiv:1807.01296 (the notes' "as shown in ?").
- **Errata.** Every misprint of the design's list was confirmed and is in the GUIDE's KNOWN HAZARDS ("ERRATA IN THE
  KZ NOTES") with the corrected form and its route; (11.26) and (12.62) are confirmed by algebra and existing entries
  (`viana_bray`'s transition equation, `sk`'s AT replicon), not by new code. Found beyond the list: the p. 33
  spinodal, p. 18's g(m), Fig. 2.2.1's axis (sqrt(Delta)), Lemmas 8-9's swapped labels, p. 230's "diverges", p. 59's
  "d = 2c", p. 106's "beta -> 0", (12.97)-(12.100)'s U* / V* swap, (16.86)'s sign, the q trace of Sec. 18.1.2, Sec.
  13.2.3's "q < 9" remark, and the p. 208 table, whose "c_col ~ 5.4 / 14.4" are the annealed thresholds (banked as
  `coloring_c_ann_q3`, `coloring_c_ann_q5`; c_col comes from ZK Table II). The "5.4" of the pre-flight scan is the
  annealed threshold once (p. 108 names it so), not a second quantity. Source errata: DKMZ p. 17, HMRT Eq. (12).
- **Bank.** No value from memory; each key has a cite and an independent route. Removed in review because no
  independent route resolved them: `cs_alpha_l1_rho0p4` (a formula evaluated in scratch, an input), `coloring_c_d_q4`,
  `coloring_c_c_q4`, `coloring_c_c_q3`, `coloring_regular_d_c_q3`. Read off figures (tolerance covering the reading):
  `cw_spinodal_fig1b4_beta1p5`, the two Fig. 2.2.1 readings, LKZ Fig. 3 (a disagreeing second source for AFUZ's
  Delta_alg). Inferred, said in the cite: Delta = 1 for the notes' scalar posteriors. UNCLAIMED (tests only): the two
  non-backtracking values of KMMNSZZ Fig. 2.
- **Not implemented.** Listed in the GUIDE ("KZ NOTES, NOT IMPLEMENTED" and the K4-K6 bullets), among them
  Curie-Weiss dynamics, the fully connected Potts model, the RFIM large-deviation free entropy, the sparse spin-glass
  game, independent sets and matchings, relaxed BP and the GLM free entropy, the probit channel, the random subcube
  model, finite-N diluted AMP and ASP, the ER first-order 1RSB path. The success criterion "every method implemented
  or listed" was audited at K8 by searching the package against the gap maps, not by a row-by-row table.
- **Pre-existing finding.** `families/sk.py`'s 1RSB at h = 0 selects a q0 > 0 stationary point (q0 = 0.2146 at
  beta = 2) by largest f, not the textbook q0 = 0 solution; ASP reproduces it. Outside this series; left to the user.
- **Time standard.** The baseline (HEAD 9db72fa, clean export, cold) was 752 tests, 56.6 s and 54.7 s on a machine at
  load average about 5 (the documented 30 s was idle-machine), so growth was measured against same-session
  baselines at each task boundary. Per task (default tier, loaded machine): K1 about +0.9 s, K2 about +1 s, K3 under
  1 s, K4 about +0.6 s, K5 about +1.5 s (accepted, ruling), K6 +0.7 s after its fix round (from +2.0 s), K7 about
  +0.9 s: about 6 s for K1-K7 against the 7 s budget. The K8 re-timing is in the CHANGELOG.
