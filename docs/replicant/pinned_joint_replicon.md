# Joint replicon of the pair problem with pinned directions

Status: physics-level (replica n -> 0 and the RS ansatz are not proved); local RS stability only.
Code: `tools/replicant/derive_pinned.py`, `pinned.py`, `pinned_modes.py`, `pinned_check.py`,
`pinned_report.py`; tests `tools/replicant/tests/test_pinned.py`; bank `tools/replicant/reference/pinned.json`;
data `docs/replicant/pinned_joint_replicon.json`. Reproduce the table:
`cd tools && python3 -m replicant.pinned_report --out ../docs/replicant/pinned_joint_replicon.json` (about 3 min).

## Answer

At the paper's five states (d, n, c) = (6, 16, 0), (10, 35, 0), (14, 70, 0), (20, 140, 0), (30, 350, 0) the
replica-symmetric solution of the JOINT problem is replicon-stable, at the capacity boundary and below it:

| d | n | joint replicon at capacity | = 1 - k/d | fixed-a "slice" replicon (paper, Table replicon) |
|---|---|---|---|---|
| 6 | 16 | +0.3177 | k = 4.09 | -0.5234 |
| 10 | 35 | +0.3484 | k = 6.52 | -0.8982 |
| 14 | 70 | +0.3612 | k = 8.94 | -1.2836 |
| 20 | 140 | +0.3965 | k = 12.07 | -1.6695 |
| 30 | 350 | +0.4362 | k = 16.92 | -2.2249 |

The slice column reproduces the paper's `data/replicant_checks.json` (-0.523385, -0.898203, -1.283637,
-1.669482, -2.224865): the paper's numbers are right as numbers, but that object is not the replicon of the pair
problem. Two things are different, and only the first is the one the task anticipated.

1. **The replicon proper does not mix with the pinned overlaps; the fixed-a slice was the wrong functional.**
   The replica-permutation group S_n acts on the fluctuations (dQ_ab, dM^a). The replicon irrep (dimension
   n(n-3)/2) occurs once, in the Q sector only, so by Schur's lemma its eigenvalue is the Q-Q block of the Hessian at
   fixed M. That block is the same function of (q, H) as the perceptron's, lambda_R = 1 - E''_R chi_R with
   chi_R = (1-q)^2 and E''_R = alpha E[((ln P)'')^2], H = sqrt(d)(M - kappa); the pinned direction enters only through
   the saddle and through H. Integer-n check (`pinned_check.hessian_integer_n`, n = 4, 5, entropy by matrix
   identities, energy by the slot rule, no class formula): the replicon vector leaks nothing into the M components
   (< 1e-9 relative) and the eigenvalue equals the class-formula value to 1e-9. The paper reduced the pair to the
   catalog perceptron at (alpha_eff = alpha/(1-a^2), kappa~ = -h). That perceptron has the same q -> 1 capacity
   condition, alpha_eff Psi(h) = 1, but a different functional: its replicon at capacity is 1 - alpha_eff Phi(-h) =
   1 - k/(d (1-a^2)), counting the k active constraints against the (1-a^2) d dimensions left after the pinned
   component, whereas the true fixed-a slice and the joint problem give 1 - alpha Phi(-h) = 1 - k/d. The negative
   slice numbers were an artefact of that substitution.
2. **The M-Q mixing is real, but it lives in the standard irrep, not in the replicon.** Fluctuations
   dQ_ab = e_a + e_b (sum e = 0) and the antisymmetric dM^a (sum dM = 0) belong to the same (n-1)-dimensional irrep
   and mix through a 2 x 2 block T = [[lam2, U - V], [(n-2)(U - V), W - X]] (class values in `pinned_modes.py`,
   verified against the integer-n Hessian, whose standard block closes on (e, mu) to 1e-9). The off-diagonal
   entries are not small (about 6 and 12 against diagonal -280 and -62 at n = 4, d = 6, kappa = 0.9 kappa*). A
   planted control that drops the mixing fails that check. The uniform (trivial-irrep) mode of M is the
   longitudinal mode.

The counting identity: at capacity lambda_R = 1 - k/d with k = N Phi(-h*) the number of active competitors, the
support-vector law of `code/rs_duals.py`. The minimum-norm point of a polyhedron is fixed by at most d active
constraints, so the joint replicon is positive exactly when the problem is not over-determined, which is the
convexity statement the question started from; here it comes out of the replica functional and is checked
against the paper's measured k (exact k = 4.13 at d = 6, n = 16 against d = 6).

## Table

`kappa/kappa*` = 1 is the capacity boundary (q -> 1 limit); the other rows are RS saddles with q < 1 at
kappa = f kappa*. "joint replicon" is lambda_R from the class values (n -> 0); "(audited)" is
`derive_replicon.derived_replicon` at a root checked against the DERIVED equations (rows f = 0.9). "longitudinal"
is the uniform M-mode (q stationary), "standard-sector modes" the two eigenvalues of T with the sign flipped
(positive = stable); a complex pair (kappa/kappa* = 0.5) has the stated real part. "slice replicon" is the paper's.
Positive means stable in every column.

| d | n | c | kappa/kappa* | q | joint replicon | (audited) | slice replicon | k/d | longitudinal | standard-sector modes | stable |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 16 | 0 | 1 (capacity) | 1 | +0.3177 |  | -0.5234 | 0.6823 | diverges | n/a (q = 1) | stable |
| 6 | 16 | 0 | 0.999 | 0.99961521 | +0.3353 |  |  |  | 1.589e+04 | 7943.3 , 1.10202e+7 | stable |
| 6 | 16 | 0 | 0.99 | 0.9960275574 | +0.3715 |  |  |  | 1558 | 778.876 , 97374.4 | stable |
| 6 | 16 | 0 | 0.9 | 0.9575464624 | +0.4752 | +0.4752 |  |  | 151.4 | 76.2816 , 725.048 | stable |
| 6 | 16 | 0 | 0.5 | 0.779408873 | +0.6366 |  |  |  | 29.68 | (17.7449 - 4.22754j) , (17.7449 + 4.22754j) (complex) | stable |
| 10 | 35 | 0 | 1 (capacity) | 1 | +0.3484 |  | -0.8982 | 0.6516 | diverges | n/a (q = 1) | stable |
| 10 | 35 | 0 | 0.999 | 0.9995688135 | +0.3692 |  |  |  | 1.984e+04 | 9922.52 , 8.56235e+6 | stable |
| 10 | 35 | 0 | 0.99 | 0.9955323135 | +0.4116 |  |  |  | 1949 | 974.749 , 74346.0 | stable |
| 10 | 35 | 0 | 0.9 | 0.9521497468 | +0.5292 | +0.5292 |  |  | 191.2 | 97.3407 , 536.986 | stable |
| 10 | 35 | 0 | 0.5 | 0.7534394605 | +0.7036 |  |  |  | 37.34 | (16.5976 - 6.70375j) , (16.5976 + 6.70375j) (complex) | stable |
| 14 | 70 | 0 | 1 (capacity) | 1 | +0.3612 |  | -1.2836 | 0.6388 | diverges | n/a (q = 1) | stable |
| 14 | 70 | 0 | 0.999 | 0.9995481639 | +0.3847 |  |  |  | 2.437e+04 | 12185.7 , 7.69641e+6 | stable |
| 14 | 70 | 0 | 0.99 | 0.9953045449 | +0.4327 |  |  |  | 2397 | 1199.31 , 65874.6 | stable |
| 14 | 70 | 0 | 0.9 | 0.9496036156 | +0.5618 | +0.5618 |  |  | 236.7 | 122.411 , 461.309 | stable |
| 14 | 70 | 0 | 0.5 | 0.7413171656 | +0.7455 |  |  |  | 45.87 | (17.3153 - 7.00719j) , (17.3153 + 7.00719j) (complex) | stable |
| 20 | 140 | 0 | 1 (capacity) | 1 | +0.3965 |  | -1.6695 | 0.6035 | diverges | n/a (q = 1) | stable |
| 20 | 140 | 0 | 0.999 | 0.9994917579 | +0.4237 |  |  |  | 2.79e+04 | 13949.8 , 5.91918e+6 | stable |
| 20 | 140 | 0 | 0.99 | 0.9947062602 | +0.4750 |  |  |  | 2751 | 1376.8 , 50044.2 | stable |
| 20 | 140 | 0 | 0.9 | 0.9432129565 | +0.6126 | +0.6126 |  |  | 272.8 | 146.859 , 335.133 | stable |
| 20 | 140 | 0 | 0.5 | 0.7116546582 | +0.7989 |  |  |  | 51.99 | (16.975 - 4.83161j) , (16.975 + 4.83161j) (complex) | stable |
| 30 | 350 | 0 | 1 (capacity) | 1 | +0.4362 |  | -2.2249 | 0.5638 | diverges | n/a (q = 1) | stable |
| 30 | 350 | 0 | 0.999 | 0.9994250341 | +0.4657 |  |  |  | 3.319e+04 | 16595.5 , 4.48441e+6 | stable |
| 30 | 350 | 0 | 0.99 | 0.9939972259 | +0.5228 |  |  |  | 3286 | 1646.25 , 37370.9 | stable |
| 30 | 350 | 0 | 0.9 | 0.9357029097 | +0.6687 | +0.6687 |  |  | 326.7 | (206.501 - 37.7765j) , (206.501 + 37.7765j) (complex) | stable |
| 30 | 350 | 0 | 0.5 | 0.6777364298 | +0.8534 |  |  |  | 60.64 | 12.7227 , 22.5385 | stable |
| 10 | 35 | 0.5 | 1 (capacity) | 1 | +0.4256 |  | -1.0180 | 0.5744 | diverges | n/a (q = 1) | stable |
| 10 | 35 | 0.5 | 0.999 | 0.9994466957 | +0.4473 |  |  |  | 1.287e+04 | 9649.86 , 4.94157e+6 | stable |
| 10 | 35 | 0.5 | 0.99 | 0.9942637897 | +0.4909 |  |  |  | 1266 | 949.996 , 42678.5 | stable |
| 10 | 35 | 0.5 | 0.9 | 0.9388802355 | +0.6097 | +0.6097 |  |  | 124.5 | 97.2068 , 304.905 | stable |
| 10 | 35 | 0.5 | 0.5 | 0.6923395451 | +0.7794 |  |  |  | 23.95 | (13.0876 - 4.54379j) , (13.0876 + 4.54379j) (complex) | stable |
| 10 | 35 | -0.5 | 1 (capacity) | 1 | +0.2104 |  | -0.7153 | 0.7896 | diverges | n/a (q = 1) | stable |
| 10 | 35 | -0.5 | 0.999 | 0.9997606329 | +0.2283 |  |  |  | 4.985e+04 | 12461.3 , 3.03383e+7 | stable |
| 10 | 35 | -0.5 | 0.99 | 0.9975303334 | +0.2642 |  |  |  | 4886 | 1221.55 , 269183.0 | stable |
| 10 | 35 | -0.5 | 0.9 | 0.9735140754 | +0.3679 | +0.3679 |  |  | 474.8 | 119.159 , 2003.6 | stable |
| 10 | 35 | -0.5 | 0.5 | 0.8590448777 | +0.5303 |  |  |  | 93.39 | 25.5975 , 53.147 | stable |

Reading: every joint replicon, every longitudinal mode and every standard-sector mode (real part) has the stable
sign at all 35 rows; the replicon decreases monotonically toward the capacity value as kappa -> kappa* and never
approaches zero. Negative overlap (c = -0.5) is the closest to marginal (+0.21 at capacity) and is still
positive.

## What was derived, and how

* **Model.** u on the unit sphere of R^d; k = 2 fixed unit vectors with Gram G = [[1, c], [c, 1]]; N = alpha d iid
  Gaussian patterns; constraint <u, v_a> - <u, xi_l>/sqrt(d) >= kappa for a = 1, 2. The pattern is shared, so
  the energy reads min_a M_a, M_a = <u, v_a>: threshold sqrt(d)(M_bind - kappa) in the catalog convention
  g(lambda) = Theta(lambda - kappa_p).
* **Entropic term** (hand-supplied, not the formula suggested in the task text, which was not used). With the
  pinned components of replica alpha fixed to M^alpha, the other d - 2 coordinates have squared norm
  1 - M^T G^-1 M and overlaps Q_ab - (M^a)^T G^-1 M^b, so (1/d) ln vol = (1/2) ln det [Q_ab - (M^a)^T G^-1 M^b];
  RS, n -> 0: (1/2)[ln(1-q) + (q - m2)/(1-q)], m2 = M^T G^-1 M = 2 M^2/(1+c) (tied). n = 1 gives
  (1/2) ln(1 - m2), the sphere-slice law. It is the package's own steps (`derive_rs`) that produce the functional
  phi(q, M) = (1/2)[ln(1-q) + (q - m2)/(1-q)] + alpha E ln Phi((H - sqrt(q) t)/sqrt(1-q)); the test compares it
  symbolically with this expression.
* **Capacity.** As q -> 1, phi = C(M)/(2(1-q)) + O(ln(1-q)), C(M) = 1 - m2(M) - alpha Psi(sqrt(d)(M - kappa)); the
  volume vanishes where C = C' = 0. That is the paper's m_RS(c; n, d) = max_a [a q_p - h_a/sqrt(d)], alpha Psi(h_a)
  = 1 - a^2, with a = M/q_p: reproduced to 1e-13 at the five states (the paper's optimiser is the limit, xatol
  1e-9 in a*).
* **Tied and untied.** The untied branch (only constraint 1 binds) has an entropic-only stationarity for the
  non-binding overlap, M_2 = c M_1 < M_1, which contradicts M_2 >= M_1; the package derives this and refuses
  `no-solution:untied-slack`. Only the tied RS branch exists (82-100% of the paper's pairs are tied in the exact
  solution; the RS theory has no untied saddle).
* **Not done by the generic solver.** `solve_derived_rs` does not find the saddle of this two-unknown functional in
  practical time (killed after 5 min, and after 200 s with a start vector near the root), and its extras-mode
  route (`solve_derived_stability._mode`) raises ZeroDivisionError in the quadrature tail of the second derivative of
  ln erfc on this functional. State points are therefore located in float64 and checked to be
  roots of the derived equations by `derived_replicon`'s independent quadrature; the longitudinal mode is written
  directly. These are the only departures from "everything after the hook is the generic steps".

## Acceptance

| item | result |
|---|---|
| M = 0 / no pinned direction is the catalog perceptron | `alpha_c(kappa, d, c, M=0)` vs `gardner_capacity("spherical", kappa=sqrt(d) kappa)`: < 1e-15 (3 cases); derived phi(q, M=0) equals the catalog's at that margin to 1e-12 |
| pair margin m_RS of `code/rs_pair.py` | five states, 1e-13 or better (tolerance 1e-8 in the bank); also `a*`, `h*` to the optimiser's 1e-7 |
| support-vector law of `code/rs_duals.py` | k = N Phi(-h*) equals (n-2) Phi(-h*) of the paper to 1e-6 |
| exact margins of `data/rs_validation.json` | the replica average over the exact law of c = <v_i, v_j> (density ~ (1-c^2)^((d-3)/2)) of the capacity, E_c[kappa*], against `mean_exact` of the table: (d, n) = (8, 30): 0.2563 vs 0.2432 (+5.4%); (10, 40): 0.2772 vs 0.2659 (+4.3%); (14, 80): 0.2805 vs 0.2741 (+2.3%); (20, 140): 0.3153 vs 0.3140 (+0.4%). E_c[kappa*] equals the paper code's own c-average (`rs_pair.m_rs`) to 1e-12 at all four. The paper's sampled biases (+0.0074, +0.0031, +0.0059, -0.0025) are smaller than these exact-law biases at d = 8, 10 by sampling noise (sem of a 450-pair mean 0.0044 by Monte Carlo over the law of c; the pairs come from only 3 document sets): the paper's "below 5% for d >= 8" holds here for d >= 10, and d = 8 is at 5.4%. Banked as `reference/pinned.json` (tolerance |paper bias| + 4 sem) |
| planted controls fail | wrong entropy term (blind to M): no interior capacity maximum, `no-solution:capacity`; wrong Gram (identity): capacity at c = 0.5 moves by > 0.02 (and is correct at c = 0, where the two coincide); dropped M-Q mixing: the standard block disagrees with the integer-n Hessian by > 1 |
| refusals named | `no-solution:untied-slack`, `:capacity`, `:above-capacity`, `:state-point`, `precondition:independent-pinned-directions`, `:dimension`, `:pinned-spherical-only`, `:overlap-range`, `:known-parameter`, `:consistent-parameters`, `:rs-saddle` (GUIDE refusal table) |
| finality | a value is `stable` only if the computed replicon is positive and the longitudinal and standard-sector modes have the stable sign; a negative replicon makes every label `unstable` (tested) |

## What this does not establish

* It is local stability of the RS saddle in the n -> 0 sense. It does not show that RS is the right ansatz away from
  the saddle, nor that a nonlocal instability is absent. Here it is supported by independent evidence: convexity
  of the polyhedron problem, and the agreement of the capacity with exact margins within the paper's stated bias.
* The criterion for the standard irrep at n = 0 (and the complex pairs at kappa/kappa* = 0.5) is the analytic
  continuation of the integer-n one, not a theorem. The finality does not depend on the complex rows being
  decided: their real parts are positive and the replicon, which is the standard criterion, is positive.
* The difference mode M_i - M_j is not smooth (the energy depends on the minimum: a concave corner, first-order
  stable) and is not in T.
* The ball-versus-sphere distinction of the paper is not touched: the entropy is the sphere's, so this is the
  sphere problem with pinned components, not the convex ball slice.
* The reading of the O(1) threshold sqrt(d)(M - kappa) at finite d is the paper's; the formulas are per d, and
  d - 2 versus d in the entropy is O(1/d).
* Spherical variables and two tied pinned vectors only. The paper's optimised-document problem is outside this.
* Nothing in the paper project was edited. The statement in its Appendix that "the replica stability of the
  joint problem itself was not computed" is answered here; the sentence about the fixed-a slice being unstable
  describes the catalog perceptron at (alpha_eff, -h), not the slice of this problem.
