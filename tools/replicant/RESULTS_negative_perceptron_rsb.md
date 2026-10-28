# Negative-margin spherical perceptron: how wrong is the RS capacity?

Question (paper project `limit-fig2-paper`, Appendix "Replica theory for random documents"): the RS capacity
`alpha_c^RS(kappa) = 1/E[(t+kappa)_+^2]` is used at effective margins kappa = -0.55, -0.85, -1.12, -1.36, -1.66, where the RS
point is unstable. What is the capacity there once replica symmetry is broken?

Answer, in one line: RS overestimates the capacity by 2.0 %, 5.5 %, 10.5 %, 16.3 % and 25.6 % (full-RSB estimate),
so `alpha_c^RSB / alpha_c^RS` = 0.981, 0.948, 0.905, 0.860, 0.796. The 1RSB value already captures 95-98 % of the
correction, and is **not** the capacity (it is unstable toward further RSB).

## Results (`alpha = P/N`, convention `J.xi/sqrt(N) >= kappa`, `J.J = N`)

| kappa | RS | 1RSB | 2RSB | 3RSB | full-RSB estimate | 1RSB / RS | full RSB / RS | AT line | AT / RS | AT / full RSB |
|---|---|---|---|---|---|---|---|---|---|---|
| -0.55 | 5.24614 | 5.14939 | 5.14543 | 5.14483 | 5.1447 | 0.9816 | 0.9807 | 4.2163 | 0.804 | 0.820 |
| -0.85 | 9.59817 | 9.11490 | 9.09719 | 9.09458 | 9.0941 | 0.9497 | 0.9475 | 6.3247 | 0.659 | 0.695 |
| -1.12 | 17.39327 | 15.80371 | 15.75339 | 15.74629 | 15.745 | 0.9086 | 0.9052 | 9.4879 | 0.545 | 0.603 |
| -1.36 | 30.78139 | 26.59894 | 26.48795 | 26.47314 | 26.471 | 0.8641 | 0.8600 | 14.3178 | 0.465 | 0.541 |
| -1.66 | 66.61077 | 53.34242 | 53.08077 | 53.04934 | 53.045 | 0.8008 | 0.7963 | 26.3023 | 0.395 | 0.496 |

* RS, 1RSB: audited to ~1e-18 (mpmath, 20 digits; the 1RSB root is a verified local minimum with stationarity residuals
  <= 1e-13). 2RSB, 3RSB: float64 grid, numerical error <= 1e-6 (change under a refined grid: <= 8e-8). Full RSB = the K = 3
  value minus a geometric tail of the K-differences (tail 0.0001, 0.0004, 0.0012, 0.0022, 0.0043 at the five kappa; the
  ratio of consecutive differences is 0.153, 0.148, 0.141, 0.133, 0.120).
* **The full-RSB column is an extrapolation of a three-point staircase, not the continuum Parisi solution.** Treat the
  last difference (3RSB - 2RSB: 0.0006, 0.0026, 0.0071, 0.0148, 0.0314) as the scale of its uncertainty: the ratio rests
  on one pair of differences per kappa, and a slower (power-law) decay would put the limit below the estimate by up to
  that order. Direct values: 3RSB / RS = 0.9807, 0.9475, 0.9053, 0.8600, 0.7964.
* AT line = the RS replicon zero (existing `perceptron.py` code). `AT / RS` reproduces the 0.81, 0.66, 0.55, 0.47, 0.40 of
  the paper's Table `replicon`.
* Contact index `c` (replicon = 1 - c) for K = 1, 2, 3: kappa = -0.55: 1.152, 1.078, 1.050; -0.85: 1.231, 1.115, 1.073;
  -1.12: 1.294, 1.143, 1.089; -1.36: 1.341, 1.160, 1.098; -1.66: 1.383, 1.171, 1.102. It is always above 1 (every level is
  `unstable` toward the next), decreases monotonically with K, and the full-RSB limit c = 1 (isostatic jamming, Franz-Parisi
  2016) is not reached by K = 3 -- it converges more slowly than alpha.
* Small-kappa behaviour: `alpha_RS - alpha_1RSB ~ 0.50 |kappa|^3` as kappa -> 0^-; for kappa >= 0 the 1RSB functional
  has no stationary point below 1/D(kappa) and `alpha_c(0) = 2` is exact.

## Checks that back the numbers (details in GUIDE, "NEGATIVE-MARGIN SPHERICAL PERCEPTRON AT T = 0")

* **Published values, read from the sources** (`reference/perceptron.json`): Stojnic, arXiv:2312.16531, Table 2 (ten margins
  kappa = -2.7 ... -0.3), Tables 3, 5, 6. His second full level of lifting equals our 1RSB capacity at all ten margins
  (largest difference 6.2e-4 relative, his numbers at or above ours), his third level equals our 2RSB at kappa = -2, -1.5, -1,
  -0.5 (124.8, 36.40, 12.29, 4.698), and his parameter `p2-hat` equals our `q0` (4 digits at the ten margins when found; only kappa = -1, -1.5 are banked and tested). His method (fully lifted random
  duality theory) shares nothing with the replica derivation used here. Also: Stojnic 2013 partially lifted bound
  (12.784 at kappa = -1) lies above our 1RSB 12.3196; Franz-Parisi's off-equilibrium jammed margin kappa* ~ -0.431 at
  alpha = 3 lies below the thermodynamic one, as it must.
* **Franz-Parisi / FPSUZ do not print a table of `alpha_c(kappa<0)`** (their Fig. 3 is a plot; the 1RSB SAT-phase entropy
  is FPSUZ Eq. (59), the K-RSB equations App. B). The derivation here is the zero-temperature scaling of that Eq. (59).
* **Independent routes**: (i) the scaling limit checked against Eq. (59) evaluated as written (float64 nested quadrature,
  no closed form): relative error 3.6e-4 at 1 - q1 = 1e-6, linear in 1 - q1, at both signs of the residual; (ii) the
  integer-n replica functional against a Gaussian orthant probability (Genz) to 3e-8 for (n, m) = (4, 2), (6, 3);
  (iii) two quadratures of the root; (iv) the nontrivial minimum is the only interior local minimum of A on a 60 x 60 grid at three margins (saddles and maxima are not excluded).
* **Planted controls** that must leave the banked number: wrong block structure, dropped m (and a dropped n/m power),
  wrong scaling of the log-determinant, and bank-level swaps (RS 13.27, the 2013 bound 12.784, 2RSB 12.287 vs the 1RSB 12.32).
* Labels: every 1RSB / K-step value is `unstable` (computed replicon 1 - c < 0) and not final; the full-RSB estimate is
  `not-computed`. No value here is `stable` for kappa < 0 as computed (every c > 1); `stable` would need a computed c <= 1.

## Limits

Physics-level (replica n -> 0 and the RSB hierarchy); the identification of the lifted-RDT levels with the RSB levels is
numerical (ten + four points), not proved. The continuum full-RSB solution (Parisi PDE, the kappa = 1.41574 scaling of
FPSUZ Sec. V) is not computed; only K <= 3 are tabulated for all five margins. FP16's kappa_1RSB ~ -2.05 (a discontinuous
dAT instability below it) is not reproduced. The 1RSB search is validated on kappa in [-3, -0.02].
The T = 0 1RSB of the *composed* level-(a) route stays refused (`precondition:polynomial-energy-for-t0`): this is a family
function plus catalog route.

## Reproduce

```
cd tools
python3 -m replicant run perceptron --route rsb --param kappa=-0.55                        # 1RSB, ~3 s
python3 -m replicant run perceptron --route rsb --param kappa=-0.55 --param ansatz=krsb --param levels=2   # ~30 s
python3 -m replicant run perceptron --route rsb --param kappa=-0.55 --param ansatz=frsb --param levels=3   # minutes
python3 -m replicant.families.perceptron_krsb -0.55 -0.85 -1.12 -1.36 -1.66          # the whole ladder (about 30 min)
```

## References (verified by fetching)

* S. Franz, G. Parisi, "The simplest model of jamming", J. Phys. A 49, 145001 (2016), arXiv:1501.03397.
* S. Franz, G. Parisi, M. Sevelev, P. Urbani, F. Zamponi, "Universality of the SAT-UNSAT (jamming) threshold in non-convex
  continuous constraint satisfaction problems", SciPost Phys. 2, 019 (2017), **arXiv:1702.06919** (not 1606.04380, a
  mathematics paper).
* M. Stojnic, "Negative spherical perceptron", arXiv:1306.3980; "Fl RDT based ultimate lowering of the negative spherical
  perceptron capacity", arXiv:2312.16531.
