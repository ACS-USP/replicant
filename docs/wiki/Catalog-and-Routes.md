# Catalog and Routes

The catalog holds 38 entries (`python3 -m replicant list` prints each entry with its routes and
citation). An entry is a published pipeline, made of the paper's steps with its banked reference
values. A route is a way of computing the entry: the same model through the replica method, the
cavity method, AMP, an exact solution, a series, a rigorous bound, the online-learning ODEs or
sampling. Every entry's default route is `replica`. An entry without that route must be run with
`--route` set, otherwise it is refused with `missing-step:route:replica`.

## Routes

| Route | Meaning | Rigor |
|---|---|---|
| `replica` | Replica method: RS, 1RSB or full RSB at a state point | `physics-level` |
| `cavity` | Dense cavity, sparse BP / population dynamics, sparse 1RSB, generating-function fixed points | `physics-level`; `statistical` for population dynamics; `certified` for `bethe_ising` and the percolation entries |
| `amp` | AMP / GAMP / Bayes AMP and their state evolution, finite-N with `--param N=...` | `physics-level` / `statistical` |
| `rsb` | `perceptron` at T = 0 with kappa < 0: 1RSB jamming, K-step RSB, full-RSB estimate | `physics-level` |
| `exact` | Solvable lattices, 2D percolation thresholds, exponents and Cardy's formula | `certified` |
| `series` | Exact high-temperature series and its holonomic operator | `physics-level` |
| `bound` | Guerra's upper bounds on the SK pressure | `certified` |
| `online` | N -> oo order-parameter ODEs of online learning in alpha = examples / N | `physics-level` |
| `sample` | Finite graphs, Newman-Ziff sweeps, Monte Carlo over named seeds | `statistical` |

## Entries by domain

| Domain | Entries | Routes |
|---|---|---|
| Spin glasses | `rem`, `sk`, `pspin`, `ising_pspin`, `viana_bray` | `replica`; `sk` also `amp`, `cavity`; `viana_bray` `cavity` only |
| Neural networks, capacity | `perceptron`, `hopfield` | `replica`, `cavity`; `perceptron` also `rsb` |
| Learning | `teacher_student`, `ridge` | `teacher_student`: `replica`, `amp`; `ridge`: `replica` |
| Dense associative memory | `dense_memory`, `dense_memory_ising` | `replica` |
| Constraint satisfaction | `xorsat`, `ksat` | `cavity` |
| Inference | `ldpc_bec`, `sbm`, `spiked_wigner` | `cavity`; `spiked_wigner` `amp` only |
| Exact lattices | `ising1d`, `ising2d`, `ising2d_anisotropic`, `ising_triangular`, `ising_honeycomb`, `bethe_ising` | `exact`; `bethe_ising` `cavity` |
| Series | `series_ising2d` | `series` |
| Bounds | `sk_bounds` | `bound` |
| Online learning | `online_perceptron`, `online_linear`, `online_drift`, `online_tcm`, `online_scm` | `online` |
| Percolation | `percolation_bethe`, `percolation_graph`, `percolation_lattice`, `percolation_kcore`, `percolation_bootstrap` | `cavity` / `exact` and `sample` |
| Monte Carlo | `cluster_mc`, `wang_landau_mc`, `histogram_reweighting_mc`, `population_annealing_mc` | `sample` |

## Where an entry lives

Each entry is assembled in a `catalog_*.py` module and has a matching `latex_*.py` module that
describes its equations. For example, `catalog_csp.py` + `latex_csp.py` for `xorsat` / `ksat`,
`catalog_mc_cluster.py` + `latex_mc_cluster.py` for `cluster_mc`, and
`catalog_percolation_bootstrap.py` for `percolation_bootstrap`. The numerical engine sits in the
domain subpackage (`csp/`, `montecarlo/`, `percolation/`, ...). The banked values are in
`reference/<entry>.json`, and some files group several entries (`lattice.json`, `percolation.json`,
`mc.json`).

## Parameters

Each route reads a fixed set of parameter names, `CatalogEntry.parameters[route]`. Any other name
is refused with `precondition:known-parameter`, and the refusal lists the accepted names. A typo
such as `kapa` is never silently ignored. The full per-entry list is the INPUTS section of
`tools/replicant/GUIDE.md`.

Examples, from the GUIDE's worked runs:

```bash
cd tools
python3 -m replicant run sk --param beta=2                     # RS saddle, negative replicon -> unstable
python3 -m replicant run viana_bray --route cavity             # sparse, cavity-only
python3 -m replicant run xorsat --route cavity                 # alpha_d 0.818469, alpha_s 0.917935
python3 -m replicant run perceptron --route rsb --param kappa=-1
python3 -m replicant run online_scm --route online
python3 -m replicant run cluster_mc --route sample
```

## Related pages

- [Derivation and Composer](Derivation-and-Composer)
- [Certified and Statistical Computation](Certified-and-Statistical-Computation)
- [CLI and Python API](CLI-and-Python-API)
