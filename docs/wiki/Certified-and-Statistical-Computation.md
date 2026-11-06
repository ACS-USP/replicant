# Certified and Statistical Computation

Most of `replicant` is `physics-level`: the replica n -> 0 continuation, the RS/RSB ansätze and
the cavity clustering assumptions are not proved, and every result of those routes says so. Two
other rigor levels exist. This page explains when each one applies and what it promises.

## Certified: ball arithmetic

`rigor: "certified"` means a ball enclosure: python-flint's arb, used through `tools/baller`, with
the radius reported in `errors`. It certifies the evaluation of a cited closed form, a cited
theorem's formula or a rigorous bound. It does not certify the cited theorem itself.

| Entry / route | What is enclosed | Code |
|---|---|---|
| `ising2d`, `ising1d` (`exact`) | Onsager's free energy, Yang's magnetization, Kaufman's finite torus (with its integer polynomial recovered from balls), the 1D ring | `lattice/onsager.py`, `lattice/kaufman.py`, `lattice/transfer.py`, `lattice/observables1d.py` |
| `ising2d_anisotropic`, `ising_triangular`, `ising_honeycomb` (`exact`) | critical points, free energies, exact tori | `lattice/anisotropic.py`, `lattice/tri_honey.py`, `lattice/geom_tori.py` |
| `bethe_ising` (`cavity`) | every BP fixed point of the Bethe-lattice ferromagnet | `lattice/bethe.py` |
| `sk_bounds` (`bound`) | Guerra's RS and K-step bounds on the SK pressure, and a chain check of the package's own RS / 1RSB / K-RSB optimisers | `bounds/guerra.py`, `bounds/krsb.py`, `bounds/chain.py` |
| percolation `cavity` routes | generating-function and k-core fixed points, birth points | `percolation/cavity.py`, `percolation/kcore.py` |
| `percolation_lattice` (`exact`) | 2D thresholds, exponents, Cardy's formula, exact crossing polynomials | `percolation/exact2d.py`, `percolation/cardy.py`, `percolation/enumerate2d.py` |

The certified routes refuse rather than round. Examples:
- a radius that does not halve when the precision doubles (`no-solution:radius-stall`);
- a fixed-point count undecided at 4096 bits;
- python-flint missing (`unsupported:python-flint`).

## Statistical: sampling over named seeds

`rigor: "statistical"` covers population dynamics, finite-N AMP / GAMP, the percolation `sample`
routes and the Monte Carlo entries. The statistical keys are listed in `Result.statistical`,
their stderr is in `errors`, and they are **never final**.

| Area | Methods | Code |
|---|---|---|
| Cluster Monte Carlo (`cluster_mc`) | Swendsen–Wang, Wolff, a checkerboard Metropolis baseline on Ising / Potts tori | `montecarlo/cluster.py`, `cluster_fast.py`, `metropolis.py` |
| Flat histogram (`wang_landau_mc`, `histogram_reweighting_mc`) | Wang–Landau 1/t, multicanonical reweighting, Ferrenberg–Swendsen single / multiple histogram | `montecarlo/wang_landau.py`, `multicanonical.py`, `reweight.py` |
| Population annealing (`population_annealing_mc`) | Machta's algorithm, 2D Ising torus or one SK / ±J matrix | `montecarlo/population.py`, `sk_metropolis.py` |
| Percolation `sample` | union-find on finite graphs, Newman–Ziff sweeps, finite-size scaling, exact peeling | `percolation/sampling.py`, `lattice2d.py`, `fss2d.py`, `kcore_sample.py` |
| Population dynamics | sparse RS / 1RSB populations | `cavity/` |

Shared statistics live in `montecarlo/stats.py`: tau_int, binning, the jackknife, Student-t bands,
the equilibration diagnostic, chi^2 and z fits. The rules every sampled route follows:

1. Seeds are named, and a band is computed over those seeds.
2. Exact small-system targets check the sampler. Small tori are compared against the exact
   distribution, and a stationarity / ergodicity chi^2 is computed (`montecarlo/exact_small.py`,
   `balance.py`). This check found a real non-ergodicity of checkerboard Metropolis on the 2×2
   torus, which is now refused below L = 4.
3. Per-run units, such as the Wolff sweep unit, are fixed once per run by a pilot chain, not
   estimated seed by seed.
4. When a diagnostic fails, the affected values are labelled undetermined. An undetermined run
   never feeds an asserted reference band.

## Independent finite-N checks

The ODE and saddle-point routes are checked by code that shares nothing with them:
- the finite-N online simulators (`online/simulate*.py`, N = 1000 in the slow tier);
- AMP (`amp/`);
- Gray-code enumeration and parallel tempering (`numerics/enumeration.py`, `numerics/pt.py`);
- exact GF(2) elimination for XORSAT;
- the peeling decoder for LDPC codes.

## Related pages

- [Catalog and Routes](Catalog-and-Routes)
- [Reference Store and Provenance](Reference-Store-and-Provenance)
- [Automated Tests](Automated-Tests)
