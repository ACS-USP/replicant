"""Inference and codes (E6).

  ldpc.py          regular (l, r) LDPC ensembles on the binary erasure channel: density
                   evolution, the BP threshold (tangency of eps(x)) and the MAP threshold
                   (Maxwell construction: root of the trial entropy, and the area theorem by
                   quadrature), through the degree-polynomial hook DEGREE_PAIRS
  ldpc_finite.py   finite N, independent of ldpc.py: configuration-model graphs, the peeling
                   decoder, success fractions over named seeds (99% Clopper-Pearson)
  spiked.py        rank-one spiked Wigner Y = (lambda/N) x x^T + W: Bayes-optimal state evolution
                   through the PRIORS denoiser hook (gaussian | rademacher), the RS potential, the
                   threshold lambda_c, and finite-N Bayes AMP with spectral start (Student-t bands)
  sbm.py           two-group stochastic block model: the Kesten-Stigum line, an exact sampler, BP
                   with the external field (DKMZ), the BP overlap over named seeds
  sbm_q.py         q-group block model (K4, KZ Ch. 10): sampler, BP with the non-edge field from
                   planted / random starts, the Bethe free entropy, the KS point (no code shared
                   with sbm.py); sbm_q_pd.py: planted-measure population dynamics and the
                   first-order thresholds c_d, c_c; sbm_learn.py: EM (BP-learning) and the group
                   count; nonbacktracking.py: the non-backtracking spectrum and spectral clustering
  spiked_diluted.py the diluted-Rademacher spiked Wigner (K5, KZ Ex. 8.2; LKZ): every SE fixed point, the
                   spinodals, the IT threshold, the hard phase, the tricritical point (spiked.py dispatches to it)
  wishart.py       rank-one spiked Wishart (K5, KZ Sec. 12.3.3): two-overlap Bayes SE, rectangular Bayes AMP,
                   the BBP edge lambda_c = alpha^(-1/4) and the top singular value
  spin_glass_game.py the dense spin-glass game of KZ Ch. 11 as the Rademacher spike (lambda = beta, derived)
"""
from .ldpc import (DEGREE_PAIRS, BecThresholds, DEResult, RegularPair, bec_bp_threshold,
                   bec_density_evolution, bec_map_threshold, bec_map_threshold_area,
                   bec_thresholds, trial_entropy)
from .ldpc_finite import (LdpcGraph, PeelResult, PeelStats, ldpc_ensemble_graph,
                          ldpc_peeling_decode, ldpc_peeling_success)

__all__ = ["BecThresholds", "DEGREE_PAIRS", "DEResult", "LdpcGraph", "PeelResult", "PeelStats",
           "RegularPair", "bec_bp_threshold", "bec_density_evolution", "bec_map_threshold",
           "bec_map_threshold_area", "bec_thresholds", "ldpc_ensemble_graph",
           "ldpc_peeling_decode", "ldpc_peeling_success", "trial_entropy"]
