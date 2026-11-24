"""Bayes-optimal estimation in Gaussian noise (KZ Ch. 7, K2).

  channels.py       the scalar channel y = sqrt(snr) x + z, priors as point-mass / Gaussian mixtures (gaussian,
                    rademacher, gauss_bernoulli, sparse), the posterior-mean denoiser, MAP, MMSE and I (float
                    quadrature; certified balls for gaussian and rademacher), the posterior of KZ's data sets in arb
  identities.py     Nishimori, Stein and I-MMSE as validators returning a CheckResult (never raising off the line)
  sparse_vector.py  denoising a one-hot vector of dimension d = 2^N (KZ Sec. 7.3, Theorem 15, App. 7.B): f(Delta),
                    Delta_c = 1/(2 log 2), the exact finite-d E log Z, and the REM route (families/rem.py)
"""
