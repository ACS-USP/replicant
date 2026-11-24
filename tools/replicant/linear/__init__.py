"""Linear estimation and convex problems (K6; Krzakala & Zdeborova lecture notes Ch. 12, 16, 18).

  gamp.py   GAMP with a Channel (awgn, sign) x Prior (bayes/channels.py) interface; finite-N instances
  se.py     its Bayes-optimal state evolution, from the uninformative and the informative start
  lasso.py  the LASSO: soft-threshold AMP (KZ Alg. 4), its state evolution, the calibration theta <-> lambda,
            proximal gradient, an exact coordinate-descent solver with a KKT audit
  cs.py     the noiseless compressed-sensing phase diagram for the Gauss-Bernoulli prior: the l1 boundary
            (Donoho-Maleki-Montanari) and the Bayes-optimal one (the SE spinodal), labelled separately; l1 by LP
"""
