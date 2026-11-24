"""Generic factor-graph belief propagation (K3; Krzakala & Zdeborova, lecture notes, Ch. 4-5).

  graph.py        FactorGraph(variables, factors): q-state variables with unary weights g_i, factors of any
                  arity with tables f_a; pairwise() builder; samplers of random trees, factor trees,
                  d-regular and Erdos-Renyi graphs (own code: shares nothing with cavity/graphs.py)
  bp.py           bp(graph, *, iters, damping, init, seed) -> BPState; marginals; bethe_free_entropy(graph,
                  state): the variable, factor and edge terms of KZ Eq. (5.1), exact on trees, labelled an
                  approximation on loopy graphs
  population.py   population_dynamics(factor_law, degree_law, *, pop, sweeps, seeds): the same equations over
                  a law of factors and degrees (statistical, over named seeds)
  enumerate.py    exact Z and marginals of a small graph by enumeration (the independent route)
"""
