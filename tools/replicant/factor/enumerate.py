"""Exact partition function and marginals of a small FactorGraph by enumerating its q^N configurations.

The independent route for factor/bp.py: it reads only graph.unary and graph.groups (scopes, tables) and sums
ln g_i(s_i) + ln f_a(s_{da}) over every configuration (log-sum-exp, zeros as -inf).  Refused above
MAX_CONFIGS configurations (precondition:enumeration-size).

  exact_log_z(graph) -> float;  exact_marginals(graph) -> (N, q);  exact(graph) -> (ln Z, marginals)
"""
import numpy as np

from ..core import Refusal

MAX_CONFIGS = 1 << 21


def _log_weights(graph):
    n, q = graph.n, graph.q
    if q ** n > MAX_CONFIGS:
        raise Refusal("precondition:enumeration-size", f"q^N = {q}^{n} > {MAX_CONFIGS} configurations")
    conf = np.indices((q,) * n, dtype=np.int8).reshape(n, -1)            # (N, q^N), row-major
    with np.errstate(divide="ignore"):
        lw = np.log(graph.unary)[np.arange(n)[:, None], conf].sum(0)
        for gr in graph.groups:
            lt = np.log(gr.tables)
            for a, sc in enumerate(gr.scopes):
                t = lt if gr.shared else lt[a]
                lw = lw + t[tuple(conf[i] for i in sc)]
    return conf, lw


def exact(graph):
    conf, lw = _log_weights(graph)
    mx = lw.max()
    w = np.exp(lw - mx)
    z = w.sum()
    marg = np.stack([np.bincount(conf[i], weights=w, minlength=graph.q) for i in range(graph.n)]) / z
    return float(mx + np.log(z)), marg


def exact_log_z(graph):
    return exact(graph)[0]


def exact_marginals(graph):
    return exact(graph)[1]


__all__ = ["MAX_CONFIGS", "exact", "exact_log_z", "exact_marginals"]
