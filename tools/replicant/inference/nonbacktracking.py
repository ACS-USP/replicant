"""The non-backtracking matrix of a graph, its leading spectrum and the spectral clustering it gives (KZ Sec. 10.4.4,
Eqs. (10.39)-(10.41); Krzakala, Moore, Mossel, Neeman, Sly, Zdeborova & Zhang, PNAS 110, 20935 (2013),
arXiv:1306.5550, "KMMNSZZ").

B is the 2M x 2M matrix on directed edges, B_{(u->v),(w->x)} = 1 if v = w and u != x (KMMNSZZ Sec. II; KZ Eq.
(10.40) writes the same matrix with the indices of the message update, B_{i->j, k->l} = delta_{il}(1 - delta_{jk})).
Every eigenvalue mu != +-1 of B is a root of det(mu^2 - mu A + (D - 1)) = 0 (KMMNSZZ Eq. (8), the Ihara-Bass
identity), i.e. an eigenvalue of the 2N x 2N matrix B' = [[0, D - 1], [-1, A]] (Eq. (9)); its right eigenvectors are
(x, f) with A f - (D - 1) f / mu = mu f, and f_v is the sum over the edges entering v of the eigenvector of B.
The leading part of the spectrum is computed from B' (ARPACK, scipy.sparse.linalg.eigs); nonbacktracking_matrix
builds B itself for the small-graph check of the identity.
Statements checked at finite N (KMMNSZZ Secs. II, IV, asymptotic in N, not theorems proved there): in the
symmetric block model the leading eigenvalue is the mean degree c, q - 1 real eigenvalues are close to
mu_c = (c_in - c_out)/q when mu_c > sqrt(c) (the KS condition |c_in - c_out| > q sqrt(c)), the bulk is in the disk
|mu| <= sqrt(c).  Spectral clustering: q = 2, the sign of Re f of the second eigenvector; q > 2, k-means
(scipy.cluster.vq.kmeans2, seeded) on the q - 1 informative eigenvectors f.  Finite-N values are statistical.
_HOOKS["backtrack"] (1.0): the coefficient of the identity in D - 1 (planted error: 0 allows backtracking,
B' = [[0, D], [-1, A]], whose bulk and leading eigenvalue move).
"""
from dataclasses import dataclass

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment
from scipy.sparse import bmat, csr_matrix, diags, identity
from scipy.sparse.linalg import eigs

from ..core import Refusal
from ..csp import _checks

_HOOKS = {"backtrack": 1.0}


def _edges(graph):
    try:
        N, E = int(graph.N), np.asarray(graph.edges, dtype=np.int64).reshape(-1, 2)
    except (AttributeError, TypeError, ValueError):
        raise Refusal("precondition:graph", "a graph with N and an (M, 2) array of edges") from None
    if len(E) == 0:
        raise Refusal("precondition:graph", "the graph has no edges")
    return N, E


def _adjacency(N, E):
    A = csr_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(N, N))
    return A, np.asarray(A.sum(1)).ravel()


def nonbacktracking_matrix(graph):
    """B on the 2M directed edges (e < M: E[e, 0] -> E[e, 1]; e >= M the reverse), as a sparse matrix."""
    N, E = _edges(graph)
    M = len(E)
    src, dst = np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]]
    out = [[] for _ in range(N)]
    for e in range(2 * M):
        out[src[e]].append(e)
    rows, cols = [], []
    for e in range(2 * M):                        # (u -> v) to (v -> x), x != u
        for f in out[dst[e]]:
            if dst[f] != src[e]:
                rows.append(e)
                cols.append(f)
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(2 * M, 2 * M))


def ihara_bass(graph):
    """B' = [[0, D - 1], [-1, A]] (2N x 2N, KMMNSZZ Eq. (9))."""
    N, E = _edges(graph)
    A, d = _adjacency(N, E)
    I = identity(N, format="csr")
    return bmat([[None, diags(d - _HOOKS["backtrack"])], [-I, A]], format="csr")


@dataclass(frozen=True)
class NBSpectrum:
    eigenvalues: np.ndarray   # the k leading eigenvalues, by decreasing modulus
    vectors: np.ndarray       # (N, k): the f halves of the right eigenvectors of B'
    mean_degree: float        # 2M / N of this graph
    label: str = "statistical"


def nonbacktracking_spectrum(graph, k) -> NBSpectrum:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise Refusal("precondition:eigen-count", f"k = {k!r}: a positive int")
    N, E = _edges(graph)
    if 2 * N - 2 <= k:
        raise Refusal("precondition:eigen-count", f"k = {k} needs 2N - 2 > k (N = {N})")
    vals, vecs = eigs(ihara_bass(graph), k=k, which="LM", tol=1e-10, maxiter=20000,
                      v0=np.random.default_rng(0).random(2 * N))
    order = np.argsort(-np.abs(vals), kind="stable")
    return NBSpectrum(vals[order], vecs[N:, order], 2 * len(E) / N)


@dataclass(frozen=True)
class NBClustering:
    overlap: float            # KZ Eq. (10.4), maximised over the label permutations
    informative: np.ndarray   # the q - 1 eigenvalues used after the leading one (real ones first; complex
                              # ones, by modulus, only when fewer than q - 1 are real: then separated is False)
    bulk_edge: float          # sqrt(2M / N)
    separated: bool           # every informative eigenvalue is outside the disk |mu| <= sqrt(c)
    third_modulus: float      # |mu| of the first eigenvalue after the informative ones
    assignment: np.ndarray    # the spectral group of each node, 0..q-1
    label: str = "statistical"


def spectral_clustering(graph, q, *, seed=0, k=None) -> NBClustering:
    """Labels from the q - 1 eigenvalues after the leading one (q = 2: sign of Re f; q > 2: k-means on Re f)."""
    seed = _checks.seed(seed)
    if isinstance(q, bool) or not isinstance(q, int) or q < 2:
        raise Refusal("precondition:group-count", f"q = {q!r}: an int >= 2")
    labels = np.asarray(getattr(graph, "labels", None))
    N, _ = _edges(graph)
    if labels.shape != (N,):
        raise Refusal("precondition:graph", "spectral_clustering needs the planted labels (an overlap is reported)")
    return cluster(graph, q, nonbacktracking_spectrum(graph, k or max(2 * q + 4, 8)), seed=seed)


def cluster(graph, q, sp, *, seed=0) -> NBClustering:
    """spectral_clustering from a computed NBSpectrum of the same graph."""
    labels, N = np.asarray(graph.labels), graph.N
    vals, vecs = sp.eigenvalues, sp.vectors
    real = [i for i in range(1, len(vals)) if abs(vals[i].imag) < 1e-8 * max(1.0, abs(vals[i]))]
    use = (real + [i for i in range(1, len(vals)) if i not in real])[:q - 1]   # complex ones only if too few real
    X = np.real(vecs[:, use])
    X = X / np.maximum(np.linalg.norm(X, axis=0), 1e-300)
    if q == 2:
        est = (X[:, 0] < 0).astype(int)
    else:
        _, est = kmeans2(X, q, seed=np.random.default_rng([seed, 31]), minit="++")
    conf = np.zeros((q, int(labels.max()) + 1 if labels.size else q))
    np.add.at(conf, (est, labels), 1.0)
    r, c = linear_sum_assignment(-conf)
    nmax = np.bincount(labels).max() / N
    edge = float(np.sqrt(sp.mean_degree))
    rest = [i for i in range(1, len(vals)) if i not in use]
    return NBClustering(float((conf[r, c].sum() / N - nmax) / (1 - nmax)), vals[use], edge,
                        bool(len(real) >= q - 1 and np.all(np.abs(vals[use]) > edge)),
                        float(abs(vals[rest[0]])) if rest else float("nan"),
                        est.astype(np.int64))


__all__ = ["NBClustering", "NBSpectrum", "cluster", "ihara_bass", "nonbacktracking_matrix", "nonbacktracking_spectrum",
           "spectral_clustering"]
