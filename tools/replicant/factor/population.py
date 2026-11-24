"""Population dynamics over a factor law: the RS cavity equations of factor/bp.py on a random factor graph
(KZ Sec. 4.3.2, Eq. (4.15) for the pairwise case; Sec. 5.2 for the messages).

A population of `pop` variable-to-factor messages chi (pop, q) represents the law of chi^{i->a}.  One sweep
replaces every member: draw the excess degree r (the number of OTHER factors of the variable), for each of the r
factors draw a table from the factor law and arity - 1 members, contract to psi^{a->i}, and set
chi_new ~ g(s) prod psi(s) with g drawn from the unary law.  Degree laws give the number of factors per variable:
("regular", d) (excess d - 1) or ("poisson", c) (Poisson(c), excess Poisson(c): KZ Eq. (4.13)).  With c_f the
mean number of factors per variable and k the arity, the Bethe free entropy per variable (Eq. (5.1) averaged):
    Phi = E ln Z_i + (c_f / k) E ln Z_a - c_f E ln Z_ia,
each term from independent draws (Z_i from a full degree, Z_a from a table and k members, Z_ia from a member and
an independent psi).  The marginal law b ~ g prod_{full degree} psi gives E b (q,) per run.
Statistics: one run per named seed (numpy default_rng(seed)); `burn_in` sweeps (default: sweeps), then `sweeps`
(>= 8) measured sweeps; per run the measured means; across runs mean, stderr and a 99% Student-t band.
Diagnostic: a run is "drifting" when its per-sweep E b(0) differs between the first and second halves of the
measured sweeps by more than 5 standard errors, or moves strictly monotonically through every measured sweep (a
deterministic relaxation; for i.i.d. noise the chance is 2/n!, n >= 8), in either case by more than `tol` (default
1e-6: a relaxation below it is converged for the stated precision), or its population holds a non-finite value;
then the result is labelled undetermined (never refused).
Factor tables with zeros (hard constraints) are refused here (unsupported:hard-constraints).

  FactorLaw(q, arity, tables(rng, n) -> (n,) + (q,) * k, unary(rng, n) -> (n, q) or None, name)
  potts_law(q, beta), rfim_law(beta, J, fields(rng, n) -> (n,))     the laws of coloring/ and the RFIM
  population_dynamics(factor_law, degree_law, *, pop, sweeps, seeds, burn_in=None, init="uniform", tol=1e-6)
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..amp._stats import student_band
from ..core import Refusal

_L = "bcdefghijklmnopqrstuvwxyz"


@dataclass(frozen=True)
class FactorLaw:
    q: int
    arity: int
    tables: Callable
    unary: Callable | None = None
    name: str = ""


def potts_law(q, beta):
    """f(s, s') = exp(-beta delta_{s s'}) (KZ Ch. 6), no field; beta = inf is the hard coloring (refused here)."""
    t = np.exp(-float(beta) * np.eye(q))
    return FactorLaw(q, 2, lambda rng, n: np.broadcast_to(t, (n, q, q)), None, f"potts(q={q}, beta={beta})")


def rfim_law(beta, J, fields):
    """Ising pair weight exp(beta J s s') and unary exp(beta h s), s = +1 for state 0, -1 for state 1."""
    s = np.array([1.0, -1.0])
    t = np.exp(float(beta) * float(J) * np.outer(s, s))
    return FactorLaw(2, 2, lambda rng, n: np.broadcast_to(t, (n, 2, 2)),
                     lambda rng, n: np.exp(float(beta) * np.outer(fields(rng, n), s)), f"rfim(beta={beta}, J={J})")


@dataclass(frozen=True)
class PDResult:
    phi: float
    phi_stderr: float
    phi_halfwidth99: float
    marginal: np.ndarray            # E b (q,), mean over runs
    marginal_stderr: np.ndarray
    per_seed_phi: tuple
    per_seed_marginal: tuple
    seeds: tuple
    status: str                     # "converged" | "undetermined"
    note: str = ""


def _degree(law):
    if not (isinstance(law, tuple) and len(law) == 2 and law[0] in ("regular", "poisson")):
        raise Refusal("unsupported:degree-law", f"{law!r}: ('regular', d) or ('poisson', c)")
    kind, p = law
    if kind == "regular" and not (isinstance(p, (int, np.integer)) and p >= 1):
        raise Refusal("unsupported:degree-law", f"{law!r}: regular needs an int d >= 1")
    if kind == "poisson" and not float(p) > 0:
        raise Refusal("unsupported:degree-law", f"{law!r}: poisson needs c > 0")
    return kind, (int(p) if kind == "regular" else float(p))


def _draw(law, n, rng, excess):
    kind, p = law
    if kind == "regular":
        return np.full(n, p - 1 if excess else p, dtype=int)
    return rng.poisson(p, n)


def _psi(fl, rng, chi, shape):
    """psi messages of shape + (q,), each from a fresh table and arity - 1 random members of chi."""
    n = int(np.prod(shape)) if shape else 1
    k, q = fl.arity, fl.q
    t = np.asarray(fl.tables(rng, n))
    if k == 1:
        out = t
    else:
        mem = chi[rng.integers(len(chi), size=(n, k - 1))]                    # (n, k-1, q)
        subs = "a" + _L[:k] + "," + ",".join("a" + _L[j] for j in range(1, k)) + "->ab"
        out = np.einsum(subs, t, *[mem[:, j - 1] for j in range(1, k)])
    out = out / out.sum(1)[:, None]
    return out.reshape(shape + (q,))


def _site(fl, rng, chi, degs):
    """(log of g prod psi over degs[i] factors) for each i: (n, q)."""
    n, q = len(degs), fl.q
    r = int(degs.max(initial=0))
    lg = np.log(fl.unary(rng, n)) if fl.unary is not None else np.zeros((n, q))
    if r:
        lp = np.log(_psi(fl, rng, chi, (n, r)))
        lp[np.arange(r)[None, :] >= degs[:, None]] = 0.0
        lg = lg + lp.sum(1)
    return lg


def _norm(lg):
    w = np.exp(lg - lg.max(1, keepdims=True))
    return w / w.sum(1)[:, None]


def _lse(lg):
    mx = lg.max(1)
    return mx + np.log(np.exp(lg - mx[:, None]).sum(1))


def _measure(fl, law, rng, chi):
    n, k = len(chi), fl.arity
    lz_i = _site(fl, rng, chi, _draw(law, n, rng, excess=False))
    t = np.asarray(fl.tables(rng, n))
    mem = chi[rng.integers(n, size=(n, k))]
    subs = "a" + _L[:k] + "," + ",".join("a" + _L[j] for j in range(k)) + "->a"
    z_a = np.einsum(subs, t, *[mem[:, j] for j in range(k)])
    z_ia = np.sum(chi[rng.integers(n, size=n)] * _psi(fl, rng, chi, (n,)), 1)
    cf = law[1]
    phi = np.mean(_lse(lz_i)) + (cf / k) * np.mean(np.log(z_a)) - cf * np.mean(np.log(z_ia))
    return phi, _norm(lz_i).mean(0)


def relaxation(series, tol, name):
    """'' for a stationary measured series, else the reason (the module's drift diagnostic, shared with
    coloring/rsb_pop.py): the halves differ by more than 5 standard errors and more than tol, or the series moves
    strictly monotonically through every measured sweep by more than tol (a deterministic relaxation; for i.i.d.
    noise the chance is 2/n!), or a value is non-finite."""
    x = np.asarray(series, dtype=float)
    if not np.all(np.isfinite(x)):
        return f"{name}: a non-finite measurement"
    h = len(x) // 2
    a, b = x[:h], x[h:]
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) + 1e-12
    steps = np.diff(x)
    if abs(a.mean() - b.mean()) > max(5 * se, tol):
        return f"{name} drifts {a.mean():.4g} -> {b.mean():.4g} ({abs(a.mean() - b.mean()) / se:.1f} se)"
    if (np.all(steps > 0) or np.all(steps < 0)) and abs(x[-1] - x[0]) > tol:
        return f"{name} moves monotonically {x[0]:.6g} -> {x[-1]:.6g} over every measured sweep (still relaxing)"
    return ""


def _start(init, pop, q, rng):
    if init == "uniform":
        return np.full((pop, q), 1.0 / q)
    if init == "planted":
        chi = np.zeros((pop, q))
        chi[:, 0] = 1.0
        return chi
    if init == "random":
        x = -np.log(rng.random((pop, q)))
        return x / x.sum(1)[:, None]
    raise Refusal("precondition:init", f"init {init!r}: 'uniform', 'planted' (all in state 0) or 'random'")


def _run(fl, law, pop, sweeps, burn_in, seed, init):
    rng = np.random.default_rng(seed)
    chi = _start(init, pop, fl.q, rng)
    phis, margs = [], []
    for t in range(burn_in + sweeps):
        chi = _norm(_site(fl, rng, chi, _draw(law, pop, rng, excess=True)))
        if not np.all(np.isfinite(chi)):
            return None
        if t >= burn_in:
            ph, mg = _measure(fl, law, rng, chi)
            phis.append(ph), margs.append(mg)
    return np.array(phis), np.array(margs)


def population_dynamics(factor_law, degree_law, *, pop, sweeps, seeds, burn_in=None, init="uniform", tol=1e-6):
    law = _degree(degree_law)
    fl = factor_law
    if not (isinstance(pop, int) and pop >= 100 and isinstance(sweeps, int) and sweeps >= 8):
        raise Refusal("precondition:population", f"pop = {pop!r} (int >= 100), sweeps = {sweeps!r} (int >= 8)")
    seeds = tuple(int(s) for s in seeds)
    if len(seeds) < 2:
        raise Refusal("precondition:named-seeds", "population dynamics needs at least two named seeds (a band)")
    burn_in = sweeps if burn_in is None else int(burn_in)
    probe = np.asarray(fl.tables(np.random.default_rng(0), 4))
    if np.any(probe <= 0):
        raise Refusal("unsupported:hard-constraints", f"{fl.name}: a factor table has zeros")
    runs, drift = [], []
    for s in seeds:
        out = _run(fl, law, pop, sweeps, burn_in, s, init)
        if out is None:
            drift.append(f"seed {s}: non-finite population")
            continue
        phis, margs = out
        why = relaxation(margs[:, 0], tol, "E b(0)")
        if why:
            drift.append(f"seed {s}: {why}")
        runs.append((phis.mean(), margs.mean(0)))
    if len(runs) < 2:
        raise Refusal("no-solution:audit", "; ".join(drift))
    band = student_band([r[0] for r in runs])
    m = np.array([r[1] for r in runs])
    status = "undetermined" if drift else "converged"
    note = ("; ".join(drift) + ": not equilibrated") if drift else (
        f"no drift between halves of {sweeps} measured sweeps in any of {len(runs)} runs")
    return PDResult(band.mean, band.stderr, band.half_width, m.mean(0), m.std(0, ddof=1) / np.sqrt(len(m)),
                    tuple(float(r[0]) for r in runs), tuple(tuple(map(float, r[1])) for r in runs), seeds, status,
                    note)


__all__ = ["FactorLaw", "PDResult", "population_dynamics", "potts_law", "relaxation", "rfim_law"]
