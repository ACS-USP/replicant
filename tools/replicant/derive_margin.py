"""Level (a): the perceptron whose margin kappa_l is drawn PER PATTERN from a law.

Spec: family "perceptron", constraint "margin kappa law", params
  kappa_law   "constant" | "gaussian" | "mixture"      (anything else: precondition:supported-margin-law)
  constant    kappa                                    kappa_l = kappa
  gaussian    kappa, kappa_s                           kappa_l = kappa + kappa_s g, g ~ N(0, 1)
  mixture     kappa_weights, kappa_values              kappa_l = kappa_values[i] w.p. kappa_weights[i]
  alpha       the load, as everywhere; kappa, kappa_s, alpha may be left symbolic (None), the
              weights and values of a mixture are numbers.
Parsing, validation and the refusals are families/margin_law.parse_margin_law (shared with the
level-(b) numerics); this module turns the law into a formal sympy object and derives with it.

What is hand-supplied and what is derived.
  hand-supplied   the storage constraint g(lambda, kappa) = Theta(lambda - kappa) and the law of
                  kappa (the formal GaussianMargin / ConstantMargin / MixtureMargin below), as the
                  catalog hook supplies g(lambda) = Theta(lambda - kappa).
  hook asserts    (rule not mechanised, as for LnGaussProd) for Gaussian patterns at fixed J the
                  lambda_a are jointly Gaussian with covariance Q_ab, and the P = alpha N patterns
                  carry iid kappa_l independent of them: (1/N) ln E V^n = site + alpha
                  LnGaussProdLaw(g, q_ab, n, law), LnGaussProdLaw = ln E_kappa E_lambda
                  prod_a g(lambda_a, kappa), the kappa average OUTSIDE the product over the
                  replicas (one kappa per pattern, shared) and inside the logarithm.
  derived (RS)    lambda_a = sqrt(q) t + sqrt(1-q) z_a.  The weight must be a function of
                  lambda - kappa (shift_form checks g(d + k, k) is free of k, else
                  precondition:margin-shift-form).  Per component kappa = m + s w, w ~ N(0, 1)
                  independent of t: sqrt(q) t - s w is a sum of independent centred Gaussians,
                  N(0, q + s^2) (merge_independent), so lambda_a - kappa = sqrt(q + s^2) u +
                  sqrt(1-q) z_a - m with u ~ N(0, 1), E_z by gauss.expect_closed, the product over
                  replicas gives [E_z g]^n, and the component enters with its weight inside ONE
                  logarithm: ln sum_i w_i E_u [H((m_i - sqrt(q + s_i^2) u)/sqrt(1-q))]^n.  n -> 0
                  (gauss.replica_limit) gives sum_i w_i E_u ln H(x_i); the stationarity equations
                  are then the generic step.  A constant law (or s = 0) reproduces the catalog
                  hook's expression exactly (sqrt(q + 0) = sqrt(q)).
  replicon        derive_replicon: E''_R = alpha sum_i w_i E_u [(R_2 - R_1^2)^2] with R_k the slot
                  ratios of the merged split (the replica measure at n -> 0 leaves the plain
                  Gaussian (t, kappa) measure: the weight M_0^n/E[M_0^n] -> 1), i.e. the
                  catalog's rule with E_t -> E_kappa E_t.
Laws with kappa_l of both signs, or kappa~ < 0, are in the same non-convex position as the
catalog's kappa < 0 (the computed replicon is the stability verdict).  The 1RSB routes do not
handle the law (refused).  Honest label: physics-level, as the rest of level (a).
"""
import sympy as sp

from . import derive as D
from . import gauss
from .core import Refusal
from .families.margin_law import LAW_PARAMS, _decimal, parse_margin_law

LnGaussProdLaw = sp.Function("LnGaussProdLaw")      # (g(lambda, kappa), c_ab, n, law)
ConstantMargin = sp.Function("ConstantMargin")      # (kappa)
GaussianMargin = sp.Function("GaussianMargin")      # (mu, s)
MixtureMargin = sp.Function("MixtureMargin")        # (w_1, m_1, ..., w_k, m_k)
KAPPA = sp.Symbol("kappa", real=True)
KAPPA_S = sp.Symbol("kappa_s", positive=True)
CONSTRAINT = "margin kappa law"
_LAM = sp.Symbol("lambda", real=True)
_KL = sp.Symbol("kappa_l", real=True)


def _is_symbolic(v):
    return isinstance(v, sp.Basic) and bool(v.free_symbols)


def _exact(v):
    """A validated finite real as an exact sympy Rational (decimals stay decimals)."""
    return sp.Rational(_decimal("law parameter", v))


def spec_law(spec):
    """The formal law of a "margin kappa law" spec; precondition:supported-margin-law for an
    unsupported one.  kappa / kappa_s absent or given as symbols stay symbolic."""
    params = dict(spec.params)
    sym = {}
    for k, default in (("kappa", KAPPA), ("kappa_s", KAPPA_S)):
        v = params.get(k)
        if k in LAW_PARAMS.get(params.get("kappa_law"), ()) and (v is None or _is_symbolic(v)):
            sym[k] = default if v is None else v
            params[k] = "1"                     # placeholder: the structure is what is checked
    law = parse_margin_law(params)
    if law is None:
        raise Refusal("precondition:supported-margin-law",
                      f"constraint {CONSTRAINT!r} needs kappa_law in constant, gaussian, mixture")
    if law.kind == "mixture":
        w = [_exact(x) for x in law.weights]
        tot = sum(w)
        args = [a for wi, v in zip(w, law.values) if wi > 0 for a in (wi / tot, _exact(v))]
        return MixtureMargin(*args)
    mu = sym.get("kappa", _exact(law.mu))
    if law.kind == "constant":
        return ConstantMargin(mu)
    return GaussianMargin(mu, sym.get("kappa_s", _exact(law.s)))


def components(law):
    """[(w, m, s)] of a formal law: kappa = m + s w~, w~ ~ N(0, 1), with weight w."""
    if law.func == ConstantMargin:
        return [(sp.S(1), law.args[0], sp.S(0))]
    if law.func == GaussianMargin:
        return [(sp.S(1), law.args[0], law.args[1])]
    if law.func == MixtureMargin:
        a = law.args
        return [(a[i], a[i + 1], sp.S(0)) for i in range(0, len(a), 2)]
    raise Refusal("precondition:supported-margin-law", f"{law}: not a margin law object")


def merge_independent(coeffs):
    """Std of sum_i c_i x_i for independent x_i ~ N(0, 1): sqrt(sum c_i^2)."""
    return sp.sqrt(sp.Add(*(sp.sympify(c)**2 for c in coeffs)))


def shift_form(g):
    """G with g(lambda, kappa) = G(lambda - kappa), checked; precondition:margin-shift-form
    otherwise (the merge of the Gaussians is valid for such weights only)."""
    if not (isinstance(g, sp.Lambda) and len(g.signature) == 2):
        raise Refusal("precondition:margin-shift-form", f"weight {g}: need a Lambda(lambda, kappa)")
    d, k = sp.Dummy("d", real=True), sp.Dummy("k", real=True)
    e = g(d + k, k)
    if sp.simplify(sp.diff(e, k)) != 0:
        raise Refusal("precondition:margin-shift-form",
                      f"weight {g.expr} is not a function of lambda - kappa")
    return sp.Lambda(d, sp.simplify(e))


def weight():
    """The hand-supplied storage constraint g(lambda, kappa) = Theta(lambda - kappa)."""
    return sp.Lambda((_LAM, _KL), sp.Heaviside(_LAM - _KL))


# ---- RS reduction (registered in derive.RS_RULES) -----------------------------------------
def rs_law_term(e, ctx):
    g, cov, nn, law = e.args
    q = ctx.free_of_indices(ctx.rs(cov), "covariance")
    G = shift_form(g)
    terms, assumed = [], []
    for w, m, s in components(law):
        c = merge_independent([sp.sqrt(q), s])
        inner = gauss.expect_closed(G(c * D._T + sp.sqrt(1 - q) * D._Z - m), D._Z, assumed)
        terms.append(w * gauss.GaussE(inner**nn, D._T))
    ctx.rules.append(
        f"LnGaussProdLaw: kappa average outside the replica product and inside the logarithm; "
        f"per component kappa = m + s w, sqrt(q) t - s w ~ N(0, q + s^2) (merged), lambda_a - kappa "
        f"= sqrt(q + s^2) u + sqrt(1-q) z_a - m, product over replicas -> [E_z g]^n; "
        f"law {law}" + "".join(f" ({x})" for x in dict.fromkeys(assumed)))
    return sp.log(sp.Add(*terms))


# ---- replicon (registered in derive.REPLICON_RULES) ---------------------------------------
def law_replicon(term, F):
    """(GaussReplicon | None, why) for term = coef LnGaussProdLaw(g, q_ab, n, law)."""
    from .derive_replicon_gauss import (GaussReplicon, _Z, _T, _gauss_prod_core,
                                        _rational_or_same)
    c, rest = term.as_independent(LnGaussProdLaw, as_Add=False)
    if not (rest.func == LnGaussProdLaw and not c.has(F.q)):
        return None, f"energy term {term} is not coef * LnGaussProdLaw(...)"
    g, cov, nn, law = rest.args
    if cov != F.q[D._a, D._b] or nn != F.n:
        return None, f"LnGaussProdLaw covariance {cov} is not q_ab"
    G, q = shift_form(g), D.Q_RS
    R = sp.symbols("R_0:5", real=True)
    pieces, notes = [], []
    for w, m, s in components(law):
        lam = sp.Lambda(_LAM, G(_LAM - m))
        _, body, _, _ = _gauss_prod_core(lam, q)       # the g-free R_k polynomial of E''_R
        M = gauss.slot_moments(lam, merge_independent([sp.sqrt(q), s]) * _T
                               + sp.sqrt(1 - q) * _Z, _Z, 2, notes)
        sub = {R[k]: M[k] / M[0] for k in (1, 2)}
        integrand = sp.Mul(*(_rational_or_same(b.xreplace(sub))**x for b, x in
                             (f.as_base_exp() for f in sp.Mul.make_args(body))))
        pieces.append(w * gauss.GaussE(integrand, _T))
    e2 = c * sp.Add(*pieces)
    note = ("energy coef ln E_kappa E_lambda prod_a g(lambda_a, kappa): the catalog rule "
            "(Price covariance derivative on four replicas, RS split, n -> 0 of the replica "
            "measure, eigenvector projection) with the Gaussian (t, kappa) measure and the "
            f"merged split lambda - kappa = sqrt(q + s^2) u + sqrt(1-q) z - m: E''_R = coef "
            f"sum_i w_i E_u [(R_2 - R_1^2)^2], law {law}" + "".join(f" ({x})" for x in notes))
    parts = (("Energy replicon eigenvalue, margin law (merged split, components weighted)",
              sp.Eq(sp.Symbol("E''_R"), e2)),)
    return GaussReplicon(e2, parts, note), ""


D.RS_RULES.append((LnGaussProdLaw, rs_law_term))
D.REPLICON_RULES.append((LnGaussProdLaw, law_replicon))
