"""Displayed equations of the spherical perceptron's T = 0 RSB route (families/perceptron_rsb.py,
perceptron_krsb.py): the derivation of route `rsb` (latex_entries.perceptron returns these blocks for it).  The
jamming functional Phi carries a check against the family code through a display function that does
the u-average by direct quadrature (no closed form); the K-RSB nesting and the replicon are cited."""
import math

import numpy as np

from .tex import Check, cited, text

_SRC = ("Franz, Parisi, Sevelev, Urbani & Zamponi, SciPost Phys. 2, 019 (2017), arXiv:1702.06919, "
        "Eq. (59) and App. B; Franz & Parisi, J. Phys. A 49, 145001 (2016), arXiv:1501.03397")
_STOJ = "Stojnic, arXiv:2312.16531 (Tables 2, 5, 6) and arXiv:1306.3980"


def _phi_display(mu, q0, alpha, kappa):
    """Phi from the displayed formula, with E_u exp(-mu g_+^2/2) done by quadrature over u (kink on a panel edge)."""
    from .families import perceptron_rsb as R
    mu, q0, alpha, kappa = (float(x) for x in (mu, q0, alpha, kappa))
    b = math.sqrt(1 - q0)
    ts, ws = R._gl_panels(kappa, q0, mu)
    acc = 0.0
    for t, w in zip(ts, ws):
        a = kappa - math.sqrt(q0) * t
        u, wu = R._u_nodes(a / b, 1.0 / (b * math.sqrt(mu)))
        acc += w * math.log(float(np.sum(wu * np.exp(-mu * np.maximum(a - b * u, 0.0)**2 / 2))))
    s = 1 - q0
    return math.log1p(mu * s) / (2 * mu) + q0 / (2 * (1 + mu * s)) + alpha / mu * acc


def blocks():
    from .families import perceptron_rsb as R
    pts = (("12", "0.6", "10", "-1"), ("5", "0.4", "3", "-0.5"), ("30", "0.8", "20", "-1.5"))
    return [
        text("T = 0 beyond RS (kappa < 0): the jamming limit of the 1RSB entropy",
             "kappa < 0 makes the version space non-convex: the RS capacity 1/D(kappa) is only an upper bound and "
             "the SAT-UNSAT threshold needs replica symmetry breaking. The 1RSB entropy (q0 between clusters, q1 "
             "inside, Parisi parameter m) is taken at 1 - q1 = delta -> 0, m = mu delta (cluster volume -> 0): "
             "s = (1/2)(1 + c) ln(delta) + Phi/delta + O(1); stationarity in q1 forces Phi = 0 and in (m, q0) "
             "dPhi/dmu = dPhi/dq0 = 0. Phi is linear in alpha, so alpha_c is the minimum of A = -ent/en over "
             "(mu, q0); mu -> 0 is the RS value", _SRC),
        cited("1RSB jamming functional Phi(mu, q0; alpha) (zero-temperature scaling of FPSUZ Eq. (59))",
              r"\Phi = \frac{\ln\!\left(1 + \mu(1-q_0)\right)}{2\mu} + \frac{q_0}{2\left(1 + \mu(1-q_0)\right)}"
              r" + \frac{\alpha}{\mu}\,\mathbb{E}_t \ln \mathbb{E}_u\, e^{-\frac{\mu}{2}\, g_+^2},\quad "
              r"g = \kappa - \sqrt{q_0}\, t - \sqrt{1-q_0}\, u", _SRC,
              check=Check(("mu", "q0", "alpha", "kappa"), R.jamming_phi, pts, display=_phi_display,
                          tol="1e-9")),
        cited("1RSB capacity",
              r"\alpha_c^{\mathrm{1RSB}}(\kappa) = \min_{\mu,\, q_0} A(\mu, q_0),\quad "
              r"A = -\frac{\mathrm{ent}}{\mathrm{en}},\quad \Phi = \mathrm{ent} + \alpha\, \mathrm{en},\quad "
              r"A(\mu \to 0) = \frac{1}{D(\kappa)}", _SRC + "; " + _STOJ),
        cited("1RSB replicon: the contact index (leading order of FPSUZ Eq. (62))",
              r"\Lambda = 1 - c,\quad c = \alpha\, \mathbb{E}_t \left[\frac{\mathbb{E}_u\, e^{-\frac{\mu}{2} g^2}"
              r"\mathbb{1}_{g > 0}}{\mathbb{E}_u\, e^{-\frac{\mu}{2} g_+^2}}\right]", _SRC),
        cited("K-step RSB at T = 0 (perceptron_krsb.py): nested averages and log-determinant",
              r"F_K(g) = -\tfrac{1}{2} g_+^2,\ \ F_i(a) = \frac{1}{\mu_i} \ln \mathbb{E}_u\, e^{\mu_i F_{i+1}"
              r"\left(a - \sqrt{q_{i+1} - q_i}\, u\right)},\ \ \Phi_K = \frac{q_0}{2 \Lambda_0} + \frac{1}{2}"
              r"\sum_{i<K} \frac{\ln(\Lambda_i / \Lambda_{i+1})}{\mu_i} + \alpha\, \mathbb{E}_t F_0\!\left("
              r"\kappa - \sqrt{q_0}\, t\right),\ \ \Lambda_K = 1,\ \Lambda_i = \Lambda_{i+1} + \mu_i (q_{i+1} - q_i)",
              _SRC),
    ]
