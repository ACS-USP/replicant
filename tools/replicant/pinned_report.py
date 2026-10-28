"""The joint-replicon table of the pinned-direction pair problem.

  python3 -m replicant.pinned_report [--out report.json] [--audit-frac 0.9]

For each of the paper's states (d, n, c) = (6, 16, 0), (10, 35, 0), (14, 70, 0), (20, 140, 0),
(30, 350, 0) (alpha = (n - 2)/d), plus c = +-0.5 at (10, 35): the capacity boundary
(pinned.capacity: kappa* = m_RS, h*, a*, the joint replicon 1 - k/d against the fixed-a slice
replicon 1 - Phi(-h*)/Psi(h*) of the catalog perceptron at (alpha_eff, -h*)) and the RS saddles at
kappa = f kappa* for f in FRACTIONS (pinned.state_point: q, the joint replicon, the longitudinal
M-mode and the standard-sector modes mixing M with the anomalous Q mode).  At f = AUDIT_FRAC the root
is audited against the derived equations and the replicon re-derived (derive_replicon).
"""
import argparse
import json
import sys

import mpmath as mp

from . import pinned as P

STATES = ((6, 16, 0), (10, 35, 0), (14, 70, 0), (20, 140, 0), (30, 350, 0), (10, 35, 0.5),
          (10, 35, -0.5))
FRACTIONS = (0.999, 0.99, 0.9, 0.5)
AUDIT_FRAC = 0.9


def _f(x, k=8):
    return float(mp.nstr(x, k)) if x is not None else None


def rows(states=STATES, fractions=FRACTIONS, audit_frac=AUDIT_FRAC, dps=20):
    out = []
    for d, n, c in states:
        alpha = mp.mpf(n - 2) / d
        cap = P.capacity(alpha, d, c, dps=dps)
        slice_rep = 1 - mp.ncdf(-cap.h) / P.psi(cap.h)
        out.append({"d": d, "n": n, "c": c, "alpha": _f(alpha), "frac": 1.0, "kappa": _f(cap.kappa, 12),
                    "kappa_star": _f(cap.kappa, 12), "a": _f(cap.a), "h": _f(cap.h),
                    "k": _f(cap.support_vectors()), "k_over_d": _f(cap.active_over_d),
                    "joint_replicon": _f(cap.replicon), "slice_replicon": _f(slice_rep),
                    "stable": bool(cap.replicon > 0), "q": 1.0})
        for f in fractions:
            sp_ = P.state_point(alpha, d, c, f * cap.kappa, audit=(f == audit_frac), dps=min(dps, 15))
            out.append({"d": d, "n": n, "c": c, "alpha": _f(alpha), "frac": f, "kappa": _f(sp_.kappa, 12),
                        "kappa_star": _f(cap.kappa, 12), "a": _f(sp_.a), "h": _f(sp_.h),
                        "q": _f(sp_.q, 10), "joint_replicon": _f(sp_.replicon),
                        "derived_replicon": _f(sp_.replicon_derived),
                        "longitudinal_mode": _f(sp_.longitudinal),
                        "standard_modes": [str(mp.nstr(m, 6)) for m in sp_.block.modes],
                        "standard_complex": bool(sp_.block.complex_pair), "stable": bool(sp_.stable)})
    return out


def markdown(table):
    lines = ["| d | n | c | kappa/kappa* | q | joint replicon | (audited) | slice replicon | k/d | "
             "longitudinal | standard-sector modes | stable |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        cap = r["frac"] == 1.0
        modes = ("n/a (q = 1)" if cap else " , ".join(r["standard_modes"])
                 + (" (complex)" if r["standard_complex"] else ""))
        lines.append(f"| {r['d']} | {r['n']} | {r['c']} | {'1 (capacity)' if cap else r['frac']} | "
                     f"{'1' if cap else r['q']} | {r['joint_replicon']:+.4f} | "
                     f"{'' if cap or r.get('derived_replicon') is None else format(r['derived_replicon'], '+.4f')} | "
                     f"{format(r['slice_replicon'], '+.4f') if cap else ''} | "
                     f"{format(r['k_over_d'], '.4f') if cap else ''} | "
                     f"{'diverges' if cap else format(r['longitudinal_mode'], '.4g')} | {modes} | "
                     f"{'stable' if r['stable'] else 'unstable'} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    table = rows()
    print(markdown(table))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(table, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
