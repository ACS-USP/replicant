# Troubleshooting

Each section starts from what you see. It then gives the cause and the designed resolution.

## `REFUSED: unsupported:python-flint` (or lattice / bounds tests fail)

**Cause:** the `python3` on `PATH` does not have python-flint. The certified routes need arb
through `tools/baller`.

**Resolution:** activate the project venv (`source .venv/bin/activate`; see `INSTALL.md`) and run
again. The batteries invoke the `python3` on `PATH`, so the activation matters. This is
deliberate: there is no silent fallback to floats. See
[Certified and Statistical Computation](Certified-and-Statistical-Computation).

## `REFUSED: missing-step:route:replica`

**Cause:** the entry is cavity-only, amp-only or sample-only (for example `viana_bray`, `xorsat`,
`spiked_wigner` or `cluster_mc`), and the default route is `replica`.

**Resolution:** pass the route, e.g. `--route cavity`. `python3 -m replicant list` shows each
entry's routes. See [Catalog and Routes](Catalog-and-Routes).

## `REFUSED: precondition:known-parameter`

**Cause:** a parameter name the route does not read: a typo, or a finite-N parameter such as
`seeds` given without `N`.

**Resolution:** the detail lists the accepted names (`CatalogEntry.parameters[route]`). Parameters
are never silently ignored.

## The run succeeds but `final` is `false`

**Cause:** at least one value is `unstable` or `not-computed`. For example, the SK RS saddle at
`beta=2` has a negative replicon, and every statistical value is `not-computed`.

**Resolution:** read `finality` and `notes`. A non-final value is a candidate, not an answer. Try
a different ansatz (e.g. `--param ansatz=1rsb` for `sk`), or use `require_final=True` in the API
to make the run refuse instead. See [Results, Finality and Refusals](Results-Finality-and-Refusals).

## `stable: null`

**Cause:** a composed model whose energy or site has no replicon rule. Stability is unknown.

**Resolution:** treat the values as unverified. Null means "unknown", not "fine".

## `REFUSED: missing-step:family:...` / `:site:...` / `:disorder:...`

**Cause:** the spec needs a level-(a) hook that does not exist.

**Resolution:** the detail names what exists and the nearest catalog entry. Adding a hook to
`derive.FAMILIES` / `SITES` / `LAWS` is the extension point. See
[Derivation and Composer](Derivation-and-Composer).

## A Monte Carlo value is labelled undetermined

**Cause:** a diagnostic failed for that value: equilibration drift, the tau_int window, a plateau
whose length cannot be fixed. The rest of the run is kept.

**Resolution:** raise the sweeps or burn-in, or add seeds. An undetermined value is never checked
against a reference band.

## `REFUSED: precondition:tex-engine` with `--pdf`

**Cause:** neither `tectonic` nor `pdflatex` is on `PATH`.

**Resolution:** install one of them, or drop `--pdf` and keep the `.tex`. See
[LaTeX Derivation Writer](LaTeX-Derivation-Writer).

## pytest runs for half an hour

**Cause:** pytest ignores the `@slow` attribute.

**Resolution:** use `python3 tools/replicant/selftest.py` (default tier) or `--full`. See
[Automated Tests](Automated-Tests).

## Related pages

- [Results, Finality and Refusals](Results-Finality-and-Refusals)
- [CLI and Python API](CLI-and-Python-API)
- [Automated Tests](Automated-Tests)
