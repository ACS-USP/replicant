# Results, Finality and Refusals

Every run ends in exactly one of two ways: a `Result` with a label on every value, or a `Refusal`
with a reason and the partial trace.

## The Result object

`compose_result.Result` (re-exported by `compose.py`):

| Field | Content |
|---|---|
| `values` | value key -> number (mpmath values are written as 20-digit decimal strings in JSON) |
| `errors` | value key -> ball radius (certified) or stderr (statistical) |
| `rigor` | `physics-level`, `certified` or `statistical` |
| `route` | the route that ran |
| `stable` | the saddle's computed flag, if any (`None` = unknown) |
| `final` | `True` only if every value is final; `None` if there are no physical values |
| `finality` | value key -> `Stability(state, ansatz, cite, note)` |
| `statistical` | the value keys estimated by sampling |
| `notes` | e.g. "q: RS value is NOT final (unstable)" |
| `report` | the `CompositionReport` (reused / adapted / missing) |
| `trace` | the `core.Trace` |
| `reference_checks(slow=False)` | re-runs the entry's banked comparisons |

## Per-value finality

`catalog.STATES = ("stable", "unstable", "not-computed", "exact-by-literature")`, and
`FINAL_STATES = ("stable", "exact-by-literature")`.

| State | Meaning | Final |
|---|---|---|
| `stable` | stability computed by this run and positive | yes |
| `exact-by-literature` | a cited source establishes the value (needs a cite) | yes |
| `unstable` | stability computed and negative, or the entropy rules the ansatz out | no |
| `not-computed` | no stability computed here, or a statistical value | no |

A value that is not final is a **candidate, not an answer**. `run(require_final=True)` refuses
when any value is not final. The reason is `unstable:<ansatz>`, `unknown-stability:rs` or
`not-final:<ansatz>`, and it is raised at the `require-final` step by `_finalise`.

Example: `python3 -m replicant run sk --param beta=2` returns RS `q`, `f`, `e` and `replicon`.
The RS saddle there has a negative replicon, so every value is `unstable` and `final` is false.

## Refusals

`core.Refusal(reason, detail, trace=, step=)`. `reason` is a `kind:detail` string. The kinds that
occur in the code:

| Kind | Example | When |
|---|---|---|
| `precondition` | `precondition:known-parameter` | an input or state precondition fails |
| `missing-step` | `missing-step:route:replica` | a hook, route or derivation does not exist |
| `no-solution` | `no-solution:rsb-audit` | a solver or audit found no admissible solution |
| `not-implemented` | `not-implemented:cavity-ising` | a named gap in the implementation |
| `unsupported` | `unsupported:python-flint` | an environment or input outside the supported set |
| `undetermined` | `undetermined:frsb-extrapolation` | the method cannot decide the value |
| `ambiguous` | `ambiguous:branch` | several admissible solutions, no rule to choose |
| `unbanked` | `unbanked:no-citation` | a reference record breaks the schema |
| `unstable`, `unknown-stability`, `not-final` | `unstable:rs` | `require_final=True` with a non-final value |

The complete reason table, entry by entry, is the REFUSAL REASONS table in
`tools/replicant/GUIDE.md`.

A refusal keeps the partial `Trace` of the steps applied before it (`refusal.trace`) and the
refusing step's name (`refusal.step`). A nested pipeline's trace is kept as `refusal.inner_trace`.
That is why `--latex` still writes a document for a refused run: the derivation up to the refusal,
followed by the reason.

## Undetermined values in sampled runs

Monte Carlo and population-dynamics routes do not refuse a correct sampler because a diagnostic
failed. They label the affected values undetermined and keep the rest. For example, an
`online_scm` plateau whose length cannot be fixed by BRW's definition is reported as
`undetermined`. Only explicit out-of-range requests are refused.

## Related pages

- [Reference Store and Provenance](Reference-Store-and-Provenance)
- [Certified and Statistical Computation](Certified-and-Statistical-Computation)
- [Troubleshooting](Troubleshooting)
