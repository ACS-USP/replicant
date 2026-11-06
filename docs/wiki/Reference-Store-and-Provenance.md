# Reference Store and Provenance

Every catalog entry is checked against numbers from the literature. These are banked in
`tools/replicant/reference/*.json`, one file per entry or per group of entries, and loaded by
`refdata.load`. The schema is `reference/_schema.md`.

## Record schema

```json
{
  "key": {
    "value": "0.7632",
    "tol": "1e-4",
    "cite": "Author Year, Eq. N",
    "independent_route": "how this was cross-checked without the cited derivation",
    "sources_disagree": [{"value": "0.7633", "cite": "Other Year, Table N"}]
  }
}
```

| Field | Rule | Refusal when broken |
|---|---|---|
| `value`, `tol` | finite decimal strings, parsed with mpmath at 30 digits (`_DPS = 30`), never floats | `unbanked:malformed` (also for `inf` / `nan`) |
| `cite` | required: the value comes from the cited paper, never from memory | `unbanked:no-citation` |
| `independent_route` | required: a second route to the same number | `unbanked:no-independent-route` |
| `sources_disagree` | optional list of differing published values; `tol` must cover the spread | — |

## Banking rules

- A value is banked only from a fetched source, at the source's printed precision. The entry
  citations say how each source was read: "PDF read", "page images read", "not fetched (as
  printed by ...)". The `list` output shows these annotations.
- When two prints disagree, both are banked: the main `value` plus a `sources_disagree` entry,
  with a `tol` wide enough to cover both. The percolation k-core threshold c4 is an example: one
  source prints 5.14, another 5.15, and both are banked with `tol` 0.012.
- Some primary sources (Onsager, Yang, Kaufman, Hukushima–Iba, Balogh–Pittel, Ferrenberg–Swendsen)
  were read through secondary sources that state them, and the citation names the secondary source.

## How a check is decided

`catalog.compare(entry, ref, values, errors)`:

- **Deterministic keys**: `refdata.check` accepts `got` within `tol` of the main value or of any
  `sources_disagree` value.
- **Statistical keys**: the band is `t(0.995, runs - 1) * hypot(stderr, tol)`, a 99% Student-t
  band, plus an optional systematic term.
- A reference quantity missing from the run's values is refused with
  `precondition:reference-quantity`.

Slow and statistical comparisons are not run by default. They run through
`Result.reference_checks(slow=True)` or the `--full` battery, and the JSON lists them under
`references_skipped_slow`.

## Planted controls

Each comparison must be shown able to fail. The tests plant a wrong value through a hook (for
example, dropping the multicanonical histogram, or swapping a degree law) and assert that the
check rejects it. They then restore the hook and assert its identity. Statistical tests state
their power against the planted error and are sized for power of at least 0.9. See
[Automated Tests](Automated-Tests).

## Related pages

- [Results, Finality and Refusals](Results-Finality-and-Refusals)
- [Certified and Statistical Computation](Certified-and-Statistical-Computation)
- [Automated Tests](Automated-Tests)
