# CLI and Python API

The package runs with `tools/` on the path: `cd tools && python3 -m replicant ...`
(`__main__.py` -> `cli.main`).

## Commands

```bash
python3 -m replicant list
python3 -m replicant run <entry|spec.json> [--route replica|cavity|amp|rsb|exact|series|bound|online|sample] \
    [--param k=v ...] [--out result.json] [--latex out.tex [--pdf]]
```

| Behaviour | Detail |
|---|---|
| Default route | `replica` (`cli.py`: `--route` default) |
| Output | JSON result to `--out` (default stdout); the rendered trace to stderr |
| Exit 0 | the run returned a `Result` |
| Exit 2 | refusal: `REFUSED: <reason>: <detail>` on stderr; with `--latex` the partial derivation and the reason are still written |
| Exit 1 | any other error: `ERROR: <type>: <message>` |
| `--pdf` without `--latex` | refused `precondition:latex-path` |

## Parameter parsing (`cli.parse_param`)

- Integers parse as `int`.
- Decimals stay **decimal strings**. mpmath converts them, so `0.1` is never rounded through a
  binary float.
- `true` / `false` become booleans, case-insensitive on the CLI.
- A comma list becomes a tuple. A trailing comma makes a one-element list: `ys=0.3,`.
- Anything else is a string: `learning=bayes`.

## JSON result

The keys are `target`, `spec`, `route`, `rigor`, `values`, `errors`, `statistical`, `stable`,
`final`, `finality`, `notes`, `params`, `report`, `trace`, `references` and
`references_skipped_slow`. See [Results, Finality and Refusals](Results-Finality-and-Refusals).

## Python API

```python
from replicant.catalog import catalog                    # dict of CatalogEntry
from replicant.compose import compose, nearest_entry
from replicant.spec import ModelSpec

spec = catalog()["perceptron"].spec
res = compose(spec, "replica").run(alpha="1.5", kappa=0)  # -> Result
res.values, res.finality, res.final
res.reference_checks(slow=False)
```

`run(require_final=True)` refuses when any value is not final. The lower-level surfaces (the
derived solvers `solve_derived_rs` and `solve_derived_1rsb`, `derived_replicon`, the `amp`,
`csp`, `inference`, `lattice`, `online` and `montecarlo` modules) are listed with their signatures
in the INVOKE section of `tools/replicant/GUIDE.md`.

## Related pages

- [Catalog and Routes](Catalog-and-Routes)
- [LaTeX Derivation Writer](LaTeX-Derivation-Writer)
- [Troubleshooting](Troubleshooting)
