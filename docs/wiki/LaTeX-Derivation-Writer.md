# LaTeX Derivation Writer

`latex.py` turns any run into the appendix of a paper: the model, the method, the derivation, the
calculation, the solution with its labels, and the check table. It also accepts a refused run and
renders the partial trace together with the refusal reason.

## Entry points

```python
from replicant.latex import render_latex, write_latex, document_blocks, report_blocks

render_latex(result, *, title=None, standalone=True, extra=(), slow=False) -> str
write_latex(result, path, *, pdf=False, **kw) -> Path     # .tex, or .pdf with pdf=True
```

`result` is a `compose.Result` (catalog or composed) or a `core.Refusal`. On the CLI, use
`--latex out.tex`, optionally followed by `--pdf`.

## Section order

1. **Model**: the spec, conventions and the coupling law.
2. **Method and route**: replica or cavity, the honest caveat, and for a composed model what was reused or adapted and relative to what.
3. **Derivation**:
   - for catalog replica entries, the level-(a) derivation of the same spec (when a `derive.FAMILIES` hook exists), rendered from the sympy objects in its `State`;
   - then its derived replicon;
   - then the equations the family code implements (`latex_entries`, `latex_cavity`, and the per-entry `latex_*.py`);
   - for composed models, the level-(a) trace itself.
4. **Calculation**: the run's trace, in order.
5. **Solution**: the values with their per-key finality.
6. **Stability**.
7. **Reference check**: key, computed, banked, tolerance, verdict. Statistical rows show stderr. Slow and statistical rows read "not run" unless `slow=True`.
8. **Labels**.
9. **Bibliography**: a `thebibliography` built from every cite string. There is no BibTeX, so the document compiles offline (`latex_bib.py`).

## The equation-source rule

Text and citations are escaped (`tex.escape`); math bodies are not. A math block that is neither
rendered from a sympy object (`tex.sym`) nor cited is refused with
`precondition:equation-source`. The writer never displays an equation the code does not hold.

## PDF compilation

`latex_pdf.compile_pdf` uses `tectonic` (preferred, `--keep-logs`) or `pdflatex` (run twice).
With neither on `PATH`, the run is refused with `precondition:tex-engine`.

## Related pages

- [Derivation and Composer](Derivation-and-Composer)
- [CLI and Python API](CLI-and-Python-API)
- [Results, Finality and Refusals](Results-Finality-and-Refusals)
