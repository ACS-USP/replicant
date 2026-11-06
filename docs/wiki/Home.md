# replicant Wiki

Last updated: 2026-11-06 (source branch: `main`)

## What this project is

`replicant` (formerly `bootloops-statmech`) is the BootLoops harness for LLM-driven precision science, together with
`replicant`, a package that runs published replica-method and cavity-method calculations of
disordered systems as typed, recorded steps. Each step names its preconditions and its citation.
A missing precondition stops the run with a named refusal, never a silent fallback. Every run
returns a trace, a finality label for each value and a comparison with banked, cited reference
values. The package targets theoretical physicists, and the agents working with them, who need a
published statistical-mechanics result reproduced together with its derivation, or the same
calculation for a nearby model.

- Python 3 with `sympy`, `mpmath`, `numpy`, `scipy` and `python-flint`. Ball arithmetic goes through `tools/baller`.
- 38 catalog entries over eight routes: `replica`, `cavity`, `amp`, `exact`, `series`, `bound`, `online` and `sample`.
- A level-(a) symbolic derivation (replicate, average, RS ansatz, n -> 0, stationarity) for models near a catalogued one.
- Three rigor levels:
  - `physics-level`: replica, cavity, AMP and series results;
  - `certified`: ball enclosures;
  - `statistical`: sampled values over named seeds.
- A LaTeX writer that turns any run, refusals included, into a derivation appendix.
- A battery runner without pytest:
  - default tier, 748 tests;
  - `--full` adds 231 `@slow` tests.

## Pages

| Page | What it covers |
|---|---|
| [Architecture Overview](Architecture-Overview) | System diagram, design principles, module boundaries, the run flow |
| [Catalog and Routes](Catalog-and-Routes) | The 38 entries grouped by domain, their routes and rigor |
| [Derivation and Composer](Derivation-and-Composer) | Level (a) and level (b), `ModelSpec`, how the composer matches, reuses and adapts |
| [Results, Finality and Refusals](Results-Finality-and-Refusals) | The `Result` object, per-value finality, `require_final`, refusal reasons |
| [Reference Store and Provenance](Reference-Store-and-Provenance) | `reference/*.json`, the schema, banking rules, checks and tolerances |
| [Certified and Statistical Computation](Certified-and-Statistical-Computation) | Ball arithmetic entries, sampled routes, bands over named seeds |
| [CLI and Python API](CLI-and-Python-API) | `python3 -m replicant list/run`, parameters, exit codes, imports |
| [LaTeX Derivation Writer](LaTeX-Derivation-Writer) | `render_latex` / `write_latex`, the section order, the equation-source rule |
| [Automated Tests](Automated-Tests) | `selftest.py`, tiers, held-out tests, planted controls |
| [BootLoops Host Repository](BootLoops-Host-Repository) | The surrounding harness: `tools/`, `toolkit/`, `upgrades/`, `run_selftests.py` |
| [Troubleshooting](Troubleshooting) | Symptom-first fixes for common failures |
| [Publishing to GitHub Wiki](Publishing-to-GitHub-Wiki) | How this bundle is synced to the wiki repository |

## Quick local run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install mpmath sympy numpy scipy python-flint pytest
python3 tools/replicant/selftest.py                 # default battery, about 30 s
cd tools && python3 -m replicant list               # the 38 catalog entries
python3 -m replicant run perceptron --param alpha=1.5 --out /tmp/result.json --latex /tmp/p.tex
```

## Repository map

- `tools/replicant/`: the replica / cavity package (see [Architecture Overview](Architecture-Overview)).
- `tools/`: the other BootLoops toolkit packages, one directory per package, indexed by `tools/README.md`.
- `toolkit/`: rosters and recipes (`ours/README.md`, `ours/RECIPES.md`, `external/TOOLS.md`).
- `upgrades/`: the house engines shipped in full source (SOFIA.jl, Eichler.jl, the Leviathan Landau engine).
- `docs/rem/`: the REM explanatory documents and their Lean files.
- `docs/replicant/`: material for the paper appendix (the joint replicon of the pinned pair problem).
- `docs/superpowers/`: design specs and implementation plans of the replicant extensions.
- `docs/wiki/`: this wiki's source bundle.
- `run_selftests.py`: the repository-wide battery runner.

## Wiki maintenance policy

When code changes, update these pages in the same PR:

1. New or removed catalog entry, or a new route -> [Catalog and Routes](Catalog-and-Routes), [CLI and Python API](CLI-and-Python-API)
2. Change to `core.py`, `compose.py`, `compose_result.py` or the finality rule -> [Architecture Overview](Architecture-Overview), [Results, Finality and Refusals](Results-Finality-and-Refusals)
3. New level-(a) hook (`derive.FAMILIES`, `derive.SITES`, `derive.LAWS`) -> [Derivation and Composer](Derivation-and-Composer)
4. Change to `reference/_schema.md` or `refdata.py` -> [Reference Store and Provenance](Reference-Store-and-Provenance)
5. New certified or sampled route -> [Certified and Statistical Computation](Certified-and-Statistical-Computation)
6. Change to `latex*.py` / `tex*.py` -> [LaTeX Derivation Writer](LaTeX-Derivation-Writer)
7. New test tier, re-timed battery or changed test counts -> [Automated Tests](Automated-Tests), this page
