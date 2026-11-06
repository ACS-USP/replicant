# BootLoops Host Repository

`replicant` ships inside BootLoops 1.0, a harness for LLMs doing precision quantitative science.
BootLoops is a scientific software package that is driven by an agent, together with working
protocols that make the model's results checkable. It does not depend on any particular model:
an agent started inside the clone gains the instruments.

## What lives where

- `tools/`: the toolkit packages, one directory per package, each with a `GUIDE.md` or `README.md` stating purpose, acceptance gates and verification class. `tools/README.md` is the index.
- `tools/BATTERIES.json`: the canonical battery command and working directory for every package.
- `toolkit/ours/README.md`, `toolkit/ours/RECIPES.md`: the tool roster by theme, and working recipes with their checks.
- `toolkit/external/TOOLS.md`: external engines, with licenses and where to obtain each.
- `upgrades/`: house engines in full source (SOFIA.jl, Eichler.jl, the Leviathan Landau engine), each with `PATCHES.md` and its license.
- `INSTALL.md`: the venv and dependencies.
- `README.md`, `CHANGELOG.md`, `REFERENCES.md`, `THIRD_PARTY.md`, `LICENSE` (MIT), `LICENSE-CONTENT` (docs, CC BY 4.0).

## Packages `replicant` depends on

| Package | Used for |
|---|---|
| `tools/baller` | ball arithmetic (python-flint arb) for every certified route (`lattice/ball.py`) |
| `tools/annihilator` | holonomic recognition of the 2D Ising series (`series/recognize.py`); missing -> `missing-tool:annihilator` |

## The repository battery

```bash
python3 run_selftests.py --par 8          # every package, 8 workers
python3 run_selftests.py replicant        # one package
```

`run_selftests.py` reads each package's verification class from `tools/README.md` (`selftest`,
`partial`, `smoke` or `data-gated`) and runs exactly the battery pinned in `tools/BATTERIES.json`.
It writes the full record to `selftest_results.json`. A data-gated package that lacks its data
exits with code 4 and is reported REFUSED, which does not count as a failure. Any other nonzero
exit is a FAIL. The default per-attempt timeout is 300 s (`--timeout`).

## Remotes

| Remote | Repository |
|---|---|
| `acs` | `ACS-USP/replicant` (public; this wiki; formerly `bootloops-statmech`) |
| `statmech` | `renatovicente/bootloops-statmech` (private mirror) |
| `origin` | `BootLoops-ai/bootloops` (upstream BootLoops, without `replicant`'s statmech history) |

## Related pages

- [Architecture Overview](Architecture-Overview)
- [Automated Tests](Automated-Tests)
- [Troubleshooting](Troubleshooting)
