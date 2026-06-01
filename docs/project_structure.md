# Project structure

```
lattice-dirac-spectra/
├── core/
│   ├── lattice_dirac_spectra/   # importable package (generic, reusable code)
│   │   ├── constants/           # operator keys, physics constants, paths, styling
│   │   ├── operators/           # stencils, fourier, gamma, dirac, overlap (physics core)
│   │   ├── gauge/               # U(1) config reader + covariant operators (Capability 2)
│   │   ├── spectra/             # free_field / gauged / mock orchestration
│   │   ├── visualization/       # spectrum scatter, BZ surfaces, dispersion (Capability 4)
│   │   └── utils/               # logging, helpers
│   ├── src/                     # thin CLI scripts (click), invoked by bash_scripts
│   └── tests/                   # unit_tests/ + integration_tests/ + mock_data/
├── bash_scripts/                # pipeline automation (library/ + workflows/)
├── data_files/                  # raw/ (gauge ensembles, gitignored) + processed/ (cached spectra)
├── output/                      # plots/ + tables/ (gitignored)
├── notebooks/                   # working notebooks
├── examples/                    # polished example notebooks (one per capability)
└── docs/                        # theory/ + api/ + tutorials/
```

## Layering

- **`core/lattice_dirac_spectra/`** — domain-agnostic, reusable functions and
  classes (the library). No script-level side effects on import.
- **`core/src/`** — domain-specific CLI entry points that orchestrate the library;
  invoked directly or by `bash_scripts/`.
- **`core/tests/`** — mirrors the library layout; `test_*.py`, run with `pytest`.

See `DESIGN_BLUEPRINT.md` (repo root) for the full architecture and roadmap.
