# lattice-dirac-spectra

Eigenvalue spectra of Wilson-type and overlap lattice Dirac operators:
free-field analytical formula, exact diagonalization on U(1) HMC gauge
configurations, stochastic mock spectra, and 3D Brillouin-zone visualization.
Wilson and Brillouin kernels in arbitrary dimension `d`.

## Status

Early development. Implemented so far (milestones M0–M1):

- **Operators core** (`operators/`): stencils (4 Laplacians × 3 derivatives,
  dimension-general), Fourier evaluation, gamma matrices, the coordinate-space
  Dirac matrix, the closed-form recipe eigenvalues, and the overlap projection.
- **Capability 1 — free-field spectra** (`spectra/free_field.py`): the recipe
  versus direct diagonalization, agreeing to machine precision.

Planned:

- **Capability 2** — exact spectra on U(1) HMC configs (`gauge/`, `spectra/gauged.py`).
- **Capability 3** — stochastic mock spectra (`spectra/mock.py`).
- **Capability 4** — Brillouin-zone surfaces and dispersion relations (`visualization/`).

See [`DESIGN_BLUEPRINT.md`](DESIGN_BLUEPRINT.md) for the full architecture,
per-capability module contracts, the U(1) config data format, the test
invariants, and the roadmap.

## Installation

```bash
git clone https://github.com/Stylianos29/lattice-dirac-spectra.git
cd lattice-dirac-spectra

# (recommended) create an environment
conda create -n lds python=3.11 && conda activate lds
# or:  python -m venv .venv && source .venv/bin/activate

# install
pip install -e .              # core
pip install -e ".[notebooks]" # + Jupyter
pip install -e ".[dev]"       # + tests/linting
```

## Quick start

```python
from lattice_dirac_spectra import (
    eigenvalues_from_formula,
    eigenvalues_from_matrix,
    compare_eigenvalues,
)

# Brillouin operator in 2D on an 8^2 lattice
ef = eigenvalues_from_formula("bri", "iso", d=2, L=8)   # recipe
ed = eigenvalues_from_matrix("bri", "iso", d=2, L=8)    # direct diagonalization
print(compare_eigenvalues(ef, ed))                       # ~1e-13
```

Or from the command line:

```bash
python core/src/verify_free_field.py --dim 2 --length 8
```

## Project structure

```
lattice-dirac-spectra/
├── core/
│   ├── lattice_dirac_spectra/   # the importable package
│   │   ├── constants/           # operator keys, physics constants, paths, styling
│   │   ├── operators/           # stencils, fourier, gamma, dirac, overlap
│   │   ├── spectra/             # free_field (+ planned gauged, mock)
│   │   ├── gauge/               # U(1) config reader + covariant operators (planned)
│   │   ├── visualization/       # spectrum scatter, BZ surfaces, dispersion (planned)
│   │   └── utils/               # logging, helpers
│   ├── src/                     # CLI scripts
│   └── tests/                   # unit + integration tests, mock data
├── bash_scripts/                # pipeline automation
├── data_files/                  # raw (gauge ensembles) + processed (cached spectra)
├── output/                      # plots, tables
├── notebooks/                   # working notebooks
├── examples/                    # polished example notebooks
└── docs/                        # theory, API, tutorials
```

## Running tests

```bash
pytest core/tests/
```

## References

- S. Dürr & G. Koutsou, *Phys. Rev. D* **83**, 114512 (2011).
- S. Dürr & G. Koutsou, arXiv:1701.00726.
- U(1) HMC gauge configurations: <https://github.com/g-koutsou/u1-hmc>.

## License

MIT (see `pyproject.toml`).
