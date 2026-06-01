# lattice-dirac-spectra — Design Blueprint

> **Status:** living design document. Implementation has begun (M0–M1 done).
> **Package (importable):** `lattice_dirac_spectra` — same name everywhere.
> **Author:** Stylianos Gregoriou.
> **Purpose:** the single authoritative context document for this project,
> written to be handed to *Claude Code* (or any collaborator) so implementation
> can proceed without re-deriving the physics or re-reading the source notebooks.
> Mirrors the conventions of `qpb-data-tools`.

A note on terminology: the stochastic model in Capability 3 is referred to
throughout as the **recipe** / **mock-spectrum recipe**. A permanent name is
still to be decided (candidates: *quadrature recipe*, *perturbed-momentum
recipe*); the word "Pythagorean" is deliberately **not** used anywhere in this
project.

---

## 1. The research arc (why these four capabilities are one project)

Everything here is about **one object**: the eigenvalue spectrum of a Wilson-type
lattice Dirac operator (and its overlap descendant) over the Brillouin zone, at
three fidelity levels plus a visualization layer.

| # | Capability | Role | Fidelity | Cost |
|---|------------|------|----------|------|
| 1 | Free-field spectra | exact baseline / verification | exact | cheap |
| 2 | U(1)-gauged spectra | ground truth on real configs | exact (per config) | expensive |
| 3 | Mock spectra (the recipe) | cheap stochastic model | approximate | very cheap |
| 4 | Brillouin-zone 3D plots | visualization of building blocks | n/a | cheap |

**The spine is the recipe.** From the Clifford algebra {γ_μ, γ_ν} = 2δ_μν, the
momentum-space Dirac block is `a D̂(k) = A(k)·1 + i Σ_μ γ_μ s_μ(k)`, whence

```
D†D = (A² + |s|²)·1   ⇒   λ±(k) = A(k) ± i |s(k)|
```

with `A(k) = -½ a²Δ̂(k) + m0` and `i s_μ(k) = a∇̂_μ(k)` (s_μ real). This formula is:

- **exact** at the discrete lattice momenta `k_μ = (2π/L) n_μ` (Capability 1);
- **a stochastic model** at *perturbed* momenta `φ_μ = k_μ + δφ_μ`,
  `δφ_μ ~ 𝒩(0, σ²)` (Capability 3): σ=0 recovers Capability 1 exactly; σ>0
  fills the spectral bulk, mimicking a gauge background.

**Capability 2 exists primarily to validate Capability 3.** The central
prediction from the working notes is that the perturbation variance tracks the
inverse coupling:

```
σ² ∝ ⟨θ²⟩ ∝ 1/β        (weak coupling, Wilson gauge action)
```

derived by a central-limit argument over lattice sites. M5 tests whether a fitted
σ(β) reproduces the actual U(1)-gauged spectra. See
`docs/theory/mock_spectra_theory.md` for the full D†D scalar/vector/tensor
decomposition and the CLT argument.

References:
- S. Dürr & G. Koutsou, *Phys. Rev. D* **83**, 114512 (2011).
- S. Dürr & G. Koutsou, arXiv:1701.00726.
- "Wilson–Dirac spectra: from free field to gauge background" (working notes, 2026).

---

## 2. Operator and convention reference (shared foundation)

Implemented in `operators/`. Fixed across the whole package.

### 2.1 Stencils (`operators/stencils.py`)
A stencil is `(3,)^d` integer array + integer norm; cell `idx` has displacement
`idx - 1`. **Laplacians** by hop count: `lap_std {0:-2d,1:1}/1`,
`lap_til {0:-2^d, d:1}/2^(d-1)`, `lap_bri {j:2^(d-j)}` with center
`-Σ C(d,j)2^j wt[j]`, norm `4^(d-1)`, `lap_iso` (per-d table). **Derivatives**
(antisymmetric on axis 0) by transverse hop count: `der_std {0:1}/2`,
`der_bri {j:2^(d-1-j)}/2^(2d-1)`, `der_iso {j:4^(d-1-j)}/(2·6^(d-1))`. μ-th
derivative = `swapaxes(stencil, 0, μ)`. Named kernels: `Wilson=(std,std)`,
`Brillouin=(iso,bri)`. 12 combinations total.

### 2.2 Fourier (`operators/fourier.py`)
`eval_stencil_fourier(stencil, norm, phi)` → `Σ w·exp(i φ·disp)/norm`. Laplacian
real; derivative imaginary. `discrete_momenta(d, L)` → the `L^d` allowed momenta.

### 2.3 Gamma (`operators/gamma.py`)
Hermitian, {γ_μ,γ_ν}=2δ. d=2: σ1,σ2; d=3: σ1,σ2,σ3; d=4: σ1⊗σ1, σ1⊗σ2, σ1⊗σ3,
σ2⊗1. `Ns(d)=2^(d//2)`.

### 2.4 Dirac matrix (`operators/dirac.py`)
`build_free_dirac_matrix`: `D = -½(Δ⊗1) + Σ_μ(∇_μ⊗γ_μ) + m0`, lexicographic
(axis 0 fastest), periodic BC, links=1. `recipe_eigenvalues(lap, der, d, phi, m0)`
returns the two branches.

### 2.5 Overlap (`operators/overlap.py`)
`overlap_eigenvalues(z, rho, mass) = (ρ+m/2) + (ρ-m/2)(z-ρ)/|z-ρ|`,
`0<ρ<2`, `0≤m<2ρ`. Free-field → GW circle (center `ρ+m/2`, radius `ρ-m/2`),
`gw_circle(rho, mass)`. `kl_sign(x, n) = tanh((2n+1)·artanh x)` via the
`((1+x)^ℓ ∓ (1-x)^ℓ)` odd/even form. (Complete massive form and partial-fraction
multishift solver: documented in `docs/theory/overlap_and_kenney_laub.md`,
implemented as needed in M4.)

---

## 3. Repository structure (as built)

```
lattice-dirac-spectra/
├── pyproject.toml
├── README.md
├── .gitignore
├── DESIGN_BLUEPRINT.md
├── core/
│   ├── lattice_dirac_spectra/        # the importable package
│   │   ├── __init__.py               # curated public API
│   │   ├── constants/                # operators, physics, paths, visualization
│   │   ├── operators/                # stencils, fourier, gamma, dirac, overlap  [M1 done]
│   │   ├── gauge/                    # u1_config, covariant, observables         [M3]
│   │   ├── spectra/                  # free_field [M1], gauged [M3], mock [M4]
│   │   ├── visualization/            # spectrum_plotter, brillouin_zone, dispersion [M2]
│   │   └── utils/                    # logging_utilities, helpers
│   ├── src/                          # CLI scripts (click); verify_free_field.py done
│   └── tests/
│       ├── unit_tests/               # test_stencils, test_fourier_gamma,
│       │                             #   test_free_field_recipe, test_overlap
│       ├── integration_tests/
│       └── mock_data/                # tiny U(1) ensemble for M3 tests
├── bash_scripts/{library,workflows}/
├── data_files/{raw,processed}/
├── output/{plots,tables}/
├── notebooks/
├── examples/
└── docs/{theory,api,tutorials}/
```

Decisions locked: package directory named `lattice_dirac_spectra` (one name
everywhere); covariant code written dimension-general, only the U(1) reader is
2D/U(1)-specific; diagonalization dense (`numpy.linalg.eigvals`) now, sparse
(`scipy.sparse.linalg.eigs`) added in M3 if lattice sizes demand it.

---

## 4. Packaging

`pyproject.toml`: setuptools, `where=["core"]`, `include=["lattice_dirac_spectra*"]`,
deps `numpy scipy h5py matplotlib click`; extras `notebooks`, `dev`.

---

## 5. Capability 1 — Free-field spectra  [IMPLEMENTED]

`spectra/free_field.py`: `eigenvalues_from_formula`, `eigenvalues_from_matrix`,
`compare_eigenvalues`. **Acceptance (passing):** all 12 ops, d∈{2,4}, L∈{4,6,8},
`compare_eigenvalues < 1e-10`.

---

## 6. Capability 2 — U(1)-gauged spectra  [M3]

### 6.1 The `u1-hmc` data contract (verified against the source repo)
HDF5 written by `g-koutsou/u1-hmc` `gauge.save()`:
- Path `"/beta{:5.3f}/traj{:08.0f}/u"` (e.g. `/beta1.000/traj00000000`).
- Dataset `u`: shape `(Ly, Lx, 2)`, complex128, unit-modulus `exp(iθ_μ(n))`;
  μ∈{0,1} = the two lattice axes.
- Group attrs: `plaquette`, `topo`, `dH`, `T`, `tau`, `NT`.
- 2D, U(1), pure gauge. Reading needs **no MPI**.
Full detail: `docs/theory/u1_hmc_data_format.md`.

### 6.2 Reader (`gauge/u1_config.py`)
`U1Config` dataclass (`links (Ly,Lx,2)`, `beta`, `traj`, `metadata`);
`list_betas`, `list_trajectories`, `load_config`, `load_ensemble`.
**Acceptance:** recomputed plaquette/topo match stored attrs to 1e-10.

### 6.3 Covariant operator (`gauge/covariant.py`) — the only hard part
Gauge-covariant `stencil_to_matrix`: a hop `n → n+disp` carries the parallel
transporter (product of links along the path). On-axis 1-hop = single link;
off-axis k-hop = **average over the k! shortest paths**. For U(1) the product is
automatically in the group — no back-projection. 2D prototype to learn from:
the old `Chiral_symmetry...` notebook (`combined_gauge_field`); rewrite clean and
dimension-general. **Acceptance (critical):** links=1 reproduces the free-field
operator bit-for-bit.

### 6.4 Orchestration (`spectra/gauged.py`)
`gauged_spectrum(lap, der, config, m0, n_ape, alpha, c_sw)`,
`ensemble_spectrum(...)`. APE smearing ported from `u1-hmc.apesmear`; optional
tree-level clover term.

---

## 7. Capability 3 — Mock spectra (the recipe)  [M4]

`spectra/mock.py`: `sample_perturbed_momenta(d, L, n_configs, sigma, seed)`,
`mock_kernel_spectrum`, `mock_overlap_spectrum`, `mock_overlap_kl`,
`condition_number`. **Research:** `fit_sigma_of_beta(...)` matches mock (varying
σ) to gauged (varying β) by a spectral distance, testing σ²∝1/β.
**Acceptance:** at σ=0, `mock_kernel_spectrum == eigenvalues_from_formula`.

---

## 8. Capability 4 — Brillouin-zone & dispersion  [M2]

`visualization/brillouin_zone.py` (PRD 83 Figs 1–2: 3D surfaces/contours of
Δ̂(k), ∇̂_μ(k) over (−π,π]^d); `visualization/dispersion.py` (1701.00726 Figs 1–2:
aE vs √(Σap²) along (1,0,0),(1,1,0),(1,1,1) vs continuum, optional
`am→exp(am)−1`); `visualization/spectrum_plotter.py` (complex-plane scatter:
formula-vs-direct overlay, ensemble clouds, GW circles).

---

## 9. Testing strategy & golden invariants

Unit tests mirror module layout (pytest). Never let these break:
1. Free-field formula == direct diagonalization (≤1e-10), all 12 ops, d∈{2,4}.  **[passing]**
2. Gauged operator with links=1 == free-field operator (bit-for-bit).  [M3]
3. Mock spectrum at σ=0 == free-field formula (exact).  [M4]
4. Plaquette/topo recomputed from a loaded config == stored attrs (≤1e-10).  [M3]
5. Overlap of any free-field kernel spectrum lies on the GW circle (≤1e-12).  **[passing]**

`core/tests/mock_data/` will hold one tiny (4×4) U(1) HDF5 ensemble for offline
CI of the reader and gauged-spectrum tests.

---

## 10. Documentation plan (`docs/`)
- `project_structure.md` — directory layout (written).
- `theory/mock_spectra_theory.md` — D†D decomposition + CLT σ²∝1/β (written).
- `theory/u1_hmc_data_format.md` — the data contract (written).
- `theory/stencils_and_conventions.md`, `theory/overlap_and_kenney_laub.md` — [M2/M4].
- `api/` — one page per public module.
- `tutorials/` — one per capability, paired with `examples/` notebooks.

---

## 11. Milestone roadmap
- **M0 — Scaffold.** Tree, packaging, constants, logging. **[done]**
- **M1 — Free field.** operators/ + spectra/free_field.py + invariants #1,#5. **[done]**
- **M2 — Visualization.** BZ surfaces, dispersion, spectrum scatter.
- **M3 — U(1) gauged.** U1Config reader (+ tiny ensemble), covariant op, gauged spectrum; invariants #2,#4.
- **M4 — Mock + overlap.** recipe, overlap, KL approximants, condition numbers; invariants #3,#5(mock).
- **M5 — Research: σ(β).** Match mock to gauged across β; test σ²∝1/β.

Order: M0 → M1 → M2 → M3 → M4 → M5.
