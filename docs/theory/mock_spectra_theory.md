# Mock-spectrum recipe — theoretical basis

This note summarizes why the **recipe**

```
z(φ) = A(φ) ± i |s(φ)|,   φ_μ = k_μ + δφ_μ,   δφ_μ ~ 𝒩(0, σ²)
```

reproduces the eigenvalue spectrum of a Wilson-type Dirac operator on a gauge
background. It follows the working notes "Wilson–Dirac spectra: from free field
to gauge background" (2026). (The temporary codename "Pythagorean" from those
notes is **not** used here.)

## 1. Operator structure

A general Wilson-type operator splits as

```
D = Σ_μ γ_μ ⊗ N_μ + 1_Ns ⊗ B,
```

where `N_μ` (V×V) is the gauge-covariant derivative in direction μ and
`B = -½Δ + m0` (V×V) the covariant Laplacian plus mass. For any gauge background,
lattice symmetries + link unitarity give

```
N_μ† = -N_μ   (anti-Hermitian),     B† = B   (Hermitian).
```

## 2. Exact decomposition of D†D

With those properties,

```
D†D = 1_Ns ⊗ (B² - Σ_μ N_μ²)            ← scalar (∝ identity in spin)
    + Σ_μ γ_μ ⊗ [B, N_μ]                ← vector
    - Σ_{μ<ν} σ_μν ⊗ [N_μ, N_ν]         ← tensor (lattice field strength)
```

with `σ_μν = ½[γ_μ, γ_ν]`. The scalar term is positive (since `-N_μ²` is PSD for
anti-Hermitian `N_μ`). The tensor commutator `[N_μ, N_ν]` is the lattice analogue
of `ig F_μν` — the same structure as the clover term.

## 3. Free field ⇒ the recipe is exact

With all links trivial, `N_μ` and `B` are simultaneously diagonalized in momentum
space: `N_μ(k) = i s_μ(k)`, `B(k) = A(k)`. Both commutators vanish, so

```
D†D|_free = 1_Ns ⊗ (A² + |s|²),     λ±(k) = A(k) ± i |s(k)|,
```

each with spinor degeneracy `Ns/2`. The directional contributions add **in
quadrature** — a direct consequence of the Clifford algebra. This is exactly the
recipe at σ=0.

## 4. Interacting case ⇒ Gaussian momentum perturbation

Write `U_μ(n) = exp(iε θ_μ(n))` and expand `N_μ = N_μ^(0) + ε N_μ^(1) + …`,
`B = B^(0) + ε B^(1) + …`. Key facts:

- The first-order commutators are **off-diagonal in momentum space** — they are
  proportional to differences of free-field eigenvalues, e.g.
  `[B^(0), N_μ^(1)](p,q) = (A(p) − A(q)) N_μ^(1)(p,q)`, which vanish at `p=q`.
  Hence they shift eigenvalues only at **second order** in perturbation theory.
- A **constant** gauge field `θ_μ(n)=θ_μ` is reproduced *exactly* by a uniform
  shift `k_μ → k_μ + θ_μ` (the recipe with `δφ_μ = θ_μ`).
- For a fluctuating field, the dominant effect on the eigenvalue at would-be
  momentum `k` is an effective shift
  `δφ_eff,μ(k) ~ V^(-1/2) Σ_n f_μ(n,k) θ_μ(n)` — a sum over many weakly
  correlated sites. By the **central limit theorem** (V → ∞) this converges to a
  Gaussian with variance

```
σ² ∝ ⟨θ_μ²⟩ ∝ 1/β
```

since `⟨θ²⟩ ∼ 1/β` for the Wilson gauge action at weak coupling.

## 5. What the recipe captures and misses

- **Captures:** the spectral boundary / overall shape (the scalar, Pythagorean-in-
  -quadrature term, robust under gauge averaging).
- **Misses (sub-leading):** the vector and tensor corrections — off-diagonal,
  second-order, and `O(a²)`-suppressed (further suppressed for the Brillouin
  operator) — which fill the spectral *interior* density. These would account for
  any mismatch between recipe and exact spectra.

## 6. Testable prediction (M5)

Fit `σ` to the exact U(1)-gauged spectra (Capability 2) at several `β`, by
matching a spectral distance (e.g. radial density of `|z|`). The prediction is
`σ²(β) ∝ 1/β` at weak coupling. Quantifying the residual (the vector/tensor
contribution) is the open research direction.
