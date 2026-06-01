# U(1) HMC gauge-configuration data format

Capability 2 reads gauge configurations produced by
[`g-koutsou/u1-hmc`](https://github.com/g-koutsou/u1-hmc) (2D, U(1), pure-gauge
Hybrid Monte Carlo). This note is the data contract the reader
(`gauge/u1_config.py`) implements against, derived from `u1-hmc/lib.py`
(`gauge.save`) and `u1-hmc/u1-hmc.py` (the driver).

## On-disk layout (HDF5)

```
<file>.h5
└── beta{:5.3f}/                 e.g. "beta1.000"
    └── traj{:08.0f}/            e.g. "traj00000000"
        └── u    (dataset)       shape (Ly, Lx, 2), complex128
        + group attrs (metadata)
```

- **Grouping:** one top-level group per β value (formatted `"beta%5.3f"`), one
  subgroup per trajectory (formatted `"traj%08.0f"`).
- **Dataset `u`:** the gauge links, shape `(Ly, Lx, 2)`, dtype `complex128`,
  **unit modulus** `U_μ(n) = exp(iθ_μ(n))`. The last index μ ∈ {0, 1} labels the
  two lattice directions.
- **Group attributes (metadata):** `plaquette` (mean plaquette), `topo`
  (topological charge `Q`), `dH`, `T`, `tau`, `NT`.

The MPI/halo layout used internally by `u1-hmc` is **not** present on disk:
`gauge.save` writes only the clean interior `(Ly, Lx, 2)` array. Reading
therefore needs **serial `h5py` only** — no MPI, no parallel HDF5.

## Index/orientation conventions

- Directions are `μ = 0, 1`. In `u1-hmc`, axis 0 and axis 1 correspond to the two
  lattice extents `(Ly, Lx)`. The reader stores `links[y, x, μ]`; the covariant
  operator must adopt the same axis↔μ correspondence as the free-field
  `stencil_to_matrix` (axis 0 fastest in the lexicographic site index).
- `U_{-μ}(n) ≡ U_μ(n - μ̂)*` (backward link is the conjugate of the shifted forward
  link). For U(1) this is just complex conjugation.

## Cross-checks (acceptance tests)

Recompute from `links` and match the stored attrs to ~1e-10:

- **Plaquette:** `P = mean_n Re[ U₀(n) · U₁(n+0̂) · U₀(n+1̂)* · U₁(n)* ]`.
- **Topological charge:** `Q = (1/2π) Σ_n arg( U₀(n) U₁(n+0̂) U₀(n+1̂)* U₁(n)* )`.

(See `u1-hmc/lib.py::gauge.plaquette` and `gauge.topo`.) These two checks
validate both the reader and the index conventions in one shot.

## Reader API (planned, `gauge/u1_config.py`)

```python
@dataclass
class U1Config:
    links: np.ndarray        # (Ly, Lx, 2) complex, unit modulus
    beta: float
    traj: str
    metadata: dict           # plaquette, topo, dH, T, tau, NT
    @property
    def L(self) -> tuple[int, int]: ...
    @property
    def dim(self) -> int: ...        # 2

def list_betas(path) -> list[float]: ...
def list_trajectories(path, beta) -> list[str]: ...
def load_config(path, beta, traj) -> U1Config: ...
def load_ensemble(path, beta, pick=slice(None)) -> Iterator[U1Config]: ...
```

## APE smearing

`u1-hmc/lib.py::gauge.apesmear(alpha)` implements 2D U(1) APE smearing (staple
average + projection back to U(1) by dividing out the modulus). Port this into
`gauge/observables.py` or `gauge/covariant.py` so smeared kernels can be studied
(the overlap-kernel results in the literature use APE-smeared links).
