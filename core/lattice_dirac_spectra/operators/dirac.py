"""
Dirac operator construction.

Two complementary representations of a free-field Wilson-type Dirac operator:

* :func:`build_free_dirac_matrix` -- the explicit ``(V*Ns) x (V*Ns)`` matrix in
  coordinate space (all gauge links set to the identity), for direct
  diagonalization.
* :func:`recipe_eigenvalues` -- the closed-form eigenvalues

      lambda_pm(phi) = A(phi) +/- i * sqrt(sum_mu s_mu(phi)**2)

  where ``A(phi) = -1/2 * Laplacian.hat(phi)`` and ``i s_mu = derivative_mu.hat(phi)``.
  This follows from the Clifford algebra and is exact at the discrete lattice
  momenta. Evaluated at perturbed momenta it becomes the stochastic "mock"
  recipe (see :mod:`lattice_dirac_spectra.spectra.mock`).

Reference: S. Duerr and G. Koutsou, Phys. Rev. D 83, 114512 (2011); and the
"Wilson-Dirac spectra" working notes (D^dag D scalar/vector/tensor decomposition).
"""

import numpy as np

from ..constants.physics import Ns
from .fourier import eval_stencil_fourier
from .gamma import gamma_matrices
from .stencils import get_der_mu, get_lap

__all__ = [
    "idx_to_coords",
    "coords_to_idx",
    "stencil_to_matrix",
    "build_free_dirac_matrix",
    "recipe_eigenvalues",
]


# --------------------------------------------------------------------------- #
# Lexicographic index <-> coordinate mapping (axis 0 runs fastest)
# --------------------------------------------------------------------------- #
def idx_to_coords(n: int, L: int, d: int) -> np.ndarray:
    """Map a linear site index to ``d``-dimensional coordinates (axis 0 fastest)."""
    coords = np.empty(d, dtype=int)
    tmp = n
    for mu in range(d):
        coords[mu] = tmp % L
        tmp //= L
    return coords


def coords_to_idx(coords: np.ndarray, L: int, d: int) -> int:
    """Map ``d``-dimensional coordinates to a linear site index (axis 0 fastest)."""
    idx = 0
    for mu in range(d):
        idx += int(coords[mu] % L) * (L**mu)
    return idx


# --------------------------------------------------------------------------- #
# Free-field stencil -> V x V matrix (periodic BC, all links = 1)
# --------------------------------------------------------------------------- #
def stencil_to_matrix(stencil: np.ndarray, norm: int, L: int, d: int) -> np.ndarray:
    """Realize a ``(3,)**d`` stencil as a ``V x V`` matrix with periodic BC.

    All gauge links are the identity (free field, ``Nc = 1``).
    """
    V = L**d
    M = np.zeros((V, V), dtype=complex)
    for n in range(V):
        cn = idx_to_coords(n, L, d)
        for idx in np.ndindex(stencil.shape):
            w = stencil[idx]
            if w == 0:
                continue
            disp = np.array([i - 1 for i in idx])
            target = (cn + disp) % L
            m = coords_to_idx(target, L, d)
            M[n, m] += w / norm
    return M


def build_free_dirac_matrix(
    lap_name: str,
    der_name: str,
    d: int,
    L: int,
    m0: float = 0.0,
) -> np.ndarray:
    """Build the explicit free-field ``(V*Ns) x (V*Ns)`` Dirac matrix.

        D = -1/2 (Laplacian (x) 1_Ns) + sum_mu (derivative_mu (x) gamma_mu) + m0

    Parameters
    ----------
    lap_name, der_name : str
        Laplacian and derivative stencil keys.
    d : int
        Lattice dimensionality.
    L : int
        Linear lattice extent.
    m0 : float, optional
        Bare mass (default 0).

    Returns
    -------
    ndarray, complex
        The dense Dirac matrix.
    """
    V = L**d
    ns = Ns(d)
    gammas = gamma_matrices(d)

    lap_s, lap_n = get_lap(lap_name, d)
    lap_mat = stencil_to_matrix(lap_s, lap_n, L, d)
    D = -0.5 * np.kron(lap_mat, np.eye(ns, dtype=complex))

    for mu in range(d):
        der_s, der_n = get_der_mu(der_name, d, mu)
        der_mat = stencil_to_matrix(der_s, der_n, L, d)
        D += np.kron(der_mat, gammas[mu])

    if m0 != 0.0:
        D += m0 * np.eye(V * ns, dtype=complex)
    return D


# --------------------------------------------------------------------------- #
# Closed-form eigenvalues: the recipe
# --------------------------------------------------------------------------- #
def recipe_eigenvalues(
    lap_name: str,
    der_name: str,
    d: int,
    phi: np.ndarray,
    m0: float = 0.0,
) -> np.ndarray:
    """Closed-form Dirac eigenvalues at a set of phase vectors.

        lambda_pm(phi) = A(phi) +/- i * |s(phi)|,
        A(phi) = -1/2 * Laplacian.hat(phi) + m0,
        i s_mu(phi) = derivative_mu.hat(phi).

    At the discrete lattice momenta this is exact (free field). At stochastically
    perturbed momenta it yields the mock spectrum.

    Parameters
    ----------
    lap_name, der_name : str
        Laplacian and derivative stencil keys.
    d : int
        Lattice dimensionality.
    phi : ndarray, shape ``(N, d)``
        Phase vectors at which to evaluate the recipe.
    m0 : float, optional
        Bare mass added to the real part (default 0).

    Returns
    -------
    ndarray, shape ``(2N,)``, complex
        The two branches ``A + i|s|`` and ``A - i|s|`` concatenated.
    """
    phi = np.asarray(phi, dtype=float)

    lap_s, lap_n = get_lap(lap_name, d)
    A = (-0.5 * eval_stencil_fourier(lap_s, lap_n, phi)).real + m0

    sum_nabla_sq = np.zeros(phi.shape[0], dtype=complex)
    for mu in range(d):
        der_s, der_n = get_der_mu(der_name, d, mu)
        nabla_mu = eval_stencil_fourier(der_s, der_n, phi)
        sum_nabla_sq += nabla_mu**2

    magnitude = np.sqrt(np.abs(sum_nabla_sq.real))
    return np.concatenate([A + 1j * magnitude, A - 1j * magnitude])
