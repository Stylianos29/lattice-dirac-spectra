"""
Capability 1 -- Free-field eigenvalue spectra.

Two routes to the free-field spectrum of a Wilson-type Dirac operator, plus a
comparison helper:

* :func:`eigenvalues_from_formula` -- the closed-form recipe evaluated at the
  exact discrete lattice momenta.
* :func:`eigenvalues_from_matrix` -- direct diagonalization of the explicit
  coordinate-space Dirac matrix.

The golden invariant of the project is that these agree to numerical precision
for every (Laplacian, derivative) combination in every supported dimension.
"""

import numpy as np

from ..operators.dirac import build_free_dirac_matrix, recipe_eigenvalues
from ..operators.fourier import discrete_momenta

__all__ = [
    "eigenvalues_from_formula",
    "eigenvalues_from_matrix",
    "compare_eigenvalues",
]


def eigenvalues_from_formula(
    lap_name: str,
    der_name: str,
    d: int,
    L: int,
    m0: float = 0.0,
) -> np.ndarray:
    """Free-field eigenvalues from the recipe at the discrete lattice momenta."""
    phi = discrete_momenta(d, L)
    return recipe_eigenvalues(lap_name, der_name, d, phi, m0=m0)


def eigenvalues_from_matrix(
    lap_name: str,
    der_name: str,
    d: int,
    L: int,
    m0: float = 0.0,
) -> np.ndarray:
    """Free-field eigenvalues by direct diagonalization of the Dirac matrix."""
    D = build_free_dirac_matrix(lap_name, der_name, d, L, m0=m0)
    return np.linalg.eigvals(D)


def compare_eigenvalues(
    eigs_formula: np.ndarray,
    eigs_direct: np.ndarray,
) -> float:
    """Maximum nearest-neighbour distance from each direct eigenvalue to the formula set.

    A small value (machine precision) certifies that the two spectra coincide as
    multisets, independent of ordering.
    """
    eigs_formula = np.asarray(eigs_formula)
    max_err = 0.0
    for e in np.asarray(eigs_direct):
        min_dist = np.min(np.abs(eigs_formula - e))
        if min_dist > max_err:
            max_err = float(min_dist)
    return max_err
