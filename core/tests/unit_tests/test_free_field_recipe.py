"""
Golden invariant #1
===================

The closed-form recipe and direct diagonalization of the free-field Dirac matrix
must produce identical spectra (as multisets), for every (Laplacian, derivative)
combination, in every supported dimension.

This is the backbone correctness check of the whole project.
"""

import numpy as np
import pytest

from lattice_dirac_spectra.constants import LAP_KEYS, DER_KEYS
from lattice_dirac_spectra.spectra.free_field import (
    eigenvalues_from_formula,
    eigenvalues_from_matrix,
    compare_eigenvalues,
)

TOL = 1e-10


@pytest.mark.parametrize("d, L", [(2, 4), (2, 6), (2, 8)])
@pytest.mark.parametrize("lap", LAP_KEYS)
@pytest.mark.parametrize("der", DER_KEYS)
def test_recipe_matches_diagonalization_2d(d, L, lap, der):
    ef = eigenvalues_from_formula(lap, der, d, L)
    ed = eigenvalues_from_matrix(lap, der, d, L)
    assert compare_eigenvalues(ef, ed) < TOL


@pytest.mark.parametrize("lap", ["std", "bri"])
@pytest.mark.parametrize("der", ["std", "iso"])
def test_recipe_matches_diagonalization_4d(lap, der):
    d, L = 4, 4  # matrix size 1024 x 1024
    ef = eigenvalues_from_formula(lap, der, d, L)
    ed = eigenvalues_from_matrix(lap, der, d, L)
    assert compare_eigenvalues(ef, ed) < TOL


def test_eigenvalue_count():
    # Direct matrix has V * Ns eigenvalues; the formula returns 2 * V (each with
    # spinor degeneracy Ns/2), so the totals match.
    d, L = 2, 6
    ef = eigenvalues_from_formula("bri", "iso", d, L)
    ed = eigenvalues_from_matrix("bri", "iso", d, L)
    assert len(ef) == 2 * L**d
    assert len(ed) == L**d * 2  # Ns = 2 in 2D


def test_mass_shift_translates_spectrum():
    # Adding a bare mass m0 shifts the whole spectrum by +m0 along the real axis.
    d, L, m0 = 2, 6, 0.3
    e0 = eigenvalues_from_formula("std", "std", d, L)
    em = eigenvalues_from_formula("std", "std", d, L, m0=m0)
    assert compare_eigenvalues(e0 + m0, em) < TOL
