"""Unit tests for stencil definitions."""

import numpy as np
import pytest

from lattice_dirac_spectra.constants import LAP_KEYS, DER_KEYS
from lattice_dirac_spectra.operators.stencils import (
    get_lap,
    get_der,
    get_der_mu,
)


@pytest.mark.parametrize("d", [2, 3, 4])
@pytest.mark.parametrize("name", LAP_KEYS)
def test_laplacian_shape_and_symmetry(d, name):
    if name == "iso" and d not in (2, 3, 4):
        pytest.skip("iso only defined for d in {2,3,4}")
    stencil, norm = get_lap(name, d)
    assert stencil.shape == (3,) * d
    assert norm > 0
    # Laplacian stencils are symmetric under full reflection (idx -> 2 - idx).
    assert np.array_equal(stencil, stencil[(slice(None, None, -1),) * d])


@pytest.mark.parametrize("d", [2, 3, 4])
@pytest.mark.parametrize("name", LAP_KEYS)
def test_laplacian_zero_mode(d, name):
    # The Laplacian should annihilate a constant field: sum of all weights = 0.
    stencil, _ = get_lap(name, d)
    assert int(stencil.sum()) == 0


@pytest.mark.parametrize("d", [2, 3, 4])
@pytest.mark.parametrize("name", DER_KEYS)
def test_derivative_antisymmetry(d, name):
    # Derivative (mu=0) is antisymmetric along axis 0.
    stencil, norm = get_der(name, d)
    assert stencil.shape == (3,) * d
    assert norm > 0
    assert np.array_equal(stencil, -stencil[::-1])
    assert int(stencil.sum()) == 0


@pytest.mark.parametrize("d", [2, 3, 4])
@pytest.mark.parametrize("name", DER_KEYS)
def test_derivative_axis_swap(d, name):
    # get_der_mu for mu != 0 is the axis-swap of the mu=0 stencil.
    base, norm = get_der(name, d)
    for mu in range(d):
        s_mu, n_mu = get_der_mu(name, d, mu)
        assert n_mu == norm
        expected = base if mu == 0 else np.swapaxes(base, 0, mu)
        assert np.array_equal(s_mu, expected)


def test_unknown_keys_raise():
    with pytest.raises(KeyError):
        get_lap("nope", 2)
    with pytest.raises(KeyError):
        get_der("nope", 2)
