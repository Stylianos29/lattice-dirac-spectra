"""Unit tests for Fourier evaluation, momentum grids, and gamma matrices."""

import numpy as np
import pytest

from lattice_dirac_spectra.constants import LAP_KEYS, DER_KEYS, Ns
from lattice_dirac_spectra.operators.fourier import (
    eval_stencil_fourier,
    discrete_momenta,
)
from lattice_dirac_spectra.operators.stencils import get_lap, get_der_mu
from lattice_dirac_spectra.operators.gamma import gamma_matrices


# ---- Fourier ------------------------------------------------------------- #
@pytest.mark.parametrize("d", [2, 3, 4])
def test_discrete_momenta_count(d):
    L = 6
    k = discrete_momenta(d, L)
    assert k.shape == (L**d, d)


@pytest.mark.parametrize("d", [2, 3, 4])
@pytest.mark.parametrize("name", LAP_KEYS)
def test_laplacian_fourier_is_real(d, name):
    k = discrete_momenta(d, 6)
    s, n = get_lap(name, d)
    vals = eval_stencil_fourier(s, n, k)
    assert np.allclose(vals.imag, 0.0, atol=1e-12)


@pytest.mark.parametrize("d", [2, 3, 4])
@pytest.mark.parametrize("name", DER_KEYS)
def test_derivative_fourier_is_imaginary(d, name):
    k = discrete_momenta(d, 6)
    for mu in range(d):
        s, n = get_der_mu(name, d, mu)
        vals = eval_stencil_fourier(s, n, k)
        assert np.allclose(vals.real, 0.0, atol=1e-12)


def test_laplacian_small_momentum_limit():
    # All Laplacians reduce to -p^2 at small momentum (continuum limit).
    d = 2
    p = np.array([[0.01, 0.0], [0.0, 0.02], [0.01, 0.02]])
    for name in LAP_KEYS:
        s, n = get_lap(name, d)
        lap_hat = eval_stencil_fourier(s, n, p).real
        p2 = (p**2).sum(axis=1)
        assert np.allclose(lap_hat, -p2, atol=1e-3)


# ---- Gamma --------------------------------------------------------------- #
@pytest.mark.parametrize("d", [2, 3, 4])
def test_clifford_algebra(d):
    gammas = gamma_matrices(d)
    ns = Ns(d)
    assert len(gammas) == d
    for mu in range(d):
        assert gammas[mu].shape == (ns, ns)
        # Hermiticity
        assert np.allclose(gammas[mu], gammas[mu].conj().T)
    # {gamma_mu, gamma_nu} = 2 delta_{mu nu}
    for mu in range(d):
        for nu in range(d):
            anticomm = gammas[mu] @ gammas[nu] + gammas[nu] @ gammas[mu]
            expected = 2 * (mu == nu) * np.eye(ns)
            assert np.allclose(anticomm, expected, atol=1e-12)


def test_gamma_unsupported_dimension():
    with pytest.raises(ValueError):
        gamma_matrices(5)
