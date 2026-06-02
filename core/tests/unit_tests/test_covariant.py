"""
Tests for the gauge-covariant operator construction.

Golden invariant #2: with all links set to 1, the gauged operator reproduces the
free-field operator bit-for-bit. Plus exact gauge covariance and
gamma5-Hermiticity on a random U(1) background.
"""

import numpy as np
import pytest

from lattice_dirac_spectra.constants import LAP_KEYS, DER_KEYS
from lattice_dirac_spectra.constants.physics import Ns
from lattice_dirac_spectra.operators.dirac import build_free_dirac_matrix
from lattice_dirac_spectra.operators.gamma import gamma5
from lattice_dirac_spectra.spectra.free_field import compare_eigenvalues
from lattice_dirac_spectra.gauge.covariant import (
    build_gauged_dirac,
    gauge_transform,
    parallel_transport,
)


# ---- Golden invariant #2: links = 1 reproduces free field ----------------- #
@pytest.mark.parametrize("d, L", [(2, 6)])
@pytest.mark.parametrize("lap", LAP_KEYS)
@pytest.mark.parametrize("der", DER_KEYS)
def test_trivial_links_equal_free_field_2d(d, L, lap, der):
    ones = np.ones((L,) * d + (d,), dtype=complex)
    Dg = build_gauged_dirac(lap, der, ones, m0=0.3)
    Df = build_free_dirac_matrix(lap, der, d, L, m0=0.3)
    assert np.abs(Dg - Df).max() < 1e-12


@pytest.mark.parametrize("lap, der", [("std", "std"), ("bri", "iso")])
def test_trivial_links_equal_free_field_3d(lap, der):
    d, L = 3, 4
    ones = np.ones((L,) * d + (d,), dtype=complex)
    Dg = build_gauged_dirac(lap, der, ones, m0=0.1)
    Df = build_free_dirac_matrix(lap, der, d, L, m0=0.1)
    assert np.abs(Dg - Df).max() < 1e-12


# ---- Parallel transport primitives --------------------------------------- #
def test_parallel_transport_basics():
    L = (4, 4)
    rng = np.random.default_rng(0)
    links = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(4, 4, 2)))
    # zero displacement -> identity
    assert parallel_transport(links, (1, 2), (0, 0), L) == 1.0 + 0.0j
    # single forward hop -> the link itself
    assert np.isclose(parallel_transport(links, (1, 2), (1, 0), L), links[1, 2, 0])
    # single backward hop -> conj of the link at the destination
    assert np.isclose(
        parallel_transport(links, (1, 2), (-1, 0), L), links[0, 2, 0].conj()
    )


# ---- Gauge covariance & gamma5-Hermiticity ------------------------------- #
@pytest.mark.parametrize("lap, der", [("std", "std"), ("bri", "iso")])
def test_operator_is_gauge_covariant(lap, der):
    d, L = 2, 5
    rng = np.random.default_rng(2)
    U = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(L, L, d)))
    g = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(L, L)))

    D = build_gauged_dirac(lap, der, U, m0=0.1)
    Dg = build_gauged_dirac(lap, der, gauge_transform(U, g), m0=0.1)

    # G = diag(g(n)) ⊗ 1_Ns, with axis-0-fastest site linearization.
    ns = Ns(d)
    gvec = np.empty(L * L, dtype=complex)
    for c1 in range(L):
        for c0 in range(L):
            gvec[c0 + c1 * L] = g[c0, c1]
    G = np.kron(np.diag(gvec), np.eye(ns))
    assert np.abs(Dg - G @ D @ G.conj().T).max() < 1e-12


@pytest.mark.parametrize("lap, der", [("std", "std"), ("bri", "iso")])
def test_gauge_invariance_of_spectrum(lap, der):
    d, L = 2, 5
    rng = np.random.default_rng(3)
    U = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(L, L, d)))
    g = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(L, L)))
    e1 = np.linalg.eigvals(build_gauged_dirac(lap, der, U, m0=0.1))
    e2 = np.linalg.eigvals(build_gauged_dirac(lap, der, gauge_transform(U, g), m0=0.1))
    assert compare_eigenvalues(e1, e2) < 1e-9


@pytest.mark.parametrize("lap, der", [("std", "std"), ("bri", "iso")])
def test_gamma5_hermiticity(lap, der):
    d, L = 2, 5
    rng = np.random.default_rng(4)
    U = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(L, L, d)))
    D = build_gauged_dirac(lap, der, U, m0=0.1)
    g5 = gamma5(d)
    G5 = np.kron(np.eye(L * L), g5)
    assert np.abs(G5 @ D @ G5 - D.conj().T).max() < 1e-12
