"""Tests for gauge observables (plaquette, topological charge)."""

import numpy as np
import pytest

from lattice_dirac_spectra.gauge.observables import (
    plaquette,
    topological_charge,
    plaquette_field,
)
from conftest import ref_plaquette_u1hmc, ref_topo_u1hmc


def _native_from_u1hmc(A):
    """Reader's convention bridge: (Ly, Lx, 2) -> native (Lx, Ly, 2)."""
    return np.ascontiguousarray(A.transpose(1, 0, 2))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_plaquette_matches_u1hmc(seed):
    rng = np.random.default_rng(seed)
    A = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(6, 5, 2)))  # (Ly, Lx, 2)
    native = _native_from_u1hmc(A)
    assert np.isclose(plaquette(native), ref_plaquette_u1hmc(A), atol=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_topo_matches_u1hmc_including_sign(seed):
    rng = np.random.default_rng(seed)
    A = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(6, 5, 2)))
    native = _native_from_u1hmc(A)
    assert np.isclose(topological_charge(native), ref_topo_u1hmc(A), atol=1e-10)


def test_topo_is_integer_for_smooth_field():
    # The geometric topological charge is an integer (here trivially 0 for a
    # nearly-uniform field).
    L = 8
    native = np.ones((L, L, 2), dtype=complex)
    Q = topological_charge(native)
    assert abs(Q - round(Q)) < 1e-9


def test_plaquette_is_one_for_trivial_field():
    native = np.ones((5, 5, 2), dtype=complex)
    assert np.isclose(plaquette(native), 1.0)


def test_plaquette_field_shape():
    native = np.ones((4, 6, 2), dtype=complex)
    assert plaquette_field(native).shape == (4, 6)


def test_topo_requires_2d():
    with pytest.raises(ValueError):
        topological_charge(np.ones((3, 3, 3, 3), dtype=complex))
