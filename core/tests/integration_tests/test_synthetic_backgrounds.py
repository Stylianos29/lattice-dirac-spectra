"""Synthetic-background generators: plaquette laws + spectrum invariants."""

import numpy as np  # type: ignore
import pytest  # type: ignore

from lattice_dirac_spectra.gauge.synthetic import (
    unit_links,
    random_phase_links,
    eps_for_plaquette,
    plaquette_2d,
    heatbath_links,
)
from lattice_dirac_spectra.gauge.observables import plaquette
from lattice_dirac_spectra.gauge.covariant import gauge_transform
from lattice_dirac_spectra.spectra.gauged import gauged_spectrum
from lattice_dirac_spectra.spectra.free_field import (
    eigenvalues_from_formula,
    compare_eigenvalues,
)

KERNELS = [("std", "std"), ("iso", "bri")]


def test_links_are_unit_modulus():
    rng = np.random.default_rng(0)
    for U in (random_phase_links(3, 4, 0.5, rng), heatbath_links(3, 4, 2.0, 30, rng)):
        assert np.allclose(np.abs(U), 1.0)


@pytest.mark.parametrize("d", [2, 4])
def test_random_phase_plaquette_law(d):
    rng = np.random.default_rng(1)
    eps = 0.4
    P = np.mean([plaquette(random_phase_links(d, 12, eps, rng)) for _ in range(20)])
    assert P == pytest.approx(np.exp(-2 * eps**2), abs=0.02)


def test_eps_plaquette_roundtrip():
    for P0 in (0.5, 0.7, 0.9):
        eps = eps_for_plaquette(P0)
        assert np.exp(-2 * eps**2) == pytest.approx(P0)


def test_random_phase_zero_eps_is_free_field():
    U = random_phase_links(2, 8, 0.0, np.random.default_rng(0))
    assert np.allclose(U, 1.0)
    g = gauged_spectrum("bri", "iso", U)
    f = eigenvalues_from_formula("bri", "iso", d=2, L=8)
    assert compare_eigenvalues(f, g) < 1e-10


@pytest.mark.slow
def test_heatbath_2d_matches_bessel():
    rng = np.random.default_rng(2)
    beta = 2.0
    Ps = [plaquette(heatbath_links(2, 16, beta, 120, rng)) for _ in range(8)]
    assert np.mean(Ps) == pytest.approx(plaquette_2d(beta), abs=0.03)


@pytest.mark.parametrize("der,lap", KERNELS)
def test_spectrum_is_gauge_invariant(der, lap):
    rng = np.random.default_rng(3)
    U = heatbath_links(2, 8, 2.0, 60, rng)
    g = np.exp(1j * rng.uniform(0, 2 * np.pi, size=(8, 8)))
    e1 = gauged_spectrum(lap, der, U)
    e2 = gauged_spectrum(lap, der, gauge_transform(U, g))
    assert compare_eigenvalues(e1, e2) < 1e-10  # gauge invariance
    assert compare_eigenvalues(e1, e1.conj()) < 1e-10  # gamma5 conjugate pairs
