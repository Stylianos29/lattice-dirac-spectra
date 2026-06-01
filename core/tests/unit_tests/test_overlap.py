"""
Tests for the overlap projection and the Kenney-Laub sign approximant.

Golden invariant #5: in the free field, the overlap spectrum lies exactly on the
Ginsparg-Wilson circle.
"""

import numpy as np
import pytest

from lattice_dirac_spectra.spectra.free_field import eigenvalues_from_formula
from lattice_dirac_spectra.operators.overlap import (
    overlap_eigenvalues,
    gw_circle,
    kl_sign,
)

TOL = 1e-12


@pytest.mark.parametrize("lap, der", [("std", "std"), ("bri", "iso")])
@pytest.mark.parametrize("rho", [0.6, 1.0, 1.4])
@pytest.mark.parametrize("mass", [0.0, 0.5])
def test_overlap_lands_on_gw_circle(lap, der, rho, mass):
    # Free-field kernel spectrum -> overlap projection -> must lie on GW circle.
    z = eigenvalues_from_formula(lap, der, d=2, L=8)
    # Drop the would-be zero modes at z == rho exactly (degenerate direction);
    # the recipe never produces exactly rho here, but guard anyway.
    z = z[np.abs(z - rho) > 1e-9]
    z_ov = overlap_eigenvalues(z, rho=rho, mass=mass)
    center, radius, _ = gw_circle(rho, mass)
    dist = np.abs((z_ov - center)) - radius
    assert np.max(np.abs(dist)) < TOL


def test_overlap_parameter_validation():
    z = np.array([0.5 + 0.2j])
    with pytest.raises(ValueError):
        overlap_eigenvalues(z, rho=2.5)
    with pytest.raises(ValueError):
        overlap_eigenvalues(z, rho=1.0, mass=2.0)  # mass must be < 2*rho


def test_gw_circle_geometry():
    center, radius, pts = gw_circle(rho=1.0, mass=0.0)
    assert np.isclose(center, 1.0)
    assert np.isclose(radius, 1.0)
    assert np.allclose(np.abs(pts - center), radius)


def test_kl_sign_converges():
    # Diagonal Kenney-Laub approximant -> sign(x) as order grows, for |x| ~ 1.
    x = np.array([0.5, 1.0, 2.0, 5.0])
    approx = kl_sign(x, n=8).real
    assert np.allclose(approx, np.sign(x), atol=1e-2)
    # Exact fixed points at x = 0, 1.
    assert np.isclose(kl_sign(np.array([0.0]), n=3)[0], 0.0)
    assert np.isclose(kl_sign(np.array([1.0]), n=3)[0], 1.0)


def test_kl_sign_is_odd():
    x = np.array([0.3, 0.7, 1.5, 3.0])
    assert np.allclose(kl_sign(x, n=4), -kl_sign(-x, n=4))
