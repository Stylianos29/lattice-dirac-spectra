"""
Tests for the visualization subsystem.

Numerical correctness of the data-producing functions (surface grids, dispersion
roots) plus smoke tests that the plotting functions execute without error on a
headless backend.
"""

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lattice_dirac_spectra.visualization.brillouin_zone import (  # noqa: E402
    laplacian_surface_data,
    derivative_surface_data,
    plot_laplacian_surface,
    plot_derivative_surface,
)
from lattice_dirac_spectra.visualization.dispersion import (  # noqa: E402
    dispersion_relation,
    onshell_value,
    default_directions,
    plot_dispersion,
)
from lattice_dirac_spectra.visualization.spectrum_plotter import (  # noqa: E402
    plot_formula_vs_direct,
    plot_overlap_spectrum,
)
from lattice_dirac_spectra.spectra.free_field import (
    eigenvalues_from_formula,
)  # noqa: E402
from lattice_dirac_spectra.operators.overlap import overlap_eigenvalues  # noqa: E402


# ---- Brillouin-zone surfaces --------------------------------------------- #
def test_surface_shapes():
    K0, K1, Z = laplacian_surface_data("std", d=2, grid_size=16)
    assert K0.shape == K1.shape == Z.shape == (16, 16)
    assert np.isrealobj(Z)


def test_brillouin_laplacian_constant_on_boundary():
    # a^2 Delta_bri.hat is constant (= -4) on the entire BZ boundary.
    _, _, Z = laplacian_surface_data("bri", d=2, grid_size=24)
    assert np.allclose(Z[:, -1], -4.0)  # k0 = pi column
    assert np.allclose(Z[-1, :], -4.0)  # k1 = pi row
    # The standard Laplacian is NOT constant on the boundary.
    _, _, Zs = laplacian_surface_data("std", d=2, grid_size=24)
    assert not np.allclose(Zs[:, -1], Zs[0, -1])


def test_standard_derivative_has_no_transverse_structure():
    # Im nabla_std_0.hat = sin(k0), independent of k1.
    _, _, Z = derivative_surface_data("std", mu=0, d=2, grid_size=20)
    for j in range(Z.shape[1]):
        assert np.allclose(Z[:, j], Z[0, j])  # constant down each column (fixed k0)


# ---- Dispersion ---------------------------------------------------------- #
def test_onshell_residual_at_solution():
    P = np.array([0.1, 0.3, 0.6])
    for lap, der in [("std", "std"), ("bri", "iso")]:
        E = dispersion_relation(lap, der, 2, (1,), P, m=0.0)
        assert np.all(np.isfinite(E))
        for p, e in zip(P, E):
            assert abs(onshell_value(lap, der, 2, np.array([p]), e, 0.0)) < 1e-8


def test_massless_lightcone_slope():
    # aE ~ aP at small momentum (slope 1) for both kernels.
    P = np.array([0.05, 0.1])
    for lap, der in [("std", "std"), ("bri", "iso")]:
        E = dispersion_relation(lap, der, 2, (1,), P, m=0.0)
        assert np.allclose(E, P, atol=2e-3)


def test_wilson_rest_energy_is_log1p_mass():
    # Wilson dispersion at p=0 gives aE = log(1 + am).
    for m in [0.1, 0.3, 0.75]:
        E = dispersion_relation("std", "std", 2, (1,), np.array([0.0]), m=m)[0]
        assert np.isclose(E, np.log(1 + m), atol=1e-6)


def test_default_directions():
    assert default_directions(2) == [(1,)]
    assert default_directions(4) == [(1, 0, 0), (1, 1, 0), (1, 1, 1)]


def test_heavy_quark_substitution_changes_rest_energy():
    # With am -> exp(am)-1, the p=0 Wilson energy becomes log(exp(am)) = am.
    m = 0.5
    E = dispersion_relation(
        "std", "std", 2, (1,), np.array([0.0]), m=m, heavy_quark=True
    )[0]
    assert np.isclose(E, m, atol=1e-6)


# ---- Plotting smoke tests ------------------------------------------------ #
def test_plot_functions_run():
    fig, _ = plot_formula_vs_direct("bri", "iso", 2, 6)
    plt.close(fig)

    z = eigenvalues_from_formula("std", "std", 2, 6)
    z_ov = overlap_eigenvalues(z, rho=1.0, mass=0.5)
    fig, _ = plot_overlap_spectrum(z_ov, rho=1.0, mass=0.5)
    plt.close(fig)

    fig, _ = plot_laplacian_surface("bri", d=2, grid_size=12, kind="surface")
    plt.close(fig)
    fig, _ = plot_derivative_surface("iso", mu=0, d=2, grid_size=12, kind="contour")
    plt.close(fig)

    fig, _ = plot_dispersion("std", "std", d=2, P_max=2.0, n_points=15)
    plt.close(fig)
