"""Integration: the unit-field HDF5 round-trip ties the gauged operator to the
free-field formula across 2D/3D/4D, and the on-disk convention is involutive."""

import numpy as np  # type: ignore
import pytest  # type: ignore

from lattice_dirac_spectra.gauge.u1_config import save_config, load_config, _to_native
from lattice_dirac_spectra.gauge.synthetic import unit_links
from lattice_dirac_spectra.spectra.gauged import gauged_spectrum
from lattice_dirac_spectra.spectra.free_field import (
    eigenvalues_from_formula,
    compare_eigenvalues,
)

KERNELS = [("std", "std"), ("iso", "bri")]  # (derivative, laplacian)


@pytest.mark.parametrize("d,L", [(2, 8), (3, 4), (4, 3)])
def test_unit_file_reproduces_free_field(tmp_path, d, L):
    path = str(tmp_path / f"unit_d{d}.h5")
    save_config(path, unit_links(d, L), beta=0.0)
    cfg = load_config(path, 0.0, "traj00000000")

    assert cfg.dim == d
    assert cfg.L == (L,) * d
    assert np.allclose(cfg.links, 1.0)
    assert cfg.metadata["plaquette"] == pytest.approx(1.0)

    for der, lap in KERNELS:
        g = gauged_spectrum(lap, der, cfg)
        f = eigenvalues_from_formula(lap, der, d=d, L=L)
        assert compare_eigenvalues(f, g) < 1e-10


@pytest.mark.parametrize("shape", [(5, 4, 2), (3, 4, 5, 3), (2, 3, 2, 3, 4)])
def test_save_load_roundtrip_identity(tmp_path, shape):
    rng = np.random.default_rng(0)
    links = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    path = str(tmp_path / "cfg.h5")
    save_config(path, links, beta=1.0, metadata={"plaquette": 0.0, "topo": 0.0})
    cfg = load_config(path, 1.0, "traj00000000")
    assert np.array_equal(cfg.links, links)  # involutive bridge


def test_2d_bridge_backward_compatible():
    arr = np.arange(4 * 3 * 2).reshape(4, 3, 2)
    assert np.array_equal(_to_native(arr), arr.transpose(1, 0, 2))
