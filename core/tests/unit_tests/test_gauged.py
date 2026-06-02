"""
Tests for the U(1) configuration reader and gauged-spectrum orchestration.

Golden invariant #4: plaquette/topo recomputed from a loaded configuration match
the metadata stored in the file (through the reader's convention bridge).
"""

from pathlib import Path

import numpy as np
import pytest

from lattice_dirac_spectra.gauge.u1_config import (
    list_betas,
    list_trajectories,
    load_config,
    load_ensemble,
)
from lattice_dirac_spectra.gauge.observables import plaquette, topological_charge
from lattice_dirac_spectra.gauge.smearing import ape_smear
from lattice_dirac_spectra.spectra.gauged import gauged_spectrum, ensemble_spectrum
from lattice_dirac_spectra.spectra.free_field import compare_eigenvalues


# ---- Reader ---------------------------------------------------------------- #
def test_list_betas_and_trajectories(synthetic_u1_file):
    path, beta, info = synthetic_u1_file
    assert list_betas(path) == [beta]
    trajs = list_trajectories(path, beta)
    assert trajs == sorted(info.keys())


def test_load_config_shape_dtype_modulus(synthetic_u1_file):
    path, beta, info = synthetic_u1_file
    cfg = load_config(path, beta, "traj00000000")
    assert cfg.dim == 2
    assert cfg.links.shape == (4, 4, 2)  # native (Lx, Ly, 2)
    assert cfg.links.dtype == np.complex128
    assert np.allclose(np.abs(cfg.links), 1.0)  # unit modulus
    assert set(["plaquette", "topo", "NT", "T", "tau", "dH"]) <= set(cfg.metadata)


def test_observables_match_stored_metadata(synthetic_u1_file):
    # Golden invariant #4 (on synthetic data with independent reference attrs).
    path, beta, info = synthetic_u1_file
    for cfg in load_ensemble(path, beta):
        assert np.isclose(plaquette(cfg.links), cfg.metadata["plaquette"], atol=1e-10)
        assert np.isclose(
            topological_charge(cfg.links), cfg.metadata["topo"], atol=1e-10
        )


def test_load_ensemble_pick(synthetic_u1_file):
    path, beta, info = synthetic_u1_file
    cfgs = list(load_ensemble(path, beta, pick=slice(0, 1)))
    assert len(cfgs) == 1


# ---- Optional: real u1-hmc file if present locally ------------------------ #
def test_real_u1hmc_file_if_present():
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    real = repo_root / "core" / "tests" / "mock_data" / "u1_L8_beta1.000.h5"
    if not real.exists():
        pytest.skip("real u1-hmc config not present (regenerate with u1-hmc)")
    cfg = load_config(str(real), 1.0, list_trajectories(str(real), 1.0)[1])
    assert cfg.links.shape == (8, 8, 2)
    assert np.allclose(np.abs(cfg.links), 1.0)
    # The true convention test: our observables vs u1-hmc's stored attrs.
    assert np.isclose(plaquette(cfg.links), cfg.metadata["plaquette"], atol=1e-8)
    assert np.isclose(topological_charge(cfg.links), cfg.metadata["topo"], atol=1e-6)


# ---- Gauged spectra -------------------------------------------------------- #
def test_gauged_spectrum_runs_and_is_conjugate_symmetric(synthetic_u1_file):
    path, beta, info = synthetic_u1_file
    cfg = load_config(path, beta, "traj00000000")
    eigs = gauged_spectrum("std", "std", cfg, m0=0.0)
    assert eigs.shape == (4 * 4 * 2,)  # V * Ns
    # gamma5-Hermiticity => spectrum closed under complex conjugation.
    assert compare_eigenvalues(eigs, eigs.conj()) < 1e-9


def test_ensemble_spectrum_concatenates(synthetic_u1_file):
    path, beta, info = synthetic_u1_file
    eigs = ensemble_spectrum("std", "std", load_ensemble(path, beta))
    assert eigs.shape == (2 * 4 * 4 * 2,)  # 2 configs


def test_ape_smearing_identity_and_modulus(synthetic_u1_file):
    path, beta, info = synthetic_u1_file
    cfg = load_config(path, beta, "traj00000000")
    # alpha = 0 is the identity
    assert np.allclose(ape_smear(cfg.links, alpha=0.0, n_steps=3), cfg.links)
    # smearing preserves unit modulus
    sm = ape_smear(cfg.links, alpha=0.5, n_steps=2)
    assert np.allclose(np.abs(sm), 1.0)


def test_smearing_changes_spectrum(synthetic_u1_file):
    path, beta, info = synthetic_u1_file
    cfg = load_config(path, beta, "traj00000000")
    e0 = gauged_spectrum("std", "std", cfg, n_ape=0)
    e1 = gauged_spectrum("std", "std", cfg, n_ape=2, alpha=0.5)
    assert compare_eigenvalues(e0, e1) > 1e-6  # genuinely different
