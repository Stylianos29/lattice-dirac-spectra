"""
Shared pytest fixtures and reference implementations for the gauge tests.

Provides a synthetic HDF5 file in the exact ``u1-hmc`` on-disk layout (so the
reader and observables are exercised end-to-end without requiring MPI or a real
HMC run), with plaquette/topo metadata computed by an *independent* literal
transcription of ``u1-hmc``'s formulas. The package's own observables must then
reproduce those values through the reader's convention bridge.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest


# --- Independent transcription of u1-hmc's observables (on (Ly, Lx, 2) layout) --- #
def ref_plaquette_u1hmc(A: np.ndarray) -> float:
    """Mean plaquette exactly as computed by u1-hmc (array layout (Ly, Lx, 2))."""
    u0 = A[..., 0]
    u1 = A[..., 1]
    P = u0 * np.roll(u1, -1, axis=1) * np.roll(u0, -1, axis=0).conj() * u1.conj()
    return float(P.real.mean())


def ref_topo_u1hmc(A: np.ndarray) -> float:
    """Topological charge exactly as computed by u1-hmc (array layout (Ly, Lx, 2))."""
    u0 = A[..., 0]
    u1 = A[..., 1]
    P = u0 * np.roll(u1, -1, axis=1) * np.roll(u0, -1, axis=0).conj() * u1.conj()
    return float(np.arctan2(P.imag, P.real).sum() / (2 * np.pi))


def _make_u1_array(Ly, Lx, seed):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, size=(Ly, Lx, 2))
    return np.exp(1j * theta)


@pytest.fixture(scope="session")
def synthetic_u1_file(tmp_path_factory):
    """Write a tiny synthetic ensemble in u1-hmc layout; return (path, beta, info).

    Two trajectories on a 4x4 lattice, beta = 1.0, with plaquette/topo attrs
    computed by the independent reference transcription above.
    """
    path = tmp_path_factory.mktemp("gauge") / "synthetic_u1_L4_beta1.000.h5"
    beta = 1.0
    Ly = Lx = 4
    info = {}
    with h5py.File(path, "w") as f:
        for i in range(2):
            A = _make_u1_array(Ly, Lx, seed=100 + i)
            traj = "traj{:08.0f}".format(i)
            grp = f.require_group(f"beta{beta:5.3f}/{traj}")
            grp.create_dataset("u", data=A)
            grp.attrs["plaquette"] = ref_plaquette_u1hmc(A)
            grp.attrs["topo"] = ref_topo_u1hmc(A)
            grp.attrs["NT"] = 8.0
            grp.attrs["T"] = 1.0
            grp.attrs["tau"] = 0.125
            grp.attrs["dH"] = 0.0
            info[traj] = {
                "A": A,
                "plaquette": ref_plaquette_u1hmc(A),
                "topo": ref_topo_u1hmc(A),
            }
    return str(path), beta, info
