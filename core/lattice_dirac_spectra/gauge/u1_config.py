"""
U(1) gauge-configuration reader (``u1-hmc`` HDF5 format).

Reads the 2D U(1) pure-gauge configurations written by
`g-koutsou/u1-hmc <https://github.com/g-koutsou/u1-hmc>`_. See
``docs/theory/u1_hmc_data_format.md`` for the on-disk contract:

    /beta{:5.3f}/traj{:08.0f}/u    dataset, shape (Ly, Lx, 2), complex128
    + group attrs: plaquette, topo, dH, T, tau, NT

Convention bridge
-----------------
``u1-hmc`` stores the array as ``(Ly, Lx, 2)`` with direction index
``mu = 0 -> x`` (array axis 1) and ``mu = 1 -> y`` (array axis 0). The rest of
this package uses the uniform convention ``links[x, mu]`` = forward link along
**lattice axis** ``mu``. The reader bridges the two with a single transpose,

    links = dataset.transpose(1, 0, 2)        # (Ly, Lx, 2) -> (Lx, Ly, 2)

so that, in the returned array, axis 0 is x with ``mu = 0`` its link and axis 1
is y with ``mu = 1`` its link. With this bridge the package's
:func:`lattice_dirac_spectra.gauge.observables.plaquette` and
:func:`~lattice_dirac_spectra.gauge.observables.topological_charge` reproduce the
``plaquette`` and ``topo`` metadata stored by ``u1-hmc`` (including the sign of
the topological charge).

Reading is serial; no MPI is required.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import h5py # type: ignore
import numpy as np # type: ignore

from .observables import plaquette, topological_charge

__all__ = [
    "U1Config",
    "list_betas",
    "list_trajectories",
    "load_config",
    "load_ensemble",
    "save_config",
]


@dataclass
class U1Config:
    """A single U(1) gauge configuration in the package's native convention.

    Attributes
    ----------
    links : ndarray, shape ``(Lx, Ly, 2)``, complex128
        Forward links ``links[x, mu] = U_mu(x)`` (unit modulus).
    beta : float
        Gauge coupling.
    traj : str
        Trajectory group name (e.g. ``"traj00000001"``).
    metadata : dict
        Group attributes from the file (``plaquette``, ``topo``, ``dH``, ...).
    """

    links: np.ndarray
    beta: float
    traj: str
    metadata: Dict[str, float] = field(default_factory=dict)

    @property
    def L(self) -> Tuple[int, ...]:
        """Lattice extents ``(L0, L1, ...)`` (link array shape without last axis)."""
        return tuple(self.links.shape[:-1])

    @property
    def dim(self) -> int:
        """Lattice dimensionality (2D for u1-hmc files; 3D/4D for
        synthetic unit fields)."""
        return self.links.shape[-1]


def _beta_group_name(beta: float) -> str:
    return "beta{:5.3f}".format(beta)


def list_betas(path: str) -> List[float]:
    """Return the beta values present in the file (parsed from group names)."""
    betas = []
    with h5py.File(path, "r") as f:
        for name in f.keys():
            if name.startswith("beta"):
                try:
                    betas.append(float(name[len("beta") :]))
                except ValueError:
                    continue
    return sorted(betas)


def list_trajectories(path: str, beta: float) -> List[str]:
    """Return the sorted trajectory group names for a given beta."""
    grp = _beta_group_name(beta)
    with h5py.File(path, "r") as f:
        if grp not in f:
            raise KeyError(f"beta group {grp!r} not found in {path!r}.")
        return sorted(f[grp].keys())


def _axis_perm(d: int) -> Tuple[int, ...]:
    """Reverse the ``d`` spatial axes, keep the ``mu`` axis last. Involutive."""
    return tuple(range(d - 1, -1, -1)) + (d,)


def _to_native(dataset: np.ndarray) -> np.ndarray:
    """Bridge the on-disk layout to native ``links[x0, ..., x_{d-1}, mu]`` order.

    The on-disk convention reverses the spatial axes relative to native; for
    ``d = 2`` this is the u1-hmc ``(Ly, Lx, 2)`` layout and the permutation
    reduces to the historical ``transpose(1, 0, 2)``.
    """
    arr = np.asarray(dataset)
    return np.ascontiguousarray(arr.transpose(_axis_perm(arr.shape[-1])))


def load_config(path: str, beta: float, traj: str) -> U1Config:
    """Load a single configuration ``/beta{beta}/{traj}``."""
    grp = _beta_group_name(beta)
    with h5py.File(path, "r") as f:
        node = f[f"{grp}/{traj}"]
        links = _to_native(node["u"][...])
        meta = {k: float(v) for k, v in node.attrs.items()}
    return U1Config(links=links, beta=beta, traj=traj, metadata=meta)


def load_ensemble(
    path: str,
    beta: float,
    pick: Optional[slice] = None,
) -> Iterator[U1Config]:
    """Iterate over configurations for a given beta.

    Parameters
    ----------
    path : str
        HDF5 file path.
    beta : float
        Gauge coupling.
    pick : slice, optional
        Slice of the sorted trajectory list to load (default: all).

    Yields
    ------
    U1Config
    """
    trajs = list_trajectories(path, beta)
    if pick is not None:
        trajs = trajs[pick]
    for traj in trajs:
        yield load_config(path, beta, traj)


def save_config(
    path: str,
    links: np.ndarray,
    beta: float,
    traj_index: int = 0,
    metadata: Optional[Dict[str, float]] = None,
    mode: str = "w",
) -> None:
    """Write one configuration to the u1-hmc-format HDF5 layout (inverse of load_config).

    ``links`` is in native ``links[x0, ..., x_{d-1}, mu]`` order and is stored on
    disk in the reversed-spatial-axis convention the reader expects, so
    ``load_config`` round-trips it to the identity in any dimension (and 2D files
    stay u1-hmc compatible). Plaquette (any d) and topological charge (d=2 only)
    are computed from the links and saved as group attributes unless overridden.

    Use ``mode="a"`` to append more trajectories/betas to an existing file.
    """
    links = np.asarray(links)
    d = links.shape[-1]
    meta = {
        "plaquette": plaquette(links),
        "topo": topological_charge(links) if d == 2 else 0.0,
        "dH": 0.0,
        "T": 0.0,
        "tau": 0.0,
        "NT": 0.0,
    }
    if metadata:
        meta.update(metadata)
    disk = np.ascontiguousarray(links.transpose(_axis_perm(d)))
    grp = _beta_group_name(beta)
    traj = "traj{:08.0f}".format(traj_index)
    with h5py.File(path, mode) as f:
        node = f.require_group(grp).create_group(traj)
        node.create_dataset("u", data=disk)
        for k, v in meta.items():
            node.attrs[k] = float(v)
