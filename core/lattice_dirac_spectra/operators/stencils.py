"""
Stencil definitions for Wilson-type lattice Dirac operators.

A *stencil* is a ``(3,) * d`` integer array together with an integer
normalization. Along each axis, the index ``1`` means "no hop", ``0`` means a
hop of ``-1``, and ``2`` means a hop of ``+1``; the physical displacement of a
cell at multi-index ``idx`` is therefore ``disp = idx - 1``.

This module provides the four Laplacian discretizations (standard, tilted,
Brillouin, isotropic) and the three derivative discretizations (standard,
Brillouin, isotropic) of

    S. Duerr and G. Koutsou, Phys. Rev. D 83, 114512 (2011),

generalized to arbitrary dimension ``d``. The combination
``(der_iso, lap_bri)`` is the "Brillouin operator"; ``(der_std, lap_std)`` is
the standard Wilson operator.
"""

from math import comb
from itertools import product as iproduct
from typing import Callable, Dict, Tuple

import numpy as np

__all__ = [
    "lap_std",
    "lap_til",
    "lap_bri",
    "lap_iso",
    "der_std",
    "der_bri",
    "der_iso",
    "get_lap",
    "get_der",
    "get_der_mu",
    "LAPLACIANS",
    "DERIVATIVES",
]

Stencil = Tuple[np.ndarray, int]


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _build_lap(d: int, weights_by_hop: Dict[int, int]) -> np.ndarray:
    """Build a symmetric Laplacian stencil from per-hop-count weights.

    ``weights_by_hop[h]`` is the weight assigned to every cell whose number of
    nonzero displacement components (the "hop count") equals ``h``.
    """
    s = np.zeros((3,) * d, dtype=int)
    for idx in iproduct(range(3), repeat=d):
        hop = sum(1 for i in idx if i != 1)
        s[idx] = weights_by_hop.get(hop, 0)
    return s


def _build_der(d: int, weights_by_thop: Dict[int, int]) -> np.ndarray:
    """Build an antisymmetric (axis-0) derivative stencil.

    The stencil is antisymmetric along axis 0; ``weights_by_thop[t]`` is the
    weight for cells whose *transverse* hop count (nonzero displacements among
    axes ``1..d-1``) equals ``t``. The sign is set by the axis-0 displacement.
    """
    s = np.zeros((3,) * d, dtype=int)
    for idx in iproduct(range(3), repeat=d):
        n0 = idx[0] - 1
        if n0 == 0:
            continue
        thop = sum(1 for i in idx[1:] if i != 1)
        w = weights_by_thop.get(thop, 0)
        s[idx] = (1 if n0 > 0 else -1) * w
    return s


# --------------------------------------------------------------------------- #
# Laplacians:  return (stencil, normalization)
# --------------------------------------------------------------------------- #
def lap_std(d: int) -> Stencil:
    """Standard Laplacian (9-point stencil in 4D): nearest-neighbour only."""
    return _build_lap(d, {0: -2 * d, 1: 1}), 1


def lap_til(d: int) -> Stencil:
    """Tilted Laplacian: support only at the hyperdiagonal corners."""
    return _build_lap(d, {0: -(2**d), d: 1}), 2 ** (d - 1)


def lap_bri(d: int) -> Stencil:
    """Brillouin Laplacian (81-point stencil in 4D).

    Constant on the entire Brillouin-zone boundary.
    """
    wt = {j: 2 ** (d - j) for j in range(1, d + 1)}
    wt[0] = -sum(comb(d, j) * (2**j) * wt[j] for j in range(1, d + 1))
    return _build_lap(d, wt), 4 ** (d - 1)


def lap_iso(d: int) -> Stencil:
    """Isotropic ("Mehrstellen") Laplacian.

    Rotational symmetry is respected through the leading deviation from the
    continuum. Defined for d in {2, 3, 4}.
    """
    _data = {
        2: ({0: -20, 1: 4, 2: 1}, 6),
        3: ({0: -200, 1: 20, 2: 6, 3: 1}, 48),
        4: ({0: -2000, 1: 100, 2: 40, 3: 7, 4: 1}, 432),
    }
    if d not in _data:
        raise ValueError(
            f"lap_iso is only defined for d in {sorted(_data)}, got d={d}."
        )
    wt, norm = _data[d]
    return _build_lap(d, wt), norm


# --------------------------------------------------------------------------- #
# Derivatives:  return (stencil, normalization)  [for direction mu = 0]
# --------------------------------------------------------------------------- #
def der_std(d: int) -> Stencil:
    """Standard 2-point symmetric derivative."""
    return _build_der(d, {0: 1}), 2


def der_bri(d: int) -> Stencil:
    """Brillouin derivative (modulates the transverse directions)."""
    wt = {j: 2 ** (d - 1 - j) for j in range(d)}
    return _build_der(d, wt), 2 ** (2 * d - 1)


def der_iso(d: int) -> Stencil:
    """Isotropic derivative."""
    wt = {j: 4 ** (d - 1 - j) for j in range(d)}
    return _build_der(d, wt), 2 * 6 ** (d - 1)


# --------------------------------------------------------------------------- #
# Registries and accessors
# --------------------------------------------------------------------------- #
LAPLACIANS: Dict[str, Callable[[int], Stencil]] = {
    "std": lap_std,
    "til": lap_til,
    "bri": lap_bri,
    "iso": lap_iso,
}

DERIVATIVES: Dict[str, Callable[[int], Stencil]] = {
    "std": der_std,
    "bri": der_bri,
    "iso": der_iso,
}


def get_lap(name: str, d: int) -> Stencil:
    """Return the ``(stencil, norm)`` for Laplacian ``name`` in dimension ``d``."""
    if name not in LAPLACIANS:
        raise KeyError(
            f"Unknown Laplacian {name!r}; expected one of {sorted(LAPLACIANS)}."
        )
    return LAPLACIANS[name](d)


def get_der(name: str, d: int) -> Stencil:
    """Return the ``(stencil, norm)`` for derivative ``name`` (mu=0) in dim ``d``."""
    if name not in DERIVATIVES:
        raise KeyError(
            f"Unknown derivative {name!r}; expected one of {sorted(DERIVATIVES)}."
        )
    return DERIVATIVES[name](d)


def get_der_mu(name: str, d: int, mu: int) -> Stencil:
    """Return the derivative stencil for direction ``mu`` (axis swap of mu=0)."""
    stencil, norm = get_der(name, d)
    if mu != 0:
        stencil = np.swapaxes(stencil, 0, mu)
    return stencil, norm
