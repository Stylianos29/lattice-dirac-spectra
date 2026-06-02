"""
Brillouin-zone visualization of stencil Fourier representations.

3D surfaces and contour maps of the momentum-space stencil functions over the
Brillouin zone ``(-pi, pi]^d``, reproducing the structure of Figs. 1-2 of
S. Duerr and G. Koutsou, Phys. Rev. D 83, 114512 (2011):

* the Laplacian ``Delta.hat(k)`` (real) -- e.g. its flatness on the BZ boundary
  for the Brillouin stencil, or its near-isotropy for the isotropic stencil;
* the derivative ``nabla_mu.hat(k)`` (imaginary) -- the transverse modulation of
  the Brillouin/isotropic derivatives versus the structureless ``sin`` of the
  standard one.

For ``d > 2`` a 2D slice is taken: two axes are scanned and the remaining
momenta are held at zero (configurable via ``axes`` and ``fixed``).
"""

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..constants.visualization import BZ_SURFACE_DEFAULTS
from ..operators.fourier import eval_stencil_fourier
from ..operators.stencils import get_der_mu, get_lap
from .style import new_axes

__all__ = [
    "laplacian_surface_data",
    "derivative_surface_data",
    "plot_surface",
    "plot_laplacian_surface",
    "plot_derivative_surface",
]


def _bz_grid(d: int, grid_size: int, axes: Tuple[int, int], fixed: Dict[int, float]):
    """Build a 2D meshgrid of phase vectors over the Brillouin zone.

    Two ``axes`` are scanned over ``(-pi, pi]``; all other components are held at
    the values in ``fixed`` (default 0). Returns ``(K0, K1, phi)`` where ``K0``,
    ``K1`` are ``(grid_size, grid_size)`` meshes and ``phi`` is the flattened
    ``(grid_size**2, d)`` array of phase vectors.
    """
    ax0, ax1 = axes
    if ax0 == ax1 or not (0 <= ax0 < d) or not (0 <= ax1 < d):
        raise ValueError(f"axes={axes} invalid for d={d}.")
    line = np.linspace(-np.pi, np.pi, grid_size, endpoint=True)
    K0, K1 = np.meshgrid(line, line, indexing="xy")
    phi = np.zeros((grid_size * grid_size, d), dtype=float)
    for mu, val in (fixed or {}).items():
        phi[:, mu] = val
    phi[:, ax0] = K0.ravel()
    phi[:, ax1] = K1.ravel()
    return K0, K1, phi


def laplacian_surface_data(
    lap_name: str,
    d: int = 2,
    grid_size: int = BZ_SURFACE_DEFAULTS["grid_size"],
    *,
    axes: Tuple[int, int] = (0, 1),
    fixed: Optional[Dict[int, float]] = None,
):
    """Return ``(K0, K1, Z)`` for the Laplacian ``Delta.hat`` over a BZ slice.

    ``Z`` is real (the Laplacian Fourier representation is real-valued).
    """
    s, n = get_lap(lap_name, d)
    K0, K1, phi = _bz_grid(d, grid_size, axes, fixed or {})
    Z = eval_stencil_fourier(s, n, phi).real.reshape(K0.shape)
    return K0, K1, Z


def derivative_surface_data(
    der_name: str,
    mu: int = 0,
    d: int = 2,
    grid_size: int = BZ_SURFACE_DEFAULTS["grid_size"],
    *,
    axes: Tuple[int, int] = (0, 1),
    fixed: Optional[Dict[int, float]] = None,
):
    """Return ``(K0, K1, Z)`` for ``nabla_mu.hat`` over a BZ slice.

    ``Z`` is the imaginary part (the derivative Fourier representation is purely
    imaginary), i.e. the real coefficient multiplying ``i``.
    """
    s, n = get_der_mu(der_name, d, mu)
    K0, K1, phi = _bz_grid(d, grid_size, axes, fixed or {})
    Z = eval_stencil_fourier(s, n, phi).imag.reshape(K0.shape)
    return K0, K1, Z


def plot_surface(
    K0: np.ndarray,
    K1: np.ndarray,
    Z: np.ndarray,
    *,
    kind: str = "surface",
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    cmap: str = BZ_SURFACE_DEFAULTS["cmap"],
):
    """Render a precomputed ``(K0, K1, Z)`` grid as a 3D surface or a contour map.

    ``kind`` is ``"surface"`` (3D) or ``"contour"`` (filled 2D).
    """
    if kind == "surface":
        if ax is None:
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure
        ax.plot_surface(K0, K1, Z, cmap=cmap, linewidth=0, antialiased=True)
        ax.set_xlabel(r"$k_0$")
        ax.set_ylabel(r"$k_1$")
        ax.set_zlabel("value")
    elif kind == "contour":
        fig, ax = new_axes(ax, figsize=(5.5, 4.5))
        cs = ax.contourf(K0, K1, Z, levels=20, cmap=cmap)
        fig.colorbar(cs, ax=ax)
        ax.set_xlabel(r"$k_0$")
        ax.set_ylabel(r"$k_1$")
        ax.set_aspect("equal")
    else:
        raise ValueError(f"kind must be 'surface' or 'contour', got {kind!r}.")
    if title:
        ax.set_title(title)
    return fig, ax


def plot_laplacian_surface(
    lap_name: str,
    d: int = 2,
    grid_size: int = BZ_SURFACE_DEFAULTS["grid_size"],
    *,
    kind: str = "surface",
    ax: Optional[plt.Axes] = None,
    axes: Tuple[int, int] = (0, 1),
    fixed: Optional[Dict[int, float]] = None,
):
    """Plot the Laplacian Fourier surface ``Delta.hat(k)`` over a BZ slice."""
    K0, K1, Z = laplacian_surface_data(lap_name, d, grid_size, axes=axes, fixed=fixed)
    return plot_surface(K0, K1, Z, kind=kind, ax=ax, title=fr"$\hat{{\Delta}}_{{\rm {lap_name}}}$")


def plot_derivative_surface(
    der_name: str,
    mu: int = 0,
    d: int = 2,
    grid_size: int = BZ_SURFACE_DEFAULTS["grid_size"],
    *,
    kind: str = "surface",
    ax: Optional[plt.Axes] = None,
    axes: Tuple[int, int] = (0, 1),
    fixed: Optional[Dict[int, float]] = None,
):
    """Plot the derivative Fourier surface ``Im nabla_mu.hat(k)`` over a BZ slice."""
    K0, K1, Z = derivative_surface_data(der_name, mu, d, grid_size, axes=axes, fixed=fixed)
    return plot_surface(
        K0, K1, Z, kind=kind, ax=ax,
        title=fr"$\mathrm{{Im}}\,\hat{{\nabla}}^{{\rm {der_name}}}_{{{mu}}}$",
    )
