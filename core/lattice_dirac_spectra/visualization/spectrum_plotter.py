"""
Complex-plane spectrum scatter plots.

Visualize eigenvalue spectra of lattice Dirac operators in the complex plane:

* :func:`plot_spectrum` -- scatter a single set of eigenvalues.
* :func:`plot_formula_vs_direct` -- overlay the recipe (Capability 1) on direct
  diagonalization, the project's signature verification figure.
* :func:`plot_overlap_spectrum` -- scatter an overlap spectrum with its ideal
  Ginsparg-Wilson circle.
* :func:`plot_operator_grid` -- the 4x3 grid over all (Laplacian, derivative)
  combinations.

These build on the verified ``spectra`` and ``operators`` modules; the plotting
adds no new physics.
"""

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..constants.operators import LAP_KEYS, DER_KEYS
from ..operators.overlap import gw_circle
from ..spectra.free_field import (
    eigenvalues_from_formula,
    eigenvalues_from_matrix,
    compare_eigenvalues,
)
from .style import new_axes

__all__ = [
    "plot_spectrum",
    "plot_formula_vs_direct",
    "plot_overlap_spectrum",
    "plot_operator_grid",
]


def plot_spectrum(
    eigs: np.ndarray,
    ax: Optional[plt.Axes] = None,
    *,
    color: str = "C0",
    size: float = 4.0,
    label: Optional[str] = None,
    equal_aspect: bool = True,
):
    """Scatter a complex eigenvalue set in the (Re, Im) plane."""
    fig, ax = new_axes(ax, figsize=(5, 5))
    eigs = np.asarray(eigs)
    ax.scatter(eigs.real, eigs.imag, s=size, c=color, edgecolors="none", label=label)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    if equal_aspect:
        ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    return fig, ax


def plot_formula_vs_direct(
    lap_name: str,
    der_name: str,
    d: int,
    L: int,
    ax: Optional[plt.Axes] = None,
    *,
    m0: float = 0.0,
):
    """Overlay the recipe spectrum (red x) on direct diagonalization (blue +).

    The plot title reports the maximum nearest-neighbour error between the two.
    """
    fig, ax = new_axes(ax, figsize=(5.5, 5.5))
    ef = eigenvalues_from_formula(lap_name, der_name, d, L, m0=m0)
    ed = eigenvalues_from_matrix(lap_name, der_name, d, L, m0=m0)
    err = compare_eigenvalues(ef, ed)

    ax.plot(ef.real, ef.imag, "rx", ms=7, mew=1.4, label="recipe", zorder=3)
    ax.plot(ed.real, ed.imag, "b+", ms=6, mew=1.0, label="direct", zorder=2)
    ax.set_title(f"lap_{lap_name}, der_{der_name}  (max err = {err:.1e})")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    return fig, ax


def plot_overlap_spectrum(
    z_ov: np.ndarray,
    rho: float = 1.0,
    mass: float = 0.0,
    ax: Optional[plt.Axes] = None,
    *,
    color: str = "C1",
    size: float = 2.0,
):
    """Scatter an overlap spectrum together with its ideal GW circle."""
    fig, ax = new_axes(ax, figsize=(5, 5))
    center, radius, pts = gw_circle(rho, mass)
    ax.plot(pts.real, pts.imag, "k--", lw=0.8, alpha=0.4, label="GW circle")
    z_ov = np.asarray(z_ov)
    ax.scatter(z_ov.real, z_ov.imag, s=size, c=color, edgecolors="none")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    return fig, ax


def plot_operator_grid(
    d: int,
    L: int,
    *,
    lap_keys: Sequence[str] = LAP_KEYS,
    der_keys: Sequence[str] = DER_KEYS,
    figsize=None,
):
    """Grid of recipe-vs-direct overlays over all (Laplacian, derivative) pairs.

    Rows are Laplacians, columns are derivatives -- the 4x3 layout of the source
    verification notebook.
    """
    nrows, ncols = len(lap_keys), len(der_keys)
    if figsize is None:
        figsize = (4 * ncols, 4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for i, lap in enumerate(lap_keys):
        for j, der in enumerate(der_keys):
            plot_formula_vs_direct(lap, der, d, L, ax=axes[i, j])
            axes[i, j].legend().set_visible(False)
    fig.suptitle(f"All operators -- {d}D free field, L = {L}", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig, axes
