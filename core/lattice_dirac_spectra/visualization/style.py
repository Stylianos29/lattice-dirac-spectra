"""
Shared plotting style helpers.

Light-touch styling utilities used across the visualization subsystem. Nothing
here forces a matplotlib backend or mutates global state on import; call
:func:`apply_style` explicitly if you want the project look.
"""

from typing import Optional

import matplotlib.pyplot as plt

from ..constants.visualization import DEFAULT_COLORS

__all__ = ["apply_style", "kernel_color", "new_axes"]


def apply_style() -> None:
    """Apply a clean, publication-leaning matplotlib rcParams profile."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def kernel_color(name: str) -> str:
    """Return the consistent color for a named kernel/series (fallback ``C0``)."""
    return DEFAULT_COLORS.get(name, "C0")


def new_axes(ax: Optional[plt.Axes] = None, **subplots_kw):
    """Return ``(fig, ax)``, creating a new figure/axes only if ``ax`` is None.

    Lets every plotting function accept an optional ``ax`` for composition while
    still working standalone.
    """
    if ax is not None:
        return ax.figure, ax
    return plt.subplots(**subplots_kw)
