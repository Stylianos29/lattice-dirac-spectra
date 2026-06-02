"""
Free-field dispersion relations.

For a free Wilson-type operator the on-shell condition (pole of the propagator)
at spatial momentum ``p`` and Euclidean energy ``E`` is obtained by continuing
the temporal momentum ``k_d -> i E`` and solving

    G(E; p) = (A + m)**2 - sum_mu nabla_mu.hat(phi)**2 = 0,
    phi = (p_1, ..., p_{d-1}, i E),   A = -1/2 * Delta.hat(phi),

where the last axis (index ``d-1``) is the temporal direction. Along this
continuation ``A`` and ``nabla_mu.hat`` are real, so ``G`` is a real function of
``E`` and the lowest positive root is the physical dispersion branch.

This reproduces the free-field dispersion curves of Figs. 1-2 of
Duerr & Koutsou (PRD 83, 114512 / arXiv:1701.00726): ``aE`` versus
``sqrt(sum a^2 p_i^2)`` along the spatial directions (1,0,0), (1,1,0), (1,1,1),
compared with the continuum relation. The heavy-quark substitution
``am -> exp(am) - 1`` is available via ``heavy_quark=True``.
"""

from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from ..operators.fourier import eval_stencil_fourier
from ..operators.stencils import get_der_mu, get_lap
from .style import new_axes

__all__ = [
    "onshell_value",
    "dispersion_relation",
    "default_directions",
    "plot_dispersion",
]


def default_directions(d: int) -> List[Tuple[int, ...]]:
    """Canonical spatial directions for a ``d``-dimensional lattice.

    Spatial dimension is ``d - 1`` (the last axis is temporal). E.g. d=4 gives
    [(1,0,0), (1,1,0), (1,1,1)].
    """
    ds = d - 1
    base = [tuple([1] * k + [0] * (ds - k)) for k in range(1, ds + 1)]
    return base


def onshell_value(
    lap_name: str,
    der_name: str,
    d: int,
    p_spatial: np.ndarray,
    E: float,
    m: float = 0.0,
) -> float:
    """Evaluate the on-shell function ``G(E; p)`` (zero at a dispersion branch)."""
    phi = np.zeros((1, d), dtype=complex)
    phi[0, : d - 1] = np.asarray(p_spatial, dtype=float)
    phi[0, d - 1] = 1j * E

    lap_s, lap_n = get_lap(lap_name, d)
    A = (-0.5 * eval_stencil_fourier(lap_s, lap_n, phi)[0]).real

    nabla_sq_sum = 0.0 + 0.0j
    for mu in range(d):
        ds, dn = get_der_mu(der_name, d, mu)
        nab = eval_stencil_fourier(ds, dn, phi)[0]
        nabla_sq_sum += nab**2

    return float(((A + m) ** 2 - nabla_sq_sum).real)


def dispersion_relation(
    lap_name: str,
    der_name: str,
    d: int,
    direction: Sequence[float],
    P_values: np.ndarray,
    *,
    m: float = 0.0,
    heavy_quark: bool = False,
    E_max: float = 10.0,
    n_scan: int = 800,
) -> np.ndarray:
    """Energy ``aE`` along a spatial ``direction`` for each ``|ap|`` in ``P_values``.

    Parameters
    ----------
    lap_name, der_name : str
        Laplacian and derivative stencil keys.
    d : int
        Lattice dimensionality (last axis is temporal).
    direction : sequence of length ``d - 1``
        Spatial direction; normalized internally.
    P_values : array
        Spatial-momentum magnitudes ``|ap| = sqrt(sum a^2 p_i^2)``.
    m : float
        Bare mass.
    heavy_quark : bool
        If True, use the substitution ``m -> exp(m) - 1`` in the operator.
    E_max, n_scan : float, int
        Energy scan range and resolution for bracketing the lowest root.

    Returns
    -------
    ndarray
        ``aE`` for each entry of ``P_values`` (``nan`` where no root was found).
    """
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (d - 1,):
        raise ValueError(
            f"direction must have length d-1 = {d - 1}, got {direction.shape}."
        )
    norm = np.linalg.norm(direction)
    nhat = direction / norm if norm > 0 else direction
    m_eff = (np.exp(m) - 1.0) if heavy_quark else m

    E_grid = np.linspace(0.0, E_max, n_scan)
    out = np.full(len(P_values), np.nan)

    for idx, P in enumerate(np.asarray(P_values, dtype=float)):
        p = P * nhat

        def G(E):
            return onshell_value(lap_name, der_name, d, p, E, m_eff)

        g_vals = np.array([G(E) for E in E_grid])

        if abs(g_vals[0]) < 1e-12:
            out[idx] = 0.0
            continue
        for i in range(len(E_grid) - 1):
            if g_vals[i] == 0.0:
                out[idx] = E_grid[i]
                break
            if g_vals[i] * g_vals[i + 1] < 0:
                out[idx] = brentq(G, E_grid[i], E_grid[i + 1])
                break
    return out


def plot_dispersion(
    lap_name: str,
    der_name: str,
    d: int = 4,
    *,
    directions: Optional[Sequence[Sequence[float]]] = None,
    P_max: float = 5.0,
    n_points: int = 60,
    m: float = 0.0,
    heavy_quark: bool = False,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
):
    """Plot dispersion curves along several directions versus the continuum line."""
    if directions is None:
        directions = default_directions(d)
    fig, ax = new_axes(ax, figsize=(6, 5))
    P_values = np.linspace(0.0, P_max, n_points)

    styles = ["-", "-.", "--", ":"]
    for k, direction in enumerate(directions):
        E = dispersion_relation(
            lap_name,
            der_name,
            d,
            direction,
            P_values,
            m=m,
            heavy_quark=heavy_quark,
        )
        label = "along (" + ",".join(str(int(c)) for c in direction) + ")"
        ax.plot(P_values, E, styles[k % len(styles)], label=label)

    # Continuum relation: aE = sqrt(P^2 + m^2)
    ax.plot(
        P_values,
        np.sqrt(P_values**2 + m**2),
        "k:",
        lw=1.0,
        alpha=0.7,
        label="continuum",
    )

    ax.set_xlabel(r"$\sqrt{\sum_i a^2 p_i^2}$")
    ax.set_ylabel(r"$aE$")
    ax.set_xlim(0, P_max)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    if title is None:
        title = f"lap_{lap_name}, der_{der_name}  (am={m}{', heavy' if heavy_quark else ''})"
    ax.set_title(title)
    return fig, ax
