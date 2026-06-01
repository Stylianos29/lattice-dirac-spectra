"""
Overlap operator construction (eigenvalue level) and sign-function approximants.

Given the eigenvalues ``z`` of a (gamma5-Hermitian) kernel Dirac operator, the
massive overlap eigenvalues are

    z_ov = (rho + m/2) + (rho - m/2) * (z - rho) / |z - rho|,
    0 < rho < 2,   0 <= m < 2 rho.

In the free field this traces a Ginsparg-Wilson circle of radius ``rho - m/2``
centered at ``(rho + m/2, 0)``.

Also provides the diagonal Kenney-Laub approximation to the sign function,
``f_{n,n}(x) = g_{2n+1}(x) = tanh((2n+1) * artanh(x))`` (Duerr & Koutsou,
arXiv:1701.00726, eqs. 11/13), used in the approximate-overlap studies.
"""

from typing import Tuple

import numpy as np

__all__ = [
    "overlap_eigenvalues",
    "gw_circle",
    "kl_sign",
]


def overlap_eigenvalues(
    z: np.ndarray,
    rho: float = 1.0,
    mass: float = 0.0,
) -> np.ndarray:
    """Massive overlap eigenvalues from kernel eigenvalues ``z``.

    Parameters
    ----------
    z : ndarray, complex
        Eigenvalues of the kernel Dirac operator.
    rho : float, optional
        Overlap shift parameter, ``0 < rho < 2`` (default 1).
    mass : float, optional
        Quark mass, ``0 <= mass < 2*rho`` (default 0).

    Returns
    -------
    ndarray, complex
        The overlap eigenvalues.
    """
    if not 0 < rho < 2:
        raise ValueError(f"Need 0 < rho < 2, got rho = {rho}.")
    if not 0 <= mass < 2 * rho:
        raise ValueError(f"Need 0 <= mass < 2*rho = {2 * rho}, got mass = {mass}.")

    z = np.asarray(z, dtype=complex)
    z_shifted = z - rho
    x = np.abs(z_shifted)

    z_norm = np.zeros_like(z_shifted)
    mask = x > 1e-12
    z_norm[mask] = z_shifted[mask] / x[mask]

    return (rho + mass / 2) + (rho - mass / 2) * z_norm


def gw_circle(
    rho: float = 1.0,
    mass: float = 0.0,
    n_points: int = 300,
) -> Tuple[float, float, np.ndarray]:
    """Ideal Ginsparg-Wilson circle for the massive overlap spectrum.

    Returns
    -------
    center_x : float
        Real-axis center, ``rho + mass/2``.
    radius : float
        Circle radius, ``rho - mass/2``.
    points : ndarray, complex, shape ``(n_points,)``
        Sampled points on the circle (for plotting).
    """
    center_x = rho + mass / 2
    radius = rho - mass / 2
    theta = np.linspace(0, 2 * np.pi, n_points)
    points = center_x + radius * np.cos(theta) + 1j * radius * np.sin(theta)
    return center_x, radius, points


def kl_sign(x: np.ndarray, n: int) -> np.ndarray:
    """Diagonal Kenney-Laub approximation to ``sign(x)`` of order ``n``.

    ``f_{n,n}(x) = g_{2n+1}(x) = tanh((2n+1) * artanh(x))``, exact in the limit
    ``n -> inf`` for real ``x`` in ``(-1, 1)`` extended by oddness/monotonicity.

    Parameters
    ----------
    x : ndarray
        Argument(s).
    n : int
        Approximation order (``n >= 1``).

    Returns
    -------
    ndarray
        The approximant ``f_{n,n}(x)``.
    """
    if n < 1:
        raise ValueError(f"Need n >= 1, got n = {n}.")
    x = np.asarray(x, dtype=complex)
    ell = 2 * n + 1
    num = (1 + x) ** ell - (1 - x) ** ell
    den = (1 + x) ** ell + (1 - x) ** ell
    return num / den
