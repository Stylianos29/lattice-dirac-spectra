"""
Capability 2 -- U(1)-gauged eigenvalue spectra.

Build the gauge-covariant Dirac operator on a gauge background and diagonalize
it to obtain the exact (per-configuration) spectrum -- the ground truth against
which the cheap stochastic mock spectra (Capability 3) are validated.

Dense diagonalization (``numpy.linalg.eigvals``) is used by default, which is
fine for the small lattices accessible here; for larger volumes pass
``k``/``sigma`` to switch to a sparse partial solve
(``scipy.sparse.linalg.eigs``).
"""

from typing import Iterable, Optional

import numpy as np

from ..gauge.covariant import build_gauged_dirac
from ..gauge.smearing import ape_smear

__all__ = ["gauged_spectrum", "ensemble_spectrum"]


def _links_of(config_or_links) -> np.ndarray:
    """Accept either a ``U1Config`` (uses ``.links``) or a raw link array."""
    return getattr(config_or_links, "links", config_or_links)


def gauged_spectrum(
    lap_name: str,
    der_name: str,
    config_or_links,
    *,
    m0: float = 0.0,
    n_ape: int = 0,
    alpha: float = 0.72,
    k: Optional[int] = None,
    sigma: Optional[complex] = None,
) -> np.ndarray:
    """Eigenvalues of the gauged Dirac operator on a single configuration.

    Parameters
    ----------
    lap_name, der_name : str
        Laplacian and derivative stencil keys.
    config_or_links : U1Config or ndarray
        A loaded configuration, or a raw ``(*L, d)`` link array.
    m0 : float
        Bare mass.
    n_ape : int
        Number of APE smearing steps applied to the links first (0 = none).
    alpha : float
        APE smearing parameter.
    k : int, optional
        If given, compute only ``k`` eigenvalues near ``sigma`` with a sparse
        solver instead of the full dense spectrum.
    sigma : complex, optional
        Shift for the sparse solve (eigenvalues nearest ``sigma``).

    Returns
    -------
    ndarray, complex
        The eigenvalues.
    """
    links = _links_of(config_or_links)
    if n_ape > 0:
        links = ape_smear(links, alpha=alpha, n_steps=n_ape)

    D = build_gauged_dirac(lap_name, der_name, links, m0=m0)

    if k is None:
        return np.linalg.eigvals(D)

    from scipy.sparse.linalg import eigs

    vals = eigs(D, k=k, sigma=sigma, return_eigenvectors=False)
    return vals


def ensemble_spectrum(
    lap_name: str,
    der_name: str,
    configs: Iterable,
    *,
    m0: float = 0.0,
    n_ape: int = 0,
    alpha: float = 0.72,
) -> np.ndarray:
    """Stack the dense gauged spectra over an ensemble of configurations.

    Returns
    -------
    ndarray, complex
        All eigenvalues from all configurations concatenated.
    """
    chunks = [
        gauged_spectrum(lap_name, der_name, c, m0=m0, n_ape=n_ape, alpha=alpha)
        for c in configs
    ]
    return np.concatenate(chunks) if chunks else np.array([], dtype=complex)
