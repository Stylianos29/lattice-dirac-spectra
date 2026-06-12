"""Synthetic gauge backgrounds (no Monte Carlo).

Trivial and (later) perturbed U(1) link fields for validating the gauged
operator against the free-field limit and for cheap exploratory spectra where
no thermalized ensemble is available.
"""

import numpy as np  # type: ignore

__all__ = ["unit_links"]


def unit_links(d: int, L: int) -> np.ndarray:
    """All-ones (free-field) U(1) links of shape ``(L,) * d + (d,)``."""
    return np.ones((L,) * d + (d,), dtype=complex)
