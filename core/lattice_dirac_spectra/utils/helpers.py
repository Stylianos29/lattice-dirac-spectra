"""
General-purpose helper functions.

Small, domain-agnostic utilities used across the library. Populated as needed.
"""

import numpy as np


def spectral_bounds(eigs: np.ndarray) -> dict:
    """Summary statistics of a complex eigenvalue set.

    Returns a dict with the real/imaginary extents and the min/max modulus.
    """
    eigs = np.asarray(eigs)
    mod = np.abs(eigs)
    return {
        "re_min": float(eigs.real.min()),
        "re_max": float(eigs.real.max()),
        "im_min": float(eigs.imag.min()),
        "im_max": float(eigs.imag.max()),
        "abs_min": float(mod.min()),
        "abs_max": float(mod.max()),
    }
