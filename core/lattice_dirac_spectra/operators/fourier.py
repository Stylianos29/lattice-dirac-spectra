"""
Fourier-space evaluation of stencils and momentum-grid helpers.

The Fourier transform of a stencil at a set of phase vectors gives the
momentum-space representation of the corresponding lattice operator:
``Laplacian.hat(phi)`` is real, ``derivative.hat(phi)`` is purely imaginary.
"""

from itertools import product as iproduct

import numpy as np

__all__ = [
    "eval_stencil_fourier",
    "discrete_momenta",
]


def eval_stencil_fourier(stencil: np.ndarray, norm: int, phi: np.ndarray) -> np.ndarray:
    """Evaluate the Fourier transform of a stencil at ``N`` phase vectors.

    Computes ``sum_idx (w[idx] / norm) * exp(i * phi . (idx - 1))``.

    Parameters
    ----------
    stencil : ndarray, shape ``(3,) * d``
        Integer stencil weights.
    norm : int
        Stencil normalization.
    phi : ndarray, shape ``(N, d)``
        Phase vectors (lattice momenta, possibly perturbed). May be complex --
        e.g. a temporal momentum continued to ``i E`` for dispersion relations --
        in which case the result is evaluated analytically at the complex phase.

    Returns
    -------
    ndarray, shape ``(N,)``, complex
        The momentum-space representation at each phase vector.
    """
    phi = np.asarray(phi)
    if not np.iscomplexobj(phi):
        phi = phi.astype(float)
    result = np.zeros(phi.shape[0], dtype=complex)
    for idx in np.ndindex(stencil.shape):
        w = int(stencil[idx])
        if w == 0:
            continue
        disp = np.array([i - 1 for i in idx], dtype=float)
        result += w * np.exp(1j * (phi @ disp))
    return result / norm


def discrete_momenta(d: int, L: int) -> np.ndarray:
    """Allowed lattice momenta with periodic BC.

    ``k_mu = (2*pi / L) * n_mu`` with ``n_mu = -L/2 + 1, ..., L/2``.

    Parameters
    ----------
    d : int
        Lattice dimensionality.
    L : int
        Linear lattice extent (same in every direction).

    Returns
    -------
    ndarray, shape ``(L**d, d)``
        The ``L**d`` distinct momentum vectors in the Brillouin zone.
    """
    n_vals = np.arange(-L // 2 + 1, L // 2 + 1)
    mom_tuples = np.array(list(iproduct(n_vals, repeat=d)))
    return (2 * np.pi / L) * mom_tuples
