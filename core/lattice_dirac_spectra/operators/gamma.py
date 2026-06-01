"""
Euclidean gamma matrices.

Hermitian gamma matrices satisfying the Clifford algebra
``{gamma_mu, gamma_nu} = 2 * delta_{mu,nu} * 1``, for ``d`` in {2, 3, 4}.
The spinor dimension is ``Ns = 2 ** (d // 2)``.
"""

from typing import List

import numpy as np

__all__ = ["gamma_matrices", "pauli_matrices"]


def pauli_matrices() -> List[np.ndarray]:
    """Return the three Pauli matrices ``[sigma_1, sigma_2, sigma_3]``."""
    s1 = np.array([[0, 1], [1, 0]], dtype=complex)
    s2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    s3 = np.array([[1, 0], [0, -1]], dtype=complex)
    return [s1, s2, s3]


def gamma_matrices(d: int) -> List[np.ndarray]:
    """Return ``d`` Hermitian gamma matrices for a ``d``-dimensional lattice.

    Parameters
    ----------
    d : int
        Lattice dimensionality (2, 3, or 4).

    Returns
    -------
    list of ndarray
        ``d`` matrices of size ``Ns x Ns`` with ``Ns = 2 ** (d // 2)``,
        satisfying ``{gamma_mu, gamma_nu} = 2 delta_{mu nu}`` and
        ``gamma_mu^dagger = gamma_mu``.

    Raises
    ------
    ValueError
        If ``d`` is not in {2, 3, 4}.
    """
    s1, s2, s3 = pauli_matrices()
    if d == 2:
        return [s1, s2]
    if d == 3:
        return [s1, s2, s3]
    if d == 4:
        I2 = np.eye(2, dtype=complex)
        return [
            np.kron(s1, s1),
            np.kron(s1, s2),
            np.kron(s1, s3),
            np.kron(s2, I2),
        ]
    raise ValueError(f"gamma_matrices is only defined for d in {{2,3,4}}, got d={d}.")
