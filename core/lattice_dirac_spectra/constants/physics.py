"""
Physical and algebraic constants.

Spinor dimensions, default overlap parameters, and the set of supported
lattice dimensionalities.
"""

from typing import Tuple

# Lattice dimensionalities for which gamma representations are defined.
SUPPORTED_DIMENSIONS: Tuple[int, ...] = (2, 3, 4)

# Canonical overlap shift parameter (0 < rho < 2).
RHO_DEFAULT: float = 1.0

# "Magic" rho that lifts the leading O(a^4) discretization error of the
# free-field Brillouin-overlap dispersion relation to O(a^6):
#   rho = (3 - sqrt(3)) / 2 ~= 0.6340
# Reference: Duerr & Koutsou, arXiv:1701.00726, Sec. 2.
RHO_MAGIC: float = (3.0 - 3.0**0.5) / 2.0


def Ns(d: int) -> int:
    """Spinor dimension ``Ns = 2 ** (d // 2)`` for a ``d``-dimensional lattice.

    Parameters
    ----------
    d : int
        Lattice dimensionality.

    Returns
    -------
    int
        Number of spinor components (2 for d in {2, 3}, 4 for d = 4).
    """
    return 2 ** (d // 2)
