"""
Operator type constants.

Defines the valid stencil keys for the Laplacian and derivative discretizations,
the named Wilson-type kernels assembled from them, and the overlap-operator
sign-function methods.

Reference: S. Duerr and G. Koutsou, Phys. Rev. D 83, 114512 (2011).
"""

from typing import Tuple

# Laplacian stencil keys (standard, tilted, Brillouin, isotropic)
LAP_KEYS: Tuple[str, ...] = ("std", "til", "bri", "iso")

# Derivative stencil keys (standard, Brillouin, isotropic)
DER_KEYS: Tuple[str, ...] = ("std", "bri", "iso")

# Named Wilson-type kernels as (derivative_key, laplacian_key) pairs.
#   - "Wilson":    standard derivative + standard Laplacian
#   - "Brillouin": isotropic derivative + Brillouin Laplacian
NAMED_OPERATORS = {
    "Wilson": ("std", "std"),
    "Brillouin": ("iso", "bri"),
}

# Valid Dirac operator types used as overlap-procedure kernels
KERNEL_OPERATOR_TYPES = frozenset(NAMED_OPERATORS.keys())

# Valid matrix sign-function methods used in overlap operators
OVERLAP_OPERATOR_METHODS = frozenset(
    ["Bare", "Chebyshev", "KL", "Neuberger", "Zolotarev"]
)


def operator_keys(name: str) -> Tuple[str, str]:
    """Resolve a named kernel to its ``(derivative_key, laplacian_key)`` pair.

    Parameters
    ----------
    name : str
        A key of :data:`NAMED_OPERATORS` (e.g. ``"Wilson"``, ``"Brillouin"``).

    Returns
    -------
    (str, str)
        The ``(der_key, lap_key)`` pair.

    Raises
    ------
    KeyError
        If ``name`` is not a recognized named operator.
    """
    if name not in NAMED_OPERATORS:
        raise KeyError(
            f"Unknown named operator {name!r}; "
            f"expected one of {sorted(NAMED_OPERATORS)}."
        )
    return NAMED_OPERATORS[name]
