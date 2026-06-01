"""
lattice_dirac_spectra
=====================

Eigenvalue spectra of Wilson-type and overlap lattice Dirac operators.

Subsystems
----------
- ``operators``     : the physics core -- stencils, Fourier evaluation, gamma
                      matrices, the coordinate-space Dirac matrix, the recipe
                      eigenvalue formula, and the overlap projection.
- ``spectra``       : orchestration of the three fidelity levels
                      (free-field implemented; gauged and mock planned).
- ``gauge``         : U(1) HMC config reading and gauge-covariant operators (planned).
- ``visualization`` : spectrum scatter, Brillouin-zone surfaces, dispersion (planned).
- ``constants``     : operator keys, physics constants, paths, plot styling.
- ``utils``         : logging and helpers.

Quick start
-----------
>>> from lattice_dirac_spectra import eigenvalues_from_formula, eigenvalues_from_matrix
>>> from lattice_dirac_spectra import compare_eigenvalues
>>> ef = eigenvalues_from_formula("bri", "iso", d=2, L=8)
>>> ed = eigenvalues_from_matrix("bri", "iso", d=2, L=8)
>>> compare_eigenvalues(ef, ed) < 1e-10
True
"""

__version__ = "0.1.0"

# Curated public API
from .operators import (  # noqa: F401
    get_lap,
    get_der,
    get_der_mu,
    eval_stencil_fourier,
    discrete_momenta,
    gamma_matrices,
    build_free_dirac_matrix,
    recipe_eigenvalues,
    overlap_eigenvalues,
    gw_circle,
    kl_sign,
)
from .spectra import (  # noqa: F401
    eigenvalues_from_formula,
    eigenvalues_from_matrix,
    compare_eigenvalues,
)
from .constants import (  # noqa: F401
    LAP_KEYS,
    DER_KEYS,
    NAMED_OPERATORS,
    KERNEL_OPERATOR_TYPES,
    Ns,
    RHO_DEFAULT,
    RHO_MAGIC,
)

__all__ = [
    "__version__",
    # operators
    "get_lap",
    "get_der",
    "get_der_mu",
    "eval_stencil_fourier",
    "discrete_momenta",
    "gamma_matrices",
    "build_free_dirac_matrix",
    "recipe_eigenvalues",
    "overlap_eigenvalues",
    "gw_circle",
    "kl_sign",
    # spectra
    "eigenvalues_from_formula",
    "eigenvalues_from_matrix",
    "compare_eigenvalues",
    # constants
    "LAP_KEYS",
    "DER_KEYS",
    "NAMED_OPERATORS",
    "KERNEL_OPERATOR_TYPES",
    "Ns",
    "RHO_DEFAULT",
    "RHO_MAGIC",
]
