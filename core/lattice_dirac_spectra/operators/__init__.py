"""
Operators subpackage -- the physics core.

Stencils, Fourier evaluation, gamma matrices, the coordinate-space Dirac matrix,
the closed-form recipe eigenvalues, and the overlap projection. These are the
dimension-general building blocks shared by every fidelity level (free-field,
gauged, mock).
"""

from .stencils import (  # noqa: F401
    get_lap,
    get_der,
    get_der_mu,
    LAPLACIANS,
    DERIVATIVES,
)
from .fourier import eval_stencil_fourier, discrete_momenta  # noqa: F401
from .gamma import gamma_matrices, pauli_matrices  # noqa: F401
from .dirac import (  # noqa: F401
    build_free_dirac_matrix,
    recipe_eigenvalues,
    stencil_to_matrix,
    idx_to_coords,
    coords_to_idx,
)
from .overlap import overlap_eigenvalues, gw_circle, kl_sign  # noqa: F401
