"""
Constants Package
=================

Central repository for constants used throughout the lattice-dirac-spectra
library, organized into thematic modules:

    - **operators**:     operator-type names and stencil keys
    - **physics**:       physical/algebraic constants (Ns(d), default rho, ...)
    - **paths**:         filesystem paths and directory structure
    - **visualization**: plot styling defaults

All names are exposed at the package level, so both forms work:

    >>> from lattice_dirac_spectra.constants import LAP_KEYS
    >>> from lattice_dirac_spectra.constants.operators import LAP_KEYS
"""

from .operators import *  # noqa: F401,F403
from .physics import *  # noqa: F401,F403
from .paths import *  # noqa: F401,F403
from .visualization import *  # noqa: F401,F403

__all__ = [
    # operators
    "KERNEL_OPERATOR_TYPES",
    "OVERLAP_OPERATOR_METHODS",
    "LAP_KEYS",
    "DER_KEYS",
    "NAMED_OPERATORS",
    "operator_keys",
    # physics
    "Ns",
    "RHO_DEFAULT",
    "RHO_MAGIC",
    "SUPPORTED_DIMENSIONS",
    # paths
    "ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "OUTPUT_DIR",
    "PLOTS_DIR",
    "TABLES_DIR",
    # visualization
    "DEFAULT_COLORS",
    "MARKER_STYLES",
    "BZ_SURFACE_DEFAULTS",
]
