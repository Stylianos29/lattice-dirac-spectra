"""
Spectra subpackage -- orchestration of the three fidelity levels.

* :mod:`free_field` (Capability 1) -- recipe vs direct diagonalization.
* :mod:`gauged`     (Capability 2) -- exact spectra on U(1) gauge configs.
* :mod:`mock`       (Capability 3) -- stochastic perturbed-momenta recipe (planned, M4).
"""

from .free_field import (  # noqa: F401
    eigenvalues_from_formula,
    eigenvalues_from_matrix,
    compare_eigenvalues,
)
from .gauged import gauged_spectrum, ensemble_spectrum  # noqa: F401
