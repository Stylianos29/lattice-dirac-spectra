"""
Spectra subpackage -- orchestration of the three fidelity levels.

* :mod:`free_field` (Capability 1) -- implemented.
* :mod:`gauged`     (Capability 2) -- planned (M3): exact spectra on U(1) configs.
* :mod:`mock`       (Capability 3) -- planned (M4): stochastic perturbed-momenta recipe.
"""

from .free_field import (  # noqa: F401
    eigenvalues_from_formula,
    eigenvalues_from_matrix,
    compare_eigenvalues,
)
