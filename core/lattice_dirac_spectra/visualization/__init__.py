"""
Visualization subpackage (Capability 4 + spectrum scatter).

* :mod:`spectrum_plotter` -- complex-plane spectrum scatter (recipe vs direct,
  ensemble clouds, overlap GW circles, the 4x3 operator grid).
* :mod:`brillouin_zone` -- 3D surfaces / contours of the stencil Fourier
  representations over the Brillouin zone (PRD 83 Figs 1-2).
* :mod:`dispersion` -- free-field dispersion relations via on-shell root finding
  (PRD 83 / arXiv:1701.00726 Figs 1-2).
* :mod:`style` -- shared matplotlib styling helpers.

Importing this subpackage pulls in matplotlib; the top-level ``lattice_dirac_spectra``
package deliberately does not, so the numerical core stays import-light.
"""

from .style import apply_style, kernel_color, new_axes  # noqa: F401
from .spectrum_plotter import (  # noqa: F401
    plot_spectrum,
    plot_formula_vs_direct,
    plot_overlap_spectrum,
    plot_operator_grid,
)
from .brillouin_zone import (  # noqa: F401
    laplacian_surface_data,
    derivative_surface_data,
    plot_surface,
    plot_laplacian_surface,
    plot_derivative_surface,
)
from .dispersion import (  # noqa: F401
    onshell_value,
    dispersion_relation,
    default_directions,
    plot_dispersion,
)
