"""
Gauge subpackage -- Capability 2 support.

* :mod:`u1_config`   -- U(1) HMC configuration reader (``u1-hmc`` HDF5 format).
* :mod:`observables` -- plaquette and topological charge (metadata cross-checks).
* :mod:`covariant`   -- gauge-covariant stencil-to-matrix, the gauged Dirac
                        operator, and gauge transformations.
* :mod:`smearing`    -- U(1) APE link smearing.

Importing this subpackage pulls in ``h5py`` (via the reader); the top-level
``lattice_dirac_spectra`` package does not, to keep the numerical core light.
"""

from .u1_config import (  # noqa: F401
    U1Config,
    list_betas,
    list_trajectories,
    load_config,
    load_ensemble,
)
from .observables import plaquette, topological_charge, plaquette_field  # noqa: F401
from .covariant import (  # noqa: F401
    parallel_transport,
    covariant_stencil_to_matrix,
    build_gauged_dirac,
    gauge_transform,
)
from .smearing import ape_smear  # noqa: F401
