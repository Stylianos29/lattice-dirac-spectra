"""
Gauge-covariant operator construction.

Promotes the free-field stencil-to-matrix machinery to a gauge background: a hop
from site ``n`` to ``n + disp`` carries the parallel transporter (the product of
link variables along the connecting path). For off-axis (multi-hop) stencil
displacements there are several shortest paths, and the transporter is the
**average over all shortest paths** -- the prescription that preserves
hypercubic symmetry and gamma5-Hermiticity (Duerr & Koutsou, PRD 83, App. C /
Sec. V.A). For U(1) the path product is automatically in the group, so no
back-projection is needed.

Conventions match :mod:`lattice_dirac_spectra.gauge.observables`:
``links`` has shape ``(*L, d)`` with ``links[x, mu] = U_mu(x)`` the forward link
along lattice axis ``mu``. Site linearization (axis 0 fastest) matches the
free-field :func:`lattice_dirac_spectra.operators.dirac.stencil_to_matrix`, so
setting all links to 1 reproduces the free-field operator exactly.

The construction is dimension-general; only the U(1) reader is dimension- and
group-specific.
"""

from itertools import permutations, product as iproduct
from typing import Tuple

import numpy as np

from ..constants.physics import Ns
from ..operators.gamma import gamma_matrices
from ..operators.stencils import get_der_mu, get_lap

__all__ = [
    "parallel_transport",
    "covariant_stencil_to_matrix",
    "build_gauged_dirac",
    "gauge_transform",
]


# --------------------------------------------------------------------------- #
# Site linearization (axis 0 fastest) -- matches operators.dirac
# --------------------------------------------------------------------------- #
def _strides(L: Tuple[int, ...]) -> Tuple[int, ...]:
    strides = [1] * len(L)
    for mu in range(1, len(L)):
        strides[mu] = strides[mu - 1] * L[mu - 1]
    return tuple(strides)


def _lin(coords, strides) -> int:
    return int(sum(c * s for c, s in zip(coords, strides)))


# --------------------------------------------------------------------------- #
# Parallel transport along shortest paths
# --------------------------------------------------------------------------- #
def parallel_transport(links: np.ndarray, n, disp, L) -> complex:
    """Shortest-path-averaged transporter from ``n`` to ``n + disp``.

    Parameters
    ----------
    links : ndarray, shape ``(*L, d)``
        Forward link variables ``U_mu(x)``.
    n : sequence of int
        Starting site coordinates.
    disp : sequence of int
        Displacement (components in ``{-1, 0, +1}`` for the standard stencils).
    L : sequence of int
        Lattice extents.

    Returns
    -------
    complex
        The averaged parallel transporter (1.0 for the zero displacement).
    """
    steps = [(mu, int(np.sign(disp[mu]))) for mu in range(len(disp)) if disp[mu] != 0]
    k = len(steps)
    if k == 0:
        return 1.0 + 0.0j

    acc = 0.0 + 0.0j
    n_perm = 0
    for order in permutations(steps):
        pos = list(n)
        prod = 1.0 + 0.0j
        for mu, s in order:
            if s > 0:
                prod *= links[tuple(pos) + (mu,)]
                pos[mu] = (pos[mu] + 1) % L[mu]
            else:
                pos[mu] = (pos[mu] - 1) % L[mu]
                prod *= np.conj(links[tuple(pos) + (mu,)])
        acc += prod
        n_perm += 1
    return acc / n_perm


def covariant_stencil_to_matrix(
    stencil: np.ndarray,
    norm: int,
    links: np.ndarray,
    d: int,
) -> np.ndarray:
    """Realize a stencil as a gauge-covariant ``V x V`` matrix with periodic BC.

    With ``links`` all equal to 1 this returns exactly the free-field
    :func:`lattice_dirac_spectra.operators.dirac.stencil_to_matrix`.
    """
    L = tuple(links.shape[:d])
    V = int(np.prod(L))
    strides = _strides(L)
    M = np.zeros((V, V), dtype=complex)

    nonzero = [
        (idx, int(stencil[idx]))
        for idx in np.ndindex(stencil.shape)
        if stencil[idx] != 0
    ]

    for n in iproduct(*[range(Lk) for Lk in L]):
        row = _lin(n, strides)
        for idx, w in nonzero:
            disp = tuple(i - 1 for i in idx)
            target = tuple((n[mu] + disp[mu]) % L[mu] for mu in range(d))
            col = _lin(target, strides)
            T = parallel_transport(links, n, disp, L)
            M[row, col] += (w / norm) * T
    return M


def build_gauged_dirac(
    lap_name: str,
    der_name: str,
    links: np.ndarray,
    m0: float = 0.0,
) -> np.ndarray:
    """Build the gauge-covariant ``(V*Ns) x (V*Ns)`` Dirac matrix on a background.

        D = -1/2 (Delta[U] (x) 1_Ns) + sum_mu (nabla_mu[U] (x) gamma_mu) + m0

    Parameters
    ----------
    lap_name, der_name : str
        Laplacian and derivative stencil keys.
    links : ndarray, shape ``(*L, d)``
        Gauge links (forward, ``links[x, mu] = U_mu(x)``).
    m0 : float, optional
        Bare mass (default 0).

    Returns
    -------
    ndarray, complex
        The dense gauged Dirac matrix.
    """
    d = links.shape[-1]
    L = tuple(links.shape[:d])
    V = int(np.prod(L))
    ns = Ns(d)
    gammas = gamma_matrices(d)

    lap_s, lap_n = get_lap(lap_name, d)
    lap_mat = covariant_stencil_to_matrix(lap_s, lap_n, links, d)
    D = -0.5 * np.kron(lap_mat, np.eye(ns, dtype=complex))

    for mu in range(d):
        der_s, der_n = get_der_mu(der_name, d, mu)
        der_mat = covariant_stencil_to_matrix(der_s, der_n, links, d)
        D += np.kron(der_mat, gammas[mu])

    if m0 != 0.0:
        D += m0 * np.eye(V * ns, dtype=complex)
    return D


def gauge_transform(links: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Apply a U(1) gauge transformation ``U_mu(x) -> g(x) U_mu(x) g(x+e_mu)^*``.

    Parameters
    ----------
    links : ndarray, shape ``(*L, d)``
        Forward links.
    g : ndarray, shape ``(*L,)``
        Unit-modulus gauge function at each site.

    Returns
    -------
    ndarray
        The transformed links (used to test gauge invariance of the spectrum).
    """
    d = links.shape[-1]
    out = np.empty_like(links)
    for mu in range(d):
        g_shift = np.roll(g, -1, axis=mu)  # g(x + e_mu)
        out[..., mu] = g * links[..., mu] * g_shift.conj()
    return out
