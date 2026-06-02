"""
Gauge observables.

Plaquette and (2D) topological charge of a lattice gauge configuration, used to
cross-check the configuration reader against the metadata stored by the
generator (e.g. ``u1-hmc``).

Convention
----------
The link array has shape ``(*L, d)`` and ``links[x, mu]`` is the forward link
``U_mu(x)`` connecting site ``x`` to ``x + e_mu`` along **lattice axis** ``mu``.
The plaquette in the ``(mu, nu)`` plane is

    P_{mu nu}(n) = U_mu(n) U_nu(n + e_mu) U_mu(n + e_nu)^dagger U_nu(n)^dagger.

For U(1), ``U_mu(x)`` is a unit-modulus complex number and ``^dagger`` is complex
conjugation.
"""

from typing import Tuple

import numpy as np

__all__ = ["plaquette_field", "plaquette", "topological_charge"]


def _plaquette_plane(links: np.ndarray, mu: int, nu: int) -> np.ndarray:
    """Per-site plaquette ``P_{mu nu}(n)`` (complex), via periodic shifts."""
    U_mu = links[..., mu]
    U_nu = links[..., nu]
    U_nu_shift_mu = np.roll(U_nu, -1, axis=mu)  # U_nu(n + e_mu)
    U_mu_shift_nu = np.roll(U_mu, -1, axis=nu)  # U_mu(n + e_nu)
    return U_mu * U_nu_shift_mu * U_mu_shift_nu.conj() * U_nu.conj()


def plaquette_field(links: np.ndarray, mu: int = 0, nu: int = 1) -> np.ndarray:
    """Complex plaquette ``P_{mu nu}(n)`` at every site."""
    return _plaquette_plane(links, mu, nu)


def plaquette(links: np.ndarray) -> float:
    """Mean plaquette ``<Re P>`` averaged over all sites and all ``mu < nu`` planes.

    For ``d = 2`` there is a single plane ``(0, 1)``, matching the standard
    scalar plaquette.
    """
    d = links.shape[-1]
    planes = [(mu, nu) for mu in range(d) for nu in range(mu + 1, d)]
    total = 0.0
    for mu, nu in planes:
        total += _plaquette_plane(links, mu, nu).real.mean()
    return float(total / len(planes))


def topological_charge(links: np.ndarray) -> float:
    """Topological charge ``Q = (1/2pi) sum_n arg P_{01}(n)`` (2D only).

    Uses the geometric (field-theoretic) definition via the principal argument
    of the plaquette. Defined for ``d = 2``.
    """
    if links.shape[-1] != 2:
        raise ValueError(
            f"topological_charge is defined for d=2, got d={links.shape[-1]}."
        )
    P = _plaquette_plane(links, 0, 1)
    return float(np.arctan2(P.imag, P.real).sum() / (2 * np.pi))


def lattice_shape(links: np.ndarray) -> Tuple[int, ...]:
    """Return the lattice extents ``L`` (the link array shape without the last axis)."""
    return tuple(links.shape[:-1])
