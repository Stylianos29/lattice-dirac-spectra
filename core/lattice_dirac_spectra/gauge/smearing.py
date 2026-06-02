"""
U(1) APE link smearing.

Dimension-general APE smearing for U(1) gauge fields: replace each link by a
weighted average of itself and the sum of its staples, then project back onto
U(1) by dividing out the modulus. Smearing reduces ultraviolet fluctuations and
improves the locality and chiral properties of the resulting overlap operator
(the overlap-kernel studies in the literature use APE-smeared links).

Convention matches the rest of the package: ``links[x, mu]`` is the forward link
``U_mu(x)`` along lattice axis ``mu``.
"""

import numpy as np

__all__ = ["ape_smear"]


def _staple_sum(links: np.ndarray, mu: int) -> np.ndarray:
    """Sum of forward and backward staples for direction ``mu`` at every site."""
    d = links.shape[-1]
    total = np.zeros(links[..., mu].shape, dtype=complex)
    for nu in range(d):
        if nu == mu:
            continue
        U_mu = links[..., mu]
        U_nu = links[..., nu]
        # Forward staple:  U_nu(n) U_mu(n+e_nu) U_nu(n+e_mu)^*
        fwd = U_nu * np.roll(U_mu, -1, axis=nu) * np.roll(U_nu, -1, axis=mu).conj()
        # Backward staple: U_nu(n-e_nu)^* U_mu(n-e_nu) U_nu(n-e_nu+e_mu)
        U_nu_bm = np.roll(U_nu, +1, axis=nu)
        U_mu_bm = np.roll(U_mu, +1, axis=nu)
        U_nu_bm_pm = np.roll(np.roll(U_nu, +1, axis=nu), -1, axis=mu)
        bwd = U_nu_bm.conj() * U_mu_bm * U_nu_bm_pm
        total += fwd + bwd
    return total


def ape_smear(links: np.ndarray, alpha: float = 0.72, n_steps: int = 1) -> np.ndarray:
    """Apply ``n_steps`` of U(1) APE smearing with parameter ``alpha``.

    Each step performs ``U_mu(x) -> Proj_U(1)[ (1-alpha) U_mu(x) + alpha * staples ]``.
    ``alpha = 0`` (or ``n_steps = 0``) is the identity.

    Parameters
    ----------
    links : ndarray, shape ``(*L, d)``
        Input links (unit modulus).
    alpha : float
        Smearing weight on the staple sum.
    n_steps : int
        Number of smearing iterations.

    Returns
    -------
    ndarray
        The smeared links (unit modulus preserved).
    """
    d = links.shape[-1]
    out = links.copy()
    for _ in range(n_steps):
        new = np.empty_like(out)
        for mu in range(d):
            blended = (1.0 - alpha) * out[..., mu] + alpha * _staple_sum(out, mu)
            modulus = np.abs(blended)
            # Avoid division by zero (staples can momentarily cancel).
            modulus = np.where(modulus < 1e-15, 1.0, modulus)
            new[..., mu] = blended / modulus
        out = new
    return out
