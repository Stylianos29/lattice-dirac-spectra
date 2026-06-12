"""Synthetic gauge backgrounds (no Monte Carlo).

Trivial and (later) perturbed U(1) link fields for validating the gauged
operator against the free-field limit and for cheap exploratory spectra where
no thermalized ensemble is available.
"""

import numpy as np  # type: ignore

__all__ = ["unit_links", "generate_heatbath_ensemble"]


def unit_links(d: int, L: int) -> np.ndarray:
    """All-ones (free-field) U(1) links of shape ``(L,) * d + (d,)``."""
    return np.ones((L,) * d + (d,), dtype=complex)


from scipy.special import i0, i1  # 2D eps<->beta convenience only  # type: ignore

__all__ = [
    "unit_links",
    "random_phase_links",
    "eps_for_plaquette",
    "plaquette_2d",
    "heatbath_links",
]


# ----------------------------------------------------------------------------- #
# Method A -- independent random link phases (fast, not Boltzmann-distributed)
# ----------------------------------------------------------------------------- #
def random_phase_links(d, L, eps, rng=None):
    """U_mu(x) = exp(i * eps * theta), theta ~ N(0,1) i.i.d. (any dimension).

    Mean plaquette ``P = exp(-2*eps**2)`` independent of ``d``. As ``eps -> 0``
    this returns the free field. The link-phase distribution is Gaussian, not
    the Boltzmann weight, and there are no spatial correlations -- a controlled
    "semblance" of a gauge background, not a thermalized ensemble.
    """
    rng = np.random.default_rng() if rng is None else rng
    theta = rng.normal(0.0, eps, size=(L,) * d + (d,))
    return np.exp(1j * theta)


def eps_for_plaquette(P):
    """epsilon giving mean plaquette ``P`` for :func:`random_phase_links`."""
    return float(np.sqrt(-np.log(P) / 2.0))


def plaquette_2d(beta):
    """Exact 2D U(1) mean plaquette ``I1(beta)/I0(beta)`` (the eps<->beta bridge)."""
    return float(i1(beta) / i0(beta))


# ----------------------------------------------------------------------------- #
# Method B -- red-black von Mises heatbath (genuinely thermalized, any dimension)
# ----------------------------------------------------------------------------- #
def _heatbath_staples(links, mu):
    """Complex staple sum A_mu(n) for the heatbath.

    Defined so the plaquettes through ``U_mu(n)`` equal ``Re[U_mu(n) * conj(A)]``.
    (This is the heatbath/action convention; it is intentionally distinct from
    the APE-smearing staple in ``smearing.py``, which uses a different grouping.)
    """
    d = links.shape[-1]
    Umu = links[..., mu]
    A = np.zeros_like(Umu)
    for nu in range(d):
        if nu == mu:
            continue
        Unu = links[..., nu]
        # forward staple
        A += np.roll(Unu, -1, axis=mu).conj() * np.roll(Umu, -1, axis=nu) * Unu
        # backward staple
        Unu_b = np.roll(Unu, 1, axis=nu)
        A += np.roll(Umu, 1, axis=nu) * np.roll(Unu_b, -1, axis=mu) * Unu_b.conj()
    return A


def heatbath_links(d, L, beta, n_sweeps=200, rng=None, start="unit"):
    """Thermalized U(1) links at coupling ``beta`` via red-black von Mises heatbath.

    Dimension-general. In 2D the mean plaquette matches ``I1(beta)/I0(beta)``;
    for ``d >= 3`` it is the (higher) coupled-plaquette value. Sweeps are cheap
    and vectorized; the staple sum is recomputed for each (direction, parity)
    sub-update so that detailed balance holds.

    Parameters
    ----------
    n_sweeps : int
        Full lattice sweeps (thermalization). For an ensemble, call repeatedly
        with a handful of extra sweeps between draws to decorrelate.
    start : {"unit", "hot"}
        Initial field: ordered (links = 1) or random.
    """
    rng = np.random.default_rng() if rng is None else rng
    shape = (L,) * d
    if start == "unit":
        links = np.ones(shape + (d,), dtype=complex)
    elif start == "hot":
        links = np.exp(1j * rng.uniform(0.0, 2 * np.pi, size=shape + (d,)))
    else:
        raise ValueError("start must be 'unit' or 'hot'.")

    parity = np.indices(shape).sum(axis=0) % 2
    for _ in range(n_sweeps):
        for mu in range(d):
            for p in (0, 1):
                A = _heatbath_staples(links, mu)  # recompute each sub-update
                R = np.abs(A)
                phi = np.angle(A)
                m = parity == p
                links[..., mu][m] = np.exp(
                    1j * (rng.vonmises(0.0, beta * R[m]) + phi[m])
                )
    return links


def _heatbath_sweep(links, beta, parity, rng):
    """One full red-black sweep over every direction (in place)."""
    for mu in range(links.shape[-1]):
        for p in (0, 1):
            A = _heatbath_staples(links, mu)  # recompute per sub-update
            R = np.abs(A)
            phi = np.angle(A)
            m = parity == p
            links[..., mu][m] = np.exp(1j * (rng.vonmises(0.0, beta * R[m]) + phi[m]))
    return links


# def heatbath_links(d, L, beta, n_sweeps=200, rng=None, start="unit"):
#     """(unchanged behavior) thermalize a fresh field for n_sweeps and return it."""
#     rng = np.random.default_rng() if rng is None else rng
#     shape = (L,) * d
#     if start == "unit":
#         links = np.ones(shape + (d,), dtype=complex)
#     elif start == "hot":
#         links = np.exp(1j * rng.uniform(0.0, 2 * np.pi, size=shape + (d,)))
#     else:
#         raise ValueError("start must be 'unit' or 'hot'.")
#     parity = np.indices(shape).sum(axis=0) % 2
#     for _ in range(n_sweeps):
#         _heatbath_sweep(links, beta, parity, rng)
#     return links


def generate_heatbath_ensemble(
    path, d, L, beta, n_configs, *, n_therm=200, n_decorr=20, rng=None, start="unit"
):
    """Thermalize, then write ``n_configs`` decorrelated configs to an HDF5 file.

    Thermalizes for ``n_therm`` sweeps, then stores a configuration every
    ``n_decorr`` sweeps via :func:`save_config` under
    ``beta{beta:5.3f}/traj{i:08.0f}``. Returns the trajectory names written.
    """
    from .u1_config import save_config  # local import keeps h5py out of module import

    rng = np.random.default_rng() if rng is None else rng
    shape = (L,) * d
    if start == "unit":
        links = np.ones(shape + (d,), dtype=complex)
    else:
        links = np.exp(1j * rng.uniform(0.0, 2 * np.pi, size=shape + (d,)))
    parity = np.indices(shape).sum(axis=0) % 2

    for _ in range(n_therm):
        _heatbath_sweep(links, beta, parity, rng)

    trajs = []
    for i in range(n_configs):
        if i > 0:
            for _ in range(n_decorr):
                _heatbath_sweep(links, beta, parity, rng)
        save_config(path, links, beta=beta, traj_index=i, mode=("w" if i == 0 else "a"))
        trajs.append("traj{:08.0f}".format(i))
    return trajs
