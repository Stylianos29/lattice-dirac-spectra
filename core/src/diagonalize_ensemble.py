#!/usr/bin/env python3
"""
Diagonalize the gauged Dirac operator over a U(1) ensemble and plot the spectrum.

Examples
--------
    python core/src/diagonalize_ensemble.py \
        --file core/tests/mock_data/u1_L8_beta1.000.h5 --beta 1.0 \
        --kernel Wilson --n-configs 5

    python core/src/diagonalize_ensemble.py \
        --file data_files/raw/u1_L16_beta2.000.h5 --beta 2.0 \
        --kernel Brillouin --n-ape 3 -o output/plots/gauged_brillouin.png
"""

from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lattice_dirac_spectra.constants.operators import operator_keys  # noqa: E402
from lattice_dirac_spectra.gauge.u1_config import load_ensemble  # noqa: E402
from lattice_dirac_spectra.spectra.gauged import gauged_spectrum  # noqa: E402
from lattice_dirac_spectra.visualization.spectrum_plotter import (
    plot_spectrum,
)  # noqa: E402


@click.command()
@click.option("--file", "path", required=True, help="HDF5 ensemble path.")
@click.option("--beta", required=True, type=float, help="Gauge coupling.")
@click.option("--kernel", default="Wilson", help="Wilson or Brillouin.")
@click.option("--mass", "m0", default=0.0, show_default=True, help="Bare mass am0.")
@click.option("--n-ape", default=0, show_default=True, help="APE smearing steps.")
@click.option(
    "--n-configs", default=5, show_default=True, help="How many configs to stack."
)
@click.option("-o", "--output", "out", default=None, help="Output PNG path.")
def main(path, beta, kernel, m0, n_ape, n_configs, out):
    """Stack gauged spectra over the first N configurations and scatter-plot them."""
    der_name, lap_name = operator_keys(kernel)
    chunks = []
    for cfg in load_ensemble(path, beta, pick=slice(0, n_configs)):
        chunks.append(gauged_spectrum(lap_name, der_name, cfg, m0=m0, n_ape=n_ape))
    eigs = np.concatenate(chunks)

    fig, ax = plot_spectrum(eigs, size=3.0)
    ax.set_title(
        f"{kernel} gauged spectrum  (beta={beta}, {len(chunks)} configs, n_ape={n_ape})"
    )
    tag = f"{kernel}_beta{beta}_nape{n_ape}"
    out_path = Path(out) if out else Path("output/plots") / f"gauged_{tag}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    click.echo(
        f"diagonalized {len(chunks)} configs, {len(eigs)} eigenvalues -> {out_path}"
    )


if __name__ == "__main__":
    main()
