#!/usr/bin/env python3
"""
Plot free-field dispersion relations for a named or explicit operator.

Examples
--------
    python core/src/plot_dispersion.py --kernel Brillouin --dim 4 --mass 0.0
    python core/src/plot_dispersion.py --lap std --der std --dim 4 --mass 0.75
    python core/src/plot_dispersion.py --kernel Wilson --dim 4 --mass 0.75 --heavy-quark
"""

from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lattice_dirac_spectra.constants.operators import operator_keys  # noqa: E402
from lattice_dirac_spectra.visualization.dispersion import plot_dispersion  # noqa: E402


@click.command()
@click.option("--kernel", default=None, help="Named kernel: Wilson or Brillouin.")
@click.option("--lap", "lap_name", default=None, help="Laplacian key (overrides --kernel).")
@click.option("--der", "der_name", default=None, help="Derivative key (overrides --kernel).")
@click.option("--dim", "d", default=4, show_default=True, help="Lattice dimensionality.")
@click.option("--mass", "m", default=0.0, show_default=True, help="Bare mass am.")
@click.option("--heavy-quark", is_flag=True, help="Use the am -> exp(am)-1 substitution.")
@click.option("--pmax", default=5.0, show_default=True, help="Max |ap| on the x-axis.")
@click.option("-o", "--output", "out", default=None, help="Output PNG path.")
def main(kernel, lap_name, der_name, d, m, heavy_quark, pmax, out):
    """Plot dispersion curves along the canonical directions vs the continuum."""
    if lap_name is None or der_name is None:
        if kernel is None:
            kernel = "Brillouin"
        der_name, lap_name = operator_keys(kernel)

    fig, _ = plot_dispersion(
        lap_name, der_name, d=d, P_max=pmax, m=m, heavy_quark=heavy_quark
    )
    tag = f"{lap_name}_{der_name}_d{d}_m{m}{'_hq' if heavy_quark else ''}"
    out_path = Path(out) if out else Path("output/plots") / f"dispersion_{tag}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"wrote {out_path}")


if __name__ == "__main__":
    main()
