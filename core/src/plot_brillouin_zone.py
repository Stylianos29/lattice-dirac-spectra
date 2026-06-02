#!/usr/bin/env python3
"""
Plot Brillouin-zone surfaces of the stencil Fourier representations.

Examples
--------
    python core/src/plot_brillouin_zone.py --lap bri --kind surface
    python core/src/plot_brillouin_zone.py --der iso --kind contour -o output/plots/der_iso.png
"""

from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")  # headless / file output
import matplotlib.pyplot as plt  # noqa: E402

from lattice_dirac_spectra.visualization.brillouin_zone import (  # noqa: E402
    plot_laplacian_surface,
    plot_derivative_surface,
)


@click.command()
@click.option("--lap", "lap_name", default=None, help="Laplacian key (std/til/bri/iso).")
@click.option("--der", "der_name", default=None, help="Derivative key (std/bri/iso).")
@click.option("--mu", default=0, show_default=True, help="Derivative direction (for --der).")
@click.option("--dim", "d", default=2, show_default=True, help="Lattice dimensionality.")
@click.option("--grid", "grid_size", default=48, show_default=True, help="Samples per axis.")
@click.option("--kind", default="surface", type=click.Choice(["surface", "contour"]), show_default=True)
@click.option("-o", "--output", "out", default=None, help="Output PNG path.")
def main(lap_name, der_name, mu, d, grid_size, kind, out):
    """Render a Laplacian (--lap) or derivative (--der) Brillouin-zone surface."""
    if (lap_name is None) == (der_name is None):
        raise click.UsageError("Provide exactly one of --lap or --der.")

    if lap_name is not None:
        fig, _ = plot_laplacian_surface(lap_name, d=d, grid_size=grid_size, kind=kind)
        default_name = f"lap_{lap_name}_{kind}.png"
    else:
        fig, _ = plot_derivative_surface(der_name, mu=mu, d=d, grid_size=grid_size, kind=kind)
        default_name = f"der_{der_name}_mu{mu}_{kind}.png"

    out_path = Path(out) if out else Path("output/plots") / default_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"wrote {out_path}")


if __name__ == "__main__":
    main()
