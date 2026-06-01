#!/usr/bin/env python3
"""
Verify the free-field golden invariant from the command line.

For every (Laplacian, derivative) combination, print the maximum nearest-neighbour
distance between the recipe spectrum and the direct-diagonalization spectrum.
A small value (~1e-13) confirms the two agree.

Example
-------
    python core/src/verify_free_field.py --dim 2 --length 8
"""

import click

from lattice_dirac_spectra.constants import LAP_KEYS, DER_KEYS
from lattice_dirac_spectra.spectra.free_field import (
    eigenvalues_from_formula,
    eigenvalues_from_matrix,
    compare_eigenvalues,
)


@click.command()
@click.option(
    "--dim", "d", default=2, show_default=True, help="Lattice dimensionality."
)
@click.option(
    "--length", "-L", "L", default=8, show_default=True, help="Linear extent."
)
def main(d: int, L: int) -> None:
    """Print recipe-vs-diagonalization agreement for all 12 operators."""
    header = f"{'Operator':<22s}{'max |err|':>14s}"
    click.echo(header)
    click.echo("-" * len(header))
    worst = 0.0
    for lap in LAP_KEYS:
        for der in DER_KEYS:
            ef = eigenvalues_from_formula(lap, der, d, L)
            ed = eigenvalues_from_matrix(lap, der, d, L)
            err = compare_eigenvalues(ef, ed)
            worst = max(worst, err)
            click.echo(f"{'lap_' + lap + ', der_' + der:<22s}{err:>14.2e}")
    click.echo("-" * len(header))
    status = "PASS" if worst < 1e-10 else "FAIL"
    click.echo(f"worst case: {worst:.2e}  [{status}]")


if __name__ == "__main__":
    main()
