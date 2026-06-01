"""
Filesystem path constants.

Resolves the project root and the standard data/output directories relative to
the installed package location, so scripts and notebooks can reference them
without hard-coding absolute paths.
"""

from pathlib import Path

# This file lives at:  <ROOT>/core/lattice_dirac_spectra/constants/paths.py
# so the project root is four parents up.
ROOT: Path = Path(__file__).resolve().parents[3]

DATA_DIR: Path = ROOT / "data_files"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

OUTPUT_DIR: Path = ROOT / "output"
PLOTS_DIR: Path = OUTPUT_DIR / "plots"
TABLES_DIR: Path = OUTPUT_DIR / "tables"
