"""
Visualization styling constants.

Shared colors, marker styles, and Brillouin-zone plotting defaults used across
the visualization subsystem.
"""

# Consistent colors for the two reference kernels and for recipe-vs-direct overlays.
DEFAULT_COLORS = {
    "Wilson": "C0",
    "Brillouin": "C1",
    "formula": "red",
    "direct": "blue",
    "continuum": "black",
}

# Marker styles for spectrum scatter overlays.
MARKER_STYLES = {
    "formula": "x",
    "direct": "+",
}

# Defaults for Brillouin-zone surface/contour plots.
BZ_SURFACE_DEFAULTS = {
    "grid_size": 48,  # samples per axis over (-pi, pi]
    "cmap": "viridis",
    "figsize": (12, 5),
}
