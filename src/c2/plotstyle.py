"""Shared matplotlib styling for C2 figures (static PNGs for slides)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# categorical slots, fixed order (validated reference palette)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_COLOR = {"pythia410m": SERIES[0], "qwen25_05b": SERIES[1]}

# diverging blue <-> red, neutral gray midpoint (for signed Δ heatmaps)
DIVERGING = LinearSegmentedColormap.from_list(
    "c2_div", ["#104281", "#3987e5", "#cde2fb", "#f0efec", "#f9d3d1", "#e34948", "#8c1d1d"]
)


def apply() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2,
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "legend.frameon": False,
    })


def pilot_title(fig, title: str, n_sets: int) -> None:
    """Every pilot figure carries the dev-pilot label and its matched-set count."""
    from . import PILOT_LABEL

    fig.suptitle(f"{PILOT_LABEL}\n{title}  (n = {n_sets} matched sets)",
                 color="#d03b3b", fontsize=11, fontweight="bold")
