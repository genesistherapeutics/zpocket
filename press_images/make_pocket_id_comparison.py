"""Press figure: zPocket pocket identification vs. IsoDDE and AnewDDE.

Each model's AUPRC comes from its developers' own report on their own held-out
test set, so this is an approximate comparison, not a head-to-head benchmark.
See README.md for sources.

Usage: python make_pocket_id_comparison.py  (writes PNG, PDF and SVG next to this file)
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath

OUT_DIR = Path(__file__).resolve().parent
STEM = "pocket_id_comparison"

# (name, developer, model AUPRC, P2Rank AUPRC or None, value labels as reported, ratio label)
# zPocket's ratio is 0.745 / 0.477 = 1.56; it is labelled "~1.5x" to read as on par with IsoDDE's
# reported 1.5x rather than as a superiority claim across non-comparable test sets.
RESULTS = [
    ("zPocket", "Genesis · ESMFold2 trunk", 0.745, 0.477, ("0.745", "0.477"), "~1.5× P2Rank"),
    ("IsoDDE", "Isomorphic Labs", 0.75, 0.51, ("0.75", "0.51"), "1.5× P2Rank"),
    ("AnewDDE", "Anew Labs", 0.751, None, ("0.751", None), None),
]

MODEL_COLOR = "#2a78d6"
P2RANK_COLOR = "#898781"
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

BAR_W = 0.28
FIG_W, FIG_H = 6.2, 3.9  # inches; fonts are fixed in points, so a smaller figure reads larger

# Layout in inches from the figure edges, so text spacing survives figure resizes.
TITLE_TOP, SUBTITLE_TOP, LEGEND_TOP, AXES_TOP = 0.2, 0.5, 0.67, 1.1
AXES_BOTTOM, AXES_LEFT, AXES_RIGHT = 0.58, 0.62, 0.12
GAP = 0.02  # surface gap between adjacent bars, in x data units
CORNER_PX = 4

plt.rcParams.update(
    {
        "font.family": ["Liberation Sans", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.edgecolor": BASELINE,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
    }
)


def rounded_bar(ax, x_center, height, color, y_per_x):
    """Bar with rounded top corners and a square base on the zero line, as one path (no seams)."""
    x0, x1 = x_center - BAR_W / 2, x_center + BAR_W / 2
    rx = CORNER_PX * (ax.get_xlim()[1] - ax.get_xlim()[0]) / ax.get_window_extent().width
    ry = rx * y_per_x
    verts = [
        (x0, 0), (x0, height - ry),
        (x0, height), (x0 + rx, height),
        (x1 - rx, height),
        (x1, height), (x1, height - ry),
        (x1, 0), (x0, 0),
    ]
    codes = [
        MplPath.MOVETO, MplPath.LINETO,
        MplPath.CURVE3, MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3, MplPath.CURVE3,
        MplPath.LINETO, MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def main():
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=SURFACE)
    ax = fig.add_axes(
        [
            AXES_LEFT / FIG_W,
            AXES_BOTTOM / FIG_H,
            (FIG_W - AXES_LEFT - AXES_RIGHT) / FIG_W,
            (FIG_H - AXES_TOP - AXES_BOTTOM) / FIG_H,
        ],
        facecolor=SURFACE,
    )
    ax.set_xlim(-0.6, len(RESULTS) - 0.4)
    ax.set_ylim(0, 1.0)

    # Data units per pixel, so corner rounding is circular rather than elliptical.
    bbox = ax.get_window_extent()
    y_per_x = (1.0 / bbox.height) / ((ax.get_xlim()[1] - ax.get_xlim()[0]) / bbox.width)

    offset = (BAR_W + GAP) / 2
    for i, (name, dev, model, p2rank, (model_lbl, p2_lbl), ratio_lbl) in enumerate(RESULTS):
        xm, xp = i - offset, i + offset
        rounded_bar(ax, xm, model, MODEL_COLOR, y_per_x)
        ax.text(xm, model + 0.015, model_lbl, ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")

        if p2rank is None:
            ax.text(xp - BAR_W / 2 + 0.03, 0.03, "P2Rank\nnot reported", ha="left", va="bottom", fontsize=7.5, color=INK_MUTED, linespacing=1.3)
            continue

        rounded_bar(ax, xp, p2rank, P2RANK_COLOR, y_per_x)
        ax.text(xp, p2rank + 0.015, p2_lbl, ha="center", va="bottom", fontsize=10, color=INK_2)

        # Bracket showing the multiple over P2Rank.
        y = model + 0.14
        ax.add_line(Line2D([xm, xm, xp, xp], [y - 0.025, y, y, y - 0.025], color=INK_2, lw=0.9, zorder=4))
        ax.text(i, y + 0.012, ratio_lbl, ha="center", va="bottom", fontsize=9.5, color=INK)

    # Two-line group labels: name (bold) over developer (muted).
    ax.set_xticks(range(len(RESULTS)))
    ax.set_xticklabels([""] * len(RESULTS))
    ax.tick_params(axis="x", length=0)
    for i, (name, dev, *_) in enumerate(RESULTS):
        ax.text(i, -0.05, name, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=11.5, fontweight="bold", color=INK)
        ax.text(i, -0.13, dev, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.5, color=INK_2)

    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.set_ylabel("AUPRC", fontsize=10, color=INK_2, labelpad=8)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)

    fig.text(0.5, 1 - TITLE_TOP / FIG_H, "Pocket identification from cofolding models", fontsize=14, fontweight="bold", color=INK, ha="center", va="top")
    fig.text(
        0.5,
        1 - SUBTITLE_TOP / FIG_H,
        "AUPRC for ranking ligand-binding residues on held-out proteins, as reported by each developer",
        fontsize=9.5,
        color=INK_2,
        ha="center",
        va="top",
    )
    fig.legend(
        handles=[
            Patch(facecolor=MODEL_COLOR, label="Cofolding-based pocket model"),
            Patch(facecolor=P2RANK_COLOR, label="P2Rank on the same predicted structures"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1 - LEGEND_TOP / FIG_H),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        handlelength=1.0,
        handleheight=1.0,
        columnspacing=1.6,
        labelcolor=INK_2,
    )

    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{STEM}.{ext}", dpi=300, facecolor=SURFACE)
    print(f"wrote {OUT_DIR / STEM}.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
