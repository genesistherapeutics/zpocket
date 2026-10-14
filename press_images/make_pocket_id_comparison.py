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

# (name, developer, model AUPRC, P2Rank AUPRC or None, value labels as reported, ratio label,
#  secondary bar or None)
# The ratio bracket spans the released ESMFold2 model and P2Rank: 0.745 / 0.477 = 1.56, labelled
# "~1.5x" to read as on par with IsoDDE's reported 1.5x rather than as a superiority claim across
# non-comparable test sets. The secondary bar is an internal trunk shown for context only: it is
# drawn in a lighter shade and carries no bracket, so it does not compete with the released bar.
RESULTS = [
    ("zPocket", "Genesis", 0.745, 0.477, ("0.745", "0.477"), "~1.5× P2Rank",
     (0.764, "0.764", "Pearl")),
    ("IsoDDE", "Isomorphic Labs", 0.75, 0.51, ("0.75", "0.51"), "1.5× P2Rank", None),
    ("AnewDDE", "Anew Labs", 0.751, None, ("0.751", None), None, None),
]

MODEL_COLOR = "#2a78d6"
# Deliberately low-contrast: the internal-trunk bar is context, not the headline.
SECONDARY_COLOR = "#aecbee"
P2RANK_COLOR = "#898781"
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
TAG_BG = "#eaf2fc"
TAG_INK = "#1c5cab"

# zPocket's test set is public, so its result can be reproduced; the competitors' sets are not.
# Counts from the paper's Table 1: 1,275 targets (1,562 pockets) in the effective test set.
ZPOCKET_TAG = "Open benchmark (PocketWatch)\nN = 1,275 test proteins"
# The figure has no title, so the caveat sits under the y-axis label.
AXIS_NOTE = "(as reported by each developer)"

BAR_W = 0.28
GAP = 0.02  # surface gap between adjacent bars, in x data units
CORNER_PX = 4

# Layout in inches. Fonts are fixed in points, so a smaller figure reads larger.
# LEGEND_TOP and AXES_TOP are measured down from the top edge; AXES_HEIGHT sets how tall the bars are.
FIG_W = 7.4
LEGEND_TOP, AXES_TOP, AXES_HEIGHT = 0.08, 0.55, 1.45
AXES_BOTTOM, AXES_LEFT, AXES_RIGHT = 0.98, 0.8, 0.12

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
    fig_h = AXES_TOP + AXES_HEIGHT + AXES_BOTTOM
    fig = plt.figure(figsize=(FIG_W, fig_h), facecolor=SURFACE)
    ax = fig.add_axes(
        [AXES_LEFT / FIG_W, AXES_BOTTOM / fig_h, (FIG_W - AXES_LEFT - AXES_RIGHT) / FIG_W, AXES_HEIGHT / fig_h],
        facecolor=SURFACE,
    )
    ax.set_xlim(-0.75, len(RESULTS) - 0.4)  # the zPocket group is three bars wide
    ax.set_ylim(0, 1.0)

    # y data units per point, so labels and brackets keep fixed spacing whatever the bar height.
    pt = 1.0 / (AXES_HEIGHT * 72)
    # Data units per pixel, so corner rounding is circular rather than elliptical.
    bbox = ax.get_window_extent()
    y_per_x = (1.0 / bbox.height) / ((ax.get_xlim()[1] - ax.get_xlim()[0]) / bbox.width)

    offset = (BAR_W + GAP) / 2
    step = BAR_W + GAP
    for i, (name, dev, model, p2rank, (model_lbl, p2_lbl), ratio_lbl, secondary) in enumerate(RESULTS):
        # A group with a secondary bar is three wide and stays centred on the tick, so the released
        # model keeps the middle slot and the bracket below still spans it and P2Rank.
        xm, xp = (i, i + step) if secondary else (i - offset, i + offset)
        if secondary:
            value, value_lbl, bar_lbl = secondary
            xs = i - step
            rounded_bar(ax, xs, value, SECONDARY_COLOR, y_per_x)
            ax.text(xs, value + 2.5 * pt, value_lbl, ha="center", va="bottom", fontsize=10, color=INK_2)
            # Trunk name inside the bar: naming the two model bars is what distinguishes them, and
            # putting it inside avoids a second row of text under the axis.
            ax.text(xs, 6 * pt, bar_lbl, ha="center", va="bottom", fontsize=7.5, color=INK_2, zorder=4)
        rounded_bar(ax, xm, model, MODEL_COLOR, y_per_x)
        ax.text(xm, model + 2.5 * pt, model_lbl, ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")
        if secondary:
            ax.text(xm, 6 * pt, "ESMFold2", ha="center", va="bottom", fontsize=7.5, color=SURFACE, zorder=4)

        if p2rank is None:
            ax.text(xp - BAR_W / 2 + 0.03, 5 * pt, "P2Rank\nnot reported", ha="left", va="bottom", fontsize=7.5, color=INK_MUTED, linespacing=1.3)
            continue

        rounded_bar(ax, xp, p2rank, P2RANK_COLOR, y_per_x)
        ax.text(xp, p2rank + 2.5 * pt, p2_lbl, ha="center", va="bottom", fontsize=10, color=INK_2)

        # Bracket showing the multiple over P2Rank.
        y = model + 22 * pt
        ax.add_line(Line2D([xm, xm, xp, xp], [y - 4 * pt, y, y, y - 4 * pt], color=INK_2, lw=0.9, zorder=4, clip_on=False))
        ax.text((xm + xp) / 2, y + 2 * pt, ratio_lbl, ha="center", va="bottom", fontsize=9.5, color=INK, zorder=4, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1))

    # Group labels: name (bold) over developer (muted), offset in points below the axis.
    ax.set_xticks(range(len(RESULTS)))
    ax.set_xticklabels([""] * len(RESULTS))
    ax.tick_params(axis="x", length=0)

    def below_axis(x, dy, text, **kw):
        ax.annotate(text, xy=(x, 0), xycoords=("data", "axes fraction"), xytext=(0, dy), textcoords="offset points", ha="center", va="top", **kw)

    for i, (name, dev, *_) in enumerate(RESULTS):
        below_axis(i, -8, name, fontsize=11.5, fontweight="bold", color=INK)
        below_axis(i, -22, dev, fontsize=8.5, color=INK_2)
    below_axis(
        0,
        -38,
        ZPOCKET_TAG,
        fontsize=8.5,
        color=TAG_INK,
        linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.6", facecolor=TAG_BG, edgecolor="none"),
    )

    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
    ax.tick_params(axis="y", length=0, pad=6)
    # Rotated text reads bottom to top, so the line "below" the label sits to its right.
    for text, dx, kw in (("AUPRC", -47, dict(fontsize=10, color=INK_2)), (AXIS_NOTE, -33, dict(fontsize=7.5, color=INK_MUTED))):
        ax.annotate(text, xy=(0, 0.5), xycoords="axes fraction", xytext=(dx, 0), textcoords="offset points", rotation=90, ha="center", va="center", **kw)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)

    fig.legend(
        handles=[
            Patch(facecolor=MODEL_COLOR, label="Cofolding-based pocket model"),
            Patch(facecolor=P2RANK_COLOR, label="P2Rank on the same predicted structures"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1 - LEGEND_TOP / fig_h),
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
