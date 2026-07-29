"""Combine all 5 per-algorithm learning-curve charts into one figure: 3 on top,
2 on the bottom row, with the empty 3rd bottom slot used for a single shared
legend instead of repeating it in every panel.
"""
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from plot_learning_curves import (
    ALGOS,
    ALGO_TITLES,
    APPROACHES,
    BASELINE_AXIS,
    GRIDLINE,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
    compute_normalized_curves,
    format_x,
)

OUT_DIR = # Output directory for the combined figure
OUT_PATH = os.path.join(OUT_DIR, "combined_grid.png")

GRID_LAYOUT = [
    ["ia2c", "maa2c", "itrpo"],
    ["matrpo", "hatrpo", None],  # None -> legend panel
]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_x))
    ax.tick_params(axis="both", labelsize=18, colors=INK_SECONDARY, width=1.3, length=5)
    ax.grid(True, color=GRIDLINE, linewidth=1.0)
    ax.set_axisbelow(True)
    for spine_name, spine in ax.spines.items():
        if spine_name in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_color(BASELINE_AXIS)
            spine.set_linewidth(1.3)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), dpi=200)
    fig.patch.set_facecolor(SURFACE)

    handles, labels = None, None

    for row_idx, row in enumerate(GRID_LAYOUT):
        for col_idx, algo in enumerate(row):
            ax = axes[row_idx][col_idx]

            if algo is None:
                ax.axis("off")
                continue

            grid, normalized, _ = compute_normalized_curves(algo)

            for approach_folder, label, color in APPROACHES:
                line, = ax.plot(
                    grid, normalized[approach_folder],
                    label=label, color=color, linewidth=3, solid_capstyle="round",
                )

            if handles is None:
                handles, labels = ax.get_legend_handles_labels()

            style_axes(ax)
            ax.set_title(ALGO_TITLES[algo], fontsize=26, fontweight="bold", color=INK_PRIMARY, pad=10)
            ax.set_xlabel("Training timesteps", fontsize=20, color=INK_SECONDARY, labelpad=8)
            ax.set_ylabel("Normalized reward mean", fontsize=20, color=INK_SECONDARY, labelpad=8)

    legend_ax = axes[1][2]
    legend_ax.legend(
        handles, labels,
        frameon=False,
        fontsize=26,
        labelcolor=INK_SECONDARY,
        loc="center",
        handlelength=2.5,
        handleheight=2.0,
    )

    fig.tight_layout()
    fig.savefig(OUT_PATH, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
