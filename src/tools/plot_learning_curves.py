"""Build per-algorithm learning-curve comparisons across the 3 approaches.

For each algorithm:
  1. Load the 5-seed parquet files for each approach.
  2. Interpolate each seed's episode_reward_mean onto a common timesteps_total grid
     (shared across all 3 approaches for that algorithm, so no extrapolation is needed).
  3. Average across seeds -> one mean-reward curve per approach.
  4. Smooth with a Gaussian filter (sigma=3), matching wesley's existing charts pipeline.
  5. Min-max normalize the 3 smoothed curves together (shared vmin/vmax per algorithm),
     so the 3 lines stay on a directly comparable 0-1 scale.
  6. Plot the 3 lines and save one PNG per algorithm.
"""
import os

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DATA_ROOT = # Data root with RL execution data converted to parquet
OUT_DIR = # Output directory for the per-algorithm learning-curve charts

ALGOS = ["ia2c", "maa2c", "itrpo", "matrpo", "hatrpo"]
ALGO_TITLES = {
    "ia2c": "IA2C",
    "maa2c": "MAA2C",
    "itrpo": "ITRPO",
    "matrpo": "MATRPO",
    "hatrpo": "HATRPO",
}

# fixed order + labels + palette (dataviz skill categorical slots 1-3, light mode)
APPROACHES = [
    ("baseline", "Baseline", "#2a78d6"),
    ("timesfm", "Gaussian-Based Occupancy", "#eb6834"),
    ("tfm_Prev3", "Gaussian-based + Forecasting", "#1baf7a"),
]

SMOOTH_SIGMA = 3.0
GRID_POINTS = 300

# chart chrome (dataviz skill, light mode)
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def load_seed_curves(algo, approach_folder):
    folder = os.path.join(DATA_ROOT, algo, approach_folder)
    curves = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".parquet"):
            continue
        df = pd.read_parquet(os.path.join(folder, fname), columns=["timesteps_total", "episode_reward_mean"])
        df = df.dropna(subset=["episode_reward_mean"]).sort_values("timesteps_total")
        df = df.drop_duplicates(subset="timesteps_total")
        if len(df) < 2:
            continue
        curves.append((df["timesteps_total"].to_numpy(dtype=float), df["episode_reward_mean"].to_numpy(dtype=float)))
    return curves


def aggregate_approach(curves, grid):
    """Interpolate each seed curve onto grid (only within its own range) and average, ignoring points beyond a seed's range."""
    stacked = np.full((len(curves), len(grid)), np.nan)
    for i, (x, y) in enumerate(curves):
        interp = np.interp(grid, x, y, left=np.nan, right=np.nan)
        stacked[i] = interp
    return np.nanmean(stacked, axis=0)


def format_x(x, pos):
    if x == 0:
        return "0"
    return f"{x / 1e3:.0f}k"


def compute_normalized_curves(algo):
    """Returns (grid, {approach_folder: normalized_curve}, grid_max)."""
    per_approach_curves = {}
    min_ts = []
    max_ts = []
    for approach_folder, _, _ in APPROACHES:
        curves = load_seed_curves(algo, approach_folder)
        per_approach_curves[approach_folder] = curves
        min_ts.append(max(x.min() for x, _ in curves))
        max_ts.append(min(x.max() for x, _ in curves))

    # keep the grid inside every seed's data range across all 3 approaches,
    # so no seed needs extrapolation and no grid point is all-NaN
    grid_min = max(min_ts)
    grid_max = min(max_ts)
    grid = np.linspace(grid_min, grid_max, GRID_POINTS)

    # each curve normalized independently (own min-max -> 0-1): reward scale is
    # not comparable in magnitude across approaches here, only the learning trend is
    normalized = {}
    for approach_folder, _, _ in APPROACHES:
        mean_curve = aggregate_approach(per_approach_curves[approach_folder], grid)
        smooth_curve = gaussian_filter1d(mean_curve, sigma=SMOOTH_SIGMA)
        vmin, vmax = np.nanmin(smooth_curve), np.nanmax(smooth_curve)
        denom = (vmax - vmin) if (vmax - vmin) > 1e-12 else 1.0
        normalized[approach_folder] = (smooth_curve - vmin) / denom

    return grid, normalized, grid_max


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for algo in ALGOS:
        grid, normalized, grid_max = compute_normalized_curves(algo)

        fig, ax = plt.subplots(figsize=(8, 6.5), dpi=200)
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)

        for approach_folder, label, color in APPROACHES:
            ax.plot(grid, normalized[approach_folder], label=label, color=color, linewidth=3.5, solid_capstyle="round")

        ax.set_xlabel("Training timesteps", fontsize=26, color=INK_SECONDARY, labelpad=12)
        ax.set_ylabel("Normalized reward mean", fontsize=26, color=INK_SECONDARY, labelpad=12)
        ax.set_ylim(-0.02, 1.02)

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_x))
        ax.tick_params(axis="both", labelsize=22, colors=INK_SECONDARY, width=1.5, length=6)

        ax.grid(True, color=GRIDLINE, linewidth=1.2)
        ax.set_axisbelow(True)
        for spine_name, spine in ax.spines.items():
            if spine_name in ("top", "right"):
                spine.set_visible(False)
            else:
                spine.set_color(BASELINE_AXIS)
                spine.set_linewidth(1.5)

        legend = ax.legend(
            frameon=False,
            fontsize=22,
            labelcolor=INK_SECONDARY,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.02),
            borderaxespad=0,
        )

        out_path = os.path.join(OUT_DIR, f"{algo}.png")
        fig.tight_layout()
        fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path} (grid_max={grid_max:.0f})")


if __name__ == "__main__":
    main()
