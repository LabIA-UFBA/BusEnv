"""Merge stops-experiment predictions -> deliverable CSVs, plots, metrics,
and the method-comparison report (instance = stop position).

Per route:
  exports_stops/<label>/stops_preds_<label>_<start>_to_<end>.csv
      trip_id, pt_sequence, stop_id, service_day, bin, bin_start, y_true,
      y_pred_naive, y_pred_naive_std,
      y_pred_timesfm[, _p10, _p90], y_pred_timesfm_ft[, _p10, _p90]
  exports_stops/plots_avg_day/<label>/stop_XX.png — average-day profile with
      P10–P90 band (TimesFM methods) / ±1 std band (naive)
Overall:
  exports_stops/metrics.csv          per (route, stop, method)
  exports_stops/comparison_report.md + win_rates.csv + pie_<metric>.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from route_flow import config, evaluate, report
from route_flow.data import dataset as trips_dataset
from route_flow.stops.dataset import StopsData, bin_label
from route_flow.stops.runner import stops_predictions_path

log = logging.getLogger(__name__)

BAND_COLORS = {"timesfm": "tab:blue", "timesfm_ft": "tab:orange",
               "naive": "tab:green"}


def _load_predictions(results_dir: Path, label: str) -> pd.DataFrame | None:
    merged = None
    for method in config.METHODS:
        path = stops_predictions_path(results_dir, label, method)
        if not path.exists():
            log.warning("%s: stops predictions missing for %r (%s)",
                        label, method, path)
            continue
        df = pd.read_parquet(path)
        merged = df if merged is None else merged.merge(
            df, on=["service_day", "pt_sequence", "bin"], how="outer")
    return merged


def _avg_day_plot(stop_table: pd.DataFrame, label: str, seq: int,
                  stop_id: str, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    avg = stop_table.groupby("bin").mean(numeric_only=True)
    x = [bin_label(b) for b in avg.index]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, avg["y_true"], "k.-", lw=2, label="observed (mean)")
    for method, col in config.METHODS.items():
        if col not in avg.columns:
            continue
        name = col.removeprefix("y_pred_")
        color = BAND_COLORS.get(name, None)
        ax.plot(x, avg[col], "--", lw=1.5, color=color, label=name)
        if f"{col}_p10" in avg.columns:  # averaged deciles (P10-P90 band)
            ax.fill_between(x, avg[f"{col}_p10"], avg[f"{col}_p90"],
                            alpha=0.15, color=color,
                            label=f"{name} P10-P90 (mean)")
        elif f"{col}_std" in avg.columns:  # naive: +-1 std band
            ax.fill_between(x, avg[col] - avg[f"{col}_std"],
                            avg[col] + avg[f"{col}_std"],
                            alpha=0.15, color=color, label=f"{name} ±1 std")
    ax.set_xticks(x[::6])
    ax.set_xlabel(f"time of day ({config.BIN_MINUTES}-min bins)")
    ax.set_ylabel("mean loading")
    ax.set_title(f"{label} stop {seq} (id {stop_id}) — average forecast day "
                 f"({config.EVAL_START}..{config.EVAL_END})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def run_aggregate_stops(od_cache: Path, results_dir: Path,
                        max_plot_stops: int | None = None) -> Path | None:
    results_dir = Path(results_dir)
    exports = results_dir / "exports_stops"
    od = trips_dataset.load_od_cache(str(od_cache))

    all_metrics = []
    for spec in tqdm(config.ROUTE_SPECS, desc="Aggregating stops", unit="route"):
        label = spec.trip_label
        preds = _load_predictions(results_dir, label)
        if preds is None:
            log.error("%s: no stops predictions — skipping", label)
            continue
        pred_cols = [c for c in config.METHODS.values() if c in preds.columns]

        data = StopsData(od, spec)
        table = data.y_true_frame(config.EVAL_START, config.EVAL_END).merge(
            preds, on=["service_day", "pt_sequence", "bin"], how="left")
        table.insert(0, "trip_id", label)
        table["bin_start"] = table["bin"].map(bin_label)
        order = ["trip_id", "pt_sequence", "stop_id", "service_day", "bin",
                 "bin_start", "y_true"]
        table = table[order + [c for c in table.columns if c not in order]]
        table = table.sort_values(["pt_sequence", "service_day", "bin"])

        route_dir = exports / label
        route_dir.mkdir(parents=True, exist_ok=True)
        name = f"stops_preds_{label}_{config.EVAL_START}_to_{config.EVAL_END}"
        table.to_csv(route_dir / f"{name}.csv", index=False)

        plot_dir = exports / "plots_avg_day" / label
        plot_dir.mkdir(parents=True, exist_ok=True)
        seqs = sorted(table["pt_sequence"].unique())
        for seq in seqs[:max_plot_stops] if max_plot_stops else seqs:
            sub = table[table["pt_sequence"] == seq]
            _avg_day_plot(sub, label, seq, data.stop_ids.get(seq, "?"),
                          plot_dir / f"stop_{seq:03d}.png")

        for seq in seqs:
            sub = table[table["pt_sequence"] == seq]
            for col in pred_cols:
                m = evaluate.metrics(sub["y_true"], sub[col])
                all_metrics.append({
                    "trip_label": label, "pt_sequence": seq,
                    "stop_id": data.stop_ids.get(seq, "?"),
                    "method": col.removeprefix("y_pred_"), **m})
        log.info("%s: exported %d rows (%d with y_true)", label, len(table),
                 int(table["y_true"].notna().sum()))

    if not all_metrics:
        log.error("Nothing aggregated — run forecast-stops first")
        return None
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(exports / "metrics.csv", index=False)
    return report.write_comparison_report(
        metrics_df, ["trip_label", "pt_sequence"], exports,
        "Stops experiment (loading vs time of day)", "stops")
