"""Merge rates predictions -> simulator-ready CSVs, plots and reports.

Per route and target:
  exports_rates/<label>/rates_<target>_<label>_<start>_to_<end>.csv
      trip_id, pt_sequence, stop_id, service_day, hour, hour_start, y_true,
      y_pred_<method>_<target>[, _p10, _p90, _std] ...
  exports_rates/plots_avg_day/<target>/<label>/stop_XXX.png
      average day: observed vs each method, P10-P90 band (TimesFM) or
      +-1 std band (naive)
Per target:
  exports_rates/<target>/metrics.csv, comparison_report.md, win_rates.csv,
  pie_<metric>.png   (instance = stop position)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from route_flow import config, evaluate, report
from route_flow.data import dataset as trips_dataset
from route_flow.rates.dataset import RatesData, hour_label
from route_flow.rates.runner import rates_predictions_path

log = logging.getLogger(__name__)

BAND_COLORS = {"naive": "tab:green", "timesfm": "tab:blue",
               "timesfm-ft": "tab:orange",
               # derived (component-ratio) variants: same hue family, lighter
               "naive-ratio": "tab:olive", "timesfm-ratio": "tab:cyan",
               "timesfm-ft-ratio": "tab:red"}


def _load_predictions(results_dir: Path, label: str,
                      target: str) -> tuple[pd.DataFrame | None, dict]:
    merged, cols = None, {}
    for method in config.RATE_METHODS:
        path = rates_predictions_path(results_dir, label, target, method)
        if not path.exists():
            log.debug("%s [%s]: no predictions for %r", label, target, method)
            continue
        df = pd.read_parquet(path)
        cols[method] = f"{config.RATE_METHODS[method]}_{target}"
        merged = df if merged is None else merged.merge(
            df, on=["service_day", "pt_sequence", "hour"], how="outer")
    return merged, cols


def _avg_day_plot(stop_table: pd.DataFrame, cols: dict, label: str,
                  target: str, seq: int, stop_id: str, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    avg = stop_table.groupby("hour").mean(numeric_only=True)
    x = [hour_label(h) for h in avg.index]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x, avg["y_true"], "k.-", lw=2, label="observed (mean)")
    for method, col in cols.items():
        if col not in avg.columns:
            continue
        color = BAND_COLORS.get(method)
        ax.plot(x, avg[col], "--", lw=1.5, color=color, label=method)
        if f"{col}_p10" in avg.columns:
            ax.fill_between(x, avg[f"{col}_p10"], avg[f"{col}_p90"],
                            alpha=0.15, color=color,
                            label=f"{method} P10-P90 (mean)")
        elif f"{col}_std" in avg.columns:
            ax.fill_between(x, (avg[col] - avg[f"{col}_std"]).clip(lower=0),
                            avg[col] + avg[f"{col}_std"], alpha=0.15,
                            color=color, label=f"{method} ±1 std")
    ax.set_xticks(x[::2])
    ax.set_xlabel("hour of day")
    ax.set_ylabel(config.RATE_TARGETS[target][1])
    ax.set_title(f"{label} stop {seq} (id {stop_id}) — {target}, average day "
                 f"({config.EVAL_START}..{config.EVAL_END})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def run_aggregate_rates(od_cache: Path, results_dir: Path,
                        targets: list[str] | None = None,
                        max_plot_stops: int | None = None) -> list[Path]:
    results_dir = Path(results_dir)
    exports = results_dir / "exports_rates"
    targets = targets or list(config.RATE_TARGETS)
    od = trips_dataset.load_od_cache(str(od_cache))

    rates_by_route = {}
    for spec in tqdm(config.ROUTE_SPECS, desc="Building rate series",
                     unit="route"):
        rates_by_route[spec.trip_label] = RatesData(od, spec)

    reports = []
    for target in targets:
        if not any(rates_predictions_path(results_dir, s.trip_label, target, m)
                   .exists()
                   for s in config.ROUTE_SPECS for m in config.RATE_METHODS):
            log.info("[%s] no predictions on disk — skipping this target",
                     target)
            continue
        all_metrics = []
        for spec in tqdm(config.ROUTE_SPECS, desc=f"Aggregating {target}",
                         unit="route"):
            label = spec.trip_label
            preds, cols = _load_predictions(results_dir, label, target)
            if preds is None:
                log.warning("%s [%s]: no predictions — skipping", label, target)
                continue
            series = rates_by_route[label].series(target)

            table = series.y_true_frame(
                config.EVAL_START, config.EVAL_END).merge(
                preds, on=["service_day", "pt_sequence", "hour"], how="left")
            table.insert(0, "trip_id", label)
            table["hour_start"] = table["hour"].map(hour_label)
            order = ["trip_id", "pt_sequence", "stop_id", "service_day",
                     "hour", "hour_start", "y_true"]
            table = table[order + [c for c in table.columns if c not in order]]
            table = table.sort_values(["pt_sequence", "service_day", "hour"])

            route_dir = exports / label
            route_dir.mkdir(parents=True, exist_ok=True)
            table.to_csv(route_dir / f"rates_{target}_{label}_"
                         f"{config.EVAL_START}_to_{config.EVAL_END}.csv",
                         index=False)

            plot_dir = exports / "plots_avg_day" / target / label
            plot_dir.mkdir(parents=True, exist_ok=True)
            seqs = sorted(table["pt_sequence"].unique())
            for seq in (seqs[:max_plot_stops] if max_plot_stops else seqs):
                _avg_day_plot(table[table["pt_sequence"] == seq], cols, label,
                              target, seq, series.stop_ids.get(seq, "?"),
                              plot_dir / f"stop_{seq:03d}.png")

            for seq in seqs:
                sub = table[table["pt_sequence"] == seq]
                for method, col in cols.items():
                    m = evaluate.metrics(sub["y_true"], sub[col])
                    all_metrics.append({
                        "trip_label": label, "pt_sequence": seq,
                        "stop_id": series.stop_ids.get(seq, "?"),
                        "method": method, **m})
            log.info("%s [%s]: exported %d rows (%d with y_true), methods=%s",
                     label, target, len(table),
                     int(table["y_true"].notna().sum()), list(cols))

        if not all_metrics:
            log.error("[%s] nothing aggregated — run forecast-rates first",
                      target)
            continue
        target_dir = exports / target
        target_dir.mkdir(parents=True, exist_ok=True)
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(target_dir / "metrics.csv", index=False)
        rep = report.write_comparison_report(
            metrics_df, ["trip_label", "pt_sequence"], target_dir,
            f"Rates experiment — {config.RATE_TARGETS[target][1]}", "stops")
        if rep:
            reports.append(rep)
    return reports
