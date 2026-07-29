"""Merge per-method predictions into the collaborators' CSV format + plots.

Per route:
  exports/<label>/preds_<label>_<start>_to_<end>.csv
      trip_id, service_day, pt_sequence, y_true, y_pred_naive,
      y_pred_timesfm, y_pred_timesfm_ft   (whichever methods have run)
  exports/plots_trip_means/<label>/mean_by_stop_....csv + plot PNG
      (mirrors their mean_by_stop schema: *_mean, y_true_count, n_rows,
       y_true_coverage)
  exports/plots_sample_week/<label>.png — first eval week, day by day
Overall: exports/metrics.csv — RMSE/MAE/MAPE/WAPE per route x method,
computed only where y_true is observed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from route_flow import config, evaluate, report
from route_flow.data import dataset
from route_flow.forecasting.base import predictions_path

log = logging.getLogger(__name__)


def _load_predictions(results_dir: Path, label: str) -> pd.DataFrame | None:
    merged = None
    for method in config.METHODS:
        path = predictions_path(results_dir, label, method)
        if not path.exists():
            log.warning("%s: no predictions for method %r (%s missing)",
                        label, method, path)
            continue
        df = pd.read_csv(path)
        merged = df if merged is None else merged.merge(
            df, on=["service_day", "pt_sequence"], how="outer")
    return merged


def _mean_by_stop(table: pd.DataFrame, pred_cols: list[str]) -> pd.DataFrame:
    g = table.groupby("pt_sequence")
    out = pd.DataFrame({"y_true_mean": g["y_true"].mean()})
    for col in pred_cols:
        out[f"{col}_mean"] = g[col].mean()
    out["y_true_count"] = g["y_true"].count()
    out["n_rows"] = g.size()
    out["y_true_coverage"] = out["y_true_count"] / out["n_rows"]
    return out.reset_index()


def _plot_mean_by_stop(mbs: pd.DataFrame, pred_cols: list[str], label: str,
                       out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(mbs["pt_sequence"], mbs["y_true_mean"], "k-", lw=2, label="observed")
    for col in pred_cols:
        ax.plot(mbs["pt_sequence"], mbs[f"{col}_mean"], "--", lw=1.5,
                label=col.removeprefix("y_pred_"))
    ax.set_xlabel("pt_sequence (stop index)")
    ax.set_ylabel("mean loading over days")
    ax.set_title(f"{label} — mean loading per stop ({config.EVAL_START}..{config.EVAL_END})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def _plot_sample_week(table: pd.DataFrame, pred_cols: list[str], label: str,
                      out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    days = sorted(table["service_day"].unique())[:7]
    fig, axes = plt.subplots(len(days), 1, figsize=(11, 2.2 * len(days)),
                             sharex=True)
    for ax, day in zip(axes, days):
        sl = table[table["service_day"] == day]
        ax.plot(sl["pt_sequence"], sl["y_true"], "k.-", lw=1.5, label="observed")
        for col in pred_cols:
            ax.plot(sl["pt_sequence"], sl[col], "--", lw=1, label=col.removeprefix("y_pred_"))
        ax.set_ylabel(str(day), fontsize=8)
    axes[0].legend(fontsize=8, ncol=len(pred_cols) + 1)
    axes[-1].set_xlabel("pt_sequence")
    fig.suptitle(f"{label} — first eval week, per-day loading profiles")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def run_aggregate(od_cache: Path, results_dir: Path) -> Path:
    results_dir = Path(results_dir)
    exports = results_dir / "exports"
    od = dataset.load_od_cache(str(od_cache))

    all_metrics = []
    for spec in tqdm(config.ROUTE_SPECS, desc="Aggregating", unit="route"):
        label = spec.trip_label
        preds = _load_predictions(results_dir, label)
        if preds is None:
            log.error("%s: no predictions at all — skipping", label)
            continue
        pred_cols = [c for c in config.METHODS.values() if c in preds.columns]

        data = dataset.RouteData(od, spec)
        y_true = (
            data.raw_slice(config.EVAL_START, config.EVAL_END)
            .stack(future_stack=True).rename("y_true").reset_index()
        )
        y_true.columns = ["service_day", "pt_sequence", "y_true"]
        y_true["service_day"] = y_true["service_day"].dt.date.astype(str)

        table = y_true.merge(preds, on=["service_day", "pt_sequence"], how="left")
        table.insert(0, "trip_id", label)
        table = table.sort_values(["service_day", "pt_sequence"])

        route_dir = exports / label
        route_dir.mkdir(parents=True, exist_ok=True)
        name = f"preds_{label}_{config.EVAL_START}_to_{config.EVAL_END}"
        table.to_csv(route_dir / f"{name}.csv", index=False)

        mbs = _mean_by_stop(table, pred_cols)
        plots_dir = exports / "plots_trip_means" / label
        plots_dir.mkdir(parents=True, exist_ok=True)
        mbs.to_csv(plots_dir / f"mean_by_stop_{name}.csv", index=False)
        _plot_mean_by_stop(mbs, pred_cols, label,
                           plots_dir / f"plot_mean_loading_{name}.png")

        week_dir = exports / "plots_sample_week"
        week_dir.mkdir(parents=True, exist_ok=True)
        _plot_sample_week(table, pred_cols, label, week_dir / f"{label}.png")

        for col in pred_cols:
            m = evaluate.metrics(table["y_true"], table[col])
            all_metrics.append({"trip_label": label,
                                "method": col.removeprefix("y_pred_"), **m})
        log.info("%s: exported %d rows (%d with y_true), methods=%s",
                 label, len(table), int(table["y_true"].notna().sum()),
                 [c.removeprefix("y_pred_") for c in pred_cols])

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = exports / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    if not metrics_df.empty:
        log.info("Metrics:\n%s", metrics_df.to_string(index=False))
        report.write_comparison_report(
            metrics_df, ["trip_label"], exports,
            "Trips experiment (loading vs stops traveled)", "trips")
    return metrics_path
