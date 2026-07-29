"""Method-comparison report, shared by both experiments.

Input: a long-format DataFrame of per-instance metrics — one row per
(instance, method) with metric columns. An "instance" is a trip for the trips
experiment and a (route, stop position) for the stops experiment.

Produces, under <out_dir>:
  comparison_report.md   overall metrics per method + win-rate tables
  win_rates.csv          per (metric, method) share of instances won
  pie_<metric>.png       one pie chart per metric with each method's win share
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

REPORT_METRICS = ("mse", "mae", "mape_pct")
METRIC_TITLES = {"mse": "MSE", "mae": "MAE", "mape_pct": "MAPE (%)"}


def win_rates(metrics_df: pd.DataFrame, instance_cols: list[str],
              metric: str) -> pd.Series:
    """Share of instances where each method has the lowest error.

    Instances missing the metric for some method are judged among the methods
    that do have it; instances with no valid values at all are dropped. Exact
    ties are split equally between the tied methods.
    """
    d = metrics_df.dropna(subset=[metric])
    if d.empty:
        return pd.Series(dtype=float)
    best = d.groupby(instance_cols)[metric].transform("min")
    winners = d[d[metric] == best].copy()
    n_tied = winners.groupby(instance_cols)["method"].transform("count")
    winners["_weight"] = 1.0 / n_tied
    shares = winners.groupby("method")["_weight"].sum()
    return (shares / shares.sum() * 100).sort_values(ascending=False)


def _pie(rates: pd.Series, metric: str, out_png: Path, experiment: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(rates.values, labels=rates.index, autopct="%1.1f%%", startangle=90)
    ax.set_title(f"{experiment}: share of instances won — {METRIC_TITLES[metric]}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def write_comparison_report(metrics_df: pd.DataFrame, instance_cols: list[str],
                            out_dir: Path, experiment: str,
                            instance_noun: str) -> Path | None:
    """metrics_df: columns = instance_cols + ['method'] + metric columns."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = sorted(metrics_df["method"].unique())
    if len(metrics_df) == 0:
        log.warning("No metrics available — skipping comparison report")
        return None
    if len(methods) < 2:
        log.warning("Only %s has results; comparison report will be trivial "
                    "until other methods run", methods)

    n_instances = metrics_df[instance_cols].drop_duplicates().shape[0]
    lines = [
        f"# {experiment} — method comparison", "",
        f"Instances: **{n_instances} {instance_noun}** | methods: "
        f"{', '.join(methods)}. Metrics computed only where y_true is "
        "observed; MAPE excludes y_true = 0 points.", "",
        "## Overall metrics (all instances pooled, weighted by points)", "",
    ]

    overall = metrics_df.groupby("method").apply(
        lambda g: pd.Series({
            "n_points": g["n_points"].sum(),
            **{m: (g[m] * g["n_points"]).sum() / g["n_points"].sum()
               for m in REPORT_METRICS if m in g},
        }), include_groups=False).round(4)
    lines += [overall.to_markdown(), ""]

    lines += [f"## Win rates (per {instance_noun}: lowest error wins; "
              "ties split equally)", ""]
    rate_rows = []
    for metric in REPORT_METRICS:
        if metric not in metrics_df.columns:
            continue
        rates = win_rates(metrics_df, instance_cols, metric)
        if rates.empty:
            continue
        lines += [f"### {METRIC_TITLES[metric]}", ""]
        lines += [f"- **{m}**: won {v:.1f}% of {instance_noun}"
                  for m, v in rates.items()]
        lines += [f"", f"![pie]({f'pie_{metric}.png'})", ""]
        _pie(rates, metric, out_dir / f"pie_{metric}.png", experiment)
        rate_rows += [{"metric": metric, "method": m, "win_pct": round(v, 2)}
                      for m, v in rates.items()]

    pd.DataFrame(rate_rows).to_csv(out_dir / "win_rates.csv", index=False)
    report = out_dir / "comparison_report.md"
    report.write_text("\n".join(lines))
    log.info("Comparison report -> %s", report)
    return report
