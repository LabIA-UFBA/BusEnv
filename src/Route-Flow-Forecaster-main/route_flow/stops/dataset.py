"""Stops experiment data: loading per stop as a function of TIME.

For each route+direction (same variant-filtered pooling as the trips
experiment) and each stop position (pt_sequence, which is unambiguous even
when a physical stop appears twice in a trip), we build a univariate series:
mean `loading` of the route's vehicles passing that stop per BIN_MINUTES
(20-min) interval — BINS_PER_DAY (72) bins per day.

Storage: one DataFrame per route with MultiIndex rows (pt_sequence,
service_day) and columns bin 0..71. NaN = no vehicle passed in that bin
(overnight bins are mostly NaN); model input NaNs are filled with day-type
per-bin means computed from history only.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from route_flow import config
from route_flow.config import RouteSpec
from route_flow.data import dataset as trips_dataset

log = logging.getLogger(__name__)


class StopsData:
    """Per-(route, stop position) binned loading series."""

    def __init__(self, od: pd.DataFrame, spec: RouteSpec,
                 start: str = config.DATA_START, end: str = config.DATA_END):
        self.spec = spec
        base = trips_dataset.RouteData(od, spec, start, end)
        self.n_stops = base.n_stops
        rows = base.rows[base.rows["pt_sequence"].between(1, self.n_stops)].copy()

        # canonical stop_id per position (mode across occurrences)
        self.stop_ids: dict[int, str] = (
            rows.groupby("pt_sequence")["stop_id"]
            .agg(lambda s: s.mode().iloc[0]).astype(str).to_dict()
        )

        rows["bin"] = (
            pd.to_datetime(rows["stop_time"]).dt.hour * 60
            + pd.to_datetime(rows["stop_time"]).dt.minute
        ) // config.BIN_MINUTES
        rows = rows.dropna(subset=["bin"])
        rows["bin"] = rows["bin"].astype(int).clip(0, config.BINS_PER_DAY - 1)

        pivot = rows.pivot_table(index=["pt_sequence", "service_day"],
                                 columns="bin", values="loading",
                                 aggfunc="mean")
        days = pd.date_range(start, end, freq="D")
        idx = pd.MultiIndex.from_product(
            [range(1, self.n_stops + 1), days],
            names=["pt_sequence", "service_day"])
        self.raw = pivot.reindex(index=idx,
                                 columns=range(config.BINS_PER_DAY))
        cov = float(self.raw.notna().mean().mean())
        log.info("%s: stops matrix %d stops x %d days x %d bins "
                 "(bin coverage %.1f%% — overnight bins are empty by nature)",
                 spec.trip_label, self.n_stops, len(days),
                 config.BINS_PER_DAY, cov * 100)

    def stop_matrix(self, seq: int) -> pd.DataFrame:
        """(service_day x bin) raw matrix for one stop position."""
        return self.raw.xs(seq, level="pt_sequence")

    def day_type_bin_stats(self, seq: int, upto: pd.Timestamp
                           ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(mean, std) per (day_type, bin) from history up to `upto` incl."""
        hist = self.stop_matrix(seq).loc[:upto]
        keys = [trips_dataset.day_type(d) for d in hist.index]
        means = hist.groupby(keys).mean().reindex(trips_dataset.DayTypes)
        stds = hist.groupby(keys).std().reindex(trips_dataset.DayTypes)
        return means, stds

    def filled_history(self, seq: int, end_day: pd.Timestamp,
                       n_days: int) -> pd.DataFrame:
        """Last n_days rows (day x bin) ending at end_day, NaN-filled with
        leakage-free day-type per-bin means (fallback overall bin mean, 0)."""
        means, _ = self.day_type_bin_stats(seq, end_day)
        window = self.stop_matrix(seq).loc[:end_day].iloc[-n_days:]
        return trips_dataset.fill_matrix(window, means)

    def y_true_frame(self, eval_start: str, eval_end: str) -> pd.DataFrame:
        """Long format ground truth for the eval window: pt_sequence,
        stop_id, service_day, bin, y_true (NaN kept)."""
        sl = self.raw.loc[
            (slice(None), slice(pd.Timestamp(eval_start),
                                pd.Timestamp(eval_end))), :]
        long = sl.stack(future_stack=True).rename("y_true").reset_index()
        long.columns = ["pt_sequence", "service_day", "bin", "y_true"]
        long["stop_id"] = long["pt_sequence"].map(self.stop_ids)
        long["service_day"] = long["service_day"].dt.date.astype(str)
        return long


def bin_label(b: int) -> str:
    minutes = b * config.BIN_MINUTES
    return f"{minutes // 60:02d}:{minutes % 60:02d}"
