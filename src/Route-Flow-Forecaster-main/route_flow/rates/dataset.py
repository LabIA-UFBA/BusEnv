"""Rates experiment data: the two inputs the simulator's queue model needs.

For each route+direction and each stop position, at HOURLY resolution:

  boardings_per_min = sum(n_boardings) / RATE_BIN_MINUTES
      Passengers joining that stop's queue per minute, for this route. The
      simulator accumulates the queue with it and empties it when a bus calls.

  alighting_rate = sum(n_alighting) / sum(lag_loading)
      Fraction of the passengers already aboard who get off at this stop.
      `lag_loading` is the load BEFORE the vehicle serves the stop (i.e. the
      load carried from the previous stop), which is exactly the denominator
      the simulator applies to its own computed occupancy.

Both are summed over every vehicle/occurrence of the route in the hour, so
they describe the route's behaviour at that stop-hour rather than one bus.

Missing vs. zero — an important distinction:
  * No vehicle of the route passed in that hour -> NaN (unobserved: it is not
    a real zero, and it must not train the model or count in metrics).
  * Vehicles passed but carried nobody (sum(lag_loading) == 0) -> the ratio is
    0/0, resolved by config.ALIGHTING_RATE_ZERO_DIV (default 0.0, "nobody
    alights"). Set it to NaN there to treat those hours as unobserved instead.

Each (route, target) pair is exposed as a `RateSeries`, which implements the
same interface as `StopsData` so the stops forecasters work unchanged.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from route_flow import config
from route_flow.config import RouteSpec
from route_flow.data import dataset as trips_dataset

log = logging.getLogger(__name__)


def hour_label(b: int) -> str:
    return f"{b * config.RATE_BIN_MINUTES // 60:02d}:00"


class RateSeries:
    """One target, one route: (pt_sequence, service_day) x hour matrices.

    Duck-types `StopsData` (n_stops / stop_ids / stop_matrix /
    day_type_bin_stats / filled_history / y_true_frame) so the existing
    forecasters and finetuning code accept it as-is.
    """

    def __init__(self, spec: RouteSpec, target: str, raw: pd.DataFrame,
                 stop_ids: dict[int, str], n_stops: int):
        self.spec = spec
        self.target = target
        self.raw = raw
        self.stop_ids = stop_ids
        self.n_stops = n_stops
        self.upper_bound = config.RATE_TARGETS[target][2]

    def stop_matrix(self, seq: int) -> pd.DataFrame:
        return self.raw.xs(seq, level="pt_sequence")

    def day_type_bin_stats(self, seq: int, upto: pd.Timestamp
                           ) -> tuple[pd.DataFrame, pd.DataFrame]:
        hist = self.stop_matrix(seq).loc[:upto]
        keys = [trips_dataset.day_type(d) for d in hist.index]
        means = hist.groupby(keys).mean().reindex(trips_dataset.DayTypes)
        stds = hist.groupby(keys).std().reindex(trips_dataset.DayTypes)
        return means, stds

    def filled_history(self, seq: int, end_day: pd.Timestamp,
                       n_days: int) -> pd.DataFrame:
        means, _ = self.day_type_bin_stats(seq, end_day)
        window = self.stop_matrix(seq).loc[:end_day].iloc[-n_days:]
        return trips_dataset.fill_matrix(window, means)

    def y_true_frame(self, eval_start: str, eval_end: str) -> pd.DataFrame:
        sl = self.raw.loc[
            (slice(None), slice(pd.Timestamp(eval_start),
                                pd.Timestamp(eval_end))), :]
        long = sl.stack(future_stack=True).rename("y_true").reset_index()
        long.columns = ["pt_sequence", "service_day", "hour", "y_true"]
        long["stop_id"] = long["pt_sequence"].map(self.stop_ids)
        long["service_day"] = long["service_day"].dt.date.astype(str)
        return long


class RatesData:
    """Builds every rate target for one route+direction."""

    def __init__(self, od: pd.DataFrame, spec: RouteSpec,
                 start: str = config.DATA_START, end: str = config.DATA_END):
        self.spec = spec
        base = trips_dataset.RouteData(od, spec, start, end)
        self.n_stops = base.n_stops
        rows = base.rows[base.rows["pt_sequence"].between(1, self.n_stops)].copy()
        self.stop_ids = (
            rows.groupby("pt_sequence")["stop_id"]
            .agg(lambda s: s.mode().iloc[0]).astype(str).to_dict()
        )

        stop_time = pd.to_datetime(rows["stop_time"])
        rows["hour"] = (
            (stop_time.dt.hour * 60 + stop_time.dt.minute)
            // config.RATE_BIN_MINUTES
        )
        rows = rows.dropna(subset=["hour"])
        rows["hour"] = rows["hour"].astype(int).clip(
            0, config.RATE_BINS_PER_DAY - 1)

        keys = ["pt_sequence", "service_day", "hour"]
        agg = rows.groupby(keys).agg(
            n_boardings=("n_boardings", "sum"),
            n_alighting=("n_alighting", "sum"),
            lag_loading=("lag_loading", "sum"),
            n_occurrences=("trip_id", "count"),
        )

        boardings = agg["n_boardings"] / config.RATE_BIN_MINUTES

        denom = agg["lag_loading"]
        rate = agg["n_alighting"].div(denom.where(denom > 0))
        # 0/0: vehicles served the stop but carried nobody -> policy value.
        rate = rate.mask(denom <= 0, config.ALIGHTING_RATE_ZERO_DIV)
        rate = rate.clip(upper=config.RATE_TARGETS["alighting_rate"][2])

        # Components of the rate, as per-vehicle means over the stop-hour.
        # Dividing them reproduces `rate` exactly (same occurrence count in
        # both), so the "-ratio" methods forecast the same quantity by a
        # different route rather than a subtly different definition.
        occ = agg["n_occurrences"].where(agg["n_occurrences"] > 0)
        alighting_per_veh = agg["n_alighting"] / occ
        lag_loading_per_veh = agg["lag_loading"] / occ

        self.targets: dict[str, RateSeries] = {}
        for target, values in (("boardings_per_min", boardings),
                               ("alighting_rate", rate),
                               ("alighting_per_veh", alighting_per_veh),
                               ("lag_loading_per_veh", lag_loading_per_veh)):
            raw = self._to_matrix(values, start, end)
            self.targets[target] = RateSeries(
                spec, target, raw, self.stop_ids, self.n_stops)
            observed = float(raw.notna().mean().mean())
            log.info("%s [%s]: %d stops x %d days x %d hours "
                     "(observed %.1f%% of cells)", spec.trip_label, target,
                     self.n_stops, raw.index.get_level_values(1).nunique(),
                     config.RATE_BINS_PER_DAY, observed * 100)

        zero_div = int((denom <= 0).sum())
        if zero_div:
            log.info("%s: %d/%d observed stop-hours had 0 onboard passengers "
                     "(alighting rate set to %s by ALIGHTING_RATE_ZERO_DIV)",
                     spec.trip_label, zero_div, len(denom),
                     config.ALIGHTING_RATE_ZERO_DIV)

    def _to_matrix(self, values: pd.Series, start: str, end: str
                   ) -> pd.DataFrame:
        pivot = values.unstack("hour")
        days = pd.date_range(start, end, freq="D")
        idx = pd.MultiIndex.from_product(
            [range(1, self.n_stops + 1), days],
            names=["pt_sequence", "service_day"])
        return pivot.reindex(index=idx,
                             columns=range(config.RATE_BINS_PER_DAY))

    def series(self, target: str) -> RateSeries:
        return self.targets[target]
