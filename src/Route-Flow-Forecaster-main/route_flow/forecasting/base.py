"""Common forecaster interface + the rolling day-by-day evaluation driver.

Protocol for all methods (naive, timesfm, timesfm-ft, future ones):
  1. `fit(route_data)` — anything the method needs from the TRAIN window.
  2. For each day D in the eval window: `predict_day(context, day)` where
     `context` is the NaN-filled (CONTEXT_DAYS x n_stops) matrix of the days
     strictly before D — including *observed* values of already-forecast days
     (rolling origin with real data, per the agreed protocol).
Output contract: one CSV per (route, method) with service_day, pt_sequence,
prediction — merged later by `aggregate`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from route_flow import config
from route_flow.data.dataset import RouteData

log = logging.getLogger(__name__)


class Forecaster(ABC):
    method: str  # key in config.METHODS

    @abstractmethod
    def fit(self, data: RouteData) -> None:
        """Prepare for one route using training data only."""

    @abstractmethod
    def predict_day(self, context: pd.DataFrame, day: pd.Timestamp) -> np.ndarray:
        """Forecast one day's (n_stops,) loading profile from the context
        matrix (CONTEXT_DAYS x n_stops, NaN-filled, raw counts)."""

    def close(self) -> None:  # optional resource cleanup
        pass


def predictions_path(results_dir: Path, label: str, method: str) -> Path:
    return Path(results_dir) / "predictions" / f"{label}__{method}.csv"


def run_forecaster(
    forecaster: Forecaster,
    data: RouteData,
    results_dir: Path,
    eval_start: str = config.EVAL_START,
    eval_end: str = config.EVAL_END,
) -> Path:
    spec = data.spec
    days = pd.date_range(eval_start, eval_end, freq="D")
    log.info("[%s] %s: fitting", forecaster.method, spec.trip_label)
    forecaster.fit(data)

    records = []
    for day in tqdm(days, desc=f"{forecaster.method}:{spec.trip_label}", unit="day"):
        context = data.filled_history(day - pd.Timedelta(days=1), config.CONTEXT_DAYS)
        pred = np.asarray(forecaster.predict_day(context, day), dtype=np.float64)
        if pred.shape != (data.n_stops,):
            raise ValueError(
                f"{forecaster.method} returned shape {pred.shape}, "
                f"expected ({data.n_stops},)")
        pred = np.clip(pred, 0.0, None)  # loading is a non-negative count
        records.append(pd.DataFrame({
            "service_day": day.date().isoformat(),
            "pt_sequence": np.arange(1, data.n_stops + 1),
            config.METHODS[forecaster.method]: pred,
        }))

    out = predictions_path(results_dir, spec.trip_label, forecaster.method)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(records, ignore_index=True).to_csv(out, index=False)
    log.info("[%s] %s: wrote %s", forecaster.method, spec.trip_label, out)
    return out
