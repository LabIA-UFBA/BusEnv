"""Rolling day-by-day driver for the rates experiment (one target at a time).

Same protocol as the other experiments: context = previous CONTEXT_DAYS days
at hourly resolution (real observed data, NaN-filled leakage-free), horizon =
the next day's 24 hours. Predictions are clipped to the target's valid range
(>= 0, and <= 1 for alighting_rate). One parquet per (route, target, method).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from route_flow import config
from route_flow.rates.dataset import RateSeries
from route_flow.stops.forecasters import StopsForecaster

log = logging.getLogger(__name__)


def rates_predictions_path(results_dir: Path, label: str, target: str,
                           method: str) -> Path:
    return (Path(results_dir) / "predictions_rates"
            / f"{label}__{target}__{method}.parquet")


def method_columns(method: str, target: str,
                   provides: tuple[str, ...]) -> dict[str, str]:
    base = f"{config.RATE_METHODS[method]}_{target}"
    cols = {"point": base}
    if "p10" in provides:
        cols["p10"], cols["p90"] = f"{base}_p10", f"{base}_p90"
    if "std" in provides:
        cols["std"] = f"{base}_std"
    return cols


def run_rates_forecaster(forecaster: StopsForecaster, series: RateSeries,
                         results_dir: Path,
                         eval_start: str = config.EVAL_START,
                         eval_end: str = config.EVAL_END) -> Path:
    spec, target = series.spec, series.target
    days = pd.date_range(eval_start, eval_end, freq="D")
    log.info("[rates:%s:%s] %s: fitting", forecaster.method, target,
             spec.trip_label)
    forecaster.fit(series)
    cols = method_columns(forecaster.method, target, forecaster.provides)
    hi = series.upper_bound

    frames = []
    for day in tqdm(days, unit="day",
                    desc=f"rates:{forecaster.method}:{target}:{spec.trip_label}"):
        contexts = np.stack([
            series.filled_history(seq, day - pd.Timedelta(days=1),
                                  config.CONTEXT_DAYS)
            .to_numpy(dtype=np.float32).reshape(-1)
            for seq in range(1, series.n_stops + 1)
        ])
        preds = forecaster.predict_day(contexts, day, series)
        expected = (series.n_stops, config.RATE_BINS_PER_DAY)
        for key, arr in preds.items():
            if arr.shape != expected:
                raise ValueError(f"{forecaster.method}[{key}]: shape "
                                 f"{arr.shape}, expected {expected}")

        point = np.clip(preds["point"], 0.0, hi)
        frame = pd.DataFrame({
            "pt_sequence": np.repeat(np.arange(1, series.n_stops + 1),
                                     config.RATE_BINS_PER_DAY),
            "hour": np.tile(np.arange(config.RATE_BINS_PER_DAY),
                            series.n_stops),
            cols["point"]: point.reshape(-1),
        })
        frame.insert(0, "service_day", day.date().isoformat())
        if "p10" in preds:
            p10 = np.clip(np.minimum(preds["p10"], point), 0.0, hi)
            p90 = np.clip(np.maximum(preds["p90"], point), 0.0, hi)
            frame[cols["p10"]] = p10.reshape(-1)
            frame[cols["p90"]] = p90.reshape(-1)
        if "std" in preds:
            frame[cols["std"]] = np.clip(preds["std"], 0.0, None).reshape(-1)
        frames.append(frame)

    out = rates_predictions_path(results_dir, spec.trip_label, target,
                                 forecaster.method)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
    log.info("[rates:%s:%s] %s: wrote %s", forecaster.method, target,
             spec.trip_label, out)
    return out
