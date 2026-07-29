"""Rolling day-by-day driver for the stops experiment.

Same protocol as the trips experiment: for each eval day, each stop's context
is the previous CONTEXT_DAYS days at 20-min resolution (real observed data,
NaN-filled leakage-free), and the forecaster emits the next day's 72 bins.
Predictions are clipped at >= 0. One parquet per (route, method) under
results/predictions_stops/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from route_flow import config
from route_flow.stops.dataset import StopsData
from route_flow.stops.forecasters import StopsForecaster

log = logging.getLogger(__name__)


def stops_predictions_path(results_dir: Path, label: str, method: str) -> Path:
    return Path(results_dir) / "predictions_stops" / f"{label}__{method}.parquet"


def method_columns(method: str, provides: tuple[str, ...]) -> dict[str, str]:
    base = config.METHODS[method]
    cols = {"point": base}
    if "p10" in provides:
        cols["p10"], cols["p90"] = f"{base}_p10", f"{base}_p90"
    if "std" in provides:
        cols["std"] = f"{base}_std"
    return cols


def run_stops_forecaster(forecaster: StopsForecaster, data: StopsData,
                         results_dir: Path,
                         eval_start: str = config.EVAL_START,
                         eval_end: str = config.EVAL_END) -> Path:
    spec = data.spec
    days = pd.date_range(eval_start, eval_end, freq="D")
    log.info("[stops:%s] %s: fitting", forecaster.method, spec.trip_label)
    forecaster.fit(data)
    cols = method_columns(forecaster.method, forecaster.provides)

    frames = []
    for day in tqdm(days, desc=f"stops:{forecaster.method}:{spec.trip_label}",
                    unit="day"):
        contexts = np.stack([
            data.filled_history(seq, day - pd.Timedelta(days=1),
                                config.CONTEXT_DAYS)
            .to_numpy(dtype=np.float32).reshape(-1)
            for seq in range(1, data.n_stops + 1)
        ])
        preds = forecaster.predict_day(contexts, day, data)
        expected = (data.n_stops, config.BINS_PER_DAY)
        for key, arr in preds.items():
            if arr.shape != expected:
                raise ValueError(f"{forecaster.method}[{key}]: shape "
                                 f"{arr.shape}, expected {expected}")
        point = np.clip(preds["point"], 0.0, None)
        p10 = np.clip(np.minimum(preds["p10"], point), 0.0, None) \
            if "p10" in preds else None
        p90 = np.maximum(preds["p90"], point) if "p90" in preds else None

        frame = pd.DataFrame({
            "pt_sequence": np.repeat(np.arange(1, data.n_stops + 1),
                                     config.BINS_PER_DAY),
            "bin": np.tile(np.arange(config.BINS_PER_DAY), data.n_stops),
            cols["point"]: point.reshape(-1),
        })
        frame.insert(0, "service_day", day.date().isoformat())
        if p10 is not None:
            frame[cols["p10"]] = p10.reshape(-1)
            frame[cols["p90"]] = p90.reshape(-1)
        if "std" in preds:
            frame[cols["std"]] = preds["std"].reshape(-1)
        frames.append(frame)

    out = stops_predictions_path(results_dir, spec.trip_label,
                                 forecaster.method)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
    log.info("[stops:%s] %s: wrote %s", forecaster.method, spec.trip_label, out)
    return out
