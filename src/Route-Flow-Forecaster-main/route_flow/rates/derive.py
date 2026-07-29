"""Derive a rate forecast by dividing two component forecasts.

Motivation: forecasting `alighting_rate` directly gave poor results. A ratio
of small, noisy quantities is hard to model — it is heteroscedastic, bounded,
and spikes when few passengers are aboard. Forecasting the numerator
(alightings/vehicle) and the denominator (onboard load arriving at the stop)
separately, then dividing, keeps the model on quantities that behave like
ordinary counts.

The components are per-vehicle means over the same stop-hour, so
    mean(n_alighting) / mean(lag_loading) == sum(n_alighting) / sum(lag_loading)
i.e. the derived quantity is *exactly* the directly-forecast target, obtained
a different way — the two are directly comparable in the metrics report.

Division policy matches the dataset: a non-positive predicted denominator is
the 0/0 case and resolves to `config.ALIGHTING_RATE_ZERO_DIV`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from route_flow import config
from route_flow.rates.runner import rates_predictions_path

log = logging.getLogger(__name__)

KEYS = ["service_day", "pt_sequence", "hour"]


def _col(method: str, target: str) -> str:
    return f"{config.RATE_METHODS[method]}_{target}"


def _safe_div(num: np.ndarray, den: np.ndarray, hi: float | None) -> np.ndarray:
    out = np.divide(num, den, out=np.full_like(num, np.nan, dtype=float),
                    where=den > 0)
    out = np.where(den > 0, out, config.ALIGHTING_RATE_ZERO_DIV)
    return np.clip(out, 0.0, hi)


def derive_ratio_predictions(results_dir: Path, label: str, target: str,
                             base_method: str) -> Path:
    """Write predictions for `target` under method '<base_method>-ratio'."""
    num_target, den_target = config.RATE_RATIO_OF[target]
    ratio_method = f"{base_method}{config.RATIO_SUFFIX}"
    hi = config.RATE_TARGETS[target][2]

    frames = {}
    for comp in (num_target, den_target):
        path = rates_predictions_path(results_dir, label, comp, base_method)
        if not path.exists():
            raise FileNotFoundError(
                f"{label}: component forecast missing for {comp!r} "
                f"({path}). Run `forecast-rates --method {base_method} "
                f"--targets {num_target} {den_target}` first.")
        frames[comp] = pd.read_parquet(path)

    merged = frames[num_target].merge(frames[den_target], on=KEYS, how="inner")
    num_col, den_col = _col(base_method, num_target), _col(base_method, den_target)
    num = merged[num_col].to_numpy(dtype=float)
    den = merged[den_col].to_numpy(dtype=float)

    out_col = _col(ratio_method, target)
    out = merged[KEYS].copy()
    out[out_col] = _safe_div(num, den, hi)

    # Uncertainty, propagated approximately (documented in PROJECT.md):
    # quantiles -> outer envelope (low numerator over high denominator, and
    # vice versa); std -> first-order delta method.
    if f"{num_col}_p10" in merged and f"{den_col}_p10" in merged:
        out[f"{out_col}_p10"] = _safe_div(
            merged[f"{num_col}_p10"].to_numpy(float),
            merged[f"{den_col}_p90"].to_numpy(float), hi)
        out[f"{out_col}_p90"] = _safe_div(
            merged[f"{num_col}_p90"].to_numpy(float),
            merged[f"{den_col}_p10"].to_numpy(float), hi)
        # keep the band around the point forecast
        out[f"{out_col}_p10"] = np.minimum(out[f"{out_col}_p10"], out[out_col])
        out[f"{out_col}_p90"] = np.maximum(out[f"{out_col}_p90"], out[out_col])
    elif f"{num_col}_std" in merged and f"{den_col}_std" in merged:
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.sqrt(
                np.where(num > 0, (merged[f"{num_col}_std"] / num) ** 2, 0.0)
                + np.where(den > 0, (merged[f"{den_col}_std"] / den) ** 2, 0.0))
        out[f"{out_col}_std"] = np.nan_to_num(out[out_col] * rel, nan=0.0,
                                              posinf=0.0)

    path = rates_predictions_path(results_dir, label, target, ratio_method)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    zero_den = int((den <= 0).sum())
    log.info("[rates:%s:%s] %s: derived from %s / %s (%d/%d predicted "
             "denominators <= 0 -> %s) -> %s", ratio_method, target, label,
             num_target, den_target, zero_den, len(den),
             config.ALIGHTING_RATE_ZERO_DIV, path)
    return path
