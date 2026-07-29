"""Forecasters for the rates experiment.

These are the stops-experiment forecasters at hourly resolution — the only
difference is `bins_per_day` (24 instead of 72), so the classes are thin
subclasses and all inference/finetuning logic is shared. See EXTENDING.md.
"""

from __future__ import annotations

import logging
from pathlib import Path

from route_flow import config
from route_flow.stops import forecasters as stops_fc

log = logging.getLogger(__name__)

# Re-exported so a new rates method only needs this module.
StopsForecaster = stops_fc.StopsForecaster
finetune_global = stops_fc.finetune_global


class RatesNaive(stops_fc.StopsNaive):
    """Day-type hourly mean per stop; std reported as dispersion."""

    bins_per_day = config.RATE_BINS_PER_DAY


class RatesTimesFMZeroShot(stops_fc.StopsTimesFMZeroShot):
    bins_per_day = config.RATE_BINS_PER_DAY


class RatesTimesFMFinetuned(stops_fc.StopsTimesFMFinetuned):
    """One global adapter PER TARGET (boarding rates and alighting rates have
    different scales and dynamics, so they do not share an adapter)."""

    bins_per_day = config.RATE_BINS_PER_DAY


def rates_adapter_dir(results_dir: Path, target: str) -> Path:
    return Path(results_dir) / "adapters_rates" / target


def make_rates_forecaster(method: str, target: str, args):
    if method == "naive":
        return RatesNaive()
    if method == "timesfm":
        return RatesTimesFMZeroShot(args.model_path, device=args.device)
    if method == "timesfm-ft":
        return RatesTimesFMFinetuned(
            args.model_path, rates_adapter_dir(args.results, target),
            device=args.device)
    raise ValueError(method)
