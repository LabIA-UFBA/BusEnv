"""Command-line entry points.

  python -m route_flow prepare-data     # build/refresh the OD cache (needs internet)
  python -m route_flow availability     # data availability report (run first)

Trips experiment (deliverable format for the collaborators):
  python -m route_flow forecast --method {naive,timesfm,timesfm-ft} [--routes ...]
  python -m route_flow aggregate        # merged CSVs + plots + metrics + comparison report
  python -m route_flow all              # availability + all 3 methods + aggregate

Stops experiment (loading vs time-of-day, 20-min bins, with deciles):
  python -m route_flow forecast-stops --method {naive,timesfm,timesfm-ft} [--routes ...]
  python -m route_flow aggregate-stops  # merged CSVs + avg-day plots + comparison report
  python -m route_flow all-stops        # all 3 methods + aggregate-stops

Rates experiment (simulator inputs: hourly boardings/min + alighting rate):
  python -m route_flow forecast-rates --method {naive,timesfm,timesfm-ft,
                                                naive-ratio,timesfm-ratio,...}
                                      [--targets ...] [--routes ...]
      A "<base>-ratio" method forecasts alighting_rate indirectly: it forecasts
      alightings/vehicle and onboard load with <base>, then divides them.
  python -m route_flow aggregate-rates  # per-target CSVs + plots + comparison reports
  python -m route_flow all-rates        # naive + timesfm + timesfm-ratio (NO
                                        #   finetuning); use --methods to change
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from route_flow import config
from route_flow.logging_utils import setup_logging

log = logging.getLogger(__name__)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--od-cache", type=Path, default=config.DEFAULT_OD_CACHE,
                   help="Path to the OD parquet cache")
    p.add_argument("--results", type=Path, default=config.DEFAULT_RESULTS_DIR,
                   help="Results output directory")
    p.add_argument("--log-level", default="INFO")


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model-path", type=Path, default=config.DEFAULT_MODEL_PATH,
                   help="Local dir with google/timesfm-2.5-200m-transformers weights")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")


def _add_finetune_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--refit", action="store_true",
                   help="Retrain adapters even if saved ones exist")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="route_flow", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare-data", help="Fetch OD slice via suntdataset "
                       "(needs internet) and cache as parquet")
    _add_common(p)
    p.add_argument("--start", default=config.DATA_START)
    p.add_argument("--end", default=config.DATA_END)
    p.add_argument("--source", default="hf", choices=["hf", "suntdataset"],
                   help="hf = direct HuggingFace download (default, no "
                        "suntdataset needed); suntdataset = official package")

    p = sub.add_parser("availability", help="Data availability report")
    _add_common(p)
    p.add_argument("--start", default=config.DATA_START)
    p.add_argument("--end", default=config.DATA_END)

    p = sub.add_parser("forecast", help="Run one forecasting method")
    _add_common(p)
    _add_model_args(p)
    _add_finetune_args(p)
    p.add_argument("--method", required=True, choices=list(config.METHODS))
    p.add_argument("--routes", nargs="*", default=None,
                   help="Subset of trip labels (default: all 5)")

    p = sub.add_parser("aggregate", help="Merge predictions into export CSVs, "
                       "plots, metrics and comparison report")
    _add_common(p)

    p = sub.add_parser("all", help="availability + naive + timesfm + "
                       "timesfm-ft + aggregate")
    _add_common(p)
    _add_model_args(p)
    _add_finetune_args(p)
    p.add_argument("--routes", nargs="*", default=None)

    p = sub.add_parser("forecast-stops", help="Stops experiment: run one method")
    _add_common(p)
    _add_model_args(p)
    _add_finetune_args(p)
    p.add_argument("--method", required=True, choices=list(config.METHODS))
    p.add_argument("--routes", nargs="*", default=None)

    p = sub.add_parser("aggregate-stops", help="Stops experiment: merged CSVs, "
                       "average-day plots, metrics and comparison report")
    _add_common(p)
    p.add_argument("--max-plot-stops", type=int, default=None,
                   help="Limit average-day plots per route (default: all stops)")

    p = sub.add_parser("all-stops", help="Stops experiment: all 3 methods + "
                       "aggregate-stops")
    _add_common(p)
    _add_model_args(p)
    _add_finetune_args(p)
    p.add_argument("--routes", nargs="*", default=None)
    p.add_argument("--max-plot-stops", type=int, default=None)

    p = sub.add_parser("forecast-rates", help="Rates experiment: run one "
                       "method over the rate targets")
    _add_common(p)
    _add_model_args(p)
    _add_finetune_args(p)
    p.add_argument("--method", required=True,
                   choices=list(config.RATE_METHODS),
                   help="'<base>-ratio' forecasts alighting_rate as "
                        "alightings/vehicle divided by onboard load instead "
                        "of forecasting the rate directly")
    p.add_argument("--targets", nargs="*", default=None,
                   choices=list(config.RATE_TARGETS),
                   help="Default: boardings_per_min and alighting_rate; the "
                        "*_per_veh components are produced automatically by "
                        "the -ratio methods")
    p.add_argument("--routes", nargs="*", default=None)

    p = sub.add_parser("aggregate-rates", help="Rates experiment: per-target "
                       "CSVs, average-day plots and comparison reports")
    _add_common(p)
    p.add_argument("--targets", nargs="*", default=None,
                   choices=list(config.RATE_TARGETS))
    p.add_argument("--max-plot-stops", type=int, default=None)

    p = sub.add_parser("all-rates", help="Rates experiment: naive + timesfm "
                       "(no finetuning by default) + aggregate-rates")
    _add_common(p)
    _add_model_args(p)
    _add_finetune_args(p)
    p.add_argument("--methods", nargs="*",
                   default=["naive", "timesfm", "timesfm-ratio"],
                   choices=list(config.RATE_METHODS),
                   help="Methods to run (default: naive timesfm "
                        "timesfm-ratio — no finetuning; add timesfm-ft "
                        "and/or naive-ratio to include them)")
    p.add_argument("--targets", nargs="*", default=None,
                   choices=list(config.RATE_TARGETS))
    p.add_argument("--routes", nargs="*", default=None)
    p.add_argument("--max-plot-stops", type=int, default=None)

    return parser


def _specs(routes: list[str] | None) -> list[config.RouteSpec]:
    if not routes:
        return config.ROUTE_SPECS
    return [config.spec_by_label(r) for r in routes]


def _make_forecaster(method: str, args):
    if method == "naive":
        from route_flow.forecasting.naive import NaiveForecaster

        return NaiveForecaster()
    if method == "timesfm":
        from route_flow.forecasting.timesfm_zeroshot import TimesFMZeroShot

        return TimesFMZeroShot(args.model_path, device=args.device)
    if method == "timesfm-ft":
        from route_flow.forecasting.timesfm_finetune import (
            FinetuneConfig, TimesFMFinetuned)

        ft = FinetuneConfig(epochs=args.epochs, batch_size=args.batch_size,
                            lr=args.lr, lora_r=args.lora_r,
                            lora_alpha=args.lora_alpha)
        return TimesFMFinetuned(args.model_path, args.results, ft=ft,
                                device=args.device, refit=args.refit)
    raise ValueError(method)


def cmd_forecast(args, method: str) -> None:
    from route_flow.data import dataset
    from route_flow.forecasting.base import run_forecaster

    od = dataset.load_od_cache(str(args.od_cache))
    forecaster = _make_forecaster(method, args)
    specs = _specs(args.routes)
    log.info("Method %r on %d route(s): %s", method, len(specs),
             [s.trip_label for s in specs])
    try:
        for spec in specs:
            data = dataset.RouteData(od, spec)
            run_forecaster(forecaster, data, args.results)
    finally:
        forecaster.close()


def _make_stops_forecaster(method: str, args):
    from route_flow.stops import forecasters as sf

    if method == "naive":
        return sf.StopsNaive()
    if method == "timesfm":
        return sf.StopsTimesFMZeroShot(args.model_path, device=args.device)
    if method == "timesfm-ft":
        return sf.StopsTimesFMFinetuned(
            args.model_path, sf.global_adapter_dir(args.results),
            device=args.device)
    raise ValueError(method)


def cmd_forecast_stops(args, method: str) -> None:
    from route_flow.data import dataset
    from route_flow.stops import forecasters as sf
    from route_flow.stops.dataset import StopsData
    from route_flow.stops.runner import run_stops_forecaster

    od = dataset.load_od_cache(str(args.od_cache))
    specs = _specs(args.routes)
    log.info("Stops experiment, method %r on %d route(s)", method, len(specs))
    all_data = [StopsData(od, spec) for spec in
                tqdm_specs(specs, "building stop series")]

    if method == "timesfm-ft":
        adir = sf.global_adapter_dir(args.results)
        if args.refit or not (adir / "adapter_config.json").exists():
            from route_flow.forecasting.timesfm_finetune import FinetuneConfig

            ft = FinetuneConfig(epochs=args.epochs, batch_size=args.batch_size,
                                lr=args.lr, lora_r=args.lora_r,
                                lora_alpha=args.lora_alpha)
            # the global adapter trains on ALL routes regardless of --routes
            train_data = all_data if args.routes is None else [
                StopsData(od, spec) for spec in config.ROUTE_SPECS
                if spec.trip_label not in {d.spec.trip_label for d in all_data}
            ] + all_data
            sf.finetune_global(train_data, args.model_path, adir, ft,
                               device=args.device)
        else:
            log.info("Reusing global stops adapter %s (use --refit to retrain)",
                     adir)

    forecaster = _make_stops_forecaster(method, args)
    try:
        for data in all_data:
            run_stops_forecaster(forecaster, data, args.results)
    finally:
        forecaster.close()


def tqdm_specs(specs, desc):
    from tqdm import tqdm

    return tqdm(specs, desc=desc, unit="route")


def cmd_forecast_rates(args, method: str) -> None:
    from route_flow.data import dataset
    from route_flow.rates import forecasters as rf
    from route_flow.rates.dataset import RatesData
    from route_flow.rates.derive import derive_ratio_predictions
    from route_flow.rates.runner import (rates_predictions_path,
                                         run_rates_forecaster)

    od = dataset.load_od_cache(str(args.od_cache))
    specs = _specs(args.routes)
    targets = args.targets or list(config.RATE_PRIMARY_TARGETS)

    # A "-ratio" method does not forecast the rate directly: it forecasts the
    # rate's two components with its base method and divides them.
    base = config.ratio_base_method(method)
    if base is not None:
        derived = [t for t in targets if t in config.RATE_RATIO_OF]
        skipped = [t for t in targets if t not in config.RATE_RATIO_OF]
        if skipped:
            log.info("%r only applies to %s — ignoring %s",
                     method, list(config.RATE_RATIO_OF), skipped)
        if not derived:
            log.error("%r produces none of the requested targets", method)
            return
        components = sorted({c for t in derived
                             for c in config.RATE_RATIO_OF[t]})
        rates = [RatesData(od, spec)
                 for spec in tqdm_specs(specs, "building rate series")]
        # forecast any component that is not already on disk
        missing = [c for c in components
                   if not all(rates_predictions_path(
                       args.results, d.spec.trip_label, c, base).exists()
                       for d in rates)]
        if missing:
            log.info("%r: forecasting components %s with base method %r",
                     method, missing, base)
            comp_args = argparse.Namespace(**vars(args))
            comp_args.targets = missing
            cmd_forecast_rates(comp_args, base)
        else:
            log.info("%r: reusing existing %s component forecasts",
                     method, base)
        for target in derived:
            for data in rates:
                derive_ratio_predictions(args.results, data.spec.trip_label,
                                         target, base)
        return

    log.info("Rates experiment, method %r, targets %s, %d route(s)",
             method, targets, len(specs))
    rates = [RatesData(od, spec)
             for spec in tqdm_specs(specs, "building rate series")]

    for target in targets:
        if method == "timesfm-ft":
            adir = rf.rates_adapter_dir(args.results, target)
            if args.refit or not (adir / "adapter_config.json").exists():
                from route_flow.forecasting.timesfm_finetune import FinetuneConfig

                ft = FinetuneConfig(epochs=args.epochs,
                                    batch_size=args.batch_size, lr=args.lr,
                                    lora_r=args.lora_r,
                                    lora_alpha=args.lora_alpha)
                # one adapter per target, trained across ALL routes/stops
                # (independently of --routes, which only limits forecasting)
                built = {d.spec.trip_label: d for d in rates}
                train_series = [
                    built.get(s.trip_label, None) or RatesData(od, s)
                    for s in config.ROUTE_SPECS
                ]
                train_series = [d.series(target) for d in train_series]
                rf.finetune_global(train_series, args.model_path, adir, ft,
                                   device=args.device,
                                   max_windows=config.RATES_MAX_TRAIN_WINDOWS)
            else:
                log.info("[%s] reusing adapter %s (use --refit to retrain)",
                         target, adir)

        forecaster = rf.make_rates_forecaster(method, target, args)
        try:
            for data in rates:
                run_rates_forecaster(forecaster, data.series(target),
                                     args.results)
        finally:
            forecaster.close()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging(getattr(args, "results", config.DEFAULT_RESULTS_DIR),
                  args.log_level)

    if args.command == "prepare-data":
        from route_flow.data.prepare import prepare_od_cache

        prepare_od_cache(args.od_cache, args.start, args.end, args.source)
    elif args.command == "availability":
        from route_flow.data.availability import run_availability

        run_availability(args.od_cache, args.results, args.start, args.end)
    elif args.command == "forecast":
        cmd_forecast(args, args.method)
    elif args.command == "aggregate":
        from route_flow.aggregate import run_aggregate

        run_aggregate(args.od_cache, args.results)
    elif args.command == "all":
        from route_flow.aggregate import run_aggregate
        from route_flow.data.availability import run_availability

        run_availability(args.od_cache, args.results,
                         config.DATA_START, config.DATA_END)
        for method in config.METHODS:
            cmd_forecast(args, method)
        run_aggregate(args.od_cache, args.results)
    elif args.command == "forecast-stops":
        cmd_forecast_stops(args, args.method)
    elif args.command == "aggregate-stops":
        from route_flow.stops.aggregate import run_aggregate_stops

        run_aggregate_stops(args.od_cache, args.results, args.max_plot_stops)
    elif args.command == "all-stops":
        from route_flow.stops.aggregate import run_aggregate_stops

        for method in config.METHODS:
            cmd_forecast_stops(args, method)
        run_aggregate_stops(args.od_cache, args.results, args.max_plot_stops)
    elif args.command == "forecast-rates":
        cmd_forecast_rates(args, args.method)
    elif args.command == "aggregate-rates":
        from route_flow.rates.aggregate import run_aggregate_rates

        run_aggregate_rates(args.od_cache, args.results, args.targets,
                            args.max_plot_stops)
    elif args.command == "all-rates":
        from route_flow.rates.aggregate import run_aggregate_rates

        log.info("Rates experiment methods: %s", args.methods)
        for method in args.methods:
            cmd_forecast_rates(args, method)
        run_aggregate_rates(args.od_cache, args.results, args.targets,
                            args.max_plot_stops)
    log.info("Done.")
