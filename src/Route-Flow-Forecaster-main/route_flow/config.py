"""Central configuration: routes, date windows, paths, method registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OD_CACHE = DEFAULT_DATA_DIR / "od_cache.parquet"
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "timesfm-hf"

# ---------------------------------------------------------------------------
# Date windows (all inclusive, YYYY-MM-DD)
# ---------------------------------------------------------------------------
# Finetuning + naive-baseline "training" data.
TRAIN_START = "2024-03-01"
TRAIN_END = "2024-03-31"
# Forecast target: May 2024, one day at a time. May 31 is kept in the window —
# the availability report will tell us whether the dataset actually covers it;
# if not, its y_true is simply NaN (predictions are still produced).
EVAL_START = "2024-05-01"
EVAL_END = "2024-05-31"
# Rolling context fed to TimesFM for each forecast day.
CONTEXT_DAYS = 15
# Everything the OD cache must cover (context for May 1 reaches back into April).
DATA_START = TRAIN_START
DATA_END = EVAL_END


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
# The collaborators' trip IDs (e.g. 20001_0310_1) name ONE vehicle's Nth trip
# of the day on a route+direction. We pool ALL vehicles / trip occurrences of
# the same (route, direction) and average loading per (service_day,
# pt_sequence) — that is the series we forecast. `trip_label` matches the
# collaborators' zero-padded naming so our outputs line up with theirs.
@dataclass(frozen=True)
class RouteSpec:
    trip_label: str          # output name, matches collaborators' files
    route_short_name: str    # normalized: no leading zeros
    direction_id: str        # 'I' (ida/outbound) or 'V' (volta/inbound)
    reference_trip_id: str   # the specific trip_id from bus_lines.txt


ROUTE_SPECS: list[RouteSpec] = [
    RouteSpec("20001_0310_1", "310", "I", "20001_310_1"),
    RouteSpec("20001_0310_2", "310", "V", "20001_310_2"),
    RouteSpec("20002_1320_1", "1320", "I", "20002_1320_1"),
    RouteSpec("20002_1320_10", "1320", "V", "20002_1320_10"),
    RouteSpec("20002_1367_5", "1367", "I", "20002_1367_5"),
    # Added 2026-07-19: the most consistent route in the dataset (91/91 days
    # in both directions, ZERO occurrence-length variation, ~90 occ/day) —
    # offered to the collaborators as an alternative to the problematic 310
    # (weekday-express vs weekend-local variants, see PROJECT.md §8b).
    RouteSpec("20056_1346_1", "1346", "I", "20056_1346_1"),
    RouteSpec("20056_1346_2", "1346", "V", "20056_1346_2"),
]


def spec_by_label(label: str) -> RouteSpec:
    for spec in ROUTE_SPECS:
        if spec.trip_label == label:
            return spec
    raise KeyError(
        f"Unknown route label {label!r}. Known: {[s.trip_label for s in ROUTE_SPECS]}"
    )


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------
# method key -> prediction column name in the aggregated output CSVs
METHODS: dict[str, str] = {
    "naive": "y_pred_naive",
    "timesfm": "y_pred_timesfm",
    "timesfm-ft": "y_pred_timesfm_ft",
}

# TimesFM 2.5 hard constraints (see TIMESFM_2.5_FINETUNING_GUIDE.md)
PATCH_LEN = 32          # training context length must be a multiple of this
MAX_FORWARD_HORIZON = 128  # steps per forward pass; longer needs chunking

# ---------------------------------------------------------------------------
# Stops experiment (loading per stop as a function of TIME)
# ---------------------------------------------------------------------------
BIN_MINUTES = 20
BINS_PER_DAY = 24 * 60 // BIN_MINUTES  # 72 (<= 128: one day = one forward pass)
# Quantile column suffixes for methods that provide them (TimesFM deciles).
QUANTILE_SUFFIXES = ("_p10", "_p90")
# Global finetune for the stops experiment: cap on total training windows
# sampled across all (route, stop) series.
STOPS_MAX_TRAIN_WINDOWS = 2000

# ---------------------------------------------------------------------------
# Rates experiment (inputs for the simulator's queue model)
# ---------------------------------------------------------------------------
# The RL environment no longer reads occupancy from SUNT: it queues passengers
# per stop from a boarding RATE and drops them using an alighting RATE, then
# derives occupancy itself. So we forecast those two rates, hourly.
RATE_BIN_MINUTES = 60
RATE_BINS_PER_DAY = 24 * 60 // RATE_BIN_MINUTES  # 24 (<= 128: one forward/day)

# target key -> (output column stem, human label, value upper bound or None)
RATE_TARGETS: dict[str, tuple[str, str, float | None]] = {
    "boardings_per_min": ("boardings_per_min",
                          "boardings per minute", None),
    "alighting_rate": ("alighting_rate",
                       "fraction of onboard passengers alighting", 1.0),
    # Components of alighting_rate, forecast separately so the rate can also
    # be obtained as numerator/denominator (the "-ratio" methods). Both are
    # per-vehicle means over the stop-hour, so their quotient is identical to
    # the direct definition: mean(alight)/mean(lag) == sum(alight)/sum(lag).
    "alighting_per_veh": ("alighting_per_veh",
                          "mean alightings per vehicle", None),
    "lag_loading_per_veh": ("lag_loading_per_veh",
                            "mean onboard load arriving at the stop", None),
}
# Forecast by default; the components are only produced on demand by the
# "-ratio" methods, so existing runs cost the same as before.
RATE_PRIMARY_TARGETS = ("boardings_per_min", "alighting_rate")
RATE_COMPONENT_TARGETS = ("alighting_per_veh", "lag_loading_per_veh")
# derived target <- (numerator component, denominator component)
RATE_RATIO_OF = {"alighting_rate": ("alighting_per_veh",
                                    "lag_loading_per_veh")}
RATIO_SUFFIX = "-ratio"

# Methods available to the rates experiment: the shared ones, plus a
# "<base>-ratio" variant that forecasts the two components with <base> and
# divides, instead of forecasting the rate directly. Kept separate from
# METHODS so the trips/stops experiments are unaffected.
RATE_METHODS: dict[str, str] = {
    **METHODS,
    **{f"{m}{RATIO_SUFFIX}": f"{col}_ratio" for m, col in METHODS.items()},
}


def ratio_base_method(method: str) -> str | None:
    """'timesfm-ratio' -> 'timesfm'; None if not a ratio method."""
    if method.endswith(RATIO_SUFFIX):
        return method[: -len(RATIO_SUFFIX)]
    return None

# Alighting rate = sum(n_alighting) / sum(lag_loading) over the hour.
# When the denominator is 0 (vehicles passed, but nobody was aboard) the rate
# is mathematically undefined. Policy — change here to change it everywhere:
#   0.0        -> treat as "nobody alights" (current choice)
#   float("nan") -> treat as unobserved (dropped from training and metrics)
ALIGHTING_RATE_ZERO_DIV = 0.0

# Global finetune for the rates experiment (one adapter PER TARGET).
RATES_MAX_TRAIN_WINDOWS = 2000


def normalize_route(value: object) -> str:
    """'0310' / 310 / ' 310' -> '310' (keep a single '0' if all zeros)."""
    s = str(value).strip().lstrip("0")
    return s if s else "0"


def normalize_direction(value: object) -> str:
    """Map direction encodings to 'I'/'V'.

    Docs disagree between tables: AFC/LTI use 'I' (ida) / 'V' (volta), while
    AVL-lines uses 1 = one-way, 0 = return. We map 1->'I', 0->'V'. The
    availability report prints the raw values observed so this can be verified
    against the real OD dump.
    """
    s = str(value).strip().upper()
    if s in ("I", "V"):
        return s
    if s in ("1", "1.0"):
        return "I"
    if s in ("0", "0.0"):
        return "V"
    raise ValueError(f"Unrecognized direction_id value: {value!r}")
