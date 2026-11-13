"""Train on some days and test on others, without editing the simulator.

Each episode simulates one day, and the simulator loops over every daily data file it
finds. For a fair test, training uses only 2024 days and testing only 2025 days.
apply_day_split() keeps just the chosen days in the simulator's list. Call it right after
creating the simulator and before the first reset().
"""
from __future__ import annotations

from typing import Optional

DEFAULT_TRAIN_START = "2024-03-01"
DEFAULT_TRAIN_END = "2024-12-31"


def _date_of(fname: str) -> str:
    """'daily_data_2024-03-01.pkl' -> '2024-03-01'"""
    return fname.replace("daily_data_", "").replace(".pkl", "")


def apply_day_split(env, split: str, train_start: str = DEFAULT_TRAIN_START, train_end: str = DEFAULT_TRAIN_END,
                    test_end: Optional[str] = None) -> int:
    """split: "train" (up to train_end), "test" (after train_end) or "none" (all days).
    Returns how many days are kept."""
    if split in (None, "", "none", "all"):
        return env.total_days
    files = list(env.daily_files)
    if split == "train":
        keep = [f for f in files if train_start <= _date_of(f) <= train_end]
    elif split == "test":
        keep = [f for f in files if _date_of(f) > train_end and (test_end is None or _date_of(f) <= test_end)]
    else:
        raise ValueError(f"unknown split {split!r}")
    if not keep:
        raise ValueError(f"day split {split!r} keeps no daily files")
    env.daily_files = keep
    env.total_days = len(keep)
    env.current_day_index = 0
    env.date = None  # the simulator picks the first kept day on reset()
    env.day_done = False
    return env.total_days
