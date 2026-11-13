"""Turns the chunks chosen for a key into 10 numbers the bus policy can read.

The policy is a small neural network that only reads numbers, not text, so the chosen
chunks are summarised into a fixed list of 10 values (always the same order and meaning).

Every experiment arm uses this same function and the same scaling constants. That way,
arms differ only in WHICH chunks they chose, which is what the experiment compares.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

FEATURE_NAMES = (
    "n_sel_frac",          # 0  how many chunks were chosen, divided by the maximum allowed
    "mean_arrival_rate",   # 1  average passenger arrival rate at the chosen upcoming stops (scaled)
    "max_boardings",       # 2  largest average boardings per visit among the chosen stops (scaled)
    "hops_to_peak",        # 3  how many stops ahead that busiest stop is (0..1)
    "mean_alight_frac",    # 4  average share of passengers who get off at the chosen stops
    "mean_segment_tt",     # 5  average travel time of the chosen road segments (/ 30 min)
    "max_segment_z",       # 6  how much slower than usual the slowest chosen segment is (-1..1)
    "mean_segment_cv",     # 7  how variable the chosen segments' travel times are
    "mean_activity_share", # 8  how busy the chosen stops are at this hour (scaled)
    "any_selected",        # 9  1 if at least one chunk was chosen, else 0
)
D = len(FEATURE_NAMES)
FEATURIZER_VERSION = "pooled-v1"


def compute_norms(chunks: pd.DataFrame) -> Dict[str, float]:
    """Scaling constants (99th percentiles), so the numbers mostly fall between 0 and 1."""
    sh = chunks[chunks["ctype"] == "stop_hour"]
    act = chunks[chunks["ctype"] == "stop_activity"]
    return {
        "p99_arrival_rate": float(max(np.nanpercentile(sh["value_main"], 99), 1e-9)),
        "p99_boardings": float(max(np.nanpercentile(sh["boardings"], 99), 1e-9)),
        "p99_activity_share": float(max(np.nanpercentile(act["value_main"], 99), 1e-9)) if len(act) else 1.0,
    }


def featurize(selected: pd.DataFrame, chunks: pd.DataFrame, all_keys: np.ndarray, norms: Dict[str, float],
              k_max: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Builds the 10 numbers for every key at once.

    selected: the chosen rows (columns qkey, chunk_id, hop, z).
    Returns (keys, numbers) with one row per key in all_keys; keys where nothing was
    chosen get all zeros.
    """
    all_keys = np.asarray(sorted(set(int(k) for k in all_keys)), dtype=np.int64)
    feats = np.zeros((len(all_keys), D), dtype=np.float32)
    if len(selected) == 0:
        return all_keys, feats
    # attach each chosen chunk's values to its row
    c = chunks.loc[selected["chunk_id"].to_numpy()].reset_index(drop=True)
    s = selected.reset_index(drop=True).copy()
    for col in ("ctype", "value_main", "value_aux", "value_std", "boardings"):
        s[col] = c[col].to_numpy()
    pos = {int(k): i for i, k in enumerate(all_keys)}
    s["_row"] = np.array([pos[int(k)] for k in s["qkey"].to_numpy()])  # output row of each chosen chunk

    n = s.groupby("_row").size()
    feats[n.index, 0] = np.minimum(n.to_numpy() / float(k_max), 1.0)
    feats[n.index, 9] = 1.0

    # passenger-demand chunks
    sh = s[s["ctype"] == "stop_hour"]
    if len(sh):
        g = sh.groupby("_row")
        feats[g.size().index, 1] = np.clip(g["value_main"].mean().to_numpy() / norms["p99_arrival_rate"], 0, 1)
        feats[g.size().index, 2] = np.clip(g["boardings"].max().to_numpy() / norms["p99_boardings"], 0, 1)
        peak = sh.loc[g["boardings"].idxmax().to_numpy()]
        feats[peak["_row"].to_numpy(), 3] = np.clip(peak["hop"].to_numpy() / float(horizon), 0, 1)
        feats[g.size().index, 4] = np.clip(g["value_aux"].mean().to_numpy(), 0, 1)

    # travel-time chunks
    seg = s[s["ctype"] == "segment"].copy()
    if len(seg):
        seg["cv"] = np.where(seg["value_main"] > 0, np.nan_to_num(seg["value_std"].astype(float)) / seg["value_main"], 0.0)
        g = seg.groupby("_row")
        feats[g.size().index, 5] = np.clip(g["value_main"].mean().to_numpy() / 1800.0, 0, 1)
        feats[g.size().index, 6] = np.clip(g["z"].max().to_numpy() / 3.0, -1, 1)
        feats[g.size().index, 7] = np.clip(g["cv"].mean().to_numpy(), 0, 1)

    # stop-activity chunks
    act = s[s["ctype"] == "stop_activity"]
    if len(act):
        g = act.groupby("_row")
        feats[g.size().index, 8] = np.clip(g["value_main"].mean().to_numpy() / norms["p99_activity_share"], 0, 1)
    return all_keys, feats
