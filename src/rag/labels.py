"""Training labels: which candidate chunks are "useful" to show the bus. No LLM is used.

Rule ("deviation coverage"): a chunk is interesting when its value is unusual for the
route (for example, a stop with much more demand than normal at that hour, or a segment
much slower than normal), and more so when it is close. So each candidate gets a weight:

    weight = |z| / sqrt(stops ahead)        (z = how unusual the value is)

Candidates with |z| below `z_min` get weight 0. The highest-weight candidates are marked
useful until they add up to `coverage` (60%) of the key's total weight.
n_support = how many chunks were marked useful for that key (at least 1).
A key can end up with no useful chunk, meaning "nothing unusual ahead".

MA-Chunk is trained to find these chunks. Note that the "salience" scorer measures |z|
too, so it agrees strongly with these labels; ablation A1 removes it to check that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def l1_labels(pools: pd.DataFrame, coverage: float = 0.6, z_min: float = 0.5) -> pd.DataFrame:
    p = pools.copy()
    w = np.abs(p["z"].to_numpy(dtype=float)) / np.sqrt(p["hop"].to_numpy(dtype=float))
    w[np.abs(p["z"].to_numpy(dtype=float)) < z_min] = 0.0
    p["_w"] = w
    useful = np.zeros(len(p), dtype=np.int64)
    n_support = np.ones(len(p), dtype=np.int64)
    for _, g in p.groupby("qkey", sort=False):
        wg = g["_w"].to_numpy()
        tot = wg.sum()
        if tot <= 0:
            continue
        order = np.argsort(-wg, kind="stable")
        cum = np.cumsum(wg[order])
        k = int(np.searchsorted(cum, coverage * tot) + 1)  # smallest top-k reaching the coverage
        chosen = g.index.to_numpy()[order[:k]]
        useful[chosen] = 1
        n_support[g.index.to_numpy()] = max(1, k)
    p["useful"] = useful
    p["n_support"] = n_support
    return p.drop(columns=["_w"])
