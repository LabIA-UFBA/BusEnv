"""Error metrics. Computed ONLY where y_true is observed (their instruction).

MAPE is undefined at y_true == 0 (common at route ends); we exclude those
points from MAPE and additionally report WAPE, which handles zeros gracefully.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    mask = y_true.notna() & y_pred.notna()
    t = y_true[mask].to_numpy(dtype=float)
    p = y_pred[mask].to_numpy(dtype=float)
    if len(t) == 0:
        return {"n_points": 0, "mse": np.nan, "rmse": np.nan, "mae": np.nan,
                "mape_pct": np.nan, "wape_pct": np.nan}
    err = p - t
    nonzero = np.abs(t) > 1e-9
    return {
        "n_points": int(len(t)),
        "mse": float(np.mean(err ** 2)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "mape_pct": float(np.mean(np.abs(err[nonzero] / t[nonzero])) * 100)
        if nonzero.any() else np.nan,
        "wape_pct": float(np.sum(np.abs(err)) / np.sum(np.abs(t)) * 100)
        if np.abs(t).sum() > 0 else np.nan,
    }
