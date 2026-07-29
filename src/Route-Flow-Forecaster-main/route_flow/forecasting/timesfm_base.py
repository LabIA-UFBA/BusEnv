"""Shared TimesFM 2.5 utilities: offline model loading + chunked forecasting.

All the hard constraints here come from TIMESFM_2.5_FINETUNING_GUIDE.md:
- weights must be the `google/timesfm-2.5-200m-transformers` HF-format repo;
- inputs are RAW counts (the model applies RevIN internally);
- one forward yields <= config.horizon_length (128) steps -> chunk if needed,
  feeding the median (decode_index column) back as context;
- for inference do NOT pass `forecast_context_len` (the model front-pads the
  context to a multiple of 32 itself);
- ONE context policy everywhere: we always feed the full rolling context
  (CONTEXT_DAYS * n_stops <= 1140 steps, far below the 16384 max).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def set_offline_env() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def resolve_device(device: str | None) -> str:
    import torch

    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_timesfm(model_path: Path, device: str | None = None,
                 dtype: str = "bfloat16"):
    """Load the base model fully offline. Returns (model, device)."""
    set_offline_env()
    import torch
    from transformers import AutoModelForTimeSeriesPrediction

    device = resolve_device(device)
    torch_dtype = getattr(torch, dtype)
    if device == "cpu" and torch_dtype is torch.bfloat16:
        torch_dtype = torch.float32
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"TimesFM weights not found at {model_path}. Download "
            f"`google/timesfm-2.5-200m-transformers` on an internet-connected "
            f"machine (hf download google/timesfm-2.5-200m-transformers "
            f"--local-dir models/timesfm-hf) and copy it there."
        )
    log.info("Loading TimesFM 2.5 from %s (device=%s, dtype=%s)",
             model_path, device, torch_dtype)
    model = AutoModelForTimeSeriesPrediction.from_pretrained(
        str(model_path), dtype=torch_dtype, local_files_only=True,
    ).to(device)
    model.eval()
    return model, device


def forecast_batch(model, contexts: np.ndarray, horizon: int,
                   device: str) -> dict[str, np.ndarray]:
    """Batched autoregressive forecast with quantiles.

    contexts: (B, L) raw values, all series sharing one context length.
    Returns {'point': (B, horizon), 'p10': ..., 'p90': ...} — point is the
    median (decode_index column); the median is fed back as context between
    chunks (never a non-median quantile).
    """
    import torch

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    chunk = base.config.horizon_length  # 128
    decode_idx = base.config.decode_index
    qs = list(base.config.quantiles)  # [0.1, ..., 0.9]
    col_p10 = 1 + qs.index(0.1)
    col_p90 = 1 + qs.index(0.9)

    rolling = np.asarray(contexts, dtype=np.float32).copy()
    out = {"point": [], "p10": [], "p90": []}
    remaining = horizon
    with torch.no_grad():
        while remaining > 0:
            ctx_t = torch.tensor(rolling, dtype=torch.float32, device=device)
            res = model(past_values=ctx_t)
            full = res.full_predictions.float().cpu().numpy()  # (B, <=128, 10)
            step = min(chunk, remaining)
            out["point"].append(full[:, :step, decode_idx])
            out["p10"].append(full[:, :step, col_p10])
            out["p90"].append(full[:, :step, col_p90])
            rolling = np.concatenate(
                [rolling, full[:, :step, decode_idx].astype(np.float32)], axis=1)
            remaining -= step
    return {k: np.concatenate(v, axis=1)[:, :horizon] for k, v in out.items()}


def forecast(model, context_1d: np.ndarray, horizon: int, device: str) -> np.ndarray:
    """Autoregressive point forecast of `horizon` steps from raw context."""
    import torch

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    chunk = base.config.horizon_length  # 128
    decode_idx = base.config.decode_index
    rolling = np.asarray(context_1d, dtype=np.float32).copy()
    out: list[np.ndarray] = []
    remaining = horizon
    with torch.no_grad():
        while remaining > 0:
            ctx_t = torch.tensor(rolling, dtype=torch.float32,
                                 device=device).unsqueeze(0)
            res = model(past_values=ctx_t)
            full = res.full_predictions[0].float().cpu().numpy()  # (<=128, 10)
            median = full[:, decode_idx]
            step = min(chunk, remaining)
            out.append(median[:step])
            rolling = np.concatenate([rolling, median[:step].astype(np.float32)])
            remaining -= step
    return np.concatenate(out)[:horizon]
