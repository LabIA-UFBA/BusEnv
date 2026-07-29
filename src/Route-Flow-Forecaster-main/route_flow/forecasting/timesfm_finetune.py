"""TimesFM 2.5 + LoRA, one adapter per route.

Training data: March windows of (CONTEXT_DAYS days context -> 1 day horizon),
day-aligned, stride 1 day. Hard constraints honored (see the finetuning
guide): training context is trimmed from the OLDEST end to a multiple of 32
and passed as `forecast_context_len`; horizon per window is n_stops <= 128;
raw counts in, loss computed by the model itself in normalized space.

Windows whose horizon day has too little observed data are skipped (we refuse
to train the model to reproduce our own NaN-filling).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from route_flow import config
from route_flow.data import dataset
from route_flow.data.dataset import RouteData, flatten
from route_flow.forecasting.base import Forecaster
from route_flow.forecasting import timesfm_base

log = logging.getLogger(__name__)


class FinetuneConfig:
    def __init__(self, epochs: int = 20, batch_size: int = 4, lr: float = 1e-4,
                 lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05,
                 min_target_coverage: float = 0.5, seed: int = 42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.min_target_coverage = min_target_coverage
        self.seed = seed


def adapter_dir(results_dir: Path, label: str) -> Path:
    return Path(results_dir) / "adapters" / label


def _training_windows(data: RouteData, ft: FinetuneConfig,
                      train_start: str, train_end: str):
    """Yield (context_1d, target_1d) pairs, context trimmed to multiple of 32."""
    raw = data.raw
    days = pd.date_range(train_start, train_end, freq="D")
    n = data.n_stops
    windows = []
    for i in range(config.CONTEXT_DAYS, len(days)):
        target_day = days[i]
        target_raw = raw.loc[target_day]
        coverage = float(target_raw.notna().mean())
        if coverage < ft.min_target_coverage:
            log.debug("skip window ending %s (target coverage %.2f)",
                      target_day.date(), coverage)
            continue
        context = data.filled_history(target_day - pd.Timedelta(days=1),
                                      config.CONTEXT_DAYS)
        hist_means = dataset.day_type_stop_means(raw.loc[:target_day])
        target = dataset.fill_matrix(target_raw.to_frame().T, hist_means)
        ctx = flatten(context)
        c_train = (len(ctx) // config.PATCH_LEN) * config.PATCH_LEN
        windows.append((ctx[-c_train:], flatten(target)[:n]))
    log.info("%s: %d training windows (of %d candidate days)",
             data.spec.trip_label, len(windows),
             max(0, len(days) - config.CONTEXT_DAYS))
    return windows


def finetune_route(data: RouteData, model_path: Path, out_dir: Path,
                   ft: FinetuneConfig, device: str | None = None,
                   train_start: str = config.TRAIN_START,
                   train_end: str = config.TRAIN_END) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model

    if data.n_stops > config.MAX_FORWARD_HORIZON:
        raise ValueError(
            f"{data.spec.trip_label}: horizon {data.n_stops} > "
            f"{config.MAX_FORWARD_HORIZON}; per-window horizon must fit one "
            f"forward pass — split the day into chunks before finetuning.")

    windows = _training_windows(data, ft, train_start, train_end)
    if not windows:
        raise RuntimeError(f"No usable training windows for {data.spec.trip_label}")

    torch.manual_seed(ft.seed)
    # float32 for training stability with LoRA
    model, device = timesfm_base.load_timesfm(model_path, device, dtype="float32")
    lora = LoraConfig(r=ft.lora_r, lora_alpha=ft.lora_alpha,
                      lora_dropout=ft.lora_dropout, bias="none",
                      target_modules="all-linear")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.train()

    # All contexts in a route share one length (same trim of the same window
    # shape), so we can batch them as a single tensor.
    c_train = len(windows[0][0])
    assert all(len(c) == c_train for c, _ in windows)
    past = torch.tensor(np.stack([c for c, _ in windows]), dtype=torch.float32)
    future = torch.tensor(np.stack([t for _, t in windows]), dtype=torch.float32)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=ft.lr)
    n_win = len(windows)
    for epoch in tqdm(range(ft.epochs), desc=f"finetune:{data.spec.trip_label}",
                      unit="epoch"):
        perm = torch.randperm(n_win)
        losses = []
        for s in range(0, n_win, ft.batch_size):
            idx = perm[s: s + ft.batch_size]
            out = model(
                past_values=past[idx].to(device),
                future_values=future[idx].to(device),
                forecast_context_len=c_train,
            )
            out.loss.backward()
            optim.step()
            optim.zero_grad()
            losses.append(out.loss.item())
        log.info("%s epoch %d/%d loss=%.4f", data.spec.trip_label,
                 epoch + 1, ft.epochs, float(np.mean(losses)))

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    log.info("%s: adapter saved -> %s", data.spec.trip_label, out_dir)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out_dir


class TimesFMFinetuned(Forecaster):
    method = "timesfm-ft"

    def __init__(self, model_path: Path, results_dir: Path,
                 ft: FinetuneConfig | None = None, device: str | None = None,
                 refit: bool = False):
        self.model_path = Path(model_path)
        self.results_dir = Path(results_dir)
        self.ft = ft or FinetuneConfig()
        self._device = device
        self.refit = refit
        self._model = None

    def fit(self, data: RouteData) -> None:
        from peft import PeftModel

        self.n_stops = data.n_stops
        adir = adapter_dir(self.results_dir, data.spec.trip_label)
        if self.refit or not (adir / "adapter_config.json").exists():
            finetune_route(data, self.model_path, adir, self.ft, self._device)
        else:
            log.info("%s: reusing existing adapter %s (use --refit to retrain)",
                     data.spec.trip_label, adir)
        base, self._device = timesfm_base.load_timesfm(
            self.model_path, self._device, dtype="float32")
        self._model = PeftModel.from_pretrained(base, str(adir))
        self._model.eval()

    def predict_day(self, context: pd.DataFrame, day: pd.Timestamp) -> np.ndarray:
        return timesfm_base.forecast(
            self._model, flatten(context), self.n_stops, self._device)

    def close(self) -> None:
        import torch

        self._model = None
        if self._device == "cuda":
            torch.cuda.empty_cache()
