"""Forecasters for the stops experiment.

Interface (see also EXTENDING.md): implement `StopsForecaster` —
  fit(data)                    once per route, training data only
  predict_day(contexts, day, data) -> {'point': (n_stops, BINS_PER_DAY), ...}
`contexts` is the method-agnostic model input: (n_stops, CONTEXT_DAYS *
BINS_PER_DAY) raw filled values. Optional extra keys: 'p10'/'p90' (quantile
methods) or 'std' (dispersion for methods without quantiles, e.g. naive).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from route_flow import config
from route_flow.data import dataset as trips_dataset
from route_flow.forecasting import timesfm_base
from route_flow.stops.dataset import StopsData

log = logging.getLogger(__name__)

INFER_BATCH = 32  # stop series per TimesFM forward pass


class StopsForecaster(ABC):
    method: str
    provides: tuple[str, ...] = ("point",)
    # Bins produced per forecast day. Subclasses for other resolutions (e.g.
    # the hourly rates experiment) override this; must stay <= 128 so one day
    # fits a single TimesFM forward pass.
    bins_per_day: int = config.BINS_PER_DAY

    @abstractmethod
    def fit(self, data: StopsData) -> None: ...

    @abstractmethod
    def predict_day(self, contexts: np.ndarray, day: pd.Timestamp,
                    data: StopsData) -> dict[str, np.ndarray]: ...

    def close(self) -> None:
        pass


class StopsNaive(StopsForecaster):
    """Day-type per-bin mean, per stop; std reported as dispersion."""

    method = "naive"
    provides = ("point", "std")

    def __init__(self, train_start: str = config.TRAIN_START,
                 train_end: str = config.TRAIN_END):
        self.train_start, self.train_end = train_start, train_end

    def fit(self, data: StopsData) -> None:
        end = pd.Timestamp(self.train_end)
        self._means, self._stds = {}, {}
        for seq in range(1, data.n_stops + 1):
            train = data.stop_matrix(seq).loc[self.train_start: self.train_end]
            keys = [trips_dataset.day_type(d) for d in train.index]
            means = train.groupby(keys).mean().reindex(trips_dataset.DayTypes)
            stds = train.groupby(keys).std().reindex(trips_dataset.DayTypes)
            overall = train.mean(axis=0)
            self._means[seq] = means.apply(
                lambda r: r.fillna(overall), axis=1).fillna(0.0)
            self._stds[seq] = stds.fillna(0.0)

    def predict_day(self, contexts, day, data):
        dt = trips_dataset.day_type(day)
        point = np.stack([self._means[s].loc[dt].to_numpy()
                          for s in range(1, data.n_stops + 1)])
        std = np.stack([self._stds[s].loc[dt].to_numpy()
                        for s in range(1, data.n_stops + 1)])
        return {"point": point, "std": std}


class _TimesFMStopsBase(StopsForecaster):
    provides = ("point", "p10", "p90")

    def _predict_batched(self, contexts: np.ndarray) -> dict[str, np.ndarray]:
        outs = []
        for s in range(0, len(contexts), INFER_BATCH):
            outs.append(timesfm_base.forecast_batch(
                self._model, contexts[s: s + INFER_BATCH],
                self.bins_per_day, self._device))
        return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}

    def predict_day(self, contexts, day, data):
        return self._predict_batched(np.asarray(contexts, dtype=np.float32))

    def close(self) -> None:
        import torch

        self._model = None
        if getattr(self, "_device", None) == "cuda":
            torch.cuda.empty_cache()


class StopsTimesFMZeroShot(_TimesFMStopsBase):
    method = "timesfm"

    def __init__(self, model_path: Path, device: str | None = None):
        self.model_path = Path(model_path)
        self._model, self._device = None, device

    def fit(self, data: StopsData) -> None:
        if self._model is None:
            self._model, self._device = timesfm_base.load_timesfm(
                self.model_path, self._device)


class StopsTimesFMFinetuned(_TimesFMStopsBase):
    """ONE global LoRA adapter shared by every stop of every route — all
    series share the same 20-min quantization, and the prior lab project
    found global adapters beat per-series ones."""

    method = "timesfm-ft"

    def __init__(self, model_path: Path, adapter_path: Path,
                 device: str | None = None):
        self.model_path = Path(model_path)
        self.adapter_path = Path(adapter_path)
        self._model, self._device = None, device

    def fit(self, data: StopsData) -> None:
        if self._model is not None:
            return
        from peft import PeftModel

        if not (self.adapter_path / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"Global stops adapter missing at {self.adapter_path} — the "
                f"runner should have trained it first (see finetune_global).")
        base, self._device = timesfm_base.load_timesfm(
            self.model_path, self._device, dtype="float32")
        self._model = PeftModel.from_pretrained(base, str(self.adapter_path))
        self._model.eval()


def global_adapter_dir(results_dir: Path) -> Path:
    return Path(results_dir) / "adapters_stops" / "global"


def finetune_global(all_data: list[StopsData], model_path: Path,
                    out_dir: Path, ft, device: str | None = None,
                    train_start: str = config.TRAIN_START,
                    train_end: str = config.TRAIN_END,
                    max_windows: int = config.STOPS_MAX_TRAIN_WINDOWS,
                    min_target_coverage: float = 0.2) -> Path:
    """Train ONE LoRA adapter on day-aligned windows sampled across every
    (route, stop) series. Window: CONTEXT_DAYS days -> 1 day (72 bins <= 128).
    Context flattened and trimmed from the oldest end to a multiple of 32."""
    import torch
    from peft import LoraConfig, get_peft_model

    rng = np.random.default_rng(ft.seed)
    days = pd.date_range(train_start, train_end, freq="D")
    ctxs, tgts = [], []
    for data in tqdm(all_data, desc="ft-stops:windows", unit="route"):
        for seq in range(1, data.n_stops + 1):
            mat = data.stop_matrix(seq)
            for i in range(config.CONTEXT_DAYS, len(days)):
                target_day = days[i]
                target_raw = mat.loc[target_day]
                if float(target_raw.notna().mean()) < min_target_coverage:
                    continue
                context = data.filled_history(
                    seq, target_day - pd.Timedelta(days=1), config.CONTEXT_DAYS)
                means, _ = data.day_type_bin_stats(seq, target_day)
                target = trips_dataset.fill_matrix(
                    target_raw.to_frame().T, means)
                ctx = context.to_numpy(dtype=np.float32).reshape(-1)
                c_len = (len(ctx) // config.PATCH_LEN) * config.PATCH_LEN
                ctxs.append(ctx[-c_len:])
                tgts.append(target.to_numpy(dtype=np.float32).reshape(-1))
    if not ctxs:
        raise RuntimeError("No usable stops training windows")
    if len(ctxs) > max_windows:
        keep = rng.choice(len(ctxs), size=max_windows, replace=False)
        ctxs = [ctxs[i] for i in keep]
        tgts = [tgts[i] for i in keep]
    log.info("Global stops finetune: %d windows (ctx len %d, horizon %d)",
             len(ctxs), len(ctxs[0]), len(tgts[0]))

    torch.manual_seed(ft.seed)
    model, device = timesfm_base.load_timesfm(model_path, device, dtype="float32")
    lora = LoraConfig(r=ft.lora_r, lora_alpha=ft.lora_alpha,
                      lora_dropout=ft.lora_dropout, bias="none",
                      target_modules="all-linear")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.train()

    c_train = len(ctxs[0])
    past = torch.tensor(np.stack(ctxs), dtype=torch.float32)
    future = torch.tensor(np.stack(tgts), dtype=torch.float32)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=ft.lr)
    n = len(ctxs)
    for epoch in tqdm(range(ft.epochs), desc="ft-stops:train", unit="epoch"):
        perm = torch.randperm(n)
        losses = []
        for s in range(0, n, ft.batch_size):
            idx = perm[s: s + ft.batch_size]
            out = model(past_values=past[idx].to(device),
                        future_values=future[idx].to(device),
                        forecast_context_len=c_train)
            out.loss.backward()
            optim.step()
            optim.zero_grad()
            losses.append(out.loss.item())
        log.info("stops-ft epoch %d/%d loss=%.4f",
                 epoch + 1, ft.epochs, float(np.mean(losses)))

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    log.info("Global stops adapter -> %s", out_dir)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out_dir
