"""TimesFM 2.5 zero-shot forecaster: 15-day context in, next day out."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from route_flow.data.dataset import RouteData, flatten
from route_flow.forecasting.base import Forecaster
from route_flow.forecasting import timesfm_base

log = logging.getLogger(__name__)


class TimesFMZeroShot(Forecaster):
    method = "timesfm"

    def __init__(self, model_path: Path, device: str | None = None,
                 model=None, resolved_device: str | None = None):
        # `model` lets the runner share one loaded model across all routes.
        self.model_path = Path(model_path)
        self._model = model
        self._device = resolved_device or device

    def fit(self, data: RouteData) -> None:
        if self._model is None:
            self._model, self._device = timesfm_base.load_timesfm(
                self.model_path, self._device)
        self.n_stops = data.n_stops

    def predict_day(self, context: pd.DataFrame, day: pd.Timestamp) -> np.ndarray:
        return timesfm_base.forecast(
            self._model, flatten(context), self.n_stops, self._device)
