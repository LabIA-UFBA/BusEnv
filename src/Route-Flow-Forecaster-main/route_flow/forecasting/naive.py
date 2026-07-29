"""Naive day-type baseline.

"Trained" on March: for each pt_sequence, the mean observed loading over
March weekdays / Saturdays / Sundays. Every eval day gets the profile of its
day type. Missing (day_type, stop) means fall back to the stop's overall
March mean, then 0.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from route_flow import config
from route_flow.data import dataset
from route_flow.data.dataset import RouteData
from route_flow.forecasting.base import Forecaster

log = logging.getLogger(__name__)


class NaiveForecaster(Forecaster):
    method = "naive"

    def __init__(self, train_start: str = config.TRAIN_START,
                 train_end: str = config.TRAIN_END):
        self.train_start = train_start
        self.train_end = train_end
        self._profiles: pd.DataFrame | None = None

    def fit(self, data: RouteData) -> None:
        train = data.raw_slice(self.train_start, self.train_end)
        means = dataset.day_type_stop_means(train)
        overall = train.mean(axis=0)
        self._profiles = means.apply(
            lambda row: row.fillna(overall), axis=1).fillna(0.0)
        for dt in dataset.DayTypes:
            n = train.groupby([dataset.day_type(d) for d in train.index]).size()
            log.debug("%s: %s days in train window: %s", data.spec.trip_label,
                      dt, int(n.get(dt, 0)))

    def predict_day(self, context: pd.DataFrame, day: pd.Timestamp) -> np.ndarray:
        assert self._profiles is not None, "fit() not called"
        return self._profiles.loc[dataset.day_type(day)].to_numpy()
