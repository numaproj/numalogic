# Copyright 2022 The Numaproj Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Tuple

import numpy as np
import pandas as pd
from datetime import date

from numpy.typing import NDArray


class SyntheticTSGenerator:
    def __init__(
        self,
        seq_len: int,
        num_series: int,
        freq="T",
        primary_period=1440,
        secondary_period=10080,
        seasonal_ts_prob=0.7,
        baseline_range=(200.0, 350.0),
        slope_range=(-0.001, 0.01),
        amplitude_range=(10, 40),
        cosine_ratio_range=(0.5, 0.9),
        noise_range=(5, 15),
        phase_shift_range: Tuple[int, int] = None,
    ):
        self.seq_len = seq_len
        self.num_series = num_series
        self.dt_index = pd.DatetimeIndex(
            pd.date_range(end=date.today(), periods=seq_len, freq=freq)
        )
        self.time_steps = np.arange(seq_len, dtype="float32")
        self.baseline_range = baseline_range
        self.slope_range = slope_range
        self.amplitude_range = amplitude_range
        self.cos_ratio_range = cosine_ratio_range
        self.noise_range = noise_range
        self.phase_range = phase_shift_range
        self.primary_period = primary_period
        self.secondary_period = secondary_period
        self.seasonal_ts_prob = seasonal_ts_prob

    def gen_tseries(self) -> pd.DataFrame:
        all_series = {}
        is_seasonal = np.random.binomial(1, self.seasonal_ts_prob, self.num_series)

        for s_idx in range(self.num_series):
            if is_seasonal[s_idx]:
                seasonality = self.seasonality(self.primary_period)
                if self.secondary_period:
                    seasonality += self.seasonality(self.secondary_period, amp_reduction_factor=3)
            else:
                seasonality = np.zeros(self.seq_len)
            all_series[f"s{s_idx+1}"] = self.baseline() + self.trend() + self.noise() + seasonality

        return pd.DataFrame(all_series, index=self.dt_index)

    def baseline(self) -> float:
        baseline = np.random.uniform(*self.baseline_range)
        return baseline

    def trend(self) -> NDArray[float]:
        slope = np.random.uniform(*self.slope_range)
        return slope * self.time_steps

    def seasonality(self, period: int, amp_reduction_factor=1) -> NDArray[float]:
        phase = np.random.uniform(*self.phase_range) if self.phase_range else 0
        cosine_ratio = np.random.uniform(*self.cos_ratio_range)
        amplitude = np.random.uniform(*self.amplitude_range) / amp_reduction_factor

        season_time = ((self.time_steps + phase) % period) / period

        seasonal_pattern = np.where(
            season_time < cosine_ratio,
            np.cos(season_time * 2 * np.pi),
            season_time,
        )
        return amplitude * seasonal_pattern

    def noise(self, seed=42) -> NDArray[float]:
        rnd = np.random.RandomState(seed)
        noise_level = np.random.uniform(*self.noise_range)
        return rnd.randn(self.seq_len) * noise_level

    @classmethod
    def train_test_split(
        cls, df: pd.DataFrame, test_size: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return df[:-test_size], df[-test_size:]
