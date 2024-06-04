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


import numpy as np
import pandas as pd
from datetime import date

from numpy.typing import NDArray
from typing import Optional


class SyntheticTSGenerator:
    """
    Generates synthetic time series data.

    Args:
    ----
        seq_len: int, length of the time series.
        num_series: int, number of time series to generate.
        freq: str, frequency of the time series.
        primary_period: int, period of the primary trend.
        secondary_period: int, period of the secondary trend.
        seasonal_ts_prob: float, probability of generating a seasonal time series.
        baseline_range: tuple[float, float], range of the baseline values.
        slope_range: tuple[float, float], range of the slope values.
        amplitude_range: tuple[float, float], range of the amplitude values.
        cosine_ratio_range: tuple[float, float], range of the cosine ratio values.
        noise_range: tuple[float, float], range of the noise values.
        phase_shift_range: tuple[int, int], range of the phase shift values.
        random_seed: int, random seed for reproducibility.
    """

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
        phase_shift_range: Optional[tuple[int, int]] = None,
        random_seed: int = 42,
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
        self.seed = random_seed
        self._rnd_gen = np.random.default_rng(random_seed)

    def gen_tseries(self) -> pd.DataFrame:
        all_series = {}
        is_seasonal = self._rnd_gen.binomial(1, self.seasonal_ts_prob, self.num_series)

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
        return self._rnd_gen.uniform(*self.baseline_range)

    def trend(self) -> NDArray[float]:
        slope = self._rnd_gen.uniform(*self.slope_range)
        return slope * self.time_steps

    def seasonality(self, period: int, amp_reduction_factor=1) -> NDArray[float]:
        phase = self._rnd_gen.uniform(*self.phase_range) if self.phase_range else 0
        cosine_ratio = self._rnd_gen.uniform(*self.cos_ratio_range)
        amplitude = self._rnd_gen.uniform(*self.amplitude_range) / amp_reduction_factor

        season_time = ((self.time_steps + phase) % period) / period

        seasonal_pattern = np.where(
            season_time < cosine_ratio, np.cos(season_time * 2 * np.pi), season_time
        )
        return amplitude * seasonal_pattern

    def noise(self) -> NDArray[float]:
        noise_level = self._rnd_gen.uniform(*self.noise_range)
        return self._rnd_gen.standard_normal(self.seq_len) * noise_level

    @classmethod
    def train_test_split(
        cls, df: pd.DataFrame, test_size: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return df[:-test_size], df[-test_size:]
