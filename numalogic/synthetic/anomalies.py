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


from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Optional, ClassVar


class AnomalyGenerator:
    """
    Class to inject synthetic anomaly to the input time series based on parameters.

    Args:
    ----
        ref_df: Reference Multivariate time series DataFrame
        anomaly_type: Type of anomaly to impute.
            Possible values include:
                - "global": Outliers in the global context
                - "contextual": Outliers only in the seasonal context
                - "causal": Outliers caused by a temporal causal effect
                - "collective": Outliers present simultaneously in two or more time series
        anomaly_ratio: Ratio of anomalous data points to inject wrt
            to number of samples
        anomaly_sign: Positive or Negative anomaly to be injected
            Possible values include:
                - "positive": higher outlier value injected compared to the current actual value
                - "negative": lower outliers injected compared to the current actual value
        mu: Distributions mean of the Gaussian Noise injected
        sigma: Distributions std of the Gaussian Noise injected
        random_seed: seed for random number generator.
    """

    __MIN_COLUMNS: ClassVar[dict[str, int]] = {
        "global": 1,
        "contextual": 1,
        "causal": 2,
        "collective": 2,
    }

    def __init__(
        self,
        ref_df: pd.DataFrame,
        anomaly_type: str = "global",
        anomaly_ratio: float = 0.1,
        anomaly_sign: str = "positive",
        mu: float = 0.1,
        sigma: float = 0.01,
        random_seed: int = 42,
    ):
        self.anomaly_type = anomaly_type
        self.anomaly_ratio = anomaly_ratio
        self.anomaly_sign = anomaly_sign
        self.freq = ref_df.index.freq
        self.mu, self.sigma = mu, sigma

        self.scaler = StandardScaler()
        scaled_ref_df = pd.DataFrame(
            self.scaler.fit_transform(ref_df.to_numpy()), index=ref_df.index, columns=ref_df.columns
        )
        self.ref_stats_df = scaled_ref_df.describe()
        self.__injected_cols = []
        self.block_size = None
        self._rnd_gen = np.random.default_rng(random_seed)

    @property
    def injected_cols(self) -> list[str]:
        return self.__injected_cols

    def add_impact_sign(self) -> int:
        if self.anomaly_sign == "positive":
            return 1
        if self.anomaly_sign == "negative":
            return -1
        raise ValueError(f"Invalid anomaly sign provided: {self.anomaly_sign}")

    def inject_anomalies(
        self, target_df: pd.DataFrame, cols: Optional[Sequence[str]] = None, **kwargs
    ) -> pd.DataFrame:
        """@param target_df: Target DataFrame where anomalies will be injected
        @param cols: Columns to inject anomalies
        @param kwargs: Optional kwargs for individual anomaly types.
        """
        if self.anomaly_type == "global":
            return self._inject_global_anomalies(target_df, cols, **kwargs)
        if self.anomaly_type == "contextual":
            return self._inject_contextual_anomalies(target_df, cols, **kwargs)
        if self.anomaly_type == "collective":
            return self._inject_collective_anomalies(target_df, cols, **kwargs)
        if self.anomaly_type == "causal":
            return self._inject_causal_anomalies(target_df, cols, **kwargs)
        raise AttributeError(f"Invalid anomaly type provided: {self.anomaly_type}")

    def _inject_global_anomalies(
        self, target_df: pd.DataFrame, cols: Optional[Sequence[str]] = None, impact=3
    ) -> pd.DataFrame:
        target_df = self._init_target_df(target_df, cols)
        anomaly_df = pd.DataFrame(index=target_df.index)
        anomaly_df["is_anomaly"] = 0

        for col in self.__injected_cols:
            tseries = target_df[col]
            sample = tseries[: -self.block_size].sample(1)
            idx_start = sample.index
            idx_end = idx_start + (self.block_size * self.freq)
            outlier_block = tseries[idx_start.values[0] : idx_end.values[0]]
            factor = abs(self.ref_stats_df.loc["max", col] - outlier_block.mean())

            # Add gaussian noise to the data
            noise = self._rnd_gen.normal(self.mu, self.sigma, outlier_block.shape)
            outlier_block += noise + impact * factor * abs(outlier_block) * self.add_impact_sign()

            # Add labels to the data
            anomaly_col = anomaly_df["is_anomaly"]
            anomaly_block = anomaly_col[idx_start.values[0] : idx_end.values[0]]
            anomaly_block += self.add_impact_sign()

        return pd.DataFrame(
            self.scaler.inverse_transform(target_df.to_numpy()),
            index=target_df.index,
            columns=target_df.columns,
        ).merge(anomaly_df, left_index=True, right_index=True)

    def _inject_contextual_anomalies(
        self, target_df: pd.DataFrame, cols: Sequence[str], impact=1
    ) -> pd.DataFrame:
        target_df = self._init_target_df(target_df, cols)
        anomaly_df = pd.DataFrame(index=target_df.index)
        anomaly_df["is_anomaly"] = 0

        for col in self.__injected_cols:
            tseries = target_df[col]
            sample = tseries[: -self.block_size].sample(1)
            idx_start = sample.index
            idx_end = idx_start + (self.block_size * self.freq)
            outlier_block = tseries[idx_start.values[0] : idx_end.values[0]]

            # Add gaussian noise to the data
            noise = self._rnd_gen.normal(self.mu, self.sigma, outlier_block.shape)

            dist_from_min = np.linalg.norm(
                outlier_block.to_numpy() - self.ref_stats_df.loc["min", col]
            )
            dist_from_max = np.linalg.norm(
                outlier_block.to_numpy() - self.ref_stats_df.loc["max", col]
            )

            if dist_from_min > dist_from_max:
                factor = abs(self.ref_stats_df.loc["min", col] - outlier_block.mean())
                outlier_block -= (
                    noise + impact * factor * abs(outlier_block) * self.add_impact_sign()
                )
            else:
                factor = abs(outlier_block.mean() - self.ref_stats_df.loc["max", col])
                outlier_block += (
                    noise + impact * factor * abs(outlier_block) * self.add_impact_sign()
                )

            anomaly_col = anomaly_df["is_anomaly"]
            anomaly_block = anomaly_col[idx_start.values[0] : idx_end.values[0]]
            anomaly_block += self.add_impact_sign()

        return pd.DataFrame(
            self.scaler.inverse_transform(target_df.to_numpy()),
            index=target_df.index,
            columns=target_df.columns,
        ).merge(anomaly_df, left_index=True, right_index=True)

    def _inject_collective_anomalies(
        self, target_df: pd.DataFrame, cols: Sequence[str], impact=0.8
    ) -> pd.DataFrame:
        target_df = self._init_target_df(target_df, cols)
        anomaly_df = pd.DataFrame(index=target_df.index)
        anomaly_df["is_anomaly"] = 0

        sample = target_df[: -self.block_size].sample(1)
        idx_start = sample.index
        idx_end = idx_start + (self.block_size * self.freq)

        for col in self.__injected_cols:
            tseries = target_df[col]
            outlier_block = tseries[idx_start.values[0] : idx_end.values[0]]

            # Add gaussian noise to the data
            noise = self._rnd_gen.normal(self.mu, self.sigma, outlier_block.shape)

            dist_from_min = np.linalg.norm(
                outlier_block.to_numpy() - self.ref_stats_df.loc["min", col]
            )
            dist_from_max = np.linalg.norm(
                outlier_block.to_numpy() - self.ref_stats_df.loc["max", col]
            )
            if dist_from_min > dist_from_max:
                factor = abs(self.ref_stats_df.loc["min", col] - outlier_block.mean())
                outlier_block -= (
                    noise + impact * factor * abs(outlier_block) * self.add_impact_sign()
                )
            else:
                factor = abs(outlier_block.mean() - self.ref_stats_df.loc["max", col])
                outlier_block += (
                    noise + impact * factor * abs(outlier_block) * self.add_impact_sign()
                )
            anomaly_col = anomaly_df["is_anomaly"]
            anomaly_block = anomaly_col[idx_start.values[0] : idx_end.values[0]]
            anomaly_block += self.add_impact_sign()

        return pd.DataFrame(
            self.scaler.inverse_transform(target_df.to_numpy()),
            index=target_df.index,
            columns=target_df.columns,
        ).merge(anomaly_df, left_index=True, right_index=True)

    def _inject_causal_anomalies(
        self, target_df: pd.DataFrame, cols: Sequence[str], impact=2, gap_range=(5, 20)
    ) -> pd.DataFrame:
        target_df = self._init_target_df(target_df, cols)
        anomaly_df = pd.DataFrame(index=target_df.index)
        anomaly_df["is_anomaly"] = 0

        sample = target_df[: -len(self.__injected_cols) * self.block_size].sample(1)
        idx_start = sample.index

        for col in self.__injected_cols:
            tseries = target_df[col]
            idx_end = idx_start + (self.block_size * self.freq)
            outlier_block = tseries[idx_start.values[0] : idx_end.values[0]]
            # Add gaussian noise to the data
            noise = self._rnd_gen.normal(self.mu, self.sigma, outlier_block.shape)

            if self._rnd_gen.binomial(1, 0.5):
                factor = abs(self.ref_stats_df.loc["min", col] - outlier_block.mean())
                outlier_block -= (
                    noise + impact * factor * abs(outlier_block) * self.add_impact_sign()
                )
            else:
                factor = abs(outlier_block.mean() - self.ref_stats_df.loc["max", col])
                outlier_block += (
                    noise + impact * factor * abs(outlier_block) * self.add_impact_sign()
                )

            anomaly_col = anomaly_df["is_anomaly"]
            anomaly_block = anomaly_col[idx_start.values[0] : idx_end.values[0]]
            anomaly_block += self.add_impact_sign()
            gap = self._rnd_gen.integers(*gap_range)
            idx_start = idx_end + (gap * self.freq)

        return pd.DataFrame(
            self.scaler.inverse_transform(target_df.to_numpy()),
            index=target_df.index,
            columns=target_df.columns,
        ).merge(anomaly_df, left_index=True, right_index=True)

    def _init_target_df(self, target_df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
        target_df = target_df.copy()
        target_df = pd.DataFrame(
            self.scaler.transform(target_df.to_numpy()),
            index=target_df.index,
            columns=target_df.columns,
        )
        self.block_size = np.ceil(target_df.shape[0] * self.anomaly_ratio).astype(int)
        if not cols:
            cols = self._rnd_gen.choice(target_df.columns, self.__MIN_COLUMNS[self.anomaly_type])
        self.__injected_cols = cols
        return target_df
