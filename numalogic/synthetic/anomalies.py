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


from typing import Sequence, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class AnomalyGenerator:
    __MIN_COLUMNS = {"global": 1, "contextual": 1, "causal": 2, "collective": 2}

    def __init__(self, ref_df: pd.DataFrame, anomaly_type="global", anomaly_ratio=0.1):
        """
        @param ref_df: Reference Multivariate time series DataFrame
        @param anomaly_type: Type of anomaly to impute.
            Possible values include:
            - "global": Outliers in the global context
            - "contextual": Outliers only in the seasonal context
            - "causal": Outliers caused by a temporal causal effect
            - "collective": Outliers present simultaneously in two or more time series
        @param anomaly_ratio: Ratio of anomalous data points to inject wrt
            to number of samples
        """

        self.anomaly_type = anomaly_type
        self.anomaly_ratio = anomaly_ratio
        self.freq = ref_df.index.freq

        self.scaler = StandardScaler()
        scaled_ref_df = pd.DataFrame(
            self.scaler.fit_transform(ref_df.to_numpy()),
            index=ref_df.index,
            columns=ref_df.columns,
        )
        self.ref_stats_df = scaled_ref_df.describe()
        self.__injected_cols = []
        self.block_size = None

    @property
    def injected_cols(self) -> List[str]:
        return self.__injected_cols

    def inject_anomalies(
        self, target_df: pd.DataFrame, cols: Sequence[str] = None, **kwargs
    ) -> pd.DataFrame:
        """
        @param target_df: Target DataFrame where anomalies will be injected
        @param cols: Columns to inject anomalies
        @param kwargs: Optional kwargs for individual anomaly types
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
        self, target_df: pd.DataFrame, cols: Sequence[str] = None, impact=3
    ) -> pd.DataFrame:
        target_df = self._init_target_df(target_df, cols)

        for col in self.__injected_cols:
            tseries = target_df[col]
            sample = tseries[: -self.block_size].sample(1)
            idx_start = sample.index
            idx_end = idx_start + (self.block_size * self.freq)
            outlier_block = tseries[idx_start.values[0] : idx_end.values[0]]

            factor = abs(self.ref_stats_df.loc["max", col] - outlier_block.mean())
            outlier_block += impact * factor * abs(outlier_block)

        return pd.DataFrame(
            self.scaler.inverse_transform(target_df.to_numpy()),
            index=target_df.index,
            columns=target_df.columns,
        )

    def _inject_contextual_anomalies(
        self, target_df: pd.DataFrame, cols: Sequence[str], impact=1
    ) -> pd.DataFrame:
        target_df = self._init_target_df(target_df, cols)

        for col in self.__injected_cols:
            tseries = target_df[col]
            sample = tseries[: -self.block_size].sample(1)
            idx_start = sample.index
            idx_end = idx_start + (self.block_size * self.freq)
            outlier_block = tseries[idx_start.values[0] : idx_end.values[0]]

            dist_from_min = np.linalg.norm(
                outlier_block.to_numpy() - self.ref_stats_df.loc["min", col]
            )
            dist_from_max = np.linalg.norm(
                outlier_block.to_numpy() - self.ref_stats_df.loc["max", col]
            )
            if dist_from_min > dist_from_max:
                factor = abs(self.ref_stats_df.loc["min", col] - outlier_block.mean())
                outlier_block -= impact * factor * abs(outlier_block)
            else:
                factor = abs(outlier_block.mean() - self.ref_stats_df.loc["max", col])
                outlier_block += impact * factor * abs(outlier_block)

        return pd.DataFrame(
            self.scaler.inverse_transform(target_df),
            index=target_df.index,
            columns=target_df.columns,
        )

    def _inject_collective_anomalies(
        self, target_df: pd.DataFrame, cols: Sequence[str], impact=0.8
    ) -> pd.DataFrame:
        target_df = self._init_target_df(target_df, cols)

        sample = target_df[: -self.block_size].sample(1)
        idx_start = sample.index
        idx_end = idx_start + (self.block_size * self.freq)

        for col in self.__injected_cols:
            tseries = target_df[col]
            outlier_block = tseries[idx_start.values[0] : idx_end.values[0]]

            dist_from_min = np.linalg.norm(
                outlier_block.to_numpy() - self.ref_stats_df.loc["min", col]
            )
            dist_from_max = np.linalg.norm(
                outlier_block.to_numpy() - self.ref_stats_df.loc["max", col]
            )
            if dist_from_min > dist_from_max:
                factor = abs(self.ref_stats_df.loc["min", col] - outlier_block.mean())
                outlier_block -= impact * factor * abs(outlier_block)
            else:
                factor = abs(outlier_block.mean() - self.ref_stats_df.loc["max", col])
                outlier_block += impact * factor * abs(outlier_block)

        return pd.DataFrame(
            self.scaler.inverse_transform(target_df),
            index=target_df.index,
            columns=target_df.columns,
        )

    def _inject_causal_anomalies(
        self, target_df: pd.DataFrame, cols: Sequence[str], impact=2, gap_range=(5, 20)
    ) -> pd.DataFrame:
        target_df = self._init_target_df(target_df, cols)

        sample = target_df[: -len(self.__injected_cols) * self.block_size].sample(1)
        idx_start = sample.index

        for col in self.__injected_cols:
            tseries = target_df[col]
            idx_end = idx_start + (self.block_size * self.freq)
            outlier_block = tseries[idx_start.values[0] : idx_end.values[0]]

            if np.random.binomial(1, 0.5):
                factor = abs(self.ref_stats_df.loc["min", col] - outlier_block.mean())
                outlier_block -= impact * factor * abs(outlier_block)
            else:
                factor = abs(outlier_block.mean() - self.ref_stats_df.loc["max", col])
                outlier_block += impact * factor * abs(outlier_block)

            gap = np.random.randint(*gap_range)
            idx_start = idx_end + (gap * self.freq)

        return pd.DataFrame(
            self.scaler.inverse_transform(target_df),
            index=target_df.index,
            columns=target_df.columns,
        )

    def _init_target_df(self, target_df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
        target_df = target_df.copy()
        target_df = pd.DataFrame(
            self.scaler.transform(target_df.to_numpy()),
            index=target_df.index,
            columns=target_df.columns,
        )
        self.block_size = np.ceil(target_df.shape[0] * self.anomaly_ratio).astype(int)
        if not cols:
            cols = np.random.choice(target_df.columns, self.__MIN_COLUMNS[self.anomaly_type])
        self.__injected_cols = cols
        return target_df
