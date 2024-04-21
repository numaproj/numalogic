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

"""
Module to provide timeseries transformations needed for preprocessing,
feature engineering and postprocessing.
"""

from numalogic.transforms._scaler import TanhScaler, PercentileScaler
from numalogic.transforms._stateless import (
    LogTransformer,
    StaticPowerTransformer,
    DataClipper,
    GaussianNoiseAdder,
    DifferenceTransform,
    FlattenVector,
)
from numalogic.transforms._movavg import ExpMovingAverage, expmov_avg_aggregator
from numalogic.transforms._postprocess import TanhNorm, tanh_norm, SigmoidNorm

__all__ = [
    "TanhScaler",
    "LogTransformer",
    "StaticPowerTransformer",
    "DataClipper",
    "ExpMovingAverage",
    "expmov_avg_aggregator",
    "TanhNorm",
    "tanh_norm",
    "GaussianNoiseAdder",
    "DifferenceTransform",
    "FlattenVector",
    "PercentileScaler",
    "SigmoidNorm"
]
