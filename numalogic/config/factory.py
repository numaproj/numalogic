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
from typing import Union

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

from numalogic.config._config import ModelInfo, RegistryInfo
from numalogic.models.autoencoder.variants import (
    VanillaAE,
    SparseVanillaAE,
    Conv1dAE,
    SparseConv1dAE,
    LSTMAE,
    SparseLSTMAE,
    TransformerAE,
    SparseTransformerAE,
)
from numalogic.models.threshold import StdDevThreshold, StaticThreshold, SigmoidThreshold
from numalogic.postprocess import TanhNorm, ExpMovingAverage
from numalogic.preprocess import LogTransformer, StaticPowerTransformer, TanhScaler
from numalogic.registry import MLflowRegistry, RedisRegistry
from numalogic.tools.exceptions import UnknownConfigArgsError


class _ObjectFactory:
    _CLS_MAP = {}

    def get_instance(self, object_info: Union[ModelInfo, RegistryInfo]):
        try:
            _cls = self._CLS_MAP[object_info.name]
        except KeyError as err:
            raise UnknownConfigArgsError(
                f"Invalid model info instance provided: {object_info}"
            ) from err
        return _cls(**object_info.conf)

    def get_cls(self, object_info: Union[ModelInfo, RegistryInfo]):
        try:
            return self._CLS_MAP[object_info.name]
        except KeyError as err:
            raise UnknownConfigArgsError(
                f"Invalid model info instance provided: {object_info}"
            ) from err


class PreprocessFactory(_ObjectFactory):
    _CLS_MAP = {
        "StandardScaler": StandardScaler,
        "MinMaxScaler": MinMaxScaler,
        "MaxAbsScaler": MaxAbsScaler,
        "RobustScaler": RobustScaler,
        "LogTransformer": LogTransformer,
        "StaticPowerTransformer": StaticPowerTransformer,
        "TanhScaler": TanhScaler,
    }


class PostprocessFactory(_ObjectFactory):
    _CLS_MAP = {"TanhNorm": TanhNorm, "ExpMovingAverage": ExpMovingAverage}


class ThresholdFactory(_ObjectFactory):
    _CLS_MAP = {
        "StdDevThreshold": StdDevThreshold,
        "StaticThreshold": StaticThreshold,
        "SigmoidThreshold": SigmoidThreshold,
    }


class ModelFactory(_ObjectFactory):
    _CLS_MAP = {
        "VanillaAE": VanillaAE,
        "SparseVanillaAE": SparseVanillaAE,
        "Conv1dAE": Conv1dAE,
        "SparseConv1dAE": SparseConv1dAE,
        "LSTMAE": LSTMAE,
        "SparseLSTMAE": SparseLSTMAE,
        "TransformerAE": TransformerAE,
        "SparseTransformerAE": SparseTransformerAE,
    }


class RegistryFactory(_ObjectFactory):
    _CLS_MAP = {
        "RedisRegistry": RedisRegistry,
        "MLflowRegistry": MLflowRegistry,
    }
