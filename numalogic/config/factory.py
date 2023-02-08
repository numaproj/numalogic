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


from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

from numalogic.config._config import ModelInfo
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
from numalogic.models.threshold import StdDevThreshold, StaticThreshold
from numalogic.postprocess import TanhNorm
from numalogic.preprocess import LogTransformer, StaticPowerTransformer
from numalogic.tools.exceptions import UnknownConfigArgsError


class _ObjectFactory:
    _CLS_MAP = {}

    def get_instance(self, model_info: ModelInfo):
        try:
            _cls = self._CLS_MAP[model_info.name]
        except KeyError:
            raise UnknownConfigArgsError(f"Invalid model info instance provided: {model_info}")
        return _cls(**model_info.conf)

    def get_cls(self, model_info: ModelInfo):
        try:
            return self._CLS_MAP[model_info.name]
        except KeyError:
            raise UnknownConfigArgsError(f"Invalid model info instance provided: {model_info}")


class PreprocessFactory(_ObjectFactory):
    _CLS_MAP = {
        "StandardScaler": StandardScaler,
        "MinMaxScaler": MinMaxScaler,
        "MaxAbsScaler": MaxAbsScaler,
        "RobustScaler": RobustScaler,
        "LogTransformer": LogTransformer,
        "StaticPowerTransformer": StaticPowerTransformer,
    }


class PostprocessFactory(_ObjectFactory):
    _CLS_MAP = {"TanhNorm": TanhNorm}


class ThresholdFactory(_ObjectFactory):
    _CLS_MAP = {"StdDevThreshold": StdDevThreshold, "StaticThreshold": StaticThreshold}


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
