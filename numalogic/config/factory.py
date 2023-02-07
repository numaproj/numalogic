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
from numalogic.models.threshold import StdDevThreshold
from numalogic.postprocess import TanhNorm
from numalogic.preprocess import LogTransformer, StaticPowerTransformer


class _ObjectFactory:
    _CLS_MAP = {}

    def get_model_instance(self, model_info: ModelInfo):
        try:
            _cls = self._CLS_MAP[model_info.name]
        except KeyError:
            raise RuntimeError(f"Invalid model info instance provided: {model_info}")
        return _cls(**model_info.conf)

    def get_model_cls(self, model_info: ModelInfo):
        try:
            return self._CLS_MAP[model_info.name]
        except KeyError:
            raise RuntimeError(f"Invalid model info instance provided: {model_info}")


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
    _CLS_MAP = {"StdDevThreshold": StdDevThreshold}


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
