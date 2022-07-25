from typing import TypeVar, Union, Dict

from mlflow.entities.model_registry import ModelVersion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from torch import nn

Artifact = TypeVar("Artifact", bound=Union[nn.Module, BaseEstimator, TransformerMixin, Pipeline])
ArtifactDict = TypeVar(
    "ArtifactDict",
    bound=Dict[
        str, Union[Union[nn.Module, BaseEstimator, TransformerMixin, Pipeline], Dict, ModelVersion]
    ],
)
AutoencoderModel = TypeVar("AutoencoderModel", bound="TorchAE")
