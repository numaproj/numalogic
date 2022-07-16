from typing import TypeVar, Union, Dict

from mlflow.entities.model_registry import ModelVersion
from sklearn.base import BaseEstimator
from torch import nn

Artifact = TypeVar("Artifact", bound=Union[nn.Module, BaseEstimator])
ArtifactDict = TypeVar(
    "ArtifactDict", bound=Dict[str, Union[Union[nn.Module, BaseEstimator], Dict, ModelVersion]]
)
AutoencoderModel = TypeVar("AutoencoderModel", bound="TorchAE")
