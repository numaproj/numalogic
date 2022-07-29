from typing import Union, Dict, NewType, TypeVar, Sequence, Optional

from mlflow.entities.model_registry import ModelVersion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from torch import nn

Artifact = NewType("Artifact", Union[nn.Module, BaseEstimator, TransformerMixin, Pipeline])
ArtifactDict = NewType(
    "ArtifactDict",
    Optional[
        Dict[str, Union[Sequence[Artifact], Dict[str, Artifact], Artifact, None, ModelVersion]]
    ],
)
AutoencoderModel = TypeVar("AutoencoderModel", bound="TorchAE")
