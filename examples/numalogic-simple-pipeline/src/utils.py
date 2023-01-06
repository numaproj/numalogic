import logging
import os
from dataclasses import dataclass
from typing import Sequence

from dataclasses_json import dataclass_json
from numalogic.models.autoencoder.base import BaseAE
from numalogic.registry import MLflowRegistry
from numalogic.tools.types import ArtifactDict
from numpy.typing import ArrayLike

DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.split(DIR)[0]
TRAIN_DATA_PATH = os.path.join(ROOT_DIR, "src/resources/train_data.csv")
TRACKING_URI = "http://mlflow-service.default.svc.cluster.local:5000"
LOGGER = logging.getLogger(__name__)


@dataclass_json
@dataclass(slots=True)
class Payload:
    ts_data: ArrayLike = None
    anomaly_score: float = 0.0
    uuid: str = None
    is_artifact_valid: bool = True


def save_artifact(
    artifact,
    skeys: Sequence[str],
    dkeys: Sequence[str],
) -> None:
    if isinstance(artifact, BaseAE):
        ml_registry = MLflowRegistry(tracking_uri=TRACKING_URI, artifact_type="pytorch")
    else:
        ml_registry = MLflowRegistry(tracking_uri=TRACKING_URI, artifact_type="sklearn")
    ml_registry.save(skeys=skeys, dkeys=dkeys, artifact=artifact)


def load_artifact(skeys: Sequence[str], dkeys: Sequence[str], type_: str = None) -> ArtifactDict:
    try:
        if type_ == "pytorch":
            ml_registry = MLflowRegistry(tracking_uri=TRACKING_URI, artifact_type="pytorch")
        else:
            ml_registry = MLflowRegistry(tracking_uri=TRACKING_URI, artifact_type="sklearn")
        artifact_dict = ml_registry.load(skeys=skeys, dkeys=dkeys)
        return artifact_dict
    except Exception as ex:
        LOGGER.exception("Error while loading artifact from MLFlow database: %s", ex)
        return None
