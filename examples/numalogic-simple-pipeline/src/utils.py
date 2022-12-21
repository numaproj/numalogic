import logging
import os
from dataclasses import dataclass
from typing import Sequence, Union

from dataclasses_json import dataclass_json
from numalogic.models.autoencoder import AutoencoderPipeline
from numalogic.models.autoencoder.base import TorchAE
from numalogic.models.threshold._std import StdDevThreshold
from numalogic.registry import MLflowRegistry
from numalogic.tools.types import ArtifactDict
from numpy.typing import ArrayLike

DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.split(DIR)[0]
TRAIN_DATA_PATH = os.path.join(ROOT_DIR, "src/resources/train_data.csv")
TRACKING_URI = "http://mlflow-service.default.svc.cluster.local:5000"
LOGGER = logging.getLogger(__name__)


@dataclass_json
@dataclass
class Payload:
    ts_data: ArrayLike = None
    anomaly_score: float = 0.0
    uuid: str = None


def save_artifact(
    pl: Union[AutoencoderPipeline, StdDevThreshold],
    skeys: Sequence[str],
    dkeys: Sequence[str],
) -> None:
    if isinstance(pl, TorchAE):
        ml_registry = MLflowRegistry(tracking_uri=TRACKING_URI, artifact_type="pytorch")
    else:
        ml_registry = MLflowRegistry(tracking_uri=TRACKING_URI, artifact_type="sklearn")
    ml_registry.save(skeys=skeys, dkeys=dkeys, artifact=pl)


def load_artifact(skeys: Sequence[str], dkeys: Sequence[str], type: str = None) -> ArtifactDict:
    try:
        if type == "pytorch":
            ml_registry = MLflowRegistry(tracking_uri=TRACKING_URI, artifact_type="pytorch")
        else:
            ml_registry = MLflowRegistry(tracking_uri=TRACKING_URI, artifact_type="sklearn")
        artifact_dict = ml_registry.load(skeys=skeys, dkeys=dkeys)
        return artifact_dict
    except Exception as ex:
        LOGGER.exception("Error while loading artifact from MLFlow database: %s", ex)
        return None
