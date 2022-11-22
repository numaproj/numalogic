import logging
import os
from dataclasses import dataclass
from typing import Sequence

import mlflow
from dataclasses_json import dataclass_json
from numalogic.models.autoencoder import AutoencoderPipeline
from numalogic.registry import MLflowRegistrar
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


def save_model(pl: AutoencoderPipeline, skeys: Sequence[str], dkeys: Sequence[str]) -> None:
    ml_registry = MLflowRegistrar(tracking_uri=TRACKING_URI, artifact_type="pytorch")
    mlflow.start_run()
    ml_registry.save(skeys=skeys, dkeys=dkeys, primary_artifact=pl.model, **pl.model_properties)
    mlflow.end_run()


def load_model(skeys: Sequence[str], dkeys: Sequence[str]) -> ArtifactDict:
    try:
        ml_registry = MLflowRegistrar(tracking_uri=TRACKING_URI)
        artifact_dict = ml_registry.load(skeys=skeys, dkeys=dkeys)
        return artifact_dict
    except Exception as ex:
        LOGGER.exception("Error while loading model from MLFlow database: %s", ex)
        return None
